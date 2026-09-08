#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom — ELECTRA tabanlı Türkçe deyim (VID) / eşdizim (LVC.full) span etiketleyici.

Token sınıflandırma: tek BIO head (O, B/I-VID, B/I-LVC), paylaşılan
`dbmdz/electra-base-turkish-cased-discriminator` gövdesi (diğer DizgeBERT modelleriyle aynı →
ortak subword sözlüğü). DizgeBERT-Morph'un aksine tek kaynak (PARSEME-TR) → treebank/şema
embedding'i yok, tek head yeterli.

Kullanım:
    python fetch_parseme_tr.py && python prepare_idiom_data.py && python prepare_idiom_data.py --build-label-space
    python train_idiom_bert.py --epochs 1              # smoke
    python train_idiom_bert.py --class-weights         # tam eğitim (O sınıfı baskın)
    python train_idiom_bert.py --eval --checkpoint idiom_data/best_idiom_tagger.pt
    python train_idiom_bert.py --export-hf dizgebert_idiom_hf/
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from dizgebert_idiom.modeling_dizgebert_idiom import decode_bigappy_spans, viterbi_decode

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "idiom_data"
LABEL_SPACE_PATH = DATA_DIR / "label_space.json"

ENCODER_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
DROPOUT = 0.15
IGN = -100


# ─────────────────────────────────────────────────────────────────────────────
#  Etiket uzayı
# ─────────────────────────────────────────────────────────────────────────────
class IdiomLabelSpace:
    def __init__(self, d: dict):
        self.encoder_model = d.get("encoder_model", ENCODER_MODEL)
        self.tags = d["tags"]
        self.tag_to_id = {t: i for i, t in enumerate(self.tags)}
        # bigappy-unicrossy 2. katman (yalnız gap'li span'lerin 2. parçası) — eski
        # (tek-katman) label_space.json'larla geriye dönük uyum için varsayılan ["o"].
        self.tags2 = d.get("tags2", ["o"])
        self.tag2_to_id = {t: i for i, t in enumerate(self.tags2)}

    @classmethod
    def load(cls, path: Path = LABEL_SPACE_PATH) -> "IdiomLabelSpace":
        if not path.exists():
            sys.exit(f"{path} yok — önce: python prepare_idiom_data.py --build-label-space")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def as_dict(self) -> dict:
        return {"encoder_model": self.encoder_model, "tags": self.tags, "tags2": self.tags2}


# ─────────────────────────────────────────────────────────────────────────────
#  Veri
# ─────────────────────────────────────────────────────────────────────────────
class IdiomDataset(Dataset):
    """Tokenization + hizalama __init__'te bir kez hesaplanır (her epoch değil).

    Kelime temsili = ilk subword ⊕ son subword. Etiketler KELİME düzeyinde tutulur.
    """

    def __init__(self, source, tokenizer, ls: IdiomLabelSpace, max_len: int = MAX_LEN):
        if isinstance(source, (str, Path)):
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            data = list(source)
        self.items: list[dict] = []
        for rec in data:
            enc = tokenizer(rec["words"], is_split_into_words=True,
                            truncation=True, max_length=max_len)
            word_ids = enc.word_ids()
            first: dict[int, int] = {}
            last: dict[int, int] = {}
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                first.setdefault(wid, i)
                last[wid] = i
            kept = sorted(first)  # truncation trailing kelimeleri düşürebilir
            tags2_src = rec.get("tags2", ["o"] * len(rec["words"]))  # eski/TDK kayıtları katman2 taşımaz
            self.items.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "first_pos": [first[w] for w in kept],
                "last_pos": [last[w] for w in kept],
                "tags": [ls.tag_to_id.get(rec["tags"][w], 0) for w in kept],
                "tags2": [ls.tag2_to_id.get(tags2_src[w], 0) for w in kept],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        maxL = max(len(b["input_ids"]) for b in batch)
        maxW = max(len(b["first_pos"]) for b in batch)

        def padL(seq, fill):
            return seq + [fill] * (maxL - len(seq))

        def padW(seq, fill):
            return seq + [fill] * (maxW - len(seq))

        return {
            "input_ids": torch.tensor([padL(b["input_ids"], pad_id) for b in batch]),
            "attention_mask": torch.tensor([padL(b["attention_mask"], 0) for b in batch]),
            "first_pos": torch.tensor([padW(b["first_pos"], 0) for b in batch]),
            "last_pos": torch.tensor([padW(b["last_pos"], 0) for b in batch]),
            "tags": torch.tensor([padW(b["tags"], IGN) for b in batch]),
            "tags2": torch.tensor([padW(b["tags2"], IGN) for b in batch]),
        }

    return collate


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
def _pool_first_last(hs, first_pos, last_pos):
    """[B,L,H] + [B,W] indeksler → [B,W,2H] (ilk ⊕ son subword)."""
    idx_f = first_pos.unsqueeze(-1).expand(-1, -1, hs.size(-1))
    idx_l = last_pos.unsqueeze(-1).expand(-1, -1, hs.size(-1))
    return torch.cat([hs.gather(1, idx_f), hs.gather(1, idx_l)], dim=-1)


class IdiomTagger(nn.Module):
    """`dizgebert_idiom.modeling_dizgebert_idiom.DizgeBertIdiomForTokenClassification` ile
    BİREBİR aynı katman adları (`encoder`, `dropout`, `tag_head`, `tag_head2`) → checkpoint
    doğrudan HF modeline yüklenir. `tag_head2` = bigappy-unicrossy 2. katman (yalnız gap'li
    span'lerin 2. parçası, bkz. prepare_idiom_data.py)."""

    def __init__(self, ls: IdiomLabelSpace, encoder_model: str = ENCODER_MODEL):
        super().__init__()
        self.ls = ls
        self.encoder = AutoModel.from_pretrained(encoder_model)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.tag_head = nn.Linear(2 * h, len(ls.tags))
        self.tag_head2 = nn.Linear(2 * h, len(ls.tags2))

    def forward(self, input_ids, attention_mask, first_pos, last_pos):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        w = _pool_first_last(out.last_hidden_state, first_pos, last_pos)  # [B,W,2H]
        z = self.dropout(w)
        return {"tags": self.tag_head(z), "tags2": self.tag_head2(z)}


def _class_weights(counts: np.ndarray, device) -> torch.Tensor:
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    w = np.clip(w, 0.3, 6.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


def build_class_weights(train_jsons: list[Path], ls: IdiomLabelSpace, device,
                        span_mult: float = 1.0) -> torch.Tensor:
    """Ters-karekök frekans ağırlıkları (ortalama 1, [0.3, 6] arası). BIO'da O sınıfı
    baskın (idiom span'ler seyrek) — standart NER dengesizlik lehine ağırlıklandırma.
    `train_jsons` FİİLEN eğitime giren TÜM kaynakları içermeli (yalnız train.json değil —
    `--tdk-examples` karışımı ağırlıkları etkiler, kod incelemesinde bulundu).
    span_mult > 1: B/I-* sınıflarının ağırlığını arttırır → recall↑ (iki-aşama boru hattında
    stage-2 precision'ı geri kazandığı için stage-1'i recall'a ayarlamak mantıklı)."""
    counts = np.zeros(len(ls.tags))
    for train_json in train_jsons:
        data = json.loads(train_json.read_text(encoding="utf-8"))
        for rec in data:
            for t in rec["tags"]:
                counts[ls.tag_to_id.get(t, 0)] += 1
    w = _class_weights(counts, device)
    if span_mult != 1.0:
        w[1:] *= span_mult          # O (index 0) hariç tüm B/I-* sınıfları
        w = torch.clamp(w, 0.3, 8.0)
    return w


def build_class_weights2(train_jsons: list[Path], ls: IdiomLabelSpace, device) -> torch.Tensor:
    """Katman 2 için aynı ağırlıklandırma — dengesizlik katman 1'den çok daha aşırı (~326k
    tokenden ~300'ü non-'o') → `--class-weights` bayrağından BAĞIMSIZ her zaman uygulanır,
    aksi halde ağ muhtemelen hep 'o' tahmin etmeyi öğrenir (gradyan sinyali neredeyse sıfır)."""
    counts = np.zeros(len(ls.tags2))
    for train_json in train_jsons:
        data = json.loads(train_json.read_text(encoding="utf-8"))
        for rec in data:
            for t in rec.get("tags2", []):
                counts[ls.tag2_to_id.get(t, 0)] += 1
    return _class_weights(counts, device)


def compute_loss(logits: dict, batch: dict, weights: torch.Tensor | None = None,
                 weights2: torch.Tensor | None = None) -> torch.Tensor:
    lg1 = logits["tags"]
    l1 = F.cross_entropy(lg1.reshape(-1, lg1.size(-1)), batch["tags"].reshape(-1),
                         ignore_index=IGN, weight=weights)
    lg2 = logits["tags2"]
    l2 = F.cross_entropy(lg2.reshape(-1, lg2.size(-1)), batch["tags2"].reshape(-1),
                         ignore_index=IGN, weight=weights2)
    return l1 + l2


# ─────────────────────────────────────────────────────────────────────────────
#  Eğitim / değerlendirme
# ─────────────────────────────────────────────────────────────────────────────
def move(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(model, loader, optimizer, scheduler, device, weights=None, weights2=None) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train"):
        batch = move(batch, device)
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"],
                       batch["first_pos"], batch["last_pos"])
        loss = compute_loss(logits, batch, weights, weights2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, ls: IdiomLabelSpace) -> dict:
    """Span-düzeyi (exact-match) P/R/F1, kategori başına + GAPLI (bigappy 2-katman
    birleşimiyle kurtarılan süreksiz span'ler, ayrı satırda) + ALL + token-düzeyi doğruluk."""
    model.eval()
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    correct_tok = 0
    total_tok = 0

    for batch in tqdm(loader, desc="eval"):
        b = move(batch, device)
        out = model(b["input_ids"], b["attention_mask"], b["first_pos"], b["last_pos"])
        logits1 = out["tags"].cpu()
        logits2 = out["tags2"].cpu()
        gold1, gold2 = batch["tags"], batch["tags2"]
        B, W = gold1.shape
        for bi in range(B):
            idxs = [wi for wi in range(W) if gold1[bi, wi].item() != IGN]
            g_tags1 = [ls.tags[gold1[bi, wi].item()] for wi in idxs]
            g_tags2 = [ls.tags2[gold2[bi, wi].item()] for wi in idxs]
            # Viterbi: yalnız gerçek (dolgu olmayan) kelimelerin logit'leri üzerinde,
            # yapısal olarak geçerli (yetim I-X / kategori-karışık geçiş yok) en-iyi-yol.
            p_tags1 = viterbi_decode(logits1[bi, idxs, :], ls.tags)
            p_tags2 = viterbi_decode(logits2[bi, idxs, :], ls.tags2)
            for gt, pt in zip(g_tags1, p_tags1):
                total_tok += 1
                correct_tok += int(gt == pt)
            g_spans = set(decode_bigappy_spans(g_tags1, g_tags2))
            p_spans = set(decode_bigappy_spans(p_tags1, p_tags2))
            for s in g_spans & p_spans:
                tp[s[-1]] += 1; tp["ALL"] += 1
                if len(s) == 5:
                    tp["GAPLI"] += 1
            for s in p_spans - g_spans:
                fp[s[-1]] += 1; fp["ALL"] += 1
                if len(s) == 5:
                    fp["GAPLI"] += 1
            for s in g_spans - p_spans:
                fn[s[-1]] += 1; fn["ALL"] += 1
                if len(s) == 5:
                    fn["GAPLI"] += 1

    def f1(c):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    real_cats = sorted((set(tp) | set(fp) | set(fn)) - {"ALL", "GAPLI"})
    cats = real_cats + ["GAPLI", "ALL"]
    res = {}
    for c in cats:
        p, r, f = f1(c)
        res[c] = {"p": round(100 * p, 2), "r": round(100 * r, 2), "f1": round(100 * f, 2),
                  "tp": tp[c], "fp": fp[c], "fn": fn[c]}
    res["_token_acc"] = round(100 * correct_tok / total_tok, 2) if total_tok else 0.0
    return res


def print_eval(res: dict) -> None:
    print(f"  token-düzeyi doğruluk: {res['_token_acc']}")
    print(f"  {'kategori':10s} {'P':>6s} {'R':>6s} {'F1':>6s}   (tp/fp/fn)")
    for c, m in res.items():
        if c == "_token_acc":
            continue
        print(f"  {c:10s} {m['p']:6.2f} {m['r']:6.2f} {m['f1']:6.2f}   "
              f"({m['tp']}/{m['fp']}/{m['fn']})")


def selection_score(res: dict) -> float:
    return res.get("ALL", {}).get("f1", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  HF export
# ─────────────────────────────────────────────────────────────────────────────
def export_hf(model, tokenizer, ls: IdiomLabelSpace, out_dir: Path, metrics: dict | None = None,
              stage2_ckpt: str | None = None) -> None:
    import shutil

    from safetensors.torch import save_file

    from dizgebert_idiom.configuration_dizgebert_idiom import DizgeBertIdiomConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DizgeBertIdiomConfig(
        encoder_name=ls.encoder_model, tags=ls.tags, tags2=ls.tags2, dropout=DROPOUT, max_len=MAX_LEN,
        stage2=bool(stage2_ckpt),
    )
    cfg.architectures = ["DizgeBertIdiomForTokenClassification"]
    cfg.auto_map = {
        "AutoConfig": "configuration_dizgebert_idiom.DizgeBertIdiomConfig",
        "AutoModel": "modeling_dizgebert_idiom.DizgeBertIdiomForTokenClassification",
    }
    cfg.save_pretrained(out_dir)

    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if stage2_ckpt:
        # `train_idiomaticity_clf.IdiomaticityClf` anahtarları: encoder.* + head.{weight,bias}
        # → stage2_encoder.* / stage2_head.* önekiyle aynı safetensors'a kat.
        s2 = torch.load(stage2_ckpt, map_location="cpu")["model"]
        for k, v in s2.items():
            nk = ("stage2_" + k) if k.startswith("head.") else k.replace("encoder.", "stage2_encoder.", 1)
            state[nk] = v.contiguous()
        print(f"stage-2 ağırlıkları katıldı: {stage2_ckpt} (+{len(s2)} tensör)")
    save_file(state, out_dir / "model.safetensors", metadata={"format": "pt"})
    tokenizer.save_pretrained(out_dir)

    pkg = PROJECT_ROOT / "dizgebert_idiom"
    for fn in ("configuration_dizgebert_idiom.py", "modeling_dizgebert_idiom.py"):
        shutil.copy(pkg / fn, out_dir / fn)

    _write_model_card(out_dir, metrics)
    print(f"HF paketi yazıldı: {out_dir}")
    print("  Test: python -c \"from transformers import AutoModel;"
          f" AutoModel.from_pretrained(r'{out_dir}', trust_remote_code=True)\"")


def _write_model_card(out_dir: Path, metrics: dict | None) -> None:
    tpl = PROJECT_ROOT / "dizgebert_idiom" / "MODEL_CARD.md"
    mt = ""
    if metrics:
        for c, m in metrics.items():
            if c == "_token_acc":
                continue
            mt += f"| {c} | {m['p']} | {m['r']} | {m['f1']} |\n"
    text = tpl.read_text(encoding="utf-8").replace("{{METRICS}}", mt or "| – | – | – | – |")
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="ağırlıkları yükle (warm start; optimizer/scheduler sıfırdan)")
    ap.add_argument("--resume", type=str, default=None,
                    help="latest checkpoint'ten devam et (model+scheduler+epoch)")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="HEDEF toplam epoch (resume'da schedule buna göre)")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--eval-file", type=str, default=str(DATA_DIR / "dev.json"))
    ap.add_argument("--export-hf", type=str, default=None)
    ap.add_argument("--stage2-ckpt", type=str, default=None,
                    help="idyomatiklik sınıflandırıcısı checkpoint'i — export-hf'e gömülür "
                         "(iki-aşamalı boru hattı; predict_spans stage-2 filtresi)")
    ap.add_argument("--class-weights", action="store_true",
                    help="O sınıfı baskınlığına karşı ters-frekans ağırlıklandırma")
    ap.add_argument("--span-weight-mult", type=float, default=1.0,
                    help="B/I-* sınıf ağırlıklarını bununla çarp (>1 → recall↑; iki-aşama stage-1)")
    ap.add_argument("--tdk-examples", action="store_true",
                    help="idiom_data/tdk_examples.json'u (TDK sözlüğü gömülü örnekleri, "
                         "isim/sıfat deyimler dahil) train'e ekle")
    ap.add_argument("--tdk-mult", type=int, default=1, help="TDK verisi tekrar sayısı")
    ap.add_argument("--corpus-examples", action="store_true",
                    help="idiom_data/corpus_examples.json'u (Leipzig derleminden madenlenen "
                         "gerçek bağlam cümleleri, prepare_tdk_corpus_examples.py) train'e ekle")
    ap.add_argument("--glu-examples", action="store_true",
                    help="idiom_data/glu_hard_examples.json'u (GLU kılavuzu idyomatik/literal "
                         "minimal çiftleri, prepare_glu_examples.py) train'e ekle")
    ap.add_argument("--glu-mult", type=int, default=1, help="GLU örnekleri tekrar sayısı (küçük set)")
    ap.add_argument("--corpus-glu", action="store_true",
                    help="idiom_data/corpus_examples_glu.json'u (derlem örneklerinin LLM ile "
                         "idyomatiklik filtresinden geçmiş hâli, filter_corpus_idiomaticity.py) ekle")
    ap.add_argument("--encoder", default=None,
                    help="label_space'teki encoder_model'i geçersiz kıl (encoder A/B için)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_metrics = None
    ck = None
    load_path = args.resume or args.checkpoint
    if load_path:
        ck = torch.load(load_path, map_location=device)
        ckpt_metrics = ck.get("metrics")

    ls = IdiomLabelSpace(ck["label_space"]) if ck and "label_space" in ck else IdiomLabelSpace.load()
    if args.encoder:
        ls.encoder_model = args.encoder
        print(f"encoder override: {ls.encoder_model}")
    tokenizer = AutoTokenizer.from_pretrained(ls.encoder_model)
    pad_id = tokenizer.pad_token_id or 0
    collate = make_collate(pad_id)

    model = IdiomTagger(ls, ls.encoder_model).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"checkpoint yüklendi: {load_path}")

    if args.export_hf:
        if not args.checkpoint:
            print("UYARI: --checkpoint verilmedi, eğitilmemiş ağırlıklar export ediliyor.")
        export_hf(model, tokenizer, ls, Path(args.export_hf), ckpt_metrics, args.stage2_ckpt)
        return

    if args.eval:
        ds = IdiomDataset(args.eval_file, tokenizer, ls)
        dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
        print_eval(evaluate(model, dl, device, ls))
        return

    train_ds = IdiomDataset(DATA_DIR / "train.json", tokenizer, ls)
    dev_ds = IdiomDataset(DATA_DIR / "dev.json", tokenizer, ls)
    weight_sources = [DATA_DIR / "train.json"]

    tdk_path = DATA_DIR / "tdk_examples.json"
    if args.tdk_examples and tdk_path.exists():
        tdk_ds = IdiomDataset(tdk_path, tokenizer, ls)
        train_ds = torch.utils.data.ConcatDataset([train_ds] + [tdk_ds] * args.tdk_mult)
        weight_sources += [tdk_path] * args.tdk_mult
        print(f"TDK örnekleri: +{len(tdk_ds)} × {args.tdk_mult}")

    corpus_path = DATA_DIR / "corpus_examples.json"
    if args.corpus_examples and corpus_path.exists():
        corpus_ds = IdiomDataset(corpus_path, tokenizer, ls)
        train_ds = torch.utils.data.ConcatDataset([train_ds, corpus_ds])
        weight_sources += [corpus_path]
        print(f"Derlem örnekleri: +{len(corpus_ds)}")

    glu_path = DATA_DIR / "glu_hard_examples.json"
    if args.glu_examples and glu_path.exists():
        glu_ds = IdiomDataset(glu_path, tokenizer, ls)
        train_ds = torch.utils.data.ConcatDataset([train_ds] + [glu_ds] * args.glu_mult)
        weight_sources += [glu_path] * args.glu_mult
        print(f"GLU örnekleri: +{len(glu_ds)} × {args.glu_mult}")

    corpus_glu_path = DATA_DIR / "corpus_examples_glu.json"
    if args.corpus_glu and corpus_glu_path.exists():
        cg_ds = IdiomDataset(corpus_glu_path, tokenizer, ls)
        train_ds = torch.utils.data.ConcatDataset([train_ds, cg_ds])
        weight_sources += [corpus_glu_path]
        print(f"Derlem (GLU-filtreli) örnekleri: +{len(cg_ds)}")

    weights = (build_class_weights(weight_sources, ls, device, args.span_weight_mult)
               if args.class_weights else None)
    if weights is not None:
        print(f"class-weighting açık (katman 1): {dict(zip(ls.tags, weights.tolist()))}")
    # katman 2 (gap'li 2. parça) ağırlıklandırması her zaman açık — bkz. build_class_weights2 docstring.
    weights2 = build_class_weights2(weight_sources, ls, device)
    print(f"class-weighting (katman 2, her zaman açık): {dict(zip(ls.tags2, weights2.tolist()))}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate)
    print(f"train {len(train_ds)}  dev {len(dev_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )

    best = -1.0
    start_epoch = 1
    if args.resume and ck is not None:
        if ck.get("target_epochs") not in (None, args.epochs):
            # scheduler.load_state_dict aşağıda last_epoch/_step_count'u eski koşunun
            # total_steps'ine göre geri yükler — hedef epoch değiştiyse lineer-decay
            # zamanlaması süreksizleşir (kod incelemesinde bulundu). Basit uyarı: doğru
            # düzeltme scheduler'ı ilerleme-oranına göre yeniden konumlamak, henüz yapılmadı.
            print(f"UYARI: checkpoint hedefi {ck.get('target_epochs')} epoch idi, şimdi "
                  f"{args.epochs} isteniyor — LR zamanlaması süreksiz olabilir.")
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", 0) + 1
        best = ck.get("best", -1.0)
        print(f"resume: epoch {start_epoch}'ten devam, best={best:.2f}, "
              f"hedef {args.epochs} epoch (optimizer sıfırdan, scheduler restore)")

    def save_atomic(obj, path: Path):
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)
        del obj
        gc.collect()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tl = train_epoch(model, train_dl, optimizer, scheduler, device, weights, weights2)
        print(f"train loss: {tl:.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        res = evaluate(model, dev_dl, device, ls)
        print_eval(res)
        score = selection_score(res)
        print(f"\n  selection score (span F1, ALL): {score:.2f}")
        is_best = score > best
        best = max(best, score)

        meta = {
            "epoch": epoch, "target_epochs": args.epochs, "best": best,
            "metrics": res, "label_space": ls.as_dict(), "encoder_model": ls.encoder_model,
        }
        if is_best:
            save_atomic({**meta, "model": model.state_dict()},
                        DATA_DIR / "best_idiom_tagger.pt")
            print("  → best kaydedildi")
        save_atomic({**meta, "model": model.state_dict(),
                     "scheduler": scheduler.state_dict()},
                    DATA_DIR / "idiom_tagger_latest.pt")

    print(f"\nBest selection score: {best:.2f}")


if __name__ == "__main__":
    main()
