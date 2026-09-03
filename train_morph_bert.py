#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Morph — ELECTRA tabanlı UD morfolojik belirsizlik gidericisi.

Token sınıflandırma: UPOS + XPOS + kategori-başına FEATS head'leri, paylaşılan
`dbmdz/electra-base-turkish-cased-discriminator` gövdesi (yayınlanmış DizgeBERT-Dep ile
aynı → ortak subword sözlüğü). Çok-treebank: Kenet + BOUN native şemalarında; `treebank_id` embedding'i
encoder çıktısına eklenir, çıkarımda `--scheme` ile seçilir.

Kullanım:
    python prepare_morph_data_ud.py && python prepare_morph_data_ud.py --build-label-space
    python train_morph_bert.py --epochs 1          # smoke
    python train_morph_bert.py                     # tam eğitim
    python train_morph_bert.py --eval --checkpoint morph_data/best_morph_tagger.pt
    python train_morph_bert.py --export-hf dizgebert_morph_hf/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "morph_data"
LABEL_SPACE_PATH = DATA_DIR / "label_space.json"

ENCODER_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"
TREEBANKS = ["kenet", "boun", "imst"]
TB_TO_ID = {tb: i for i, tb in enumerate(TREEBANKS)}
TB_EMB_DIM = 48

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 12
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
DROPOUT = 0.15
IGN = -100

# UPOS/XPOS loss ağırlığı — v1'de her terim eşitti, UPOS ~91-92'de takıldı (overfit).
# Ana etiketleri upweight + treebank kimliğini encoder'a token_type_ids ile besle.
UPOS_LOSS_W = 3.0
XPOS_LOSS_W = 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  Etiket uzayı
# ─────────────────────────────────────────────────────────────────────────────
class LabelSpace:
    def __init__(self, d: dict):
        self.encoder_model = d.get("encoder_model", ENCODER_MODEL)
        self.upos = d["upos"]
        self.xpos = d["xpos"]
        self.feat_names = list(d["feats"].keys())          # kanonik sıralı
        self.feat_values = d["feats"]                       # name -> [_ , ...]
        self.deprels = d.get("deprels", [])                 # joint model için (0 = "dep")
        self.upos_to_id = {v: i for i, v in enumerate(self.upos)}
        self.xpos_to_id = {v: i for i, v in enumerate(self.xpos)}
        self.feat_to_id = {
            n: {v: i for i, v in enumerate(vals)} for n, vals in self.feat_values.items()
        }

    @classmethod
    def load(cls, path: Path = LABEL_SPACE_PATH) -> "LabelSpace":
        if not path.exists():
            sys.exit(f"{path} yok — önce: python prepare_morph_data_ud.py --build-label-space")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def as_dict(self) -> dict:
        d = {
            "encoder_model": self.encoder_model,
            "treebanks": TREEBANKS,
            "upos": self.upos,
            "xpos": self.xpos,
            "feats": self.feat_values,
        }
        if self.deprels:
            d["deprels"] = self.deprels
        return d

    def parse_feats(self, fstr: str) -> dict[str, str]:
        out = {n: "_" for n in self.feat_names}
        if fstr and fstr != "_":
            for part in fstr.split("|"):
                name, _, val = part.partition("=")
                if name in out:
                    out[name] = val
        return out

    def feats_to_string(self, pairs: dict[str, str]) -> str:
        items = [(n, v) for n, v in pairs.items() if v != "_"]
        if not items:
            return "_"
        items.sort(key=lambda kv: kv[0].lower())
        return "|".join(f"{n}={v}" for n, v in items)


# ─────────────────────────────────────────────────────────────────────────────
#  Veri
# ─────────────────────────────────────────────────────────────────────────────
class MorphDataset(Dataset):
    """Tokenization + hizalama __init__'te bir kez hesaplanır (her epoch değil).

    Kelime temsili = ilk subword ⊕ son subword (Türkçe'de çekim son eklerdedir —
    `evlerinden` → `ev ##ler ##inden`; yalnız ilk subword suffiks bilgisini kaybeder).
    Etiketler KELİME düzeyinde tutulur.
    """

    def __init__(self, source, tokenizer, ls: LabelSpace, max_len: int = MAX_LEN):
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
            self.items.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "treebank_id": TB_TO_ID[rec["treebank"]],
                "first_pos": [first[w] for w in kept],
                "last_pos": [last[w] for w in kept],
                "upos": [ls.upos_to_id.get(rec["upos"][w], ls.upos_to_id["X"]) for w in kept],
                "xpos": [ls.xpos_to_id.get(rec["xpos"][w], 0) for w in kept],
                "feats": {
                    n: [ls.feat_to_id[n].get(ls.parse_feats(rec["feats"][w])[n], 0)
                        for w in kept]
                    for n in ls.feat_names
                },
            })
        self.feat_names = ls.feat_names

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def make_collate(ls: LabelSpace, pad_id: int):
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
            "treebank_id": torch.tensor([b["treebank_id"] for b in batch]),
            "first_pos": torch.tensor([padW(b["first_pos"], 0) for b in batch]),
            "last_pos": torch.tensor([padW(b["last_pos"], 0) for b in batch]),
            "upos": torch.tensor([padW(b["upos"], IGN) for b in batch]),
            "xpos": torch.tensor([padW(b["xpos"], IGN) for b in batch]),
            "feats": {n: torch.tensor([padW(b["feats"][n], IGN) for b in batch])
                      for n in ls.feat_names},
        }

    return collate


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
def _resize_token_type_embeddings(encoder, n: int) -> None:
    """ELECTRA type_vocab_size=2 → n treebank için genişlet (ekstra satır = ortalama)."""
    emb = encoder.embeddings.token_type_embeddings
    if emb.num_embeddings >= n:
        return
    new = nn.Embedding(n, emb.embedding_dim)
    with torch.no_grad():
        new.weight[: emb.num_embeddings] = emb.weight
        new.weight[emb.num_embeddings:] = emb.weight.mean(0, keepdim=True)
    encoder.embeddings.token_type_embeddings = new
    encoder.config.type_vocab_size = n


def _pool_first_last(hs, first_pos, last_pos):
    """[B,L,H] + [B,W] indeksler → [B,W,2H] (ilk ⊕ son subword)."""
    idx_f = first_pos.unsqueeze(-1).expand(-1, -1, hs.size(-1))
    idx_l = last_pos.unsqueeze(-1).expand(-1, -1, hs.size(-1))
    return torch.cat([hs.gather(1, idx_f), hs.gather(1, idx_l)], dim=-1)


class MorphTagger(nn.Module):
    def __init__(self, ls: LabelSpace, encoder_model: str = ENCODER_MODEL):
        super().__init__()
        self.ls = ls
        self.encoder = AutoModel.from_pretrained(encoder_model)
        _resize_token_type_embeddings(self.encoder, len(TREEBANKS))
        h = self.encoder.config.hidden_size
        self.tb_emb = nn.Embedding(len(TREEBANKS), TB_EMB_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        d = 2 * h + TB_EMB_DIM
        self.upos_head = nn.Linear(d, len(ls.upos))
        self.xpos_head = nn.Linear(d, len(ls.xpos))
        self.feat_heads = nn.ModuleDict(
            {n: nn.Linear(d, len(ls.feat_values[n])) for n in ls.feat_names}
        )

    def forward(self, input_ids, attention_mask, treebank_id, first_pos, last_pos):
        tti = treebank_id[:, None].expand(-1, input_ids.size(1))
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=tti
        )
        w = _pool_first_last(out.last_hidden_state, first_pos, last_pos)  # [B,W,2H]
        tb = self.tb_emb(treebank_id)[:, None, :].expand(-1, w.size(1), -1)
        z = self.dropout(torch.cat([w, tb], dim=-1))
        return {
            "upos": self.upos_head(z),
            "xpos": self.xpos_head(z),
            "feats": {n: head(z) for n, head in self.feat_heads.items()},
        }


def build_class_weights(train_json: Path, ls: LabelSpace, device) -> dict:
    """Ters-karekök frekans ağırlıkları (ortalama 1, [0.3, 6] arası). Nadir FEATS
    değerleri (Voice=Rfl, Abbr...) için. Yalnız FEATS head'lerine; UPOS/XPOS düz CE."""
    import numpy as np

    data = json.loads(train_json.read_text(encoding="utf-8"))
    counts = {n: np.zeros(len(ls.feat_values[n])) for n in ls.feat_names}
    for rec in data:
        for fstr in rec["feats"]:
            parsed = ls.parse_feats(fstr)
            for n in ls.feat_names:
                counts[n][ls.feat_to_id[n].get(parsed[n], 0)] += 1
    weights = {}
    for n, c in counts.items():
        c = np.maximum(c, 1.0)
        w = 1.0 / np.sqrt(c)
        w = w / w.mean()
        w = np.clip(w, 0.3, 6.0)
        weights[n] = torch.tensor(w, dtype=torch.float32, device=device)
    return weights


def compute_loss(logits: dict, batch: dict, ls: LabelSpace, weights: dict | None = None) -> torch.Tensor:
    def ce(name):
        lg = logits[name] if name in ("upos", "xpos") else logits["feats"][name]
        gd = batch[name] if name in ("upos", "xpos") else batch["feats"][name]
        w = weights.get(name) if weights else None
        return F.cross_entropy(lg.reshape(-1, lg.size(-1)), gd.reshape(-1),
                               ignore_index=IGN, weight=w)

    total = UPOS_LOSS_W * ce("upos") + XPOS_LOSS_W * ce("xpos")
    for n in ls.feat_names:
        total = total + ce(n)
    denom = UPOS_LOSS_W + XPOS_LOSS_W + len(ls.feat_names)
    return total / denom


# ─────────────────────────────────────────────────────────────────────────────
#  Eğitim / değerlendirme
# ─────────────────────────────────────────────────────────────────────────────
def move(batch: dict, device) -> dict:
    b = {k: v.to(device) for k, v in batch.items() if k != "feats"}
    b["feats"] = {n: t.to(device) for n, t in batch["feats"].items()}
    return b


def train_epoch(model, loader, optimizer, scheduler, device, ls, scaler=None, weights=None) -> float:
    model.train()
    total = 0.0
    use_amp = scaler is not None
    for batch in tqdm(loader, desc="train"):
        batch = move(batch, device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["treebank_id"], batch["first_pos"], batch["last_pos"])
            loss = compute_loss(logits, batch, ls, weights)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, ls: LabelSpace) -> dict:
    """Treebank başına metrikler döndürür: {tb: {...}}."""
    model.eval()
    from collections import defaultdict

    agg = {
        tb: {
            "upos_ok": 0, "xpos_ok": 0, "xpos_n": 0, "n": 0,
            "tp": 0, "fp": 0, "fn": 0, "exact": 0, "full": 0,
            "pf": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        }
        for tb in TREEBANKS
    }

    for batch in tqdm(loader, desc="eval"):
        dev_b = move(batch, device)
        logits = model(dev_b["input_ids"], dev_b["attention_mask"], dev_b["treebank_id"],
                       dev_b["first_pos"], dev_b["last_pos"])
        upos_pred = logits["upos"].argmax(-1).cpu()
        xpos_pred = logits["xpos"].argmax(-1).cpu()
        feat_pred = {n: logits["feats"][n].argmax(-1).cpu() for n in ls.feat_names}
        tb_ids = batch["treebank_id"]

        B, Ldim = batch["upos"].shape
        for bi in range(B):
            tb = TREEBANKS[tb_ids[bi].item()]
            a = agg[tb]
            for ti in range(Ldim):
                gu = batch["upos"][bi, ti].item()
                if gu == IGN:
                    continue
                a["n"] += 1
                a["upos_ok"] += int(upos_pred[bi, ti].item() == gu)

                gx = batch["xpos"][bi, ti].item()
                if gx != IGN and gx != 0:
                    a["xpos_n"] += 1
                    a["xpos_ok"] += int(xpos_pred[bi, ti].item() == gx)

                gset, pset = set(), set()
                for n in ls.feat_names:
                    gv = batch["feats"][n][bi, ti].item()
                    pv = feat_pred[n][bi, ti].item()
                    gname = ls.feat_values[n][gv] if gv > 0 else None
                    pname = ls.feat_values[n][pv] if pv > 0 else None
                    if gname is not None:
                        gset.add(f"{n}={gname}")
                    if pname is not None:
                        pset.add(f"{n}={pname}")
                    pf = a["pf"][n]
                    if gv > 0 or pv > 0:
                        if gv == pv:
                            pf["tp"] += 1
                        else:
                            if pv > 0:
                                pf["fp"] += 1
                            if gv > 0:
                                pf["fn"] += 1
                a["tp"] += len(gset & pset)
                a["fp"] += len(pset - gset)
                a["fn"] += len(gset - pset)
                a["exact"] += int(gset == pset)
                a["full"] += int(gset == pset and upos_pred[bi, ti].item() == gu)

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    res = {}
    for tb, a in agg.items():
        if a["n"] == 0:
            continue
        per_feat = {
            n: round(100 * f1(d["tp"], d["fp"], d["fn"]), 2)
            for n, d in sorted(a["pf"].items())
            if d["tp"] + d["fp"] + d["fn"] > 0
        }
        res[tb] = {
            "n": a["n"],
            "upos_acc": round(100 * a["upos_ok"] / a["n"], 2),
            "xpos_acc": round(100 * a["xpos_ok"] / a["xpos_n"], 2) if a["xpos_n"] else None,
            "ufeats_f1": round(100 * f1(a["tp"], a["fp"], a["fn"]), 2),
            "feats_exact": round(100 * a["exact"] / a["n"], 2),
            "full_tag_acc": round(100 * a["full"] / a["n"], 2),
            "per_feature_f1": per_feat,
        }
    return res


def print_eval(res: dict) -> None:
    for tb, m in res.items():
        print(f"\n  [{tb}]  n={m['n']}")
        print(f"    UPOS acc      : {m['upos_acc']}")
        print(f"    XPOS acc      : {m['xpos_acc']}")
        print(f"    UFeats F1     : {m['ufeats_f1']}")
        print(f"    FEATS exact   : {m['feats_exact']}")
        print(f"    full-tag acc  : {m['full_tag_acc']}")
        print(f"    per-feature F1: {m['per_feature_f1']}")


def selection_score(res: dict) -> float:
    vals = [m["feats_exact"] for m in res.values()]
    return sum(vals) / len(vals) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  HF export
# ─────────────────────────────────────────────────────────────────────────────
def export_hf(model, tokenizer, ls: LabelSpace, out_dir: Path, metrics: dict | None = None) -> None:
    import shutil

    from safetensors.torch import save_file

    from dizgebert_morph.configuration_dizgebert_morph import DizgeBertMorphConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DizgeBertMorphConfig(
        encoder_name=ls.encoder_model,
        upos_labels=ls.upos,
        xpos_labels=ls.xpos,
        feats_label_space=ls.feat_values,
        treebanks=TREEBANKS,
        default_scheme="kenet",
        tb_emb_dim=TB_EMB_DIM,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    )
    cfg.architectures = ["DizgeBertMorphForMorphology"]
    cfg.auto_map = {
        "AutoConfig": "configuration_dizgebert_morph.DizgeBertMorphConfig",
        "AutoModel": "modeling_dizgebert_morph.DizgeBertMorphForMorphology",
    }
    cfg.save_pretrained(out_dir)

    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, out_dir / "model.safetensors", metadata={"format": "pt"})
    tokenizer.save_pretrained(out_dir)

    pkg = PROJECT_ROOT / "dizgebert_morph"
    for fn in ("configuration_dizgebert_morph.py", "modeling_dizgebert_morph.py"):
        shutil.copy(pkg / fn, out_dir / fn)

    _write_model_card(out_dir, ls, metrics)
    print(f"HF paketi yazıldı: {out_dir}")
    print("  Test: python -c \"from transformers import AutoModel;"
          f" AutoModel.from_pretrained(r'{out_dir}', trust_remote_code=True)\"")


def _write_model_card(out_dir: Path, ls: LabelSpace, metrics: dict | None) -> None:
    tpl = PROJECT_ROOT / "dizgebert_morph" / "MODEL_CARD.md"
    mt = "| treebank | UPOS acc | XPOS acc | UFeats F1 | FEATS exact |\n|---|---|---|---|---|"
    if metrics:
        for tb, m in metrics.items():
            mt += (f"\n| {tb} | {m['upos_acc']} | {m.get('xpos_acc')} | "
                   f"{m['ufeats_f1']} | {m['feats_exact']} |")
    if tpl.exists():
        (out_dir / "README.md").write_text(
            tpl.read_text(encoding="utf-8").replace("{{METRICS}}", mt), encoding="utf-8"
        )
        return
    _mt_legacy = mt  # noqa
    card = f"""---
language: tr
license: cc-by-sa-4.0
library_name: transformers
tags:
- token-classification
- morphological-analysis
- universal-dependencies
- turkish
---

# DizgeBERT-Morph

UD-uyumlu, ELECTRA tabanlı Türkçe **morfolojik belirsizlik gidericisi**. Ön-token'lanmış bir
cümle için token başına **UPOS + XPOS + FEATS** tahmin eder. `{ls.encoder_model}` gövdesi;
kelime temsili = ilk subword ⊕ son subword (Türkçe'de çekim son eklerdedir); kategori-başına
sınıflandırma head'leri.

## Çok-treebank / şema seçimi

**UD_Turkish-Kenet + BOUN + IMST** native şemalarında eğitildi (bu üç treebank yapısal olarak
farklı işaretlenir — BOUN `Evident` ve MWT bölme kullanır, Kenet `VerbForm=Fin` + `Mood`).
`scheme` argümanı çıktı şemasını seçer:

- `scheme="kenet"` (varsayılan) — `iatagun/DizgeBERT-Dep` ile uyumlu
- `scheme="boun"` / `scheme="imst"` — ilgili treebank şeması

## Sonuçlar (dev)

| treebank | UPOS acc | XPOS acc | UFeats F1 | FEATS exact |
|---|---|---|---|---|{mt or " | – | – | – | – |"}

## Kullanım

```python
from transformers import AutoModel, AutoTokenizer
m = AutoModel.from_pretrained("iatagun/DizgeBERT-Morph", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Morph")
print(m.predict(["Yarın", "İstanbul'a", "gideceğim", "."], scheme="imst", tokenizer=tok))
```

## Kısıtlar

- Ön-token'lanmış girdi bekler (kelime listesi). Ham metin için harici tokenizer + MWT bölücü gerekir.
- `iatagun/DizgeBERT-Dep` bağımlılık ayrıştırıcısını beslemek için tasarlandı.

## Lisans

Eğitim verisi UD_Turkish-{{Kenet, BOUN, IMST}} (CC BY-SA 4.0) → model **ShareAlike** miras alır.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="ağırlıkları yükle (warm start; optimizer/scheduler sıfırdan)")
    ap.add_argument("--resume", type=str, default=None,
                    help="latest checkpoint'ten devam et (model+optimizer+scheduler+epoch)")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="HEDEF toplam epoch (resume'da schedule buna göre)")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--eval-file", type=str, default=str(DATA_DIR / "dev.json"))
    ap.add_argument("--export-hf", type=str, default=None)
    ap.add_argument("--amp", action="store_true", help="mixed-precision (fp16) eğitim")
    ap.add_argument("--synthetic", action="store_true",
                    help="morph_data/synthetic_morph.json'u train'e ekle")
    ap.add_argument("--synthetic-mult", type=int, default=2, help="sentetik veri tekrar sayısı")
    ap.add_argument("--class-weights", action="store_true",
                    help="nadir FEATS değerleri için ters-frekans ağırlıklandırma")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_metrics = None
    ck = None
    load_path = args.resume or args.checkpoint
    if load_path:
        ck = torch.load(load_path, map_location=device)
        ckpt_metrics = ck.get("metrics")

    ls = LabelSpace(ck["label_space"]) if ck and "label_space" in ck else LabelSpace.load()
    tokenizer = AutoTokenizer.from_pretrained(ls.encoder_model)
    pad_id = tokenizer.pad_token_id or 0
    collate = make_collate(ls, pad_id)

    model = MorphTagger(ls, ls.encoder_model).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"checkpoint yüklendi: {load_path}")

    if args.export_hf:
        if not args.checkpoint:
            print("UYARI: --checkpoint verilmedi, eğitilmemiş ağırlıklar export ediliyor.")
        export_hf(model, tokenizer, ls, Path(args.export_hf), ckpt_metrics)
        return

    if args.eval:
        ds = MorphDataset(args.eval_file, tokenizer, ls)
        dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
        print_eval(evaluate(model, dl, device, ls))
        return

    train_ds = MorphDataset(DATA_DIR / "train.json", tokenizer, ls)
    dev_ds = MorphDataset(DATA_DIR / "dev.json", tokenizer, ls)

    syn_path = DATA_DIR / "synthetic_morph.json"
    if args.synthetic and syn_path.exists():
        syn = MorphDataset(syn_path, tokenizer, ls)
        train_ds = torch.utils.data.ConcatDataset(
            [train_ds] + [syn] * args.synthetic_mult
        )
        print(f"sentetik: +{len(syn)} × {args.synthetic_mult}")

    weights = build_class_weights(DATA_DIR / "train.json", ls, device) if args.class_weights else None
    if weights:
        print("FEATS class-weighting açık")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate)
    print(f"train {len(train_ds)}  dev {len(dev_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )
    scaler = torch.amp.GradScaler(device.type) if (args.amp and device.type == "cuda") else None
    if scaler:
        print("AMP (fp16) açık")

    best = -1.0
    start_epoch = 1
    if args.resume and ck is not None:
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", 0) + 1
        best = ck.get("best", -1.0)
        print(f"resume: epoch {start_epoch}'ten devam, best={best:.2f}, "
              f"hedef {args.epochs} epoch (optimizer sıfırdan, scheduler restore)")

    import gc

    def save_atomic(obj, path: Path):
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)
        del obj
        gc.collect()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tl = train_epoch(model, train_dl, optimizer, scheduler, device, ls, scaler, weights)
        print(f"train loss: {tl:.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        res = evaluate(model, dev_dl, device, ls)
        print_eval(res)
        score = selection_score(res)
        print(f"\n  selection score (mean feats_exact): {score:.2f}")
        is_best = score > best
        best = max(best, score)

        meta = {
            "epoch": epoch, "target_epochs": args.epochs, "best": best,
            "metrics": res, "label_space": ls.as_dict(),
            "encoder_model": ls.encoder_model, "treebanks": TREEBANKS,
        }
        # deliverable: yalnız model (~440MB) — düşük RAM baskısı
        if is_best:
            save_atomic({**meta, "model": model.state_dict()},
                        DATA_DIR / "best_morph_tagger.pt")
            print("  → best kaydedildi")
        # resume state: optimizer state KAYDEDİLMEZ (bu makinede ~5GB boş RAM → 1.3GB
        # checkpoint save'i OOM-kill tetikliyor). Adam momentleri resume'da ~50 adımda
        # yeniden oturur; LR sürekliliği için yalnız scheduler state yeterli.
        save_atomic({**meta, "model": model.state_dict(),
                     "scheduler": scheduler.state_dict()},
                    DATA_DIR / "morph_tagger_latest.pt")

    print(f"\nBest selection score: {best:.2f}")


if __name__ == "__main__":
    main()
