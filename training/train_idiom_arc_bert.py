#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom-Arc PROTOTİP — deyimi bağımlılık-oku sınıflandırması olarak çerçeveler.

Hipotez: token-bağımsız BIO yerine, her tokenin KENDİ HEAD'İNE giden okunu {O,VID,LVC}
olarak sınıflandır. PARSEME'de VID span'lerinin %91'i, LVC'nin %97'si zaten 2 kelimelik VE
bunların ~%95'i doğrudan bir bağımlılık oku — bkz. `prepare_idiom_arc_data.py` docstring'i.

Mimari: `dizgebert_joint.Biaffine` ile BİREBİR aynı desen (root vektörü + dep/head MLP +
biaffine), ama HEAD'i TAHMİN ETMİYOR — PARSEME'nin (bu prototipte) altın HEAD'i veriliyor,
yalnız o okun deyim-türünü sınıflandırıyor. Gerçek kullanımda (2. faz) `iatagun/DizgeBERT-Joint`
ile boru hattı: önce ayrıştır, sonra her tahmin edilen ok için bu sınıflandırıcıyı çalıştır.

Bilinen kapsam sınırı: 3+ kelimelik (VID ~%9, LVC ~%3) ve doğrudan-ok-olmayan 2-kelimelik
(~%3-4) span'ler bu şemada TEMSİL EDİLEMEZ — veri hazırlığında zaten etiketlenmemiş (O
kalıyor). Kapsam tavanı ~%88 (kept/(kept+skipped), prepare_idiom_arc_data.py çıktısına bakın).
Bonus: BIO'nun aksine GAP'Lİ ("sahip ... olarak") span'ler burada sorun DEĞİL — ok, kelimelerin
cümledeki bitişikliğine bakmaz.

Kullanım:
    python prepare_idiom_arc_data.py
    python train_idiom_arc_bert.py --epochs 5      # prototip turu
    python train_idiom_arc_bert.py --eval --checkpoint idiom_arc_data/best_idiom_arc.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "idiom_arc_data"
LABEL_SPACE_PATH = DATA_DIR / "label_space.json"

ENCODER_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"
ARC_DIM = 200
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
DROPOUT = 0.15
IGN = -100


# ─────────────────────────────────────────────────────────────────────────────
#  Etiket uzayı
# ─────────────────────────────────────────────────────────────────────────────
class ArcLabelSpace:
    def __init__(self, d: dict):
        self.encoder_model = d.get("encoder_model", ENCODER_MODEL)
        self.tags = d["arc_tags"]  # ["O","VID","LVC"]
        self.tag_to_id = {t: i for i, t in enumerate(self.tags)}

    @classmethod
    def load(cls, path: Path = LABEL_SPACE_PATH) -> "ArcLabelSpace":
        if not path.exists():
            sys.exit(f"{path} yok — önce: python prepare_idiom_arc_data.py")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def as_dict(self) -> dict:
        return {"encoder_model": self.encoder_model, "arc_tags": self.tags}


# ─────────────────────────────────────────────────────────────────────────────
#  Veri
# ─────────────────────────────────────────────────────────────────────────────
class ArcDataset(Dataset):
    def __init__(self, source, tokenizer, ls: ArcLabelSpace, max_len: int = MAX_LEN):
        data = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else list(source)
        self.items: list[dict] = []
        for rec in data:
            enc = tokenizer(rec["words"], is_split_into_words=True, truncation=True, max_length=max_len)
            word_ids = enc.word_ids()
            first: dict[int, int] = {}
            last: dict[int, int] = {}
            for i, wid in enumerate(word_ids):
                if wid is None:
                    continue
                first.setdefault(wid, i)
                last[wid] = i
            kept = sorted(first)  # truncation trailing kelimeleri düşürebilir
            n_kept = len(kept)
            self.items.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "first_pos": [first[w] for w in kept],
                "last_pos": [last[w] for w in kept],
                # heads/arc_tags 0-tabanlı `kept` indeksine göre kırpılır; head değeri
                # `kept` dışına düşerse (truncation) 0 (root) sayılır — nadir, sınırda kayıp.
                "heads": [rec["heads"][w] if rec["heads"][w] <= n_kept else 0 for w in kept],
                "arc_tags": [ls.tag_to_id[rec["arc_tags"][w]] for w in kept],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
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
            "heads": torch.tensor([padW(b["heads"], 0) for b in batch]),
            "arc_tags": torch.tensor([padW(b["arc_tags"], IGN) for b in batch]),
        }

    return collate


# ─────────────────────────────────────────────────────────────────────────────
#  Model — dizgebert_joint.Biaffine ile BİREBİR aynı desen
# ─────────────────────────────────────────────────────────────────────────────
class Biaffine(nn.Module):
    def __init__(self, in1, in2, out, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x, self.bias_y = bias_x, bias_y
        self.U = nn.Parameter(torch.zeros(out, in1 + bias_x, in2 + bias_y))

    def forward(self, x1, x2):
        if self.bias_x:
            x1 = torch.cat([x1, x1.new_ones(*x1.shape[:-1], 1)], -1)
        if self.bias_y:
            x2 = torch.cat([x2, x2.new_ones(*x2.shape[:-1], 1)], -1)
        s = torch.einsum("bni,oij,bmj->bonm", x1, self.U, x2)
        return s.squeeze(1) if s.size(1) == 1 else s.permute(0, 2, 3, 1)


def _pool_first_last(hs, first_pos, last_pos):
    idx_f = first_pos.unsqueeze(-1).expand(-1, -1, hs.size(-1))
    idx_l = last_pos.unsqueeze(-1).expand(-1, -1, hs.size(-1))
    return torch.cat([hs.gather(1, idx_f), hs.gather(1, idx_l)], dim=-1)


class IdiomArcTagger(nn.Module):
    """HEAD altın veriliyor (tahmin edilmiyor) — yalnız oku {O,VID,LVC} sınıflandırır."""

    def __init__(self, ls: ArcLabelSpace, encoder_model: str = ENCODER_MODEL):
        super().__init__()
        self.ls = ls
        self.encoder = AutoModel.from_pretrained(encoder_model)
        h = self.encoder.config.hidden_size
        d = 2 * h
        self.dropout = nn.Dropout(DROPOUT)
        self.root = nn.Parameter(torch.zeros(d))
        mlp = lambda: nn.Sequential(nn.Linear(d, ARC_DIM), nn.ReLU(), nn.Dropout(DROPOUT))
        self.dep_mlp, self.head_mlp = mlp(), mlp()
        self.biaf = Biaffine(ARC_DIM, ARC_DIM, len(ls.tags))

    def forward(self, input_ids, attention_mask, first_pos, last_pos, heads):
        B = input_ids.size(0)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        z = self.dropout(_pool_first_last(out.last_hidden_state, first_pos, last_pos))  # [B,W,2H]
        ze = torch.cat([self.root.expand(B, 1, -1), z], dim=1)  # [B,W+1,2H], 0=ROOT
        dep = self.dep_mlp(ze)
        hed = self.head_mlp(ze)
        full = self.biaf(dep, hed)  # [B, W+1, W+1, n_tags]  (dependent, head, tag)
        W1 = full.size(1)
        dep_idx = torch.arange(1, W1, device=full.device)[None, :].expand(B, -1)  # [B,W] -> 1..W
        flat = full.reshape(B, W1 * W1, -1)
        flat_idx = dep_idx * W1 + heads  # heads zaten 0..W (0=root), ze indeksleriyle uyumlu
        gathered = flat.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, flat.size(-1)))
        return {"arc": gathered}  # [B,W,n_tags]


def compute_loss(logits: dict, batch: dict) -> torch.Tensor:
    lg = logits["arc"]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), batch["arc_tags"].reshape(-1), ignore_index=IGN)


# ─────────────────────────────────────────────────────────────────────────────
#  Eğitim / değerlendirme
# ─────────────────────────────────────────────────────────────────────────────
def move(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train"):
        batch = move(batch, device)
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"], batch["first_pos"],
                       batch["last_pos"], batch["heads"])
        loss = compute_loss(logits, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, ls: ArcLabelSpace) -> dict:
    """Ok-çifti düzeyinde (dependent,head) exact-match P/R/F1 — BIO span'inin bu mimarideki
    karşılığı: her `arc_tag != O` tahmini, {dependent_pos, head_pos} çiftiyle bir 'span' sayılır."""
    model.eval()
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    correct_tok = total_tok = 0

    for batch in tqdm(loader, desc="eval"):
        b = move(batch, device)
        logits = model(b["input_ids"], b["attention_mask"], b["first_pos"], b["last_pos"], b["heads"])
        pred = logits["arc"].argmax(-1).cpu()
        gold = batch["arc_tags"]
        heads = batch["heads"]
        B, W = gold.shape
        for bi in range(B):
            g_pairs, p_pairs = set(), set()
            for wi in range(W):
                gv = gold[bi, wi].item()
                if gv == IGN:
                    continue
                total_tok += 1
                pv = pred[bi, wi].item()
                correct_tok += int(gv == pv)
                dep_pos, head_pos = wi + 1, heads[bi, wi].item()
                pair = frozenset((dep_pos, head_pos))
                if gv != 0:
                    g_pairs.add((pair, ls.tags[gv]))
                if pv != 0:
                    p_pairs.add((pair, ls.tags[pv]))
            for s in g_pairs & p_pairs:
                tp[s[1]] += 1
                tp["ALL"] += 1
            for s in p_pairs - g_pairs:
                fp[s[1]] += 1
                fp["ALL"] += 1
            for s in g_pairs - p_pairs:
                fn[s[1]] += 1
                fn["ALL"] += 1

    def f1(c):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    res = {}
    for c in sorted((set(tp) | set(fp) | set(fn)) - {"ALL"}) + ["ALL"]:
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
        print(f"  {c:10s} {m['p']:6.2f} {m['r']:6.2f} {m['f1']:6.2f}   ({m['tp']}/{m['fp']}/{m['fn']})")


def selection_score(res: dict) -> float:
    return res.get("ALL", {}).get("f1", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--eval-file", type=str, default=str(DATA_DIR / "dev.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ck = None
    load_path = args.resume or args.checkpoint
    if load_path:
        ck = torch.load(load_path, map_location=device)

    ls = ArcLabelSpace(ck["label_space"]) if ck and "label_space" in ck else ArcLabelSpace.load()
    tokenizer = AutoTokenizer.from_pretrained(ls.encoder_model)
    pad_id = tokenizer.pad_token_id or 0
    collate = make_collate(pad_id)

    model = IdiomArcTagger(ls, ls.encoder_model).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"checkpoint yüklendi: {load_path}")

    if args.eval:
        ds = ArcDataset(args.eval_file, tokenizer, ls)
        dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
        print_eval(evaluate(model, dl, device, ls))
        return

    train_ds = ArcDataset(DATA_DIR / "train.json", tokenizer, ls)
    dev_ds = ArcDataset(DATA_DIR / "dev.json", tokenizer, ls)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate)
    print(f"train {len(train_ds)}  dev {len(dev_ds)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    best = -1.0

    def save_atomic(obj, path: Path):
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)
        del obj
        gc.collect()

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tl = train_epoch(model, train_dl, optimizer, scheduler, device)
        print(f"train loss: {tl:.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        res = evaluate(model, dev_dl, device, ls)
        print_eval(res)
        score = selection_score(res)
        print(f"\n  selection score (arc-pair F1, ALL): {score:.2f}")
        is_best = score > best
        best = max(best, score)
        meta = {"epoch": epoch, "best": best, "metrics": res, "label_space": ls.as_dict(),
                "encoder_model": ls.encoder_model}
        if is_best:
            save_atomic({**meta, "model": model.state_dict()}, DATA_DIR / "best_idiom_arc.pt")
            print("  → best kaydedildi")

    print(f"\nBest selection score: {best:.2f}")


if __name__ == "__main__":
    main()
