#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aşama-2: İDYOMATİKLİK sınıflandırıcısı (Fikir 3 — iki aşamalı boru hattı).

DizgeBERT-Idiom'un tek-BIO modeli precision tavanına takılıyor (~%64-71): bir öbeği
YÜZEY BİÇİM eşleşince işaretliyor, GLU kılavuzunun Aşama 3'ünü (bağlamda gerçek mi mecazi mi)
uygulamıyor. v6-v13 boyunca L→hep-O örneklerini eğitim verisine katma denemesi ayrımı bir
miktar öğretti ama span-precision'ı her yerde düşürdü (yapısal gerilim).

İki aşama bunu ayırır:
  Aşama 1 (mevcut v5 BIO)  → aday span'ler (yüksek recall)
  Aşama 2 (BU MODEL)       → (cümle, aday span) → {idyomatik, literal} → literal olanı ELE

Aşama-2 mimarisi: paylaşılan ELECTRA gövdesi + span'in ilk⊕son subword temsili → Linear(2H, 2).
Eğitim verisi (elle etiketli, `filter_corpus_idiomaticity.py` akışından):
  - `_corpus_sample_records.jsonl` (idx → words/tags/idiom/span; tags B-VID/I-VID span'i işaretler)
  - `_corpus_sample_labels.tsv`   (idx → D/L/E);  D=idyomatik(1), L=literal(0), E atılır
  - held-out: `corpus_minpair_test.json` (görülmemiş 118 deyim, D ve L kayıtları)

Kullanım:
    python filter_corpus_idiomaticity.py --apply --balance   # (test json'unu üretir)
    python train_idiomaticity_clf.py --epochs 8
    python train_idiomaticity_clf.py --eval --checkpoint idiom_data/best_idiomaticity_clf.pt
"""
from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "idiom_data"
RECS = DATA / "raw" / "_corpus_sample_records.jsonl"
LABELS = DATA / "raw" / "_corpus_sample_labels.tsv"
# not: filter_corpus_idiomaticity.py bunları idiom_data/ altına yazıyor (raw/ değil) — ikisini de dene
RECS_ALT = DATA / "_corpus_sample_records.jsonl"
LABELS_ALT = DATA / "_corpus_sample_labels.tsv"
TEST_JSON = DATA / "corpus_minpair_test.json"
CKPT = DATA / "best_idiomaticity_clf.pt"

ENCODER = "dbmdz/electra-base-turkish-cased-discriminator"
MAX_LEN = 128
BATCH = 16
EPOCHS = 8
LR = 2e-5
WARMUP = 0.1
DROPOUT = 0.15


def _find(p: Path, alt: Path) -> Path:
    if p.exists():
        return p
    if alt.exists():
        return alt
    sys.exit(f"{p} / {alt} yok — önce filter_corpus_idiomaticity.py --apply")


def span_from_tags(tags: list[str]) -> tuple[int, int] | None:
    idx = [i for i, t in enumerate(tags) if t != "O"]
    return (idx[0], idx[-1] + 1) if idx else None


def load_pairs() -> tuple[list[dict], list[dict]]:
    """→ (train, test) kayıt listeleri: {words, s, e, y}  (y: 1 idyomatik, 0 literal)."""
    recs = {json.loads(l)["idx"]: json.loads(l)
            for l in _find(RECS, RECS_ALT).read_text(encoding="utf-8").splitlines() if l.strip()}
    lab: dict[int, str] = {}
    for line in _find(LABELS, LABELS_ALT).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit() and p[1] in "DLE":
            lab[int(p[0])] = p[1]

    # held-out DEYİM düzeyinde (cümle-metni değil): _holdout_idioms.json varsa onu kullan,
    # yoksa corpus_minpair_test.json cümlelerine düş (geriye dönük uyum).
    hp = DATA / "_holdout_idioms.json"
    holdout_idioms = set(json.loads(hp.read_text(encoding="utf-8"))) if hp.exists() else None
    test_texts: set[str] = set()
    if holdout_idioms is None and TEST_JSON.exists():
        for r in json.loads(TEST_JSON.read_text(encoding="utf-8")):
            test_texts.add(" ".join(r["words"]))

    train, test = [], []
    for i, lb in lab.items():
        if lb == "E" or i not in recs:
            continue
        r = recs[i]
        sp = span_from_tags(r["tags"])
        if sp is None:
            continue
        rec = {"words": r["words"], "s": sp[0], "e": sp[1], "y": 1 if lb == "D" else 0}
        is_test = (r["idiom"] in holdout_idioms if holdout_idioms is not None
                   else " ".join(r["words"]) in test_texts)
        (test if is_test else train).append(rec)
    return train, test


class ClfDS(Dataset):
    def __init__(self, rows: list[dict], tok):
        self.items = []
        for r in rows:
            enc = tok(r["words"], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
            wid = enc.word_ids()
            first, last = {}, {}
            for i, w in enumerate(wid):
                if w is None:
                    continue
                first.setdefault(w, i)
                last[w] = i
            if r["s"] not in first or (r["e"] - 1) not in last:
                continue  # span truncation'a takıldı
            self.items.append({
                "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "sf": first[r["s"]], "sl": last[r["e"] - 1], "y": r["y"],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad_id: int):
    def f(b):
        m = max(len(x["input_ids"]) for x in b)
        pad = lambda s, v: s + [v] * (m - len(s))
        return {
            "input_ids": torch.tensor([pad(x["input_ids"], pad_id) for x in b]),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in b]),
            "sf": torch.tensor([x["sf"] for x in b]),
            "sl": torch.tensor([x["sl"] for x in b]),
            "y": torch.tensor([x["y"] for x in b]),
        }
    return f


class IdiomaticityClf(nn.Module):
    """ELECTRA + span ilk⊕son subword → Linear(2H, 2). DizgeBERT-Idiom ile aynı gövde/pooling."""

    def __init__(self, encoder=ENCODER):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(2 * h, 2)

    def forward(self, input_ids, attention_mask, sf, sl):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B, _, H = hs.shape
        f = hs[torch.arange(B), sf]
        g = hs[torch.arange(B), sl]
        return self.head(self.dropout(torch.cat([f, g], -1)))


@torch.no_grad()
def evaluate(model, dl, device) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        pred = model(b["input_ids"], b["attention_mask"], b["sf"], b["sl"]).argmax(-1)
        y = b["y"]
        tp += int(((pred == 1) & (y == 1)).sum());  fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum());  tn += int(((pred == 0) & (y == 0)).sum())
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    lit_acc = tn / (tn + fp) if tn + fp else 0.0  # literal'i doğru ELEME oranı (asıl hedef)
    return {"n": n, "acc": round(100 * acc, 1), "idyom_P": round(100 * p, 1),
            "idyom_R": round(100 * r, 1), "idyom_F1": round(100 * f1, 1),
            "literal_eleme": round(100 * lit_acc, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--encoder", default=ENCODER)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tok = AutoTokenizer.from_pretrained(args.encoder)
    pad_id = tok.pad_token_id or 0

    train_rows, test_rows = load_pairs()
    print(f"train {len(train_rows)} ({Counter(r['y'] for r in train_rows)})  "
          f"test {len(test_rows)} ({Counter(r['y'] for r in test_rows)})")

    test_ds = ClfDS(test_rows, tok)
    test_dl = DataLoader(test_ds, batch_size=BATCH, collate_fn=collate(pad_id))

    model = IdiomaticityClf(args.encoder).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
        print(f"yüklendi: {args.checkpoint}")

    if args.eval:
        print("held-out:", evaluate(model, test_dl, device))
        return

    train_ds = ClfDS(train_rows, tok)
    # sınıf ağırlığı: idyomatik(1) baskın (~820:463) → literal'e ağırlık
    cnt = Counter(x["y"] for x in train_ds.items)
    w = torch.tensor([1.0 / max(cnt[0], 1), 1.0 / max(cnt[1], 1)], device=device)
    w = (w / w.sum() * 2).float()
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate(pad_id))
    print(f"train_ds {len(train_ds)}  class-weights {w.tolist()}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(total * WARMUP), total)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for b in tqdm(train_dl, desc=f"ep{ep}"):
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            logits = model(b["input_ids"], b["attention_mask"], b["sf"], b["sl"])
            loss = F.cross_entropy(logits, b["y"], weight=w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tot += loss.item()
        res = evaluate(model, test_dl, device)
        # seçim skoru: idyomatik-F1 ve literal-eleme dengesi (macro)
        score = (res["idyom_F1"] + res["literal_eleme"]) / 2
        print(f"ep{ep} loss {tot/len(train_dl):.4f}  {res}  macro {score:.1f}")
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "encoder": args.encoder, "metrics": res}, CKPT)
            print(f"  → {CKPT.name}")
    print(f"best macro {best:.1f}")


if __name__ == "__main__":
    main()
