#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Öneri #2 — ortak-gövde çok-görevli eğitim (multi-task, hiç denenmemiş eksen).

Şu ana kadar stage-1 (BIO span) ve stage-2 (idiomatiklik sınıflandırma) TAMAMEN AYRI iki
ELECTRA gövdesi taşıyordu (~880MB-1.3GB paket). Bu script TEK paylaşılan gövde üzerine hem
BIO tagging head'lerini (`tag_head`/`tag_head2`, v5'ten değişmedi) hem idiomatiklik head'ini
(`stage2_head`, `IdiomaticityClf` ile aynı mimari: span ilk⊕son subword → Linear(2H,2))
kurup İKİ görevi AYNI ADIMDA, PAYLAŞILAN gradyanla eğitir — ne ensemble (Deney O, tahminleri
birleştirme) ne distilasyon (Deney R, bir modelin çıktısını diğerine öğretme), üçüncü bir
mekanizma: iki gözetim sinyali aynı temsili birlikte şekillendiriyor.

Veri: stage-1 tarafı vL'nin BİREBİR aynı reçetesi (train.json + tdk_examples.json +
corpus_examples_glu.json). stage-2 tarafı `train_idiomaticity_clf.py::load_pairs()`'ın
AYNI frozen havuzu (PARSEME altın ile hiç karışmaz, ayrı bir DataLoader). Her adımda stage-1
batch'i + cycle edilmiş bir stage-2 batch'i AYNI şekilde ileri geçirilir, kayıplar toplanır
(loss1 + mu*loss2), TEK backward/optimizer adımı.

Değerlendirme kolaylığı için (mevcut eval altyapısını DEĞİŞTİRMEDEN kullanmak amacıyla) en iyi
epoch'ta paylaşılan gövde iki AYRI, mevcut format uyumlu checkpoint'e "bölünür":
  - `best_idiom_tagger_vJoint.pt`      → düz `IdiomTagger` state_dict (stage2_head hariç),
                                          `benchmark/eval_idiom.py --checkpoint` ile uyumlu.
  - `best_idiomaticity_clf_vJoint.pt`  → `IdiomaticityClf`-uyumlu state_dict (aynı encoder
                                          ağırlıkları + stage2_head→head), `--stage2` ile uyumlu.
Bu yalnız DEĞERLENDİRME kolaylığı — eğitim GERÇEKTEN paylaşılan tek gövdeyle yapılıyor.

Kullanım:
    python scripts/train_joint_stage2.py --epochs 10 --mu 1.0
    python benchmark/eval_idiom.py --local --checkpoint idiom_data/best_idiom_tagger_vJoint.pt \\
        --stage2 idiom_data/best_idiomaticity_clf_vJoint.pt --gap 2 --mode external
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from training.train_idiom_bert import (
    DATA_DIR, ENCODER_MODEL, IdiomDataset, IdiomLabelSpace, IdiomTagger,
    build_class_weights, build_class_weights2, compute_loss, evaluate,
    make_collate, print_eval, selection_score,
)
from training.train_idiomaticity_clf import ClfDS, load_pairs
from training.train_idiomaticity_clf import collate as clf_collate

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
BATCH_SIZE = 16
LR = 2e-5


class JointIdiomTagger(IdiomTagger):
    """`IdiomTagger` + ikinci bir head (`stage2_head`) AYNI `self.encoder`'ı paylaşır.
    `forward()` DEĞİŞMEDİ (stage-1 davranışı birebir aynı, tek başına yüklenebilir — bkz.
    dosya-başı docstring'teki checkpoint bölme). `forward_stage2` span ilk⊕son subword →
    idiomatiklik logiti, `IdiomaticityClf.forward` ile AYNI hesap, gövde PAYLAŞILAN."""

    def __init__(self, ls: IdiomLabelSpace, encoder_model: str = ENCODER_MODEL,
                stage2_dropout: float = 0.15):
        super().__init__(ls, encoder_model)
        h = self.encoder.config.hidden_size
        self.stage2_head = nn.Linear(2 * h, 2)
        self.stage2_dropout_layer = nn.Dropout(stage2_dropout)

    def forward_stage2(self, input_ids, attention_mask, sf, sl):
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B = hs.shape[0]
        f = hs[torch.arange(B, device=hs.device), sf]
        g = hs[torch.arange(B, device=hs.device), sl]
        return self.stage2_head(self.stage2_dropout_layer(torch.cat([f, g], -1)))


def split_checkpoints(model: JointIdiomTagger, ls: IdiomLabelSpace, meta: dict) -> None:
    """En iyi epoch'ta paylaşılan gövdeyi mevcut eval altyapısıyla uyumlu iki dosyaya böler
    (dosya-başı docstring). Yalnız değerlendirme kolaylığı, eğitimi etkilemez."""
    full = model.state_dict()
    tagger_sd = {k: v for k, v in full.items()
                if not k.startswith("stage2_head") and not k.startswith("stage2_dropout_layer")}
    torch.save({**meta, "model": tagger_sd, "label_space": ls.as_dict(),
                "encoder_model": ls.encoder_model},
               DATA_DIR / "best_idiom_tagger_vJoint.pt")

    clf_sd = {k: v for k, v in full.items() if k.startswith("encoder.")}
    clf_sd["head.weight"] = full["stage2_head.weight"]
    clf_sd["head.bias"] = full["stage2_head.bias"]
    torch.save({"model": clf_sd, "encoder": ls.encoder_model, "metrics": meta.get("metrics")},
               DATA_DIR / "best_idiomaticity_clf_vJoint.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--mu", type=float, default=1.0, help="stage-2 kaybının ağırlığı (loss1 + mu*loss2)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  mu={args.mu}")

    ls = IdiomLabelSpace.load()
    tokenizer = AutoTokenizer.from_pretrained(ls.encoder_model)
    pad_id = tokenizer.pad_token_id or 0
    collate1 = make_collate(pad_id)

    # ── stage-1 veri: vL'nin BİREBİR aynı reçetesi ──────────────────────────
    train_ds = IdiomDataset(DATA_DIR / "train.json", tokenizer, ls)
    dev_ds = IdiomDataset(DATA_DIR / "dev.json", tokenizer, ls)
    weight_sources = [DATA_DIR / "train.json"]
    tdk_ds = IdiomDataset(DATA_DIR / "tdk_examples.json", tokenizer, ls)
    train_ds = torch.utils.data.ConcatDataset([train_ds, tdk_ds])
    weight_sources += [DATA_DIR / "tdk_examples.json"]
    cg_ds = IdiomDataset(DATA_DIR / "corpus_examples_glu.json", tokenizer, ls)
    train_ds = torch.utils.data.ConcatDataset([train_ds, cg_ds])
    weight_sources += [DATA_DIR / "corpus_examples_glu.json"]
    print(f"stage-1 train {len(train_ds)}  dev {len(dev_ds)}")

    weights = build_class_weights(weight_sources, ls, device)
    weights2 = build_class_weights2(weight_sources, ls, device)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate1)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate1)

    # ── stage-2 veri: train_idiomaticity_clf.py'nin AYNI frozen havuzu ──────
    s2_train_rows, _s2_test_rows = load_pairs()
    s2_train_ds = ClfDS(s2_train_rows, tokenizer, "stage2-train")
    s2_cnt = Counter(x["y"] for x in s2_train_ds.items)
    s2_w = torch.tensor([1.0 / max(s2_cnt[0], 1), 1.0 / max(s2_cnt[1], 1)], device=device)
    s2_w = (s2_w / s2_w.sum() * 2).float()
    s2_dl = DataLoader(s2_train_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=clf_collate(pad_id))
    print(f"stage-2 train {len(s2_train_ds)}  class-weights {s2_w.tolist()}")

    model = JointIdiomTagger(ls, ls.encoder_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        model.train()
        s2_iter = itertools.cycle(s2_dl)
        tot1 = tot2 = 0.0
        for batch1 in tqdm(train_dl, desc="train"):
            batch1 = {k: v.to(device) for k, v in batch1.items()}
            batch2 = {k: v.to(device) for k, v in next(s2_iter).items()}
            optimizer.zero_grad()
            logits1 = model(batch1["input_ids"], batch1["attention_mask"],
                            batch1["first_pos"], batch1["last_pos"], batch1["pos_ids"])
            loss1 = compute_loss(logits1, batch1, weights, weights2)
            logits2 = model.forward_stage2(batch2["input_ids"], batch2["attention_mask"],
                                           batch2["sf"], batch2["sl"])
            loss2 = F.cross_entropy(logits2, batch2["y"], weight=s2_w)
            loss = loss1 + args.mu * loss2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tot1 += loss1.item(); tot2 += loss2.item()
        n = len(train_dl)
        print(f"train loss1(stage-1) {tot1/n:.4f}  loss2(stage-2) {tot2/n:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        res = evaluate(model, dev_dl, device, ls)
        print_eval(res)
        score = selection_score(res)
        print(f"\n  selection score (span F1, ALL): {score:.2f}")
        if score > best:
            best = score
            split_checkpoints(model, ls, {"epoch": epoch, "target_epochs": args.epochs,
                                          "best": best, "metrics": res})
            print("  → best kaydedildi (vJoint çifti)")

    print(f"\nBest selection score: {best:.2f}")


if __name__ == "__main__":
    main()
