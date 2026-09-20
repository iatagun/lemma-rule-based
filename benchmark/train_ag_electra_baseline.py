#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aslantaş & Güngör'ün TR eğitim verisiyle (644 train + 68 val), ONLARIN yöntemiyle bire bir
aynı basit tek-katman BIO tagger'ı (B-IDIOM/I-IDIOM/O) eğitir — ELECTRA-tr backbone (onların
en iyi TR-only sonucu, F1=.877 — ve DizgeBERT-Idiom'la AYNI encoder, kontrollü kıyas).

Amaç: "onların modelini/backbone'unu bizim Çavuşoğlu bağlam-bağımlılık benchmark'ımızda
çalıştır" — checkpoint'leri yayınlamadıkları için (double-blind, "yakında HF'de") kendimiz
aynı veri+backbone ile eğitiyoruz. Bigappy 2. katman / stage-2 idyomatiklik filtresi YOK —
onların task tanımı tek-aşama span tespiti.

Kullanım:
    python data/fetch_aslantas_gungor_tr.py
    python benchmark/train_ag_electra_baseline.py --epochs 5
    python benchmark/eval_idiom.py --mode external   # karşılaştırma referansı (bizim model)
    (bu script kendi sonunda otomatik olarak Çavuşoğlu benchmark'ında da çalıştırır)
"""
from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "aslantas_gungor_multi_turkic.csv"
LABELS = ["O", "B-IDIOM", "I-IDIOM"]
L2I = {l: i for i, l in enumerate(LABELS)}
ENCODER = "dbmdz/electra-base-turkish-cased-discriminator"
OUT_DIR = PROJECT_ROOT / "idiom_data" / "ag_electra_baseline"


def load_split(split: str) -> list[dict]:
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["lang"] == "TR" and r["split"] == split]


def build_dataset(tok, rows, max_len=64):
    import torch

    all_ids, all_mask, all_labels = [], [], []
    for r in rows:
        words = ast.literal_eval(r["tokens"])
        labels = ast.literal_eval(r["labels"])
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_len,
                   padding="max_length", return_tensors=None)
        word_ids = enc.word_ids()
        lab_ids = []
        prev = None
        for wid in word_ids:
            if wid is None:
                lab_ids.append(-100)
            elif wid != prev:
                lab_ids.append(L2I[labels[wid]])
            else:
                lab_ids.append(-100)  # yalnız ilk subtoken etiketlenir
            prev = wid
        all_ids.append(enc["input_ids"])
        all_mask.append(enc["attention_mask"])
        all_labels.append(lab_ids)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(all_ids)

        def __getitem__(self, i):
            return {
                "input_ids": torch.tensor(all_ids[i]),
                "attention_mask": torch.tensor(all_mask[i]),
                "labels": torch.tensor(all_labels[i]),
            }

    return DS()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} yok — önce `python data/fetch_aslantas_gungor_tr.py`.")

    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, Trainer,
                               TrainingArguments, DataCollatorForTokenClassification)

    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModelForTokenClassification.from_pretrained(
        ENCODER, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)

    train_rows, val_rows, test_rows = load_split("train"), load_split("validation"), load_split("test")
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    train_ds = build_dataset(tok, train_rows)
    val_ds = build_dataset(tok, val_rows)

    def compute_metrics(eval_pred):
        import numpy as np
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids
        tp = fp = fn = 0
        for p_row, l_row in zip(preds, labels):
            for p, l in zip(p_row, l_row):
                if l == -100:
                    continue
                p_pos, l_pos = p != 0, l != 0
                tp += p_pos and l_pos
                fp += p_pos and not l_pos
                fn += l_pos and not p_pos
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"precision": prec, "recall": rec, "f1": f1}

    targs = TrainingArguments(
        output_dir=str(OUT_DIR / "_hf_out"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
        logging_steps=20,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                       data_collator=DataCollatorForTokenClassification(tok),
                       compute_metrics=compute_metrics)
    trainer.train()

    print("\n=== kendi TR test setlerinde (sanity-check, onların Table 2'siyle kıyas) ===")
    test_ds = build_dataset(tok, test_rows)
    metrics = trainer.evaluate(test_ds)
    print(metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"\nkaydedildi: {OUT_DIR}")


if __name__ == "__main__":
    main()
