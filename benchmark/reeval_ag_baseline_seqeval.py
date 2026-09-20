#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`train_ag_electra_baseline.py`'nin kaydettiği checkpoint'i onların KENDİ metriğiyle
(seqeval entity-F1, `MAX_LENGTH=256`) AG'nin kendi TR test setinde yeniden ölçer.

Önceki "F1=0.965 ile tekrarladık" iddiası YANLIŞTI: o sayı bizim token-düzeyi (kısmi
örtüşmeye kısmi puan veren) metriğimizle hesaplanmıştı, onların seqeval entity-düzeyi
(span sınırları BİREBİR) metriğiyle DEĞİL — iki metrik doğrudan kıyaslanamaz.

Kullanım:
    python benchmark/train_ag_electra_baseline.py --epochs 5
    python benchmark/reeval_ag_baseline_seqeval.py
"""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "aslantas_gungor_multi_turkic.csv"
CKPT_DIR = PROJECT_ROOT / "idiom_data" / "ag_electra_baseline"
MAX_LEN = 256  # onların config.py'sindeki MAX_LENGTH


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR))
    args = ap.parse_args()

    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
    except ImportError:
        sys.exit("pip install seqeval")

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.ckpt_dir).to(device).eval()
    id2label = model.config.id2label

    rows = [r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8"))
            if r["lang"] == "TR" and r["split"] == "test"]

    golds, preds = [], []
    with torch.no_grad():
        for r in rows:
            words = ast.literal_eval(r["tokens"])
            gold = ast.literal_eval(r["labels"])
            enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits[0]
            pred_ids = logits.argmax(-1).tolist()
            word_ids = enc.word_ids()
            pred = ["O"] * len(words)
            seen = set()
            for wid, pid in zip(word_ids, pred_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                pred[wid] = id2label[pid]
            golds.append(gold)
            preds.append(pred)

    print(f"=== AG-ELECTRA-tr baseline, onların KENDİ seqeval metriğiyle, TR test (n={len(rows)}) ===")
    print(f"  seqeval  P={precision_score(golds, preds):.3f}  R={recall_score(golds, preds):.3f}"
          f"  F1={f1_score(golds, preds):.3f}")
    print("  Onların bildirdiği (3-seed ortalaması): ELECTRA-tr P=.876 R=.878 F1=.877")


if __name__ == "__main__":
    main()
