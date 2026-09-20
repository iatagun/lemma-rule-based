#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aslantaş & Güngör TR test setinde (stage2=False) her entity-eşleşmezliğini otomatik
sınıflandırır — "gerçek kaçırma/yanlış-pozitif" mi yoksa "doğru deyimi buldu ama sınır
(boundary) bir kelime kısa/uzun" mı ayrımı için. Kart iddiası: entity-exact F1 (0.592) ile
token-örtüşmeli F1 (0.795) arasındaki fark BÜYÜK ÖLÇÜDE sınır-konvansiyonu farkından mı
geliyor, yoksa gerçek kaçırma/fazladan-işaretlemeden mi?

Kategori tanımı (satır = bir gold span ya da bir pred span, ikisi çakışmıyorsa ayrı sayılır):
  - EXACT     : pred span == gold span (start,end birebir)
  - BOUNDARY  : pred ve gold ÇAKIŞIYOR ama sınırları birebir aynı değil (kısmi örtüşme > 0)
  - MISS      : gold span var, hiçbir pred span'le çakışmıyor (recall kaybı)
  - SPURIOUS  : pred span var, hiçbir gold span'le çakışmıyor (precision kaybı)

Kullanım:
    python benchmark/analyze_ag_errors.py
"""
from __future__ import annotations

import ast
import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "aslantas_gungor_multi_turkic.csv"


def spans_from_tags(tags: list[str]) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-"):
            if start is not None:
                spans.append((start, i))
            start = i
        elif t.startswith("I-"):
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    return spans


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("iatagun/DizgeBERT-Idiom", trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Idiom")

    rows = [r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8"))
            if r["lang"] == "TR" and r["split"] == "test"]

    counts = Counter()
    examples = {"BOUNDARY": [], "MISS": [], "SPURIOUS": []}
    with torch.no_grad():
        for r in rows:
            words = ast.literal_eval(r["tokens"])
            gold_tags = ast.literal_eval(r["labels"])
            pred_spans_raw = model.predict_spans(words, tokenizer=tok, stage2=False)
            pred_spans = [(sp["start"], sp["end"]) for sp in pred_spans_raw]
            gold_spans = spans_from_tags(gold_tags)

            gold_matched = [False] * len(gold_spans)
            pred_matched = [False] * len(pred_spans)
            for gi, g in enumerate(gold_spans):
                for pi, p in enumerate(pred_spans):
                    if overlaps(g, p):
                        gold_matched[gi] = pred_matched[pi] = True
                        if g == p:
                            counts["EXACT"] += 1
                        else:
                            counts["BOUNDARY"] += 1
                            examples["BOUNDARY"].append((words, gold_spans, pred_spans))
            for gi, g in enumerate(gold_spans):
                if not gold_matched[gi]:
                    counts["MISS"] += 1
                    examples["MISS"].append((words, gold_spans, pred_spans))
            for pi, p in enumerate(pred_spans):
                if not pred_matched[pi]:
                    counts["SPURIOUS"] += 1
                    examples["SPURIOUS"].append((words, gold_spans, pred_spans))

    total = sum(counts.values())
    print(f"=== AG TR test (n={len(rows)}), stage2=False — span-düzeyi hata dökümü, toplam {total} span-olayı ===")
    for k in ["EXACT", "BOUNDARY", "MISS", "SPURIOUS"]:
        print(f"  {k}: {counts[k]} (%{100*counts[k]/total:.1f})")

    for cat in ("BOUNDARY", "MISS", "SPURIOUS"):
        print(f"\n--- örnek {cat} vakalar (ilk 8) ---")
        for words, gold_spans, pred_spans in examples[cat][:8]:
            def fmt(spans):
                return "; ".join(" ".join(words[s:e]) for s, e in spans) or "(yok)"
            print(f"  cümle: {' '.join(words)}")
            print(f"    gold: {fmt(gold_spans)}   pred: {fmt(pred_spans)}")


if __name__ == "__main__":
    main()
