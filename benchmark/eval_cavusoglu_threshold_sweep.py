#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v7 (yayınlanan eşik 0.5) ile v8'i (eşik-taramasıyla seçilmiş 0.3) AYNI koşulda kıyaslamak
için — ikisi de kendi eşik taramasıyla ölçülür, yalnız v8 taranıp v7 taranmazsa v8 lehine
yanlı bir kıyas olur. Bu script v7'yi de (Aşama-1 3-gövde ensemble DEĞİŞMEDİ — vE+vL+vX3,
yalnız Aşama-2 = v3, v8'de bu vSynthOnly650 oldu) AYNI 198 çift üzerinde eşik taramasından
geçirir.

Kullanım:
    python benchmark/eval_cavusoglu_threshold_sweep.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.stats_utils import proportion_ci  # noqa: E402
from benchmark.eval_idiom import make_ensemble_predictor  # noqa: E402
from training.train_idiomaticity_clf import wrap_stage2  # noqa: E402

TSV_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "turkish_idioms_benchmark.tsv"
V7_ENSEMBLE = [
    str(PROJECT_ROOT / "idiom_data" / "best_idiom_tagger_vE_leipzigglu.pt"),
    str(PROJECT_ROOT / "idiom_data" / "best_idiom_tagger_vL_idiomcoverage.pt"),
    str(PROJECT_ROOT / "idiom_data" / "best_idiom_tagger_vX3_slice3.pt"),
]
V7_STAGE2 = str(PROJECT_ROOT / "idiom_data" / "best_idiomaticity_clf_v3.pt")
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


def load_rows():
    rows = [r for r in csv.DictReader(TSV_PATH.open(encoding="utf-8"), delimiter="\t")
            if r.get("sample", "").strip() and r.get("literal", "").strip()]
    return [r for r in rows if len(r["sample"].split()) >= 2 and len(r["literal"].split()) >= 2]


def sweep(raw_predict, stage2_ckpt, rows, label):
    print(f"\n=== {label} ===")
    best = None
    for th in THRESHOLDS:
        predict = wrap_stage2(raw_predict, stage2_ckpt, th)
        sens, fp = [], []
        for r in rows:
            sens.append(bool(predict(r["sample"].split())))
            fp.append(bool(predict(r["literal"].split())))
        both = [s and not f for s, f in zip(sens, fp)]
        p_sens = sum(sens) / len(sens)
        p_fp = sum(fp) / len(fp)
        p_both, lo, hi = proportion_ci(both)
        print(f"  eşik={th}  duyarlılık=%{100*p_sens:.1f}  yanlış-poz=%{100*p_fp:.1f}"
              f"  doğru-ayırt=%{100*p_both:.1f} (95% GA %{100*lo:.1f}-%{100*hi:.1f})")
        if best is None or p_both > best[1]:
            best = (th, p_both, lo, hi)
    print(f"  → en iyi eşik: {best[0]} (doğru-ayırt %{100*best[1]:.1f}, 95% GA"
          f" %{100*best[2]:.1f}-%{100*best[3]:.1f})")
    return best


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["v7", "v8"], required=True,
                    help="Bellek baskısını önlemek için v7/v8 AYRI process'lerde koşturulur")
    args = ap.parse_args()

    rows = load_rows()
    print(f"Çavuşoğlu & Çöltekin, {len(rows)} çift — {args.which}")

    if args.which == "v7":
        v7_predict = make_ensemble_predictor(V7_ENSEMBLE, min_votes=1)
        sweep(v7_predict, V7_STAGE2, rows, "v7 (Aşama-1: vE+vL+vX3, Aşama-2: v3)")
        return

    from transformers import AutoModel, AutoTokenizer
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("iatagun/DizgeBERT-Idiom", trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Idiom")

    print("\n=== v8 (Aşama-1: aynı vE+vL+vX3 — DEĞİŞMEDİ, Aşama-2: vSynthOnly650) ===")
    best_v8 = None
    for th in THRESHOLDS:
        sens, fp = [], []
        with torch.no_grad():
            for r in rows:
                sens.append(bool(model.predict_spans(r["sample"].split(), tokenizer=tok, stage2_thresh=th)))
                fp.append(bool(model.predict_spans(r["literal"].split(), tokenizer=tok, stage2_thresh=th)))
        both = [s and not f for s, f in zip(sens, fp)]
        p_sens, p_fp = sum(sens) / len(sens), sum(fp) / len(fp)
        p_both, lo, hi = proportion_ci(both)
        print(f"  eşik={th}  duyarlılık=%{100*p_sens:.1f}  yanlış-poz=%{100*p_fp:.1f}"
              f"  doğru-ayırt=%{100*p_both:.1f} (95% GA %{100*lo:.1f}-%{100*hi:.1f})")
        if best_v8 is None or p_both > best_v8[1]:
            best_v8 = (th, p_both, lo, hi)
    print(f"  → en iyi eşik: {best_v8[0]} (doğru-ayırt %{100*best_v8[1]:.1f})")


if __name__ == "__main__":
    main()
