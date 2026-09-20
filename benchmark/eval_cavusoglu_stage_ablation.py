#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aşama 1'in TEK BAŞINA (stage2=False) Çavuşoğlu benchmark'ındaki doğru-ayırt/yanlış-poz'unu
hesaplar — MODEL_CARD.md'nin "Aşama 1 (stage2=False)" satırındaki BOŞ hücreleri dolduruyor.

Bu, "tek-aşamalı bir tagger bağlam ayrımı yapamaz" iddiasının doğrudan ablasyonudur: Aşama 1
(v7, 3-gövde ensemble, TDK+PARSEME+corpus-glu ile eğitilmiş — literal kullanımlar corpus-glu'da
zaten hep-O olarak var) TEK BAŞINA, Aşama 2 (idyomatiklik filtresi) OLMADAN çalıştırılıyor.

Kullanım:
    python benchmark/eval_cavusoglu_stage_ablation.py --hf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.stats_utils import paired_diff_ci, proportion_ci  # noqa: E402

TSV_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "turkish_idioms_benchmark.tsv"


def main() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained("iatagun/DizgeBERT-Idiom", trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Idiom")

    rows = [r for r in csv.DictReader(TSV_PATH.open(encoding="utf-8"), delimiter="\t")
            if r.get("sample", "").strip() and r.get("literal", "").strip()]
    rows = [r for r in rows if len(r["sample"].split()) >= 2 and len(r["literal"].split()) >= 2]
    print(f"=== Çavuşoğlu & Çöltekin, {len(rows)} çift — Aşama 1 tek başına vs +Aşama 2 ===")

    def run(stage2: bool):
        sens, fp = [], []
        for r in rows:
            s_spans = model.predict_spans(r["sample"].split(), tokenizer=tok, stage2=stage2)
            l_spans = model.predict_spans(r["literal"].split(), tokenizer=tok, stage2=stage2)
            sens.append(bool(s_spans))
            fp.append(bool(l_spans))
        both = [s and not f for s, f in zip(sens, fp)]
        return sens, fp, both

    sens1, fp1, both1 = run(False)
    sens2, fp2, both2 = run(True)

    for label, sens, fp, both in [("Aşama 1 tek başına (stage2=False)", sens1, fp1, both1),
                                   ("+ Aşama 2 (stage2=True, varsayılan eşik)", sens2, fp2, both2)]:
        p_sens, lo_s, hi_s = proportion_ci(sens)
        p_fp, lo_f, hi_f = proportion_ci(fp)
        p_both, lo_b, hi_b = proportion_ci(both)
        print(f"\n  {label}")
        print(f"    duyarlılık   %{100*p_sens:.1f}  (95% GA %{100*lo_s:.1f}-%{100*hi_s:.1f})")
        print(f"    yanlış-poz   %{100*p_fp:.1f}  (95% GA %{100*lo_f:.1f}-%{100*hi_f:.1f})")
        print(f"    doğru-ayırt  %{100*p_both:.1f}  (95% GA %{100*lo_b:.1f}-%{100*hi_b:.1f})")

    diff, lo_d, hi_d = paired_diff_ci(both1, both2)
    print(f"\n  eşleştirilmiş fark (doğru-ayırt, +Aşama2 − Aşama1-tek): {100*diff:+.1f}pp"
          f"  (95% GA {100*lo_d:+.1f}pp – {100*hi_d:+.1f}pp)")
    print("  GA sıfırı içermiyorsa: Aşama 2'nin katkısı istatistiksel olarak anlamlı"
          " → 'tek aşama yetmiyor' iddiası bu ölçekte (n={}) destekleniyor.".format(len(rows)))


if __name__ == "__main__":
    main()
