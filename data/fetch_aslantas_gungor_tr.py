#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aslantaş & Güngör'ün Turkic Idiom Understanding Benchmark'ının Türkçe (TR) dilimini indirir
(SIGTURK 2026, "A Unified Turkic Idiom Understanding Benchmark").

Tüm 5 Turkic dili (TR/AZ/TK/GA/UZ) tek CSV'de; yalnız TR + split=test satırları bizim
karşılaştırmamız için gerekli (131 cümle, tek-sınıf B-IDIOM/I-IDIOM/O BIO etiketi).

Kaynak: github.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark

Kullanım:
    python data/fetch_aslantas_gungor_tr.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "aslantas_gungor_multi_turkic.csv"
URL = ("https://raw.githubusercontent.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark/"
       "main/turkic_idiom_span_detection/data/processed/ner_labeled_multi_turkic.csv")


def main() -> int:
    if OUT_PATH.exists():
        print(f"atlandı (var): {OUT_PATH.relative_to(PROJECT_ROOT)}  ({OUT_PATH.stat().st_size:,} B)")
        return 0
    print(f"indiriliyor: {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "fetch_aslantas_gungor_tr/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(data)
    print(f"→ {OUT_PATH.relative_to(PROJECT_ROOT)}  ({len(data):,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
