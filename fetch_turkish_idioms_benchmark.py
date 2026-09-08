#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Çavuşoğlu & Çöltekin'in Türkçe deyim benchmark'ını indirir (CC BY 4.0).

201 deyim için gerçek **idyomatik kullanım + literal kullanım** cümle çifti — elle
yazılmış, bağımsız bir kaynak (bizim eğitim verimizde YOK). `benchmark/eval_idiom.py
--mode external`'ın bağlam-bağımlılık (idyomatik/literal ayrım) testi için kullanılır.

Kaynak: github.com/coltekin/turkish-idioms (Çavuşoğlu & Çöltekin, "An Idiom Benchmark
for Turkish", MWE 2026 Workshop).

Kullanım:
    python fetch_turkish_idioms_benchmark.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "turkish_idioms_benchmark.tsv"
URL = "https://raw.githubusercontent.com/coltekin/turkish-idioms/main/turkish-idioms.tsv"


def main() -> int:
    if OUT_PATH.exists():
        print(f"atlandı (var): {OUT_PATH.relative_to(PROJECT_ROOT)}  ({OUT_PATH.stat().st_size:,} B)")
        return 0
    print(f"indiriliyor: {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "fetch_turkish_idioms_benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(data)
    print(f"→ {OUT_PATH.relative_to(PROJECT_ROOT)}  ({len(data):,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
