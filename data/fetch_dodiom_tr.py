#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dodiom Türkçe deyim veri setini indirir (Eryiğit, Şentaş & Monti, "Gamified Crowdsourcing
for Idiom Corpora Construction", Natural Language Engineering 2022) — Umut et al. (2025)
"Exploring Turkish Idiomaticity with LLMs" (UBMK) makalesinin veri seti/kodu yayınlanmadığı
için (IEEE-arkalı, erişilemedi) aynı ITU NLP ekosisteminden, halka açık, insan-etiketli bir
YERİNE KOYMA olarak kullanılıyor — bkz. `.claude/skills/idiom/SKILL.md` dış-kaynak bölümü.

6861 crowdsourced örnek, 36 deyim, idiom/nonidiom ikili etiket + hedef span indeksleri
(`idiom_indices`) — hem idyomatiklik sınıflandırması hem span-konumlama için kullanılabilir.

Kaynak: github.com/Dodiom/dodiom

Kullanım:
    python data/fetch_dodiom_tr.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "dodiom_tr.csv"
URL = "https://raw.githubusercontent.com/Dodiom/dodiom/master/dataset/dataset_tr.csv"


def main() -> int:
    if OUT_PATH.exists():
        print(f"atlandı (var): {OUT_PATH.relative_to(PROJECT_ROOT)}  ({OUT_PATH.stat().st_size:,} B)")
        return 0
    print(f"indiriliyor: {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "fetch_dodiom_tr/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(data)
    print(f"→ {OUT_PATH.relative_to(PROJECT_ROOT)}  ({len(data):,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
