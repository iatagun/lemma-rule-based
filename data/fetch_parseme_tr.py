#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PARSEME Türkçe fiil-merkezli çok-sözcüklü ifade (VMWE) derlemi indirici.

DizgeBERT-Idiom, deyim (VID) ve eşdizim/yardımcı-fiil (LVC.full) span'lerini bu derlemden
öğrenir. Serbest birleşimler (O sınıfı) etiketlenmemiş token'lardır.

Kaynak: gitlab.com/parseme/sharedtask-data, edition 1.2, TR/ (Güngör & Yirmibeşoğlu 2018-2020).
Lisans: CC-BY-NC-SA 4.0 (ticari olmayan, paylaş-aynı-lisansla) — bkz. indirilen README.md.
Hedef:  idiom_data/raw/  (gitignore'lu)

Kullanım:
    python fetch_parseme_tr.py            # eksikleri indir
    python fetch_parseme_tr.py --force     # yeniden indir
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
sys.path.insert(0, str(PROJECT_ROOT))
RAW_DIR = PROJECT_ROOT / "idiom_data" / "raw"

_BASE = "https://gitlab.com/parseme/sharedtask-data/-/raw/master/1.2/TR"
FILES = ["train.cupt", "dev.cupt", "test.cupt", "README.md",
         "train-stats.md", "dev-stats.md", "test-stats.md"]


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"  atlandı (var): {dest.name}  ({dest.stat().st_size:,} B)")
        return
    print(f"  indiriliyor: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_parseme_tr/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"  → {dest.name}  ({len(data):,} B)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Var olsa da yeniden indir")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== PARSEME-TR 1.2 → {RAW_DIR} ===")
    try:
        for name in FILES:
            download(f"{_BASE}/{name}", RAW_DIR / name, args.force)
    except Exception as e:  # ağ hatası, 404, timeout...
        print(f"HATA: {e}", file=sys.stderr)
        print("Elle: gitlab.com/parseme/sharedtask-data/-/tree/master/1.2/TR", file=sys.stderr)
        return 1
    print("Tamam. Lisans: CC-BY-NC-SA 4.0 — bkz. idiom_data/raw/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
