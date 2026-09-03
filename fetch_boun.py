#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UD_Turkish treebank indirici (BOUN + IMST).

DizgeBERT-Morph, native şemalarını koruyarak Kenet + BOUN + IMST üzerinde eğitilir
(çok-treebank + treebank-kaynak embedding'i). Kenet repoda zaten vendored;
BOUN ve IMST train/dev/test buradan çekilir.

Kaynak: github.com/UniversalDependencies/UD_Turkish-{BOUN,IMST} (master)
Hedef:  morph_data/raw/  (gitignore'lu)

Kullanım:
    python fetch_boun.py            # eksikleri indir
    python fetch_boun.py --force    # yeniden indir
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "morph_data" / "raw"

_RAW = "https://raw.githubusercontent.com/UniversalDependencies"
SOURCES = {
    "BOUN": [f"tr_boun-ud-{s}.conllu" for s in ("train", "dev", "test")],
    "IMST": [f"tr_imst-ud-{s}.conllu" for s in ("train", "dev", "test")],
}


def download(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"  atlandı (var): {dest.name}  ({dest.stat().st_size:,} B)")
        return
    print(f"  indiriliyor: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_boun/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"  → {dest.name}  ({len(data):,} B, ~{data.count(b'\\n# sent_id')} cümle)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Var olsa da yeniden indir")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== UD_Turkish {'/'.join(SOURCES)} → {RAW_DIR} ===")
    try:
        for repo, files in SOURCES.items():
            for name in files:
                download(f"{_RAW}/UD_Turkish-{repo}/master/{name}", RAW_DIR / name, args.force)
    except Exception as e:  # ağ hatası, 404, timeout...
        print(f"HATA: {e}", file=sys.stderr)
        print("Elle: github.com/UniversalDependencies/UD_Turkish-{BOUN,IMST}", file=sys.stderr)
        return 1
    print("Tamam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
