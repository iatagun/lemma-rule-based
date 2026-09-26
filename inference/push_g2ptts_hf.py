#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dizgebert_g2ptts_hf/ klasörünü HF'ye push eder. VARSAYILAN ÖZEL (private): önce kendin gör, sonra --public.

Sıra: 1) python -X utf8 -m dizgetts.g2ptts.export_hf
      2) python -X utf8 -m dizgetts.tests.test_hf_g2ptts     (gidiş-dönüş: PUSH ÖNCESİ ZORUNLU, tümü OK olmalı)
      3) hf auth login  (veya HF_TOKEN)  ->  python inference/push_g2ptts_hf.py [--repo iatagun/DizgeBERT-G2PTTS] [--public]
Kart (README.md) dizgebert_g2ptts/MODEL_CARD.md'den üretilir; kartı değiştirdiysen 1. adımı tekrarla.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import HfApi, whoami

FOLDER = Path(__file__).resolve().parent.parent / "dizgebert_g2ptts_hf"  # repo kökü


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="iatagun/DizgeBERT-G2PTTS")
    ap.add_argument("--public", action="store_true", help="varsayılan özel; herkese açmak için")
    ap.add_argument("--message", default="DizgeBERT-G2PTTS v0 (deneysel): hibrit vurgu (kural + ELECTRA) + sınır")
    args = ap.parse_args()

    if not FOLDER.exists():
        return sys.exit(f"{FOLDER} yok — önce: python -X utf8 -m dizgetts.g2ptts.export_hf")
    try:
        who = whoami()
    except Exception:
        return sys.exit("HF girişi yok. Önce çalıştır:  hf auth login   (write yetkili token)")
    print(f"HF user: {who.get('name')}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    print(f"repo hazır ({'herkese açık' if args.public else 'ÖZEL'}): https://huggingface.co/{args.repo}")
    api.upload_folder(folder_path=str(FOLDER), repo_id=args.repo, repo_type="model", commit_message=args.message)
    print(f"✓ push tamam → https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
