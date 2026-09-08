#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dizgebert_morph_hf/ klasörünü HF'ye push eder.

Önce: hf auth login   (veya HF_TOKEN env var)
Sonra: python push_morph_hf.py [--repo iatagun/DizgeBERT-Morph] [--private]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import HfApi, whoami

FOLDER = Path(__file__).resolve().parent.parent / "dizgebert_morph_hf"  # repo kökü


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="iatagun/DizgeBERT-Morph")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="DizgeBERT-Morph v3 (first⊕last pooling, +IMST)")
    args = ap.parse_args()

    if not FOLDER.exists():
        return sys.exit(f"{FOLDER} yok — önce: python train_morph_bert.py "
                        f"--checkpoint morph_data/best_morph_tagger.pt --export-hf dizgebert_morph_hf")

    try:
        who = whoami()
    except Exception:
        return sys.exit("HF girişi yok. Önce çalıştır:  hf auth login   (write yetkili token)")
    print(f"HF user: {who.get('name')}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"repo hazır: https://huggingface.co/{args.repo}")

    api.upload_folder(
        folder_path=str(FOLDER),
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.message,
    )
    print(f"✓ push tamam → https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
