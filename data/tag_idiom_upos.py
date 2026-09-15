#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""idiom eğitim/eval JSON'larına kelime-düzeyi UPOS etiketi ekler (`--pos-features` deneyi).

DizgeBERT-Morph çıkarımı kullanılır (altın UPOS DEĞİL — gerçek çıkarım zamanında da yalnız
Morph'un tahmini olacağı için tutarlılık gerekiyor, bkz. plan). Her kayda `"upos": [id, ...]`
alanı eklenir (words ile aynı uzunlukta), dosyalar YERİNDE güncellenir.

Kullanım:
    python data/tag_idiom_upos.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MORPH_REPO = "iatagun/DizgeBERT-Morph"
UPOS_LABELS_FILE = PROJECT_ROOT / "idiom_data" / "upos_labels.json"
SCHEME = "imst"  # PARSEME-TR + genel Türkçe metin için en genel/varsayılan şema

FILES = [
    "train.json", "dev.json", "test.json",
    "tdk_examples.json", "tdk_examples_train.json", "tdk_examples_dev.json",
    "tdk_examples_test.json", "corpus_examples_glu.json",
]


def load_morph_upos_fn(device=None):
    """→ (upos_labels, upos_for(words) -> list[int]).  Tek kaynak — hem bu scriptin toplu
    veri etiketlemesi hem de `benchmark/eval_idiom.py`'nin çıkarım-zamanı `--pos-features`
    desteği bunu kullanır (aynı model/aynı hizalama, train/inference tutarlılığı için)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MORPH_REPO)
    # float16 — bu makinede sistem belleği dar (idiom modeliyle aynı süreçte çalışırken
    # iki tam-boyutlu ELECTRA gövdesi bellek baskısı yapıp süreci öldürdü); yalnız çıkarım
    # olduğu için hassasiyet kaybı önemsiz.
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MORPH_REPO, trust_remote_code=True,
                                       torch_dtype=dtype).to(device).eval()
    upos_labels = list(model.config.upos_labels)
    tb = torch.tensor([{"kenet": 0, "boun": 1, "imst": 2}[SCHEME]], device=device)

    @torch.no_grad()
    def upos_for(words: list[str]) -> list[int]:
        if not words:
            return []
        enc = tok(words, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=128).to(device)
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for i, wid in enumerate(enc.word_ids()):
            if wid is None:
                continue
            first.setdefault(wid, i)
            last[wid] = i
        kept = sorted(first)
        fp = torch.tensor([[first[w] for w in kept]], device=device)
        lp = torch.tensor([[last[w] for w in kept]], device=device)
        out = model(enc["input_ids"], enc["attention_mask"], tb, fp, lp)
        up = out.logits_upos.argmax(-1)[0].tolist()
        # truncation kelime düşürebilir — düşenlere UNK (son id) ata
        ids = up + [len(upos_labels)] * (len(words) - len(kept))
        return ids[:len(words)]

    return upos_labels, upos_for


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    upos_labels, upos_for = load_morph_upos_fn(dev)
    UPOS_LABELS_FILE.write_text(json.dumps(upos_labels, ensure_ascii=False), encoding="utf-8")
    print(f"UPOS etiket sayısı: {len(upos_labels)} → {UPOS_LABELS_FILE.name}")

    data_dir = PROJECT_ROOT / "idiom_data"
    for fname in FILES:
        path = data_dir / fname
        if not path.exists():
            print(f"atlandı (yok): {fname}")
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        for rec in records:
            rec["upos"] = upos_for(rec["words"])
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"{fname}: {len(records)} kayıt etiketlendi")


if __name__ == "__main__":
    main()
