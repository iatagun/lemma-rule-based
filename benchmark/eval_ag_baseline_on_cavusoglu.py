#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aslantaş & Güngör'ün backbone'uyla (ELECTRA-tr, kendi TR verileriyle eğitilmiş — bkz.
`train_ag_electra_baseline.py`) bizim Çavuşoğlu & Çöltekin bağlam-bağımlılık benchmark'ımızı
(`benchmark/eval_idiom.py --mode external`) çalıştırır — ters yön kıyası.

Onların modeli tek-katman, tek-sınıf (B-IDIOM/I-IDIOM/O) — bizim VID/LVC + bigappy + stage-2
mimarimiz yok. `run_external`'ın beklediği `predict(words) -> list[{'start','end'}]`
arayüzüne sarılıyor.

Kullanım:
    python benchmark/train_ag_electra_baseline.py --epochs 5
    python benchmark/eval_ag_baseline_on_cavusoglu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CKPT_DIR = PROJECT_ROOT / "idiom_data" / "ag_electra_baseline"


def main() -> None:
    if not CKPT_DIR.exists():
        sys.exit(f"{CKPT_DIR} yok — önce `python benchmark/train_ag_electra_baseline.py`.")

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(CKPT_DIR)
    model = AutoModelForTokenClassification.from_pretrained(CKPT_DIR).to(device).eval()
    id2label = model.config.id2label

    @torch.no_grad()
    def predict(words: list[str]) -> list[dict]:
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=64,
                   return_tensors="pt").to(device)
        logits = model(**enc).logits[0]
        pred_ids = logits.argmax(-1).tolist()
        word_ids = enc.word_ids()
        tags = ["O"] * len(words)
        seen = set()
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            tags[wid] = id2label[pid]
        spans, start = [], None
        for i, t in enumerate(tags + ["O"]):
            if t == "B-IDIOM":
                if start is not None:
                    spans.append({"start": start, "end": i})
                start = i
            elif t == "I-IDIOM":
                if start is None:
                    start = i
            else:
                if start is not None:
                    spans.append({"start": start, "end": i})
                    start = None
        return spans

    from benchmark.eval_idiom import run_external
    print("=== Aslantaş & Güngör backbone'u (ELECTRA-tr, kendi TR verisiyle eğitildi) ===")
    run_external(predict, label="AG-ELECTRA-tr baseline")


if __name__ == "__main__":
    main()
