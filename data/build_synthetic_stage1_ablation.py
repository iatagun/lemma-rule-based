#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gerçek "aynı veri" tek-aşama ablasyonu için: v8'in Aşama-2'sini (Deney Z) eğiten AYNI
sentetik 1866 kaydı (`_synthetic_stage2_records.jsonl` + `_synthetic_stage2_labels.tsv`)
Aşama-1'in BIO formatına çevirir — D (idyomatik) kayıtlar zaten B/I-VID taşıyor, **L
(literal) kayıtların etiketleri hep-O'ya değiştirilir** (span'in kelimeleri cümlede duruyor
ama idyomatik span SAYILMIYOR — `filter_corpus_idiomaticity.py`'nin L→hep-O kuralıyla AYNI
konvansiyon). Çıktı, `corpus_examples_glu.json`'a EKLENEREK geçici bir birleşik dosya
üretir — `train_idiom_bert.py --corpus-glu` bunu okuyabilsin diye orijinali GEÇİCİ olarak
bu birleşik dosyayla DEĞİŞTİRİR (yedeği alınır, eğitim bitince script geri yükler).

Kullanım:
    python data/build_synthetic_stage1_ablation.py --swap-in    # yedek al, birleşik dosyayı yerine koy
    python training/train_idiom_bert.py --class-weights --tdk-examples --corpus-glu --epochs 10
    # (checkpoint'i ayrı bir isimle kopyala!)
    python data/build_synthetic_stage1_ablation.py --restore    # orijinali geri yükle
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "idiom_data"
GLU = DATA / "corpus_examples_glu.json"
BACKUP = DATA / "_corpus_examples_glu_PRE_SYNTH_ABLATION_backup.json"
RECS = DATA / "_synthetic_stage2_records.jsonl"
LABELS = DATA / "_synthetic_stage2_labels.tsv"


def build_merged() -> list[dict]:
    labels = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        idx, lab = line.strip().split("\t")
        labels[int(idx)] = lab

    extra = []
    n_d = n_l = 0
    for line in RECS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lab = labels.get(r["idx"])
        if lab == "L":
            tags = ["O"] * len(r["tags"])
            n_l += 1
        else:
            tags = r["tags"]
            n_d += 1
        extra.append({"words": r["words"], "tags": tags})
    print(f"sentetik havuzdan dönüştürüldü: {n_d} D (span korunuyor) + {n_l} L (hep-O'ya çevrildi)")

    base = json.loads(GLU.read_text(encoding="utf-8"))
    print(f"taban corpus_examples_glu.json: {len(base)} kayıt")
    merged = base + extra
    print(f"birleşik: {len(merged)} kayıt")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--swap-in", action="store_true")
    g.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.swap_in:
        if BACKUP.exists():
            sys.exit(f"{BACKUP} zaten var — önce --restore ile geri yükle (yarım kalmış bir tur olabilir).")
        merged = build_merged()
        GLU.rename(BACKUP)
        GLU.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        print(f"→ {GLU} geçici olarak birleşik veriyle değiştirildi (yedek: {BACKUP.name})")
    else:
        if not BACKUP.exists():
            sys.exit(f"{BACKUP} yok — geri yüklenecek bir yedek bulunamadı.")
        if GLU.exists():
            GLU.unlink()
        BACKUP.rename(GLU)
        print(f"→ {GLU} orijinal hâline geri yüklendi.")


if __name__ == "__main__":
    main()
