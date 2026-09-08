#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TDK deyimleri için BÜYÜK DERLEMDEN gerçek bağlam cümlesi madenciliği (weak supervision).

`prepare_tdk_idiom_examples.py` yalnız TDK'nin `meaning` alanındaki GÖMÜLÜ örnek cümleyi
kullanır — deyim başına genelde tek, çeşitlilik düşük. Bu script aynı deyimleri Leipzig Türkçe
derleminde (`fetch_leipzig_tr.py`, ~3M cümle) tarar:

  - Eşleştirme kuralı `prepare_tdk_idiom_examples.py` ile AYNI — sıkı ardışık GÖVDE alt-dizisi
    (`find_span`). Gevşetme YOK (v7 dersi: gürültülü weak-label held-out'u düşürdü).
  - Deyim başına EN FAZLA `--cap` (varsayılan 12) cümle.
  - **Held-out koruması**: mevcut `tdk_examples_{dev,test}.json`'daki deyimler ATLANIR.

Hız: 3M cümleyi analizörle tek tek gövdelemek yavaş. Bunun yerine:
  1. Derlemin BENZERSİZ token'ları toplanır (~1-1.5M), her biri BİR KEZ gövdelenir →
     `_stem_map.json` cache (~5-7 dk, tekrar çalıştırmada anında).
  2. Tarama: `sent_stems = [stem_map[tok] ...]` düz sözlük araması; deyimler ilk-gövde
     indeksiyle eşleştirilir. ~birkaç dk.

Çıktı: `idiom_data/corpus_examples.json`. `train_idiom_bert.py --corpus-examples` ile eklenir.
Lisans: Leipzig Corpora Collection — CC BY (Goldhahn, Eckart & Quasthoff, LREC 2012).

Kullanım:
    python fetch_leipzig_tr.py
    python prepare_tdk_idiom_examples.py            # önce frozen dev/test
    python -u prepare_tdk_corpus_examples.py [--cap 12] [--build-stem-map]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_tdk_idiom_examples import (  # noqa: E402
    extract_examples, find_span, idiom_stems, stem, tokenize, tr_lower,
)

RAW_DIR = PROJECT_ROOT / "idiom_data" / "raw"
LEIPZIG_DIR = RAW_DIR / "leipzig"
IN_CSV = RAW_DIR / "tdk_atasozu_deyim.csv"
STEM_MAP = RAW_DIR / "_stem_map.json"
OUT_JSON = PROJECT_ROOT / "idiom_data" / "corpus_examples.json"
DEV_JSON = PROJECT_ROOT / "idiom_data" / "tdk_examples_dev.json"
TEST_JSON = PROJECT_ROOT / "idiom_data" / "tdk_examples_test.json"

MIN_WORDS, MAX_WORDS = 4, 60


def corpus_files() -> list[Path]:
    return sorted(LEIPZIG_DIR.glob("*-sentences.txt"))


def iter_sentences():
    for p in corpus_files():
        with p.open(encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    yield parts[1].strip()


def build_stem_map() -> dict[str, str]:
    """Derlemdeki benzersiz (lowercase) token → gövde. Bir kez; cache'lenir."""
    if STEM_MAP.exists():
        print(f"stem-map cache: {STEM_MAP.name}")
        return json.loads(STEM_MAP.read_text(encoding="utf-8"))

    print("benzersiz token toplanıyor…")
    t = time.time()
    vocab: set[str] = set()
    n = 0
    for sent in iter_sentences():
        n += 1
        for w in tokenize(sent):
            vocab.add(tr_lower(w))
    print(f"  {n:,} cümle, {len(vocab):,} benzersiz token, {time.time() - t:.0f}s")

    print("gövdeleme…")
    t = time.time()
    sm: dict[str, str] = {}
    for i, tok in enumerate(vocab):
        if i % 200_000 == 0 and i:
            print(f"  {i:,}/{len(vocab):,}  {time.time() - t:.0f}s")
            STEM_MAP.write_text(json.dumps(sm, ensure_ascii=False), encoding="utf-8")
        sm[tok] = stem(tok)
    STEM_MAP.write_text(json.dumps(sm, ensure_ascii=False), encoding="utf-8")
    print(f"  bitti: {len(sm):,} token, {time.time() - t:.0f}s → {STEM_MAP.name}")
    return sm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=12, help="deyim başına maksimum cümle")
    ap.add_argument("--build-stem-map", action="store_true",
                    help="yalnız stem-map cache'ini kur ve çık")
    args = ap.parse_args()

    if not IN_CSV.exists():
        sys.exit(f"{IN_CSV} yok — önce: node fetch_tdk_deyim.mjs")
    if not corpus_files():
        sys.exit(f"{LEIPZIG_DIR} boş — önce: python fetch_leipzig_tr.py")

    stem_map = build_stem_map()
    if args.build_stem_map:
        return

    # ── held-out deyimleri belirle ──
    rows = [r for r in csv.DictReader(IN_CSV.open(encoding="utf-8")) if r["kind"] == "idiom"]
    held_texts: set[str] = set()
    for p in (DEV_JSON, TEST_JSON):
        if p.exists():
            held_texts |= {" ".join(r["words"]) for r in json.loads(p.read_text(encoding="utf-8"))}

    idiom_seqs: dict[str, list[str]] = {}
    held_keys: set[str] = set()
    for r in rows:
        seq = idiom_stems(r["text"])
        if len(seq) < 2:
            continue
        idiom_seqs[r["text"]] = seq
        for ex in extract_examples(r["meaning"]):
            if " ".join(tokenize(ex)) in held_texts:
                held_keys.add(r["text"])
                break

    # ilk-gövde indeksi (yalnız train adayları)
    by_first: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for k, seq in idiom_seqs.items():
        if k not in held_keys:
            by_first[seq[0]].append((k, seq))
    print(f"deyim: {len(idiom_seqs)} eşlenebilir, {len(held_keys)} held-out, "
          f"{sum(len(v) for v in by_first.values())} train adayı, {len(by_first)} farklı ilk-gövde")

    # ── tarama ──
    out: list[dict] = []
    per_idiom: Counter[str] = Counter()
    seen_txt: set[str] = set()
    t = time.time()
    n = 0
    for sent in iter_sentences():
        n += 1
        if n % 500_000 == 0:
            print(f"  {n:,} cümle, {len(out):,} örnek, {len(per_idiom)} deyim, {time.time() - t:.0f}s")
        words = tokenize(sent)
        if not (MIN_WORDS <= len(words) <= MAX_WORDS):
            continue
        wl = [tr_lower(w) for w in words]
        stems = [stem_map.get(w, w) for w in wl]
        for i, s0 in enumerate(stems):
            bucket = by_first.get(s0)
            if not bucket:
                continue
            for k, seq in bucket:
                if per_idiom[k] >= args.cap:
                    continue
                nlen = len(seq)
                if stems[i:i + nlen] == seq:
                    txt = " ".join(words)
                    if txt not in seen_txt:
                        seen_txt.add(txt)
                        tags = ["O"] * len(words)
                        tags[i] = "B-VID"
                        for j in range(i + 1, i + nlen):
                            tags[j] = "I-VID"
                        # "idiom" (sözlük biçimi) + "span" (cümledeki yüzey) — idyomatiklik
                        # filtresi (filter_corpus_idiomaticity.py) için; IdiomDataset yok sayar.
                        out.append({"words": words, "tags": tags,
                                    "idiom": k, "span": " ".join(words[i:i + nlen])})
                    per_idiom[k] += 1

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n{n:,} cümle tarandı ({time.time() - t:.0f}s)")
    print(f"örnek: {len(out):,}  |  kapsanan deyim: {len(per_idiom):,} / "
          f"{sum(len(v) for v in by_first.values())}")
    dist = Counter(per_idiom.values())
    print(f"deyim başına dağılım: {dict(sorted(dist.items()))}")
    print(f"\nyazıldı: {OUT_JSON.relative_to(PROJECT_ROOT)}")
    print("Sonraki: python train_idiom_bert.py --class-weights --tdk-examples --corpus-examples --epochs 10")


if __name__ == "__main__":
    main()
