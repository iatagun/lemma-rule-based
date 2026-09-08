#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom veri hazırlığı — deyim (VID) + eşdizim/yardımcı-fiil (LVC.full) span BIO etiketleme.

Girdi: `.cupt` (CoNLL-U + 11. PARSEME:MWE kolonu). `conllu` paketi bu ek kolonu doğrudan
desteklemiyor (10 kolonluk sabit şema varsayıyor) — `benchmark/eval_dep.py`'deki gibi kendi
hafif satır-tabanlı ayrıştırıcımız var, kütüphaneye zorlamak yerine.

PARSEME Türkçe 1.2'de yalnız 3 VMWE kategorisi var: VID (3351), LVC.full (2858), MVC (3-5,
gürültü düzeyinde → atılır). VPC/LVC.cause/IRV/IAV Türkçe alt-derlemde hiç yok. Kanonik etiket
seti bu yüzden sabit (frekans-eşikli değil, morph'un XPOS/FEATS/deprel'inin aksine):
    O, B-VID, I-VID, B-LVC, I-LVC

**bigappy-unicrossy tarzı iki katmanlı şema** (Berk, Erden & Güngör 2019 — PARSEME-TR'nin
kendi yazarları): süreksiz (gap'li) MWE'ler standart tek-katman BIO ile temsil edilemez, ama
ampirik olarak PARSEME-TR'deki TÜM gap'li VID/LVC span'leri (313 kayıt) tam olarak **2 bitişik
parçadan** oluşuyor (asla 3+) — bkz. sohbet analizi. Bu yüzden ikinci bir etiket katmanı yeterli:
    tags  (katman 1, ilk parça): O, B-VID, I-VID, B-LVC, I-LVC   — bitişik span'lerin TAMAMI da burada
    tags2 (katman 2, ikinci parça): o, b-VID, i-VID, b-LVC, i-LVC — YALNIZ gap'li span'lerin 2. parçası
Aynı token birden fazla MWE'ye aitse (nadir, ";" ile ayrılmış id listesi) ilk MWE kazanır.
Bir katman zaten doluysa (çakışma, çok nadir) o MWE atlanır ve sayılır.

Girdi:
    idiom_data/raw/{train,dev,test}.cupt  (önce `python fetch_parseme_tr.py`)

Çıktı:
    idiom_data/{train,dev,test}.json
    idiom_data/label_space.json   (--build-label-space ile)

Kullanım:
    python fetch_parseme_tr.py
    python prepare_idiom_data.py
    python prepare_idiom_data.py --build-label-space
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "idiom_data" / "raw"
OUT_DIR = PROJECT_ROOT / "idiom_data"

SOURCES = {split: RAW_DIR / f"{split}.cupt" for split in ("train", "dev", "test")}

# cupt kategori adı -> BIO etiket kısaltması. Yalnız bu ikisi tutulur; gerisi (MVC, bilinmeyen) O'ya düşer.
KEEP_CATS = {"VID": "VID", "LVC.full": "LVC"}

ENCODER_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"


def _parse_mwe_field(s: str) -> list[tuple[str, str | None]]:
    """PARSEME:MWE kolonu → [(mwe_id, kategori|None), ...]. "*" / "_" → []."""
    if not s or s in ("*", "_"):
        return []
    out = []
    for part in s.split(";"):
        mid, sep, cat = part.partition(":")
        out.append((mid, cat if sep else None))
    return out


def iter_sentences(path: Path):
    """.cupt → cümle başına syntactic-word token listesi (MWT aralık + boş düğüm atlanır)."""
    sent: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sent:
                    yield sent
                sent = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            tid = cols[0]
            if not tid.isdigit():  # "2-3" (MWT aralık) veya "8.1" (boş düğüm) atla
                continue
            mwe = cols[10] if len(cols) > 10 else "*"
            sent.append({"id": int(tid), "form": cols[1], "mwe": mwe})
    if sent:
        yield sent


def _contiguous_runs(positions: list[int]) -> list[list[int]]:
    """Sıralı 1-tabanlı pozisyon listesini ardışık (bitişik) bloklara böler."""
    runs = [[positions[0]]]
    for p in positions[1:]:
        if p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


def sentence_to_record(toks: list[dict], stats: Counter) -> dict:
    n = len(toks)
    tags = ["O"] * n
    tags2 = ["o"] * n
    mwe_positions: dict[str, list[int]] = defaultdict(list)
    mwe_cat: dict[str, str] = {}

    for pos, tok in enumerate(toks, start=1):
        for mid, cat in _parse_mwe_field(tok["mwe"]):
            mwe_positions[mid].append(pos)
            if cat:
                mwe_cat[mid] = cat

    for mid, positions in mwe_positions.items():
        cat = mwe_cat.get(mid)
        if cat is None:
            stats["kategorisiz_atlandı"] += 1
            continue
        if cat not in KEEP_CATS:
            stats[f"kategori_dışı_atlandı:{cat}"] += 1
            continue
        tag = KEEP_CATS[cat]
        runs = _contiguous_runs(sorted(positions))

        if len(runs) == 1:
            if any(tags[p - 1] != "O" for p in runs[0]):
                stats["çakışma_atlandı"] += 1
                continue
            for i, p in enumerate(runs[0]):
                tags[p - 1] = f"{'B' if i == 0 else 'I'}-{tag}"
            stats[f"tutuldu:{cat}"] += 1
        elif len(runs) == 2:
            r1, r2 = runs
            if any(tags[p - 1] != "O" for p in r1):
                stats["gapli_çakışma1_atlandı"] += 1
                continue
            if any(tags2[p - 1] != "o" for p in r2):
                stats["gapli_çakışma2_atlandı"] += 1
                continue
            for i, p in enumerate(r1):
                tags[p - 1] = f"{'B' if i == 0 else 'I'}-{tag}"
            for i, p in enumerate(r2):
                tags2[p - 1] = f"{'b' if i == 0 else 'i'}-{tag}"
            stats[f"tutuldu_gapli:{cat}"] += 1
        else:
            # ampirik olarak hiç görülmedi (bkz. docstring) — güvenlik ağı
            stats["3+parça_atlandı"] += 1

    return {"words": [t["form"] for t in toks], "tags": tags, "tags2": tags2}


def build_split(split: str, stats: Counter) -> list[dict]:
    path = SOURCES[split]
    if not path.exists():
        print(f"  UYARI: yok, atlanıyor: {path}")
        return []
    records = [sentence_to_record(toks, stats) for toks in iter_sentences(path)]
    n_tok = sum(len(r["words"]) for r in records)
    print(f"  {split:5s}: {len(records):6d} cümle, {n_tok:7d} token  ({path.name})")
    return records


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  yazıldı: {path.relative_to(PROJECT_ROOT)}  ({len(data)} kayıt)")


def build_label_space() -> None:
    tags = ["O"] + [f"{p}-{t}" for t in KEEP_CATS.values() for p in ("B", "I")]
    tags2 = ["o"] + [f"{p}-{t}" for t in KEEP_CATS.values() for p in ("b", "i")]
    label_space = {
        "encoder_model": ENCODER_MODEL,
        "tags": tags,
        "tags2": tags2,
    }
    out = OUT_DIR / "label_space.json"
    out.write_text(json.dumps(label_space, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  yazıldı: {out.relative_to(PROJECT_ROOT)}")
    print(f"  TAGS ({len(tags)}): {tags}")
    print(f"  TAGS2 ({len(tags2)}, gap'li 2. parça): {tags2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-label-space", action="store_true")
    args = ap.parse_args()

    if args.build_label_space:
        print("=== label_space.json ===")
        build_label_space()
        return

    print("=== DizgeBERT-Idiom veri hazırlığı (PARSEME-TR 1.2) ===")
    stats: Counter = Counter()
    for split in ("train", "dev", "test"):
        recs = build_split(split, stats)
        if recs:
            write_json(OUT_DIR / f"{split}.json", recs)

    print("\n[MWE istatistikleri]")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")

    print("\nSonraki: python prepare_idiom_data.py --build-label-space")


if __name__ == "__main__":
    main()
