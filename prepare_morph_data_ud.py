#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Morph veri hazırlığı — UD morfolojik etiketleme (UPOS + XPOS + FEATS).

Kural-tabanlı `prepare_morph_data.py`'den AYRIDIR (o, dep-parser için lemma-doğruluk
filtresi yapar). Bu script UD treebank'lerini token-sınıflandırma eğitim JSON'una çevirir.

Çok-treebank: Kenet ve BOUN native şemalarında korunur; her kayıt `treebank` alanı taşır
(model içinde treebank-kaynak embedding'i olur). Normalizasyon YOK — tek istisna UD-geçersiz
`Aspect=Rapid` değerinin düşürülmesi.

Girdi:
    Kenet: ngram_pos/UD_Turkish-Kenet-master/tr_kenet-ud-{train,dev,test}.conllu  (vendored)
    BOUN:  morph_data/raw/tr_boun-ud-{train,dev,test}.conllu  (önce `python fetch_boun.py`)

Çıktı:
    morph_data/{train,dev,test}.json      # birleşik, `treebank` etiketli
    morph_data/kenet_test.json            # treebank başına test
    morph_data/boun_test.json
    morph_data/label_space.json           # --build-label-space ile

Kullanım:
    python fetch_boun.py
    python prepare_morph_data_ud.py
    python prepare_morph_data_ud.py --build-label-space
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import conllu

PROJECT_ROOT = Path(__file__).resolve().parent
KENET_DIR = PROJECT_ROOT / "ngram_pos" / "UD_Turkish-Kenet-master"
BOUN_DIR = PROJECT_ROOT / "morph_data" / "raw"
OUT_DIR = PROJECT_ROOT / "morph_data"

SOURCES = {
    "kenet": {
        "train": KENET_DIR / "tr_kenet-ud-train.conllu",
        "dev": KENET_DIR / "tr_kenet-ud-dev.conllu",
        "test": KENET_DIR / "tr_kenet-ud-test.conllu",
    },
    "boun": {
        "train": BOUN_DIR / "tr_boun-ud-train.conllu",
        "dev": BOUN_DIR / "tr_boun-ud-dev.conllu",
        "test": BOUN_DIR / "tr_boun-ud-test.conllu",
    },
    "imst": {
        "train": BOUN_DIR / "tr_imst-ud-train.conllu",
        "dev": BOUN_DIR / "tr_imst-ud-dev.conllu",
        "test": BOUN_DIR / "tr_imst-ud-test.conllu",
    },
}

# UD-geçersiz / şema-dışı değerler: şema dönüşümü değil, değer temizliği.
DROP_FEATS = {("Aspect", "Rapid")}

MIN_VALUE_COUNT = 10   # label_space FEATS: treebank başına eşik
MIN_XPOS_COUNT = 5     # label_space XPOS: BOUN XPOS'u gürültülü (yazım hataları), nadirleri ele
MIN_DEPREL_COUNT = 20  # label_space DEPREL: nadir alt-türleri "dep"e topla

# UD v2 kanonik UPOS — treebank'te görülmese de head sınıfı garanti edilsin
UD_UPOS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]


def canon_feats(feats: dict | None) -> str:
    """conllu feats dict → kanonik UD string (harf-duyarsız alfabetik), `_` boşsa."""
    if not feats:
        return "_"
    items = []
    for name, val in feats.items():
        if val is None:
            continue
        # conllu çoklu değeri virgülle verir; UD FEATS'te nadir, olduğu gibi bırak
        if (name, str(val)) in DROP_FEATS:
            continue
        items.append((name, str(val)))
    if not items:
        return "_"
    items.sort(key=lambda kv: kv[0].lower())
    return "|".join(f"{n}={v}" for n, v in items)


def iter_sentences(path: Path):
    """CoNLL-U → cümle başına syntactic-word token listesi (MWT aralık + boş düğüm atlanır)."""
    with open(path, encoding="utf-8") as f:
        for sent in conllu.parse_incr(f):
            toks = [t for t in sent if isinstance(t["id"], int)]
            if toks:
                yield toks


def sentence_to_record(toks, treebank: str) -> dict:
    # syntactic-word id → sıra indeksi (1-tabanlı; 0 = root)
    id2pos = {t["id"]: i + 1 for i, t in enumerate(toks)}
    heads = []
    for t in toks:
        h = t["head"]
        heads.append(id2pos.get(h, 0) if h not in (None, 0) else 0)
    return {
        "treebank": treebank,
        "words": [t["form"] for t in toks],
        "upos": [t["upos"] or "_" for t in toks],
        "xpos": [t["xpos"] or "_" for t in toks],
        "feats": [canon_feats(t["feats"]) for t in toks],
        "heads": heads,
        "deprels": [t["deprel"] or "dep" for t in toks],
    }


def build_split(split: str) -> list[dict]:
    records: list[dict] = []
    for tb, paths in SOURCES.items():
        p = paths[split]
        if not p.exists():
            print(f"  UYARI: yok, atlanıyor: {p}")
            continue
        n0 = len(records)
        for toks in iter_sentences(p):
            records.append(sentence_to_record(toks, tb))
        n_tok = sum(len(r["words"]) for r in records[n0:])
        print(f"  {tb:5s} {split:5s}: {len(records) - n0:6d} cümle, {n_tok:7d} token  ({p.name})")
    return records


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  yazıldı: {path.relative_to(PROJECT_ROOT)}  ({len(data)} kayıt)")


def build_label_space() -> None:
    train_path = OUT_DIR / "train.json"
    if not train_path.exists():
        sys.exit("Önce `python prepare_morph_data_ud.py` çalıştır (train.json yok).")
    data = json.loads(train_path.read_text(encoding="utf-8"))

    upos: set[str] = set(UD_UPOS)
    xpos_counts: Counter = Counter()
    deprel_counts: Counter = Counter()
    # (treebank, feat_name) -> Counter(value)
    feat_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for rec in data:
        tb = rec["treebank"]
        upos.update(rec["upos"])
        xpos_counts.update(rec["xpos"])
        deprel_counts.update(rec.get("deprels", []))
        for fstr in rec["feats"]:
            if fstr == "_":
                continue
            for part in fstr.split("|"):
                name, _, val = part.partition("=")
                feat_counts[(tb, name)][val] += 1

    feats: dict[str, list[str]] = {}
    for (tb, name), ctr in feat_counts.items():
        kept = {v for v, c in ctr.items() if c >= MIN_VALUE_COUNT}
        if not kept:
            continue
        feats.setdefault(name, set()).update(kept)

    xpos = {x for x, c in xpos_counts.items() if c >= MIN_XPOS_COUNT and x != "_"}
    dropped_xpos = sorted(x for x, c in xpos_counts.items() if c < MIN_XPOS_COUNT and x != "_")

    # deprel: sayım >= 20 olanlar; gerisi runtime'da "dep"e toplanır
    deprels = {r for r, c in deprel_counts.items() if c >= MIN_DEPREL_COUNT}
    deprels.add("dep")
    dropped_dep = sorted(f"{r}({c})" for r, c in deprel_counts.items()
                         if c < MIN_DEPREL_COUNT and r != "dep")

    label_space = {
        "encoder_model": "dbmdz/electra-base-turkish-cased-discriminator",
        "treebanks": ["kenet", "boun", "imst"],
        "upos": sorted(upos),
        "xpos": ["_"] + sorted(xpos),
        "feats": {name: ["_"] + sorted(vals) for name, vals in sorted(feats.items())},
        "deprels": ["dep"] + sorted(deprels - {"dep"}),
    }
    out = OUT_DIR / "label_space.json"
    out.write_text(json.dumps(label_space, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  yazıldı: {out.relative_to(PROJECT_ROOT)}")
    print(f"  UPOS ({len(label_space['upos'])}): {label_space['upos']}")
    print(f"  XPOS ({len(label_space['xpos'])}): {label_space['xpos']}")
    if dropped_xpos:
        print(f"  XPOS elenen ({len(dropped_xpos)}, <{MIN_XPOS_COUNT}, runtime→'_'): {dropped_xpos}")
    print(f"  DEPREL ({len(label_space['deprels'])}): {label_space['deprels']}")
    if dropped_dep:
        print(f"  DEPREL elenen (<{MIN_DEPREL_COUNT}, runtime→'dep'): {dropped_dep}")
    print(f"  FEATS kategorileri ({len(label_space['feats'])}):")
    for name, vals in label_space["feats"].items():
        print(f"    {name:16s} {len(vals) - 1:2d} değer  {vals[1:]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-label-space", action="store_true")
    args = ap.parse_args()

    if args.build_label_space:
        print("=== label_space.json ===")
        build_label_space()
        return

    print("=== DizgeBERT-Morph veri hazırlığı ===")
    for split in ("train", "dev", "test"):
        print(f"\n[{split}]")
        recs = build_split(split)
        write_json(OUT_DIR / f"{split}.json", recs)

    print("\n[treebank başına test]")
    for tb in ("kenet", "boun", "imst"):
        p = SOURCES[tb]["test"]
        if not p.exists():
            print(f"  UYARI: yok: {p}")
            continue
        recs = [sentence_to_record(toks, tb) for toks in iter_sentences(p)]
        write_json(OUT_DIR / f"{tb}_test.json", recs)

    print("\nSonraki: python prepare_morph_data_ud.py --build-label-space")


if __name__ == "__main__":
    main()
