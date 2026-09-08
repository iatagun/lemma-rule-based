#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Morph — morfolojik belirsizlik için sentetik veri üretici.

Yöntem: **gerçek taşıyıcı + kontrollü hedef.** Elle FEATS yazmak yerine, gerçek
treebank cümlelerinden (altın etiketli) uygun bir NOUN/VERB/NUM slotu bulur ve o
token'ı eş-yazımlı bir kelimeyle değiştirir. Yalnızca değişen kelimenin analizi
buradaki mini sözlükten gelir; cümlenin gerisi gerçek altın kalır.

Amaç: `at` (hayvan/at!), `koyun` (hayvan/koyun!), `gülün` (gül+Gen/gülün!),
`yüz` (yüz/100/yüz!), `dolar` (para/dol-ar), `beni` (ben/leke) gibi klasik
belirsizlikleri ve ilk⊕son pooling'in kaçırdığı nadir okumaları öğretmek.

Çıktı: morph_data/synthetic_morph.json  (train.json ile karıştırılır, ~%2)

Kullanım:
    python generate_synthetic_morph.py [--per-slot 40] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import conllu

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
sys.path.insert(0, str(PROJECT_ROOT))
OUT = PROJECT_ROOT / "morph_data" / "synthetic_morph.json"

SRC = {
    "kenet": PROJECT_ROOT / "data/treebanks/UD_Turkish-Kenet/tr_kenet-ud-train.conllu",
    "boun": PROJECT_ROOT / "morph_data/raw/tr_boun-ud-train.conllu",
    "imst": PROJECT_ROOT / "morph_data/raw/tr_imst-ud-train.conllu",
}

# XPOS: şema × UPOS
XPOS = {
    "kenet": {"NOUN": "_", "VERB": "_", "NUM": "_", "PRON": "_"},
    "boun": {"NOUN": "_", "VERB": "_", "NUM": "ANum", "PRON": "Pers"},
    "imst": {"NOUN": "Noun", "VERB": "Verb", "NUM": "ANum", "PRON": "Pers"},
}

# soyut slot → şemaya özgü FEATS string'i (treebank'lerin gerçek dağılımından alındı)
FEATS: dict[str, dict[str, str]] = {
    "NOUN_NOM": {s: "Case=Nom|Number=Sing|Person=3" for s in SRC},
    "NOUN_GEN": {s: "Case=Gen|Number=Sing|Person=3" for s in SRC},
    "NOUN_ACC": {s: "Case=Acc|Number=Sing|Person=3" for s in SRC},
    "NUM_CARD": {s: "NumType=Card" for s in SRC},
    "PRON_ACC": {s: "Case=Acc|Number=Sing|Person=1|PronType=Prs" for s in SRC},
    "VERB_IMP2SG": {
        "kenet": "Mood=Imp|Number=Sing|Person=2|Polarity=Pos|Tense=Pres|VerbForm=Fin",
        "boun": "Mood=Imp|Number=Sing|Person=2|Polarity=Pos",
        "imst": "Aspect=Perf|Mood=Imp|Number=Sing|Person=2|Polarity=Pos|Tense=Pres",
    },
    "VERB_IMP2PL": {
        "kenet": "Mood=Imp|Number=Plur|Person=2|Polarity=Pos|Tense=Pres|VerbForm=Fin",
        "boun": "Mood=Imp|Number=Plur|Person=2|Polarity=Pos",
        "imst": "Aspect=Perf|Mood=Imp|Number=Plur|Person=2|Polarity=Pos|Tense=Pres",
    },
    "VERB_AOR3SG": {
        "kenet": "Aspect=Hab|Mood=Gen|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin",
        "boun": "Aspect=Hab|Number=Sing|Person=3|Polarity=Pos|Tense=Pres",
        "imst": "Aspect=Hab|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres",
    },
    "VERB_IMP2SG_NEG": {
        "kenet": "Mood=Imp|Number=Sing|Person=2|Polarity=Neg|Tense=Pres|VerbForm=Fin",
        "boun": "Mood=Imp|Number=Sing|Person=2|Polarity=Neg",
        "imst": "Aspect=Perf|Mood=Imp|Number=Sing|Person=2|Polarity=Neg|Tense=Pres",
    },
    "VERB_VNOUN": {  # -mA ad-fiili (Kenet Vnoun'u seyrek → atlanır)
        "boun": "Case=Nom|Number=Sing|Person=3|Polarity=Pos|VerbForm=Vnoun",
        "imst": "Aspect=Perf|Case=Nom|Mood=Ind|Polarity=Pos|Tense=Pres|VerbForm=Vnoun",
    },
}

# eş-yazım: yüzey biçimi → [(slot, upos), ...]  (aynı yüzey biçimi paylaşan okumalar)
HOMOGRAPHS: dict[str, list[tuple[str, str]]] = {
    "koyun": [("NOUN_NOM", "NOUN"), ("VERB_IMP2PL", "VERB")],
    "gelin": [("NOUN_NOM", "NOUN"), ("VERB_IMP2PL", "VERB")],
    "gülün": [("NOUN_GEN", "NOUN"), ("VERB_IMP2PL", "VERB")],
    "atın": [("NOUN_GEN", "NOUN"), ("VERB_IMP2PL", "VERB")],
    "kızın": [("NOUN_GEN", "NOUN"), ("VERB_IMP2PL", "VERB")],
    "ekin": [("NOUN_NOM", "NOUN"), ("VERB_IMP2PL", "VERB")],
    "at": [("NOUN_NOM", "NOUN"), ("VERB_IMP2SG", "VERB")],
    "gül": [("NOUN_NOM", "NOUN"), ("VERB_IMP2SG", "VERB")],
    "yüz": [("NOUN_NOM", "NOUN"), ("NUM_CARD", "NUM"), ("VERB_IMP2SG", "VERB")],
    "bin": [("NUM_CARD", "NUM"), ("VERB_IMP2SG", "VERB")],
    "kına": [("NOUN_NOM", "NOUN"), ("VERB_IMP2SG", "VERB")],
    "dolar": [("NOUN_NOM", "NOUN"), ("VERB_AOR3SG", "VERB")],
    "yazar": [("NOUN_NOM", "NOUN"), ("VERB_AOR3SG", "VERB")],
    "çeker": [("NOUN_NOM", "NOUN"), ("VERB_AOR3SG", "VERB")],
    "beni": [("PRON_ACC", "PRON"), ("NOUN_ACC", "NOUN")],
    "okuma": [("VERB_IMP2SG_NEG", "VERB"), ("VERB_VNOUN", "VERB")],
    "gitme": [("VERB_IMP2SG_NEG", "VERB"), ("VERB_VNOUN", "VERB")],
    "gülme": [("VERB_IMP2SG_NEG", "VERB"), ("VERB_VNOUN", "VERB")],
    "yapma": [("VERB_IMP2SG_NEG", "VERB"), ("VERB_VNOUN", "VERB")],
}
HOMOGRAPHS = {k: v for k, v in HOMOGRAPHS.items() if v}  # boşları at


def slot_match(slot: str, upos: str, f: dict) -> bool:
    if slot.startswith("NOUN") and upos != "NOUN":
        return False
    if slot == "NUM_CARD":
        return upos == "NUM" and f.get("NumType") in (None, "Card")
    if slot == "PRON_ACC":
        return upos == "PRON" and f.get("Case") == "Acc"
    if slot.startswith("VERB") and upos != "VERB":
        return False
    if slot == "NOUN_NOM":
        return f.get("Case") in (None, "Nom") and f.get("Number") == "Sing" and "Number[psor]" not in f
    if slot == "NOUN_GEN":
        return f.get("Case") == "Gen" and "Number[psor]" not in f
    if slot == "NOUN_ACC":
        return f.get("Case") == "Acc" and "Number[psor]" not in f
    if slot in ("VERB_IMP2SG", "VERB_IMP2PL", "VERB_IMP2SG_NEG"):
        want_num = "Plur" if slot.endswith("2PL") else "Sing"
        want_pol = "Neg" if slot.endswith("NEG") else "Pos"
        return (f.get("Mood") == "Imp" and f.get("Person") == "2"
                and f.get("Number") == want_num and f.get("Polarity", "Pos") == want_pol)
    if slot == "VERB_AOR3SG":
        return (f.get("Person") == "3" and f.get("Number") == "Sing"
                and f.get("Aspect") in ("Hab", "Prog") and f.get("Tense") == "Pres"
                and "Number[psor]" not in f and f.get("VerbForm", "Fin") == "Fin")
    if slot == "VERB_VNOUN":
        return f.get("VerbForm") == "Vnoun" and f.get("Case") in (None, "Nom") and "Number[psor]" not in f
    return False


def carriers(scheme: str):
    """Her slot için: o slota uyan token'ı olan (cümle-token-listesi, idx) örnekleri."""
    path = SRC[scheme]
    if not path.exists():
        print(f"  UYARI yok: {path}")
        return {}
    pools: dict[str, list] = {s: [] for s in FEATS}
    for sent in conllu.parse_incr(open(path, encoding="utf-8")):
        toks = [t for t in sent if isinstance(t["id"], int)]
        if not (3 <= len(toks) <= 30):
            continue
        forms = [t["form"] for t in toks]
        upos = [t["upos"] or "_" for t in toks]
        for i, t in enumerate(toks):
            if i == 0:  # cümle-başı: büyük harf karmaşası
                continue
            f = t["feats"] or {}
            for slot in FEATS:
                if scheme not in FEATS[slot]:
                    continue
                if slot_match(slot, t["upos"], f):
                    pools[slot].append((forms, upos, i))
    return pools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-slot", type=int, default=40,
                    help="şema × eş-yazım × okuma başına cümle sayısı")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    records: list[dict] = []
    stats: dict[tuple, int] = {}
    for scheme in SRC:
        pools = carriers(scheme)
        if not pools:
            continue
        for surface, readings in HOMOGRAPHS.items():
            for slot, upos in readings:
                if scheme not in FEATS.get(slot, {}):
                    continue
                pool = pools.get(slot, [])
                if not pool:
                    continue
                picks = rnd.sample(pool, min(args.per_slot, len(pool)))
                feats = FEATS[slot][scheme]
                xpos = XPOS[scheme][upos]
                for forms, uposes, idx in picks:
                    nf = list(forms)
                    nu = list(uposes)
                    nx = ["_"] * len(forms)
                    nfe = ["_"] * len(forms)
                    nf[idx] = surface.capitalize() if idx == 0 else surface
                    nu[idx] = upos
                    nx[idx] = xpos
                    nfe[idx] = feats
                    records.append({"treebank": scheme, "words": nf,
                                    "upos": nu, "xpos": nx, "feats": nfe})
                stats[(scheme, surface, slot)] = len(picks)

    rnd.shuffle(records)
    OUT.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"yazıldı: {OUT.relative_to(PROJECT_ROOT)}  ({len(records)} cümle)")
    by_scheme: dict[str, int] = {}
    for (sc, _, _), n in stats.items():
        by_scheme[sc] = by_scheme.get(sc, 0) + n
    print("  şema başına:", by_scheme)
    print(f"  eş-yazım × okuma kombinasyonu: {len(stats)}")


if __name__ == "__main__":
    main()
