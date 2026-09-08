#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom-Arc PROTOTİP veri hazırlığı — deyim span'lerini bağımlılık okuna eşler.

Hipotez (ampirik doğrulandı, bkz. sohbet): PARSEME'de VID/LVC span'lerinin ezici çoğunluğu
zaten 2 kelimelik VE bu iki kelime arasında doğrudan bir bağımlılık oku var (biri diğerinin
HEAD'i) — VID %95.3, LVC %95.4 (2-kelimelik span'ler arasında; 2-kelimelik span'ler de
VID'in %91'i, LVC'nin %97'si). Yani "her token'a bağımsız bak" yerine "her bağımlılık okuna
bak, bu ok deyim mi" diye çerçevelemek mümkün — Joint'in biaffine deprel-etiketleme
mekanizmasıyla birebir aynı şekil.

`prepare_idiom_data.py`'den FARKI: o, MWE kolonunu BIO'ya çevirir, HEAD'i atar. Bu script
HEAD'i de okur, her tokenin "kendi HEAD'ine giden ok" için bir etiket üretir: `O` (deyim
değil), `VID`, `LVC`.

PROTOTİP KAPSAM SINIRI (bilerek): yalnız 2-kelimelik VE doğrudan-ok olan span'ler etiketlenir.
3+ kelimelik (VID ~%9, LVC ~%3) veya doğrudan ok olmayan 2-kelimelikler (~%3-4) bu turda
atlanır (etiketlenmez, sanki deyim değilmiş gibi kalır) — kapsam tavanı bu yüzden %100 değil,
aşağıdaki istatistiklerde raporlanır.

Çıktı: idiom_arc_data/{train,dev,test}.json  ({"words","heads","arc_tags"})
       idiom_arc_data/label_space.json        ({"encoder_model","arc_tags":["O","VID","LVC"]})

Kullanım:
    python prepare_idiom_arc_data.py   (idiom_data/raw/{train,dev,test}.cupt zaten inmiş olmalı)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo kökü (script bir alt dizinde)
sys.path.insert(0, str(PROJECT_ROOT))
RAW_DIR = PROJECT_ROOT / "idiom_data" / "raw"
OUT_DIR = PROJECT_ROOT / "idiom_arc_data"

KEEP_CATS = {"VID": "VID", "LVC.full": "LVC"}
ENCODER_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"


def _parse_mwe_field(s: str) -> list[tuple[str, str | None]]:
    if not s or s in ("*", "_"):
        return []
    out = []
    for part in s.split(";"):
        mid, sep, cat = part.partition(":")
        out.append((mid, cat if sep else None))
    return out


def iter_sentences(path: Path):
    """.cupt → cümle başına token listesi — bu sefer HEAD de taşınıyor."""
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
            if not tid.isdigit():  # MWT aralık / boş düğüm
                continue
            head = cols[6]
            mwe = cols[10] if len(cols) > 10 else "*"
            sent.append({"id": int(tid), "form": cols[1], "head": head, "mwe": mwe})
    if sent:
        yield sent


def sentence_to_record(toks: list[dict], stats: Counter) -> dict:
    n = len(toks)
    id2pos = {t["id"]: i + 1 for i, t in enumerate(toks)}  # 1-tabanlı pozisyon
    heads = []
    for t in toks:
        h = t["head"]
        heads.append(id2pos.get(int(h), 0) if h.isdigit() and int(h) != 0 else 0)

    arc_tags = ["O"] * n
    mwe_positions: dict[str, list[int]] = {}
    mwe_cat: dict[str, str] = {}
    for pos, tok in enumerate(toks, start=1):
        for mid, cat in _parse_mwe_field(tok["mwe"]):
            mwe_positions.setdefault(mid, []).append(pos)
            if cat:
                mwe_cat[mid] = cat

    for mid, positions in mwe_positions.items():
        cat = mwe_cat.get(mid)
        if cat not in KEEP_CATS:
            continue
        if len(positions) != 2:
            stats["atlandı_3+kelime"] += 1
            continue
        a, b = sorted(positions)
        # doğrudan bağımlılık oku mu? (biri digerinin HEAD'i) — dependent'ın arc'ı etiketlenir
        if heads[a - 1] == b:
            dep = a
        elif heads[b - 1] == a:
            dep = b
        else:
            stats["atlandı_doğrudan_ok_değil"] += 1
            continue
        if arc_tags[dep - 1] != "O":
            stats["atlandı_çakışma"] += 1
            continue
        arc_tags[dep - 1] = KEEP_CATS[cat]
        stats[f"tutuldu:{cat}"] += 1

    return {"words": [t["form"] for t in toks], "heads": heads, "arc_tags": arc_tags}


def main() -> None:
    print("=== DizgeBERT-Idiom-Arc PROTOTİP veri hazırlığı ===")
    stats: Counter = Counter()
    for split in ("train", "dev", "test"):
        path = RAW_DIR / f"{split}.cupt"
        if not path.exists():
            sys.exit(f"{path} yok — önce: python fetch_parseme_tr.py")
        recs = [sentence_to_record(toks, stats) for toks in iter_sentences(path)]
        n_tok = sum(len(r["words"]) for r in recs)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"{split}.json"
        out_path.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
        print(f"  {split:5s}: {len(recs):6d} cümle, {n_tok:7d} token → {out_path.relative_to(PROJECT_ROOT)}")

    print("\n[kapsam istatistikleri — TÜM split'ler toplam]")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")

    label_space = {"encoder_model": ENCODER_MODEL, "arc_tags": ["O", "VID", "LVC"]}
    (OUT_DIR / "label_space.json").write_text(
        json.dumps(label_space, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nyazıldı: {(OUT_DIR / 'label_space.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
