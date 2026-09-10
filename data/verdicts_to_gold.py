#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Claude'un aday-öbek yargılarını (`_my_verdicts.jsonl`) `gold_labels.json`'a çevirir.

Yargı kodları (aday öbek için):
  D = gerçek mecazi VID deyim, bu cümlede idyomatik → span VID/idio  (stage-2: D)
  V = eşdizim / yardımcı-fiil yapısı (LVC), normal kullanım        → span LVC/idio (stage-2: L)
  L = deyim yüzeyi ama bu cümlede LİTERAL kullanılmış             → span VID/lit  (stage-2: L)
  X = yanlış öbek / terim / kalıp söz / atasözü / özel ad / bozuk → span YOK       (stage-2: L)

`_my_verdicts.jsonl`: satır başına {"idx": N, "v": "D|V|L|X", "r": "kısa gerekçe"}
`annotation_batch.json` ile idx üzerinden birleşir.

    python data/verdicts_to_gold.py            # → gold_labels.json
    python data/ingest_gold.py gold_labels.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "idiom_data" / "annotation_batch.json"
VERD = ROOT / "idiom_data" / "_my_verdicts.jsonl"
OUT = ROOT / "gold_labels.json"


def main() -> None:
    batch = {r["idx"]: r for r in json.loads(BATCH.read_text(encoding="utf-8"))}
    verds = [json.loads(l) for l in VERD.read_text(encoding="utf-8").splitlines() if l.strip()]

    out, dist = [], Counter()
    for d in verds:
        it = batch.get(d["idx"])
        if it is None:
            print(f"  idx {d['idx']} batch'te yok, atlandı"); continue
        v = d["v"].upper()
        dist[v] += 1
        s, e = it["s"], it["e"]
        if v == "D":
            spans = [{"s": s, "e": e, "cat": "VID", "sense": "idio"}]
        elif v == "V":
            spans = [{"s": s, "e": e, "cat": "LVC", "sense": "idio"}]
        elif v == "L":
            spans = [{"s": s, "e": e, "cat": "VID", "sense": "lit"}]
        else:  # X
            spans = []
        out.append({
            "idx": it["idx"], "idiom": it["idiom"], "sentence": " ".join(it["words"]),
            "words": it["words"], "cand": {"s": s, "e": e}, "spans": spans,
            "visited": True, "note": d.get("r", "") or None,
        })

    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n_d = dist["D"]
    print(f"{len(out)} kayıt → {OUT.name}   dağılım {dict(dist)}")
    print(f"stage-2: {n_d} D  /  {len(out)-n_d} N   (V+L+X hepsi y=0)")


if __name__ == "__main__":
    main()
