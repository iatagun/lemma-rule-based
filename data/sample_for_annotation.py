#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Elle-etiketleme için altın-set örneği seçer (stage-2 idyomatiklik: D / N).

Amaç: güvenilir, donmuş bir değerlendirme seti + temiz eğitim tohumu — gümüş
(Claude/LLM) etiketlerin yerine. Kaynak `corpus_examples.json` (madenlenen deyim
cümleleri, hepsinde bir öbek işaretli).

Öncelik sırası:
  1. MİNİMAL ÇİFT adayı — bu oturumun Sonnet D/N koşusu (`_corpus_idiomaticity_labels.jsonl`)
     aynı deyime HEM D HEM N vermiş → literal/mecazi ikircik gerçek. Deyim başına
     bir Sonnet-D + bir Sonnet-N cümlesi seçilir (insan bunları teyit/düzeltir).
  2. Literal-eğilimli — Sonnet ≥1 N vermiş ama çift yok.
  3. Kapsam dolgusu — Sonnet'in işaretlemediği rastgele deyimler.

Sonnet etiketi sadece ÖRNEKLEME için ipucu; insan kararı bağlar. İpucu çıktıya
`sonnet` alanında konur, HTML aracında varsayılan GİZLİ (önyargı yaratmasın).

Kullanım:
    python data/sample_for_annotation.py --n 600
    # → idiom_data/annotation_batch.json   (HTML etiketleyici bunu yükler)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "idiom_data" / "corpus_examples.json"
SONNET = ROOT / "idiom_data" / "_corpus_idiomaticity_labels.jsonl"
TDK_CSV = ROOT / "idiom_data" / "raw" / "tdk_atasozu_deyim.csv"
HOLDOUT = ROOT / "idiom_data" / "_holdout_idioms.json"
FROZEN = ROOT / "idiom_data" / "_corpus_sample_records.jsonl"
OUT = ROOT / "idiom_data" / "annotation_batch.json"

MIN_W, MAX_W = 5, 42


def span_range(tags: list[str]) -> tuple[int, int] | None:
    idx = [i for i, t in enumerate(tags) if t != "O"]
    return (idx[0], idx[-1] + 1) if idx else None


def load_meanings() -> dict[str, str]:
    """TDK sözlük biçimi → tanım (varsa örnek cümleyle). Anahtar normalize edilir."""
    out: dict[str, str] = {}
    if not TDK_CSV.exists():
        return out
    with TDK_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("kind") != "idiom":
                continue
            key = _norm(row["text"])
            out.setdefault(key, (row.get("meaning") or "").strip())
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", s)).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="hedef kayıt sayısı")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--exclude-holdout", action="store_true",
                    help="mevcut 118 held-out deyimi dışla (varsayılan: dahil et, yeniden etiketle)")
    ap.add_argument("--exclude-frozen", action="store_true",
                    help="elle etiketlenmiş frozen deyimleri dışla (varsayılan: dahil)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    items = json.loads(CORPUS.read_text(encoding="utf-8"))
    by_idiom: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        sr = span_range(it["tags"])
        if sr is None or not (MIN_W <= len(it["words"]) <= MAX_W):
            continue
        it = {**it, "s": sr[0], "e": sr[1], "text": " ".join(it["words"])}
        by_idiom[it["idiom"]].append(it)

    # Sonnet ipuçları: cümle metni → D/N, ve deyim → {D:n, N:n}
    hint: dict[str, str] = {}
    idiom_dn: dict[str, Counter] = defaultdict(Counter)
    if SONNET.exists():
        for line in SONNET.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            lb = d.get("label", "").upper()
            if lb in ("D", "N"):
                hint[d["k"]] = lb
                idiom_dn[d["idiom"]][lb] += 1

    excl: set[str] = set()
    if args.exclude_holdout and HOLDOUT.exists():
        excl |= set(json.loads(HOLDOUT.read_text(encoding="utf-8")))
    if args.exclude_frozen and FROZEN.exists():
        excl |= {json.loads(l)["idiom"] for l in FROZEN.read_text(encoding="utf-8").splitlines() if l.strip()}

    cand = [k for k in by_idiom if k not in excl and by_idiom[k]]
    tier1 = [k for k in cand if idiom_dn[k]["D"] and idiom_dn[k]["N"]]        # minimal-çift adayı
    tier2 = [k for k in cand if (idiom_dn[k]["D"] or idiom_dn[k]["N"]) and k not in tier1]
    tier3 = [k for k in cand if k not in tier1 and k not in tier2]
    for t in (tier1, tier2, tier3):
        rng.shuffle(t)
    print(f"aday deyim: {len(cand)}  | çift-adayı {len(tier1)}  literal-eğilimli {len(tier2)}  diğer {len(tier3)}")

    meanings = load_meanings()
    picked: list[dict] = []
    seen_txt: set[str] = set()

    def take(it: dict, idiom: str) -> None:
        if it["text"] in seen_txt:
            return
        seen_txt.add(it["text"])
        picked.append({
            "idx": len(picked),
            "idiom": idiom,
            "meaning": meanings.get(_norm(idiom), ""),
            "span": it["span"],
            "words": it["words"],
            "tags": it["tags"],
            "s": it["s"], "e": it["e"],
            "sentence": it["text"],
            "sonnet": hint.get(it["text"], ""),
        })

    def pick_pair(idiom: str) -> None:
        lst = by_idiom[idiom]
        ds = [x for x in lst if hint.get(x["text"]) == "D"]
        ns = [x for x in lst if hint.get(x["text"]) == "N"]
        rng.shuffle(ds); rng.shuffle(ns)
        if ds:
            take(ds[0], idiom)
        if ns:
            take(ns[0], idiom)

    def pick_n(idiom: str, k: int) -> None:
        lst = list(by_idiom[idiom])
        rng.shuffle(lst)
        for x in lst[:k]:
            take(x, idiom)

    for k in tier1:
        if len(picked) >= args.n:
            break
        pick_pair(k)
    for k in tier2:
        if len(picked) >= args.n:
            break
        pick_n(k, 2)
    for k in tier3:
        if len(picked) >= args.n:
            break
        pick_n(k, 2)

    rng.shuffle(picked)
    for i, r in enumerate(picked):
        r["idx"] = i
    OUT.write_text(json.dumps(picked, ensure_ascii=False, indent=0), encoding="utf-8")
    dist = Counter(r["sonnet"] or "?" for r in picked)
    print(f"yazıldı: {OUT.relative_to(ROOT)}  ({len(picked)} kayıt, "
          f"{len({r['idiom'] for r in picked})} farklı deyim)  Sonnet ipucu dağılımı {dict(dist)}")
    print("Sonraki: deyim_etiketle.html'i tarayıcıda aç, bu dosyayı yükle.")


if __name__ == "__main__":
    main()
