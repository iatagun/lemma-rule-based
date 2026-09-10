#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Elle-etiketlenmiş span altın setini (`gold_labels.json`) repoya katar.

`deyim_etiketle.html` çıktısı, cümle başına:
    {idx, idiom, sentence, words, cand:{s,e}, spans:[{s,e,cat,sense}], visited, flag?, note?, sec?}
  cat  ∈ VID | LVC        sense ∈ idio (mecazi) | lit (literal kullanım)

Üç çıktı:
1. STAGE-2 verisi → `_corpus_sample_{records.jsonl,labels.tsv}`'ye APPEND-ONLY:
   aday öbeğin (cand) kaderi — spans'te idio bir öbek onu kapsıyorsa D (y=1),
   lit öbek kapsıyorsa veya kullanıcı sildiyse L (y=0). Ayrıca cand olmayan lit
   öbekler de birer L kaydı (literal kullanım örnekleri).
2. BIO ALTIN → `idiom_data/gold_spans.jsonl`: gezilen her cümle için
   {words, tags, spans} — tags yalnız idio öbeklerden (B/I-VID, B/I-LVC); lit öbek O.
   Düzgün span-tespiti değerlendirme seti (stage-1 / gelecek iş).
3. Notlar → `idiom_data/_missed_idioms.txt`.

Deyim düzeyinde bölme (`--eval-frac`): eval deyimleri `_holdout_idioms.json`'a eklenir
(stage-2 eğitici test'e yönlendirir); `_gold_eval_idioms.json` yeni temiz-altını kaydeder.

    python data/ingest_gold.py gold_labels.json
    python training/train_idiomaticity_clf.py --freeze 8 --dropout 0.3 --weight-decay 0.05 \
        --epochs 14 --out idiom_data/best_idiomaticity_clf_gold.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "idiom_data"
RECS = D / "_corpus_sample_records.jsonl"
LABS = D / "_corpus_sample_labels.tsv"
TSV = D / "_corpus_sample.tsv"
HOLD = D / "_holdout_idioms.json"
GOLD_EVAL = D / "_gold_eval_idioms.json"
GOLD_SPANS = D / "gold_spans.jsonl"
MISSED = D / "_missed_idioms.txt"


def bio(n: int, spans: list[dict]) -> list[str]:
    """idio öbeklerden BIO etiket dizisi (lit öbek O bırakılır)."""
    tags = ["O"] * n
    for sp in spans:
        if sp.get("sense") != "idio":
            continue
        cat = sp.get("cat", "VID")
        for i in range(sp["s"], min(sp["e"], n)):
            tags[i] = ("B-" if i == sp["s"] else "I-") + cat
    return tags


def covers(sp: dict, c: dict) -> float:
    """cand ile öbek örtüşme skoru (kapsıyorsa 1, yoksa IoU)."""
    if sp["s"] <= c["s"] and sp["e"] >= c["e"]:
        return 1.0
    lo, hi = max(sp["s"], c["s"]), min(sp["e"], c["e"])
    inter = max(0, hi - lo)
    union = (sp["e"] - sp["s"]) + (c["e"] - c["s"]) - inter
    return inter / union if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", help="gold_labels.json")
    ap.add_argument("--eval-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    gold = [g for g in gold if g.get("visited")]

    frozen_recs = [json.loads(l) for l in RECS.read_text(encoding="utf-8").splitlines() if l.strip()] if RECS.exists() else []
    frozen_txt = {" ".join(r["words"]) for r in frozen_recs}
    frozen_idioms = {r["idiom"] for r in frozen_recs}
    nxt = max((r["idx"] for r in frozen_recs), default=-1) + 1

    s2_rows: list[tuple] = []      # (idiom, words, tags, "D"/"L")
    span_rows: list[dict] = []     # BIO altın
    missed: list[tuple] = []
    kd = kl = k_new_lit = 0

    for g in gold:
        words, spans, cand = g["words"], g.get("spans", []), g["cand"]
        idiom = g["idiom"]
        if (g.get("note") or "").strip():
            missed.append((g["sentence"], g["note"].strip()))

        span_rows.append({"idiom": idiom, "words": words, "tags": bio(len(words), spans),
                          "spans": spans})

        # aday öbeğin stage-2 etiketi
        best, bscore = None, 0.35
        for sp in spans:
            sc = covers(sp, cand)
            if sc >= bscore:
                best, bscore = sp, sc
        if best and best.get("sense") == "idio":
            lab, tags = "D", bio(len(words), [{**best, "s": cand["s"], "e": cand["e"], "cat": "VID"}])
            kd += 1
        else:
            lab, tags = "L", ["O"] * len(words)     # literal ya da silinmiş aday
            kl += 1
        if " ".join(words) not in frozen_txt:
            s2_rows.append((idiom, words, tags, lab))

        # cand dışı literal öbekler → ek L kayıtları
        for sp in spans:
            if sp.get("sense") == "lit" and covers(sp, cand) < 0.35:
                s2_rows.append((idiom, words, ["O"] * len(words), "L"))
                k_new_lit += 1

    idioms = sorted({r[0] for r in s2_rows})
    rng.shuffle(idioms)
    n_eval = round(len(idioms) * args.eval_frac)
    eval_idioms = set(idioms[:n_eval])

    print(f"gezilen cümle: {len(gold)}  →  stage-2: {kd} D / {kl} L (+{k_new_lit} literal öbek)")
    print(f"BIO altın: {len(span_rows)} cümle, "
          f"{sum(1 for r in span_rows for t in r['tags'] if t.startswith('B-'))} idio öbek")
    print(f"bölme: {len(eval_idioms)} deyim eval / {len(idioms)-len(eval_idioms)} train  "
          f"({len(missed)} not)")
    overlap = {r[0] for r in s2_rows} & frozen_idioms
    if overlap:
        print(f"NOT: {len(overlap)} deyim zaten frozen sette (stage-2'de tekrar olmaz, "
              f"cümle-metni farklıysa eklenir)")

    if args.dry_run:
        print("(dry-run)")
        return

    new_recs, new_labs = [], []
    for idiom, words, tags, y in s2_rows:
        span = " ".join(w for w, t in zip(words, tags) if t != "O") or idiom
        new_recs.append({"idx": nxt, "words": words, "tags": tags, "idiom": idiom, "span": span})
        new_labs.append((nxt, y)); nxt += 1
    with RECS.open("a", encoding="utf-8") as f:
        for r in new_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with LABS.open("a", encoding="utf-8") as f:
        for i, y in new_labs:
            f.write(f"{i}\t{y}\n")
    with TSV.open("a", encoding="utf-8") as f:
        for r in new_recs:
            f.write(f"{r['idx']}\t{r['span']}\t{' '.join(r['words'])}\n")

    with GOLD_SPANS.open("a", encoding="utf-8") as f:
        for r in span_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hold = set(json.loads(HOLD.read_text(encoding="utf-8"))) if HOLD.exists() else set()
    hold |= eval_idioms
    HOLD.write_text(json.dumps(sorted(hold), ensure_ascii=False), encoding="utf-8")
    prev = set(json.loads(GOLD_EVAL.read_text(encoding="utf-8"))) if GOLD_EVAL.exists() else set()
    GOLD_EVAL.write_text(json.dumps(sorted(prev | eval_idioms), ensure_ascii=False), encoding="utf-8")

    if missed:
        with MISSED.open("a", encoding="utf-8") as f:
            for s, n in missed:
                f.write(f"{n}\t{s}\n")

    print(f"eklendi: stage-2 {len(new_recs)} kayıt (idx {new_recs[0]['idx']}..{nxt-1}, ilk 2173 dokunulmadı), "
          f"BIO altın {len(span_rows)} → {GOLD_SPANS.name}")
    print(f"held-out artık {len(hold)} deyim ({len(eval_idioms)} yeni)")
    print("Sonraki: python training/train_idiomaticity_clf.py --freeze 8 --dropout 0.3 "
          "--weight-decay 0.05 --epochs 14 --out idiom_data/best_idiomaticity_clf_gold.pt")


if __name__ == "__main__":
    main()
