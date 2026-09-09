#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Elle-etiketlenmiş altın seti (`gold_labels.json`) stage-2 verisine katar.

`deyim_etiketle.html` çıktısı: [{idx, idiom, span, sentence, label}]  label ∈ D/N/S.
`annotation_batch.json` ile idx üzerinden join edilip `words`/`tags` alınır.

- Deyim düzeyinde bölünür (`--eval-frac`, varsayılan 0.5): eval deyimleri
  `_holdout_idioms.json`'a EKLENİR → `train_idiomaticity_clf.py` bunları otomatik
  test setine yönlendirir. Mevcut 118 held-out KORUNUR.
- Kayıtlar `_corpus_sample_{records.jsonl,labels.tsv}`'ye APPEND-ONLY eklenir
  (D→"D" y=1, N→"L" y=0; S atlanır). İlk 2173 satır dokunulmaz.
- Yeni eval deyimleri `_gold_eval_idioms.json`'a yazılır (temiz-altın eval kaydı).

Kullanım:
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
BATCH = D / "annotation_batch.json"
RECS = D / "_corpus_sample_records.jsonl"
LABS = D / "_corpus_sample_labels.tsv"
TSV = D / "_corpus_sample.tsv"
HOLD = D / "_holdout_idioms.json"
GOLD_EVAL = D / "_gold_eval_idioms.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", help="gold_labels.json (HTML aracının çıktısı)")
    ap.add_argument("--eval-frac", type=float, default=0.5, help="eval'a giden deyim oranı")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    batch = {r["idx"]: r for r in json.loads(BATCH.read_text(encoding="utf-8"))}
    frozen_recs = [json.loads(l) for l in RECS.read_text(encoding="utf-8").splitlines() if l.strip()]
    frozen_idioms = {r["idiom"] for r in frozen_recs}
    frozen_txt = {" ".join(r["words"]) for r in frozen_recs}
    nxt = max((r["idx"] for r in frozen_recs), default=-1) + 1

    rows = []           # (idiom, words, tags, y_label "D"/"L")
    skip_s = skip_dup = skip_nojoin = 0
    for g in gold:
        lb = (g.get("label") or "").upper()
        if lb not in ("D", "N"):
            skip_s += 1
            continue
        b = batch.get(g["idx"])
        if b is None or b["sentence"] != g.get("sentence", b["sentence"]):
            skip_nojoin += 1
            continue
        if b["sentence"] in frozen_txt:
            skip_dup += 1
            continue
        tags = b["tags"] if lb == "D" else ["O"] * len(b["words"])
        rows.append((b["idiom"], b["words"], tags, "D" if lb == "D" else "L"))

    by_idiom: dict[str, list] = {}
    for r in rows:
        by_idiom.setdefault(r[0], []).append(r)
    idioms = sorted(by_idiom)
    rng.shuffle(idioms)
    n_eval = round(len(idioms) * args.eval_frac)
    eval_idioms = set(idioms[:n_eval])
    train_idioms = set(idioms[n_eval:])

    n_d = sum(1 for r in rows if r[3] == "D")
    print(f"altın: {len(gold)} kayıt → {len(rows)} kullanılabilir ({n_d} D / {len(rows)-n_d} N), "
          f"{len(idioms)} deyim  (atlandı: {skip_s} S, {skip_dup} tekrar, {skip_nojoin} join-yok)")
    print(f"bölme: {len(eval_idioms)} deyim eval (held-out'a eklenecek) / {len(train_idioms)} deyim train")
    overlap = (eval_idioms | train_idioms) & frozen_idioms
    if overlap:
        print(f"UYARI: {len(overlap)} deyim zaten frozen sette — {sorted(overlap)[:5]}...")

    if args.dry_run:
        print("(dry-run — yazılmadı)")
        return

    new_recs, new_labs = [], []
    for idiom, words, tags, y in rows:
        new_recs.append({"idx": nxt, "words": words, "tags": tags, "idiom": idiom, "span": " ".join(
            w for w, t in zip(words, tags) if t != "O") or idiom})
        new_labs.append((nxt, y))
        nxt += 1
    with RECS.open("a", encoding="utf-8") as f:
        for r in new_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with LABS.open("a", encoding="utf-8") as f:
        for i, y in new_labs:
            f.write(f"{i}\t{y}\n")
    with TSV.open("a", encoding="utf-8") as f:
        for r in new_recs:
            f.write(f"{r['idx']}\t{r['span']}\t{' '.join(r['words'])}\n")

    hold = set(json.loads(HOLD.read_text(encoding="utf-8"))) if HOLD.exists() else set()
    hold |= eval_idioms
    HOLD.write_text(json.dumps(sorted(hold), ensure_ascii=False), encoding="utf-8")
    prev_ge = set(json.loads(GOLD_EVAL.read_text(encoding="utf-8"))) if GOLD_EVAL.exists() else set()
    GOLD_EVAL.write_text(json.dumps(sorted(prev_ge | eval_idioms), ensure_ascii=False), encoding="utf-8")

    print(f"eklendi: {len(new_recs)} kayıt (idx {new_recs[0]['idx']}..{nxt-1}), ilk 2173 dokunulmadı")
    print(f"held-out artık {len(hold)} deyim  ({len(eval_idioms)} yeni temiz-altın)")
    print("Sonraki: python training/train_idiomaticity_clf.py --freeze 8 --dropout 0.3 "
          "--weight-decay 0.05 --epochs 14 --out idiom_data/best_idiomaticity_clf_gold.pt")


if __name__ == "__main__":
    main()
