#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pilot (2026-09-14) — hedefli-derin deyim etiketleme.

Deney F (`benchmark/calibrate_stage2.py`) gösterdi: stage-2 per-deyim kalibrasyonunun
darboğazı veri KALİTESİ değil, deyim başına örnek YOĞUNLUĞU (1059 deyimden yalnız 87'si
≥3 örnek + hem D hem L içeriyor). Bu script BELİRLİ bir deyim listesine (varsayılan:
Çavuşoğlu&Çöltekin dış-benchmark'ıyla örtüşen, `corpus_examples.json`'da zaten yeterli ham
örneği olan deyimler) DERİNLEMESİNE yeni etiketli örnek ekler — rastgele geniş kapsam değil,
dar-hedefli yoğunluk artışı. `filter_corpus_idiomaticity.py`'nin `--ingest-llm`'inden farkı:
o FROZEN deyimleri (zaten kapsanan) hariç tutar (kapsam genişletme amaçlı); bu script tam
tersini yapar — KISMEN kapsanan deyimlere KASITLI olarak daha fazla örnek ekler (yoğunluk
amaçlı), `call_llm_voted` (kendinden-tutarlılık oylaması) ile.

Kullanım:
    python data/deep_label_idioms.py --n-idioms 25 --per-idiom 10 --votes 3 \
        --base-url https://api.anthropic.com/v1 --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.filter_corpus_idiomaticity import call_llm_voted  # noqa: E402 — tek kaynak, yeniden yazma

CORPUS = PROJECT_ROOT / "idiom_data" / "corpus_examples.json"
BENCH_TSV = PROJECT_ROOT / "idiom_data" / "raw" / "turkish_idioms_benchmark.tsv"
RECS = PROJECT_ROOT / "idiom_data" / "_corpus_sample_records.jsonl"
LABELS = PROJECT_ROOT / "idiom_data" / "_corpus_sample_labels.tsv"


def pick_target_idioms(n: int, seed: int) -> list[str]:
    """Çavuşoğlu benchmark'ıyla isim-birebir örtüşen VE corpus_examples.json'da ≥15 ham
    örneği olan deyimlerden rastgele n tanesi — düşük maliyetli pilot için."""
    import random
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    from collections import Counter
    counts = Counter(r["idiom"] for r in corpus)
    bench = {r["idiom"] for r in csv.DictReader(BENCH_TSV.open(encoding="utf-8"), delimiter="\t")
             if r.get("sample", "").strip() and r.get("literal", "").strip()}
    eligible = sorted(k for k in bench if counts.get(k, 0) >= 15)
    random.Random(seed).shuffle(eligible)
    return eligible[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-idioms", type=int, default=25)
    ap.add_argument("--per-idiom", type=int, default=10)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true", help="yalnız hedef deyim listesini yazdır, LLM çağırma")
    args = ap.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    targets = pick_target_idioms(args.n_idioms, args.seed)
    print(f"hedef deyim ({len(targets)}): {targets}")
    if args.dry_run:
        return

    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    by_idiom: dict[str, list[dict]] = {}
    for r in corpus:
        by_idiom.setdefault(r["idiom"], []).append(r)

    seen_texts: set[str] = set()
    nxt = 0
    if RECS.exists():
        for l in RECS.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                seen_texts.add(" ".join(r["words"]))
                nxt = max(nxt, r["idx"] + 1)

    import random
    rng = random.Random(args.seed + 1)
    candidates: list[dict] = []
    for idm in targets:
        pool = [r for r in by_idiom.get(idm, []) if " ".join(r["words"]) not in seen_texts]
        rng.shuffle(pool)
        candidates.extend(pool[:args.per_idiom])
    print(f"etiketlenecek yeni cümle: {len(candidates)}")

    t = time.time()
    all_labels: dict[int, str] = {}
    for b0 in range(0, len(candidates), args.batch):
        batch = candidates[b0:b0 + args.batch]
        res, agr = call_llm_voted(args.base_url, args.model, api_key, batch, args.timeout, args.votes)
        for pos in range(len(batch)):
            all_labels[b0 + pos] = res.get(pos + 1, "N")
        print(f"  {min(b0+args.batch, len(candidates))}/{len(candidates)}  {time.time()-t:.0f}s")

    new_recs, new_lab_lines, dist = [], [], {}
    for i, it in enumerate(candidates):
        lab = "D" if all_labels.get(i) == "D" else "L"
        dist[lab] = dist.get(lab, 0) + 1
        idx = nxt + i
        new_recs.append({"idx": idx, "words": it["words"], "tags": it["tags"],
                          "idiom": it["idiom"], "span": it["span"]})
        new_lab_lines.append(f"{idx}\t{lab}")

    with RECS.open("a", encoding="utf-8") as f:
        for r in new_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with LABELS.open("a", encoding="utf-8") as f:
        for line in new_lab_lines:
            f.write(line + "\n")

    print(f"\netiket dağılımı: {dist}")
    print(f"eklendi: {len(new_recs)} kayıt (idx {nxt}..{nxt+len(new_recs)-1}) → "
          f"{RECS.name} / {LABELS.name}")
    print("Sonraki: python benchmark/calibrate_stage2.py --clf-checkpoint idiom_data/best_idiomaticity_clf_v3.pt")


if __name__ == "__main__":
    main()
