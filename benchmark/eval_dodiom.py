#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom'u Dodiom Türkçe crowdsourced deyim veri setinde (Eryiğit, Şentaş & Monti
2022, 6861 örnek / 36 deyim, idiom/nonidiom ikili etiket + hedef span) değerlendirir.

Umut et al. (2025, UBMK) "Exploring Turkish Idiomaticity with LLMs" makalesinin kendi veri
seti/kodu yayınlanmadığı (IEEE-arkalı) için YERİNE KOYMA olarak kullanılıyor.

**Metodoloji notları:**
- "Dengelenmiş doğruluk" (balanced accuracy) Çavuşoğlu'nun çift-düzeyi "doğru-ayırt"
  metriğiyle AYNI ŞEY DEĞİL — ayrı sütun/başlık altında, EŞLEŞTİRİLMEMİŞ cümle-düzeyi bir
  ortalama olarak raporlanıyor.
- **Deyim-kimliği örtüşmesi:** 36 deyimin 34'ü Aşama 1'in TAM eğitim havuzunda (TDK +
  Leipzig-madenli frozen havuz) — bu **Aşama-1 açısından** neredeyse hiç "görülmemiş deyim"
  testi değil. Aşama-2'nin (idyomatiklik/bağlam ayrımı) KENDİ etiketli eğitim havuzuyla
  örtüşme 0/36 — yani bu, **Aşama-2 açısından görülmemiş** bir test (Aşama-1 açısından değil).
  Bu ayrım kartta da böyle yazılmalı: "dış kaynak, görülmemiş" DEĞİL, "Aşama-2 açısından
  görülmemiş deyim kimlikleri, doğal crowdsourced cümleler".
- **Etkin örneklem 36 deyim, 6861 satır DEĞİL** — güven aralığı deyim-kümesi (cluster)
  bootstrap ile hesaplanıyor.
- stage2 açık/kapalı AYRI raporlanıyor VE aralarındaki fark deyim-kümesi EŞLEŞTİRİLMİŞ
  bootstrap ile test ediliyor (aynı satırlar, iki koşul — bağımsız değil).

Kullanım:
    python data/fetch_dodiom_tr.py
    python benchmark/eval_dodiom.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.stats_utils import (cluster_paired_balanced_acc_diff_ci,  # noqa: E402
                                    cluster_proportion_ci, per_cluster_rates)

CSV_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "dodiom_tr.csv"


def load_rows(strict: bool, min_rating: float) -> list[dict]:
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["category"] in ("idiom", "nonidiom")]
    if strict:
        rows = [r for r in rows if r.get("dislikes", "0") == "0" and r.get("reports", "0") == "0"]
    if min_rating > 0:
        rows = [r for r in rows if float(r.get("rating", "0") or 0) >= min_rating]
    return rows


def full_training_pool_stems() -> set[tuple]:
    from data.prepare_tdk_idiom_examples import idiom_stems

    data_dir = PROJECT_ROOT / "idiom_data"
    frozen = set()
    recs = data_dir / "_corpus_sample_records.jsonl"
    if recs.exists():
        for l in recs.read_text(encoding="utf-8").splitlines():
            if l.strip():
                frozen.add(json.loads(l)["idiom"])
    tdk_split_path = data_dir / "tdk_idioms_by_split.json"
    tdk = set()
    if tdk_split_path.exists():
        d = json.loads(tdk_split_path.read_text(encoding="utf-8"))
        for v in d.values():
            tdk.update(v)
    return {tuple(idiom_stems(i)) for i in (frozen | tdk) if idiom_stems(i)}


def stage2_pool_idioms() -> set[str]:
    p = PROJECT_ROOT / "idiom_data" / "corpus_examples_glu.json"
    if not p.exists():
        return set()
    pool = json.loads(p.read_text(encoding="utf-8"))
    return {r["idiom"] for r in pool if r.get("idiom")}


def hit_at_target(spans: list[dict], rng: tuple[int, int]) -> bool:
    ts, te = rng
    return any(sp["start"] < te and ts < sp["end"] for sp in spans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="iatagun/DizgeBERT-Idiom")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--min-rating", type=float, default=0.0)
    args = ap.parse_args()

    if not CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} yok — önce `python data/fetch_dodiom_tr.py`.")

    import torch
    from transformers import AutoModel, AutoTokenizer
    from data.prepare_tdk_idiom_examples import idiom_stems

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.hf_repo, trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.hf_repo)

    rows = load_rows(args.strict, args.min_rating)
    idiom_rows = [r for r in rows if r["category"] == "idiom"]
    lit_rows = [r for r in rows if r["category"] == "nonidiom"]
    n_idioms = sorted(set(r["idiom"] for r in rows))
    pool = full_training_pool_stems()
    s2_pool = stage2_pool_idioms()
    seen_stage1 = [i for i in n_idioms if tuple(idiom_stems(i)) in pool]
    seen_stage2 = [i for i in n_idioms if i in s2_pool]
    print(f"=== Dodiom TR — {len(rows)} satır, {len(n_idioms)} benzersiz deyim — model: {args.hf_repo} ===")
    print(f"  Aşama-1 (span) eğitim havuzuyla örtüşme: {len(seen_stage1)}/{len(n_idioms)} deyim"
          " (span-bulma açısından NEREDEYSE HİÇ görülmemiş-deyim testi değil)")
    print(f"  Aşama-2 (idyomatiklik) eğitim havuzuyla örtüşme: {len(seen_stage2)}/{len(n_idioms)} deyim"
          " (Aşama-2 açısından GERÇEKTEN görülmemiş)")

    def target_rng(r):
        s, e = ast.literal_eval(r["idiom_indices"])
        return (s, e + 1)

    results = {}
    for stage2 in (True, False):
        with torch.no_grad():
            sens_hits = [hit_at_target(model.predict_spans(ast.literal_eval(r["words"]),
                                                            tokenizer=tok, stage2=stage2),
                                        target_rng(r)) for r in idiom_rows]
            fp_hits = [hit_at_target(model.predict_spans(ast.literal_eval(r["words"]),
                                                          tokenizer=tok, stage2=stage2),
                                      target_rng(r)) for r in lit_rows]
        results[stage2] = sens_hits, fp_hits

    sens_clusters = [r["idiom"] for r in idiom_rows]
    fp_clusters = [r["idiom"] for r in lit_rows]

    for stage2 in (True, False):
        sens_hits, fp_hits = results[stage2]
        sens, sens_lo, sens_hi = cluster_proportion_ci(sens_hits, sens_clusters)
        fpr, fpr_lo, fpr_hi = cluster_proportion_ci(fp_hits, fp_clusters)
        bal_acc = (sens + (1 - fpr)) / 2
        print(f"\n--- stage2={stage2} ---")
        print(f"  duyarlılık: %{100*sens:.1f} (95% GA %{100*sens_lo:.1f}-%{100*sens_hi:.1f}, n={len(idiom_rows)} satır)")
        print(f"  yanlış-poz: %{100*fpr:.1f} (95% GA %{100*fpr_lo:.1f}-%{100*fpr_hi:.1f}, n={len(lit_rows)} satır)")
        print(f"  DENGELENMİŞ DOĞRULUK (eşleştirilmemiş): %{100*bal_acc:.1f}")
        per_idiom_sens = per_cluster_rates(sens_hits, sens_clusters)
        per_idiom_fp = per_cluster_rates(fp_hits, fp_clusters)
        all_idioms = sorted(set(per_idiom_sens) | set(per_idiom_fp))
        worst = sorted(all_idioms, key=lambda i: per_idiom_sens.get(i, 1) - per_idiom_fp.get(i, 0))[:5]
        print("  en zayıf 5 deyim: " + ", ".join(
            f"{i}(duy={100*per_idiom_sens.get(i,float('nan')):.0f}/fp={100*per_idiom_fp.get(i,float('nan')):.0f})"
            for i in worst))

    diff, lo, hi = cluster_paired_balanced_acc_diff_ci(
        results[False][0], sens_clusters, results[False][1], fp_clusters,
        results[True][0], results[True][1])
    print(f"\n  EŞLEŞTİRİLMİŞ fark (dengelenmiş doğruluk, stage2=True − stage2=False),"
          f" deyim-kümesi bootstrap: {100*diff:+.1f}pp (95% GA {100*lo:+.1f}pp – {100*hi:+.1f}pp)")
    verdict = "sıfırı İÇERMİYOR — istatistiksel olarak anlamlı" if (lo > 0 or hi < 0) \
        else "sıfırı İÇERİYOR — bu ölçekte (n=36 deyim) istatistiksel olarak ayırt edilemiyor"
    print(f"  → {verdict}")


if __name__ == "__main__":
    main()
