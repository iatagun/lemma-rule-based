#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom'u Aslantaş & Güngör'ün Turkic Idiom Understanding Benchmark'ının Türkçe
test diliminde (131 cümle, tek-sınıf B-IDIOM/I-IDIOM/O) değerlendirir.

**Metrik notu (2026-09-20 düzeltmesi):** onların `src/modeling/metrics.py`'si seqeval ile
ENTİTY-düzeyi (span sınırları BİREBİR tutmalı) P/R/F1 hesaplıyor — Table 2'deki %87.7/%88.0
bu ölçütle. İlk sürümümüz TOKEN-düzeyi (kısmi örtüşmeye kısmi puan veren) bir metrik
kullanıyordu — onlarla DOĞRUDAN kıyaslanamaz, daha toleranslı. Şimdi ikisi de raporlanıyor,
seqeval birincil (Table 2 ile karşılaştırılabilir), token-düzeyi ikincil/referans.

**Deyim-kimliği örtüşmesi notu:** bu 131 test idiomunun **69'u (%54)** bizim Aşama 1
eğitim havuzumuzda (TDK+Leipzig-madenli frozen havuz) zaten var — yani CÜMLE-düzeyinde
sıfır-atış (bu cümleleri hiç görmedik) ama DEYİM-kimliği düzeyinde sıfır-atış DEĞİL. Metrik
her iki alt-kümede AYRI raporlanıyor.

Kullanım:
    python data/fetch_aslantas_gungor_tr.py
    pip install seqeval
    python benchmark/eval_aslantas_gungor.py --hf
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

from benchmark.stats_utils import paired_diff_ci  # noqa: E402

CSV_PATH = PROJECT_ROOT / "idiom_data" / "raw" / "aslantas_gungor_multi_turkic.csv"


def load_test_rows() -> list[dict]:
    if not CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} yok — önce `python data/fetch_aslantas_gungor_tr.py`.")
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["lang"] == "TR" and r["split"] == "test"]


def full_training_pool_stems() -> set[tuple]:
    """Stage-1'in görmüş olabileceği HER idiom kimliği (frozen Leipzig-madenli havuz ∪ TDK
    split lexicon'u) — stem-eşleştirilmiş. PARSEME'nin kendi VMWE lemma'ları dahil DEĞİL
    (format farkı nedeniyle karşılaştırılmadı — bilinen bir eksik, bkz. proje notları)."""
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


def to_binary_tags(spans: list[dict], n_tokens: int) -> list[str]:
    tags = ["O"] * n_tokens
    for sp in spans:
        s, e = sp["start"], sp["end"]
        if 0 <= s < e <= n_tokens:
            tags[s] = "B-IDIOM"
            for i in range(s + 1, e):
                tags[i] = "I-IDIOM"
        if sp.get("gappy"):
            s2, e2 = sp["start2"], sp["end2"]
            if 0 <= s2 < e2 <= n_tokens:
                tags[s2] = "B-IDIOM" if tags[s2] == "O" else tags[s2]
                for i in range(s2 + 1, e2):
                    if tags[i] == "O":
                        tags[i] = "I-IDIOM"
    return tags


def token_prf(golds: list[list[str]], preds: list[list[str]]) -> tuple[float, float, float, float]:
    tp = fp_ = fn = tn = 0
    for gold, pred in zip(golds, preds):
        for g, p in zip(gold, pred):
            g_pos, p_pos = g != "O", p != "O"
            tp += g_pos and p_pos
            fp_ += p_pos and not g_pos
            fn += g_pos and not p_pos
            tn += not g_pos and not p_pos
    prec = tp / (tp + fp_) if (tp + fp_) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp_ + fn) if (tp + tn + fp_ + fn) else 0.0
    return prec, rec, f1, acc


def seqeval_f1(golds: list[list[str]], preds: list[list[str]]) -> float:
    from seqeval.metrics import f1_score
    return f1_score(golds, preds)


def bootstrap_f1_ci(golds, preds, n_boot=2000, seed=0) -> tuple[float, float, float]:
    import random
    rng = random.Random(seed)
    n = len(golds)
    point = seqeval_f1(golds, preds)
    boots = []
    idx = list(range(n))
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        boots.append(seqeval_f1([golds[i] for i in sample], [preds[i] for i in sample]))
    boots.sort()
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="iatagun/DizgeBERT-Idiom")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    try:
        import seqeval  # noqa: F401
    except ImportError:
        sys.exit("pip install seqeval  (onlarla aynı entity-düzeyi metrik için gerekli)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.hf_repo, trust_remote_code=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.hf_repo)

    rows = load_test_rows()
    pool = full_training_pool_stems()
    from data.prepare_tdk_idiom_examples import idiom_stems
    seen_mask = [tuple(idiom_stems(r["idiom"])) in pool for r in rows]
    print(f"=== Aslantaş & Güngör TR test seti, {len(rows)} cümle — model: {args.hf_repo} ===")
    print(f"  deyim-kimliği örtüşmesi (Aşama-1 eğitim havuzuyla): {sum(seen_mask)}/{len(rows)}"
          f" zaten görülmüş, {len(rows)-sum(seen_mask)} gerçekten görülmemiş\n")

    golds_all = [ast.literal_eval(r["labels"]) for r in rows]
    words_all = [ast.literal_eval(r["tokens"]) for r in rows]

    for stage2 in (False, True):
        preds_all = [to_binary_tags(model.predict_spans(ws, tokenizer=tok, stage2=stage2), len(ws))
                     for ws in words_all]
        tag = "stage2=False (adil kıyas — onların görevinde bağlam-ayrımı yok)" if not stage2 \
            else "stage2=True (yayınlanan varsayılan davranış)"
        print(f"--- {tag} ---")
        for subset_name, mask_val in [("TÜMÜ", None), ("deyim-kimliği GÖRÜLMÜŞ", True),
                                       ("deyim-kimliği GERÇEKTEN GÖRÜLMEMİŞ", False)]:
            idxs = range(len(rows)) if mask_val is None else [i for i in range(len(rows)) if seen_mask[i] == mask_val]
            if not idxs:
                continue
            g = [golds_all[i] for i in idxs]
            p = [preds_all[i] for i in idxs]
            prec_t, rec_t, f1_t, acc_t = token_prf(g, p)
            f1_point, lo, hi = bootstrap_f1_ci(g, p)
            print(f"    [{subset_name}, n={len(idxs)}]  seqeval-entity-F1={f1_point:.3f}"
                  f" (95% GA {lo:.3f}-{hi:.3f})   token-düzeyi P={prec_t:.3f} R={rec_t:.3f}"
                  f" F1={f1_t:.3f} Acc={acc_t:.3f}")
        if not stage2:
            preds_off = preds_all
        else:
            preds_on = preds_all

    # eşleştirilmiş (aynı 131 cümle) stage2 açık/kapalı seqeval-F1 farkı — kaba: her cümle
    # için entity-match'i 0/1'e indirip paired bootstrap (tam F1 değil ama yönü/anlamlılığı gösterir)
    def sent_exact(gold, pred):
        return gold == pred

    hits_off = [sent_exact(g, p) for g, p in zip(golds_all, preds_off)]
    hits_on = [sent_exact(g, p) for g, p in zip(golds_all, preds_on)]
    diff, lo, hi = paired_diff_ci(hits_off, hits_on)
    print(f"\n  eşleştirilmiş fark (tam-cümle-etiket eşleşmesi, stage2=True − stage2=False):"
          f" {100*diff:+.1f}pp (95% GA {100*lo:+.1f}pp – {100*hi:+.1f}pp)")

    print("\n  Onların Table 2'si (seqeval entity-F1, TR-only fine-tune, TR test, in-domain):"
          " ELECTRA-tr F1=.877, ConvBERT-tr F1=.880, BERTurk F1=.822.")


if __name__ == "__main__":
    main()
