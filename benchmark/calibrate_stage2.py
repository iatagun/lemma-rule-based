#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deney F — gerçek etiketli komşularla per-deyim stage-2 eşik kalibrasyonu.

Reddedilen per-deyim eşik denemesi (2026-09-10) SENTETİK referans (çıplak span / şablon
cümle) kullanmıştı; gerçek Çavuşoğlu&Çöltekin altın-çift dağılımıyla yalnız r≈0.30 korele
çıktı — o yüzden terk edildi. Bu script AYNI fikri GERÇEK etiketli komşularla dener:
`idiom_data/gold_spans.jsonl` (601 elle-doğrulanmış kayıt, sense idio/lit doğrudan verili)
ve `idiom_data/_corpus_sample_records.jsonl`+`_corpus_sample_labels.tsv` (D/L etiketli
stage-2 eğitim havuzu) — ikisi de GERÇEK deyim kimliğiyle (`idiom` alanı) etiketli.

≥`--min-examples` gerçek örneği olan (ve hem D hem L içeren) her deyim kimliği için mevcut
`span_p_literal()` (tek kaynak) ile o deyimin gerçek p(idyomatik) dağılımını ölçüp per-deyim
eşik türetir (D-ortalama ile L-ortalama arası orta nokta) → `idiom_data/_stage2_per_idiom_thresh.json`.

KRİTİK DİSİPLİN: kalibrasyon-seti sinyali (mean_D/mean_L) ile TUTULAN (held-out) Çavuşoğlu
altın-çift skoru arasındaki Pearson r'yi raporlar — reddedilen sentetik denemeyi öldüren
ölçüt buydu (r≈0.30). Gerçek veri de bunu geçemezse, "daha iyi görünen veriyle aynı hata"dır;
iterasyon yapılmaz (bkz. plan dosyası, Deney F).

Kullanım:
    python benchmark/calibrate_stage2.py --clf-checkpoint idiom_data/best_idiomaticity_clf_v3.pt
    python benchmark/eval_idiom.py --stage2 idiom_data/best_idiomaticity_clf_v3.pt \\
        --mode stage2-iso --per-idiom-thresh-file idiom_data/_stage2_per_idiom_thresh.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLD_SPANS = PROJECT_ROOT / "idiom_data" / "gold_spans.jsonl"
CORPUS_RECORDS = PROJECT_ROOT / "idiom_data" / "_corpus_sample_records.jsonl"
CORPUS_LABELS = PROJECT_ROOT / "idiom_data" / "_corpus_sample_labels.tsv"
BENCH_TSV = PROJECT_ROOT / "idiom_data" / "raw" / "turkish_idioms_benchmark.tsv"
OUT_THRESH = PROJECT_ROOT / "idiom_data" / "_stage2_per_idiom_thresh.json"


def _idiom_key(text: str, idiom_stems_fn) -> str:
    return " ".join(idiom_stems_fn(text))


def load_calibration_examples(idiom_stems_fn) -> list[tuple[str, list[str], int, int, bool]]:
    """→ [(deyim_kimlik_anahtarı, words, start, end, is_idiomatik)] — iki gerçek-etiketli
    kaynağın birleşimi. `span_from_tags` (train_idiomaticity_clf.py, tek kaynak) tags→span
    dönüşümünü tekrar yazmadan kullanır."""
    from training.train_idiomaticity_clf import span_from_tags

    out: list[tuple[str, list[str], int, int, bool]] = []
    if GOLD_SPANS.exists():
        for line in GOLD_SPANS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = _idiom_key(r["idiom"], idiom_stems_fn)
            for sp in r.get("spans", []):
                out.append((key, r["words"], sp["s"], sp["e"], sp["sense"] == "idio"))

    if CORPUS_RECORDS.exists() and CORPUS_LABELS.exists():
        labels: dict[int, str] = {}
        with CORPUS_LABELS.open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["label"] in ("D", "L"):
                    labels[int(row["idx"])] = row["label"]
        for line in CORPUS_RECORDS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            lab = labels.get(r["idx"])
            if lab is None:
                continue
            sp = span_from_tags(r["tags"])
            if sp is None:
                continue
            key = _idiom_key(r["idiom"], idiom_stems_fn)
            out.append((key, r["words"], sp[0], sp[1], lab == "D"))
    return out


def build_scorer(clf_ckpt: str):
    """→ p_idio(words, s, e) -> float|None. `span_p_literal` (tek kaynak) ile aynı hesap;
    `benchmark/eval_idiom.py: run_stage2_iso`'daki yerel kopyayla bire bir aynı mantık."""
    import torch
    from transformers import AutoTokenizer
    from dizgebert_idiom.modeling_dizgebert_idiom import align_words, span_p_literal
    from training.train_idiomaticity_clf import IdiomaticityClf, MAX_LEN

    ck = torch.load(clf_ckpt, map_location="cpu")
    enc_name = ck.get("encoder", "dbmdz/electra-base-turkish-cased-discriminator")
    tok = AutoTokenizer.from_pretrained(enc_name)
    clf = IdiomaticityClf(enc_name).eval()
    clf.load_state_dict(ck["model"])

    @torch.no_grad()
    def p_idio(words, s, e):
        enc, kept, fp, lp = align_words(tok, words, MAX_LEN)
        if s >= len(kept) or (e - 1) >= len(kept):
            return None
        hs = clf.encoder(input_ids=enc["input_ids"],
                         attention_mask=enc["attention_mask"]).last_hidden_state[0]
        return 1.0 - span_p_literal(hs, fp[0, s].item(), lp[0, e - 1].item(), clf.head)

    return p_idio


def build_thresholds(examples, scores, min_examples: int):
    """deyim-anahtarı → (eşik, mean_D, mean_L, n_D, n_L). Yalnız hem D hem L örneği olan VE
    toplamda >= min_examples örneği olan deyimler kalibre edilir — tek-sınıflı deyim için
    "orta nokta" tanımsız (ayrımı ölçecek karşıt sınıf yok)."""
    from collections import defaultdict
    by_key: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for (key, _words, _s, _e, is_idio), score in zip(examples, scores):
        if score is None:
            continue
        by_key[key].append((score, is_idio))

    thresh: dict[str, float] = {}
    rows = []
    for key, items in by_key.items():
        d = [s for s, y in items if y]
        l = [s for s, y in items if not y]
        if len(items) < min_examples or not d or not l:
            continue
        md, ml = sum(d) / len(d), sum(l) / len(l)
        thresh[key] = (md + ml) / 2
        rows.append((key, thresh[key], md, ml, len(d), len(l)))
    return thresh, rows


def score_benchmark_pairs(scorer, idiom_stems_fn, wanted_keys: set[str]) -> dict[str, tuple[float, float]]:
    """kalibre edilmiş deyimlerin Çavuşoğlu altın-çiftindeki GERÇEK (idyomatik,literal) skoru
    — `run_stage2_iso`'nun `_iso_locate` fuzzy içerik-sözcük eşleştiricisiyle aynı konumlama."""
    if not BENCH_TSV.exists():
        return {}
    from benchmark.eval_idiom import _iso_locate
    from data.prepare_tdk_idiom_examples import stem

    rows = [r for r in csv.DictReader(BENCH_TSV.open(encoding="utf-8"), delimiter="\t")
            if r.get("sample", "").strip() and r.get("literal", "").strip()]
    out: dict[str, tuple[float, float]] = {}
    for r in rows:
        key = _idiom_key(r["idiom"], idiom_stems_fn)
        if key not in wanted_keys or key in out:
            continue
        sw, lw = r["sample"].split(), r["literal"].split()
        if len(sw) < 2 or len(lw) < 2:
            continue
        sr, lr = _iso_locate(r["idiom"], sw, stem), _iso_locate(r["idiom"], lw, stem)
        if not sr or not lr:
            continue
        pi, pl = scorer(sw, *sr), scorer(lw, *lr)
        if pi is None or pl is None:
            continue
        out[key] = (pi, pl)
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf-checkpoint", default=str(PROJECT_ROOT / "idiom_data" / "best_idiomaticity_clf_v3.pt"))
    ap.add_argument("--min-examples", type=int, default=3,
                    help="deyim başına gereken minimum GERÇEK etiketli örnek (D+L toplam)")
    ap.add_argument("--out", default=str(OUT_THRESH))
    args = ap.parse_args()

    from data.prepare_tdk_idiom_examples import idiom_stems

    examples = load_calibration_examples(idiom_stems)
    print(f"kalibrasyon havuzu: {len(examples)} etiketli örnek "
          f"({len({k for k, *_ in examples})} benzersiz deyim kimliği)")

    scorer = build_scorer(args.clf_checkpoint)
    scores = [scorer(words, s, e) for _, words, s, e, _ in examples]
    n_scored = sum(s is not None for s in scores)
    print(f"skorlanan: {n_scored}/{len(examples)} (kalan MAX_LEN kırpmasına takıldı)")

    thresh, rows = build_thresholds(examples, scores, args.min_examples)
    print(f"kalibre edilebilen deyim (≥{args.min_examples} örnek, hem D hem L): {len(thresh)}")

    Path(args.out).write_text(json.dumps(thresh, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"yazıldı: {args.out}")

    bench = score_benchmark_pairs(scorer, idiom_stems, set(thresh))
    print(f"\nbunlardan dış-benchmark'ta (idiom-kimliği eşleşen + konumlanabilen) satır: "
          f"{len(bench)}/{len(thresh)}")
    if bench:
        md_l = [r[2] for r in rows if r[0] in bench]
        pi_l = [bench[r[0]][0] for r in rows if r[0] in bench]
        ml_l = [r[3] for r in rows if r[0] in bench]
        pl_l = [bench[r[0]][1] for r in rows if r[0] in bench]
        r_d, r_l = pearson(md_l, pi_l), pearson(ml_l, pl_l)
        print(f"  korelasyon r(kalibrasyon mean_D, altın-çift pi):  "
              f"{r_d:.3f}" if r_d is not None else "  r(mean_D, pi): tanımsız (n<2 veya varyans 0)")
        print(f"  korelasyon r(kalibrasyon mean_L, altın-çift pl):  "
              f"{r_l:.3f}" if r_l is not None else "  r(mean_L, pl): tanımsız (n<2 veya varyans 0)")
        print(f"  (reddedilen sentetik-referans denemesinin ölçütü buydu: r≈0.30 → terk. "
              f"Bu gerçek de benzer/daha düşükse aynı hüküm geçerli.)")
    else:
        print("  UYARI: hiçbir kalibre deyim dış-benchmarkta konumlanamadı — korelasyon "
              "hesaplanamıyor (kapsam kesişimi boş).")


if __name__ == "__main__":
    main()
