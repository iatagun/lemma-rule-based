#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deney Z — LLM-ÜRETİLMİŞ (etiketlenmiş DEĞİL) dengeli D+L minimal çiftler, stage-2 için.

9 stage-2 negatifinin (Deney P/Q/R/S/T/U/V + focal loss + LLM-ölçekli ETİKETLEME) hepsi AYNI
doğal-derlem-madenli veriye dayanıyordu — LLM doğal cümleleri D/L diye SINIFLANDIRIYORDU
(κ~0.57-0.66 sınır gürültüsü). Literatür (EDM2025 "Bridging the Data Gap", bkz. plan dosyası
`~/.claude/plans/swift-finding-token.md`) farklı bir rejim öneriyor: LLM'e doğal cümle
etiketletmek yerine, her deyim için dengeli D (idyomatik) + L (literal, aynı yüzey biçimi
başka bağlamda) cümle YAZDIRMAK — üretim, sınıflandırmadan daha az belirsiz bir görev.

Bu script İKİ aşamalıdır (LLM çağrısı script İÇİNDE yapılmaz — Claude Code alt-ajanlarıyla,
Aşama 2/3'teki aynı ücretsiz yöntemle üretilir, script yalnız seçim + doğrulama/birleştirmeyi
yapar):

  1. --select   : Çavuşoğlu'nun 198 GERÇEK eval çiftiyle (yalnız sample+literal dolu satırlar)
                  VE frozen/held-out deyimlerle örtüşmeyen bir aday deyim havuzu seçer,
                  alt-ajan dispatch'i için parçalara böler (`idiom_data/_synth_batch_N.json`).
  2. --build    : alt-ajanların yazdığı ham üretim çıktılarını (`idiom_data/_synth_raw_*.json`,
                  {"idiom":..., "D":[...], "L":[...]}) okuyup `find_span` ile her cümlede span'i
                  doğrular (bulunamayan cümle SESSİZCE atılır — ucuz kalite filtresi), eğitime
                  hazır kayıtları AYRI bir havuza yazar (`_synthetic_stage2_records.jsonl` +
                  `_synthetic_stage2_labels.tsv`, idx uzayı 900000+ — mevcut frozen havuzla
                  ASLA karışmaz, `training/train_idiomaticity_clf.py --synthetic-file` ile
                  opsiyonel olarak train'e eklenir, test seti HİÇ etkilenmez).

Kullanım:
    python data/prepare_synthetic_stage2_pairs.py --select --n-idioms 150 --batch-size 25
    # ... alt-ajanlar idiom_data/_synth_raw_*.json'a üretim yazar ...
    python data/prepare_synthetic_stage2_pairs.py --build
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA = PROJECT_ROOT / "idiom_data"
BENCH_TSV = DATA / "raw" / "turkish_idioms_benchmark.tsv"
SAMPLE_RECS = DATA / "_corpus_sample_records.jsonl"
HOLDOUT_JSON = DATA / "_holdout_idioms.json"
CANDIDATES_JSON = DATA / "_synthetic_candidates.json"
OUT_RECS = DATA / "_synthetic_stage2_records.jsonl"
OUT_LABELS = DATA / "_synthetic_stage2_labels.tsv"
IDX_BASE = 900000  # mevcut frozen idx uzayından (şu an ~13.5k) uzak — asla çakışmaz


def _excluded_idioms() -> set[str]:
    """Çavuşoğlu'nun 198 GERÇEK eval çifti (yalnız sample+literal dolu) + frozen + held-out
    + ÖNCEKİ turlarda zaten seçilmiş deyimler (`_synthetic_candidates.json`, kümülatif —
    2. tur 1. turun 150'sini TEKRAR seçmesin diye)."""
    rows = list(csv.DictReader(BENCH_TSV.open(encoding="utf-8"), delimiter="\t"))
    eval_idioms = {r["idiom"].strip() for r in rows
                   if r.get("sample", "").strip() and r.get("literal", "").strip()}
    frozen = {json.loads(l)["idiom"] for l in SAMPLE_RECS.read_text(encoding="utf-8").splitlines()
              if l.strip()} if SAMPLE_RECS.exists() else set()
    holdout = set(json.loads(HOLDOUT_JSON.read_text(encoding="utf-8"))) if HOLDOUT_JSON.exists() else set()
    prev_rounds = set(json.loads(CANDIDATES_JSON.read_text(encoding="utf-8"))) if CANDIDATES_JSON.exists() else set()
    return rows, eval_idioms | frozen | holdout | prev_rounds


def select(n_idioms: int, batch_size: int, seed: int) -> None:
    import random
    rows, excl = _excluded_idioms()
    by_idiom = {}
    for r in rows:
        name = r.get("idiom", "").strip()
        if name and name not in by_idiom:
            by_idiom[name] = r.get("explanation", "").strip()
    candidates = sorted(set(by_idiom) - excl)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    picked = candidates[:n_idioms]
    # kümülatif — sonraki tur bunu da hariç tutsun
    prev = set(json.loads(CANDIDATES_JSON.read_text(encoding="utf-8"))) if CANDIDATES_JSON.exists() else set()
    CANDIDATES_JSON.write_text(json.dumps(sorted(prev | set(picked)), ensure_ascii=False), encoding="utf-8")
    print(f"aday havuzu: {len(by_idiom)} TDK deyim, {len(excl)} hariç tutuldu "
          f"(198 eval + frozen + held-out + önceki turlar) → {len(candidates)} uygun, "
          f"{len(picked)} bu turda seçildi")
    # önceki turlardan kalan _synth_batch_*.json dosya sayısı kadar öteden başla — üzerine yazma
    start = len(list(DATA.glob("_synth_batch_*.json")))
    batches = [picked[i:i + batch_size] for i in range(0, len(picked), batch_size)]
    for j, b in enumerate(batches):
        i = start + j
        entries = [{"idiom": name, "meaning": by_idiom.get(name, "")} for name in b]
        (DATA / f"_synth_batch_{i}.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    print(f"{len(batches)} parça yazıldı: idiom_data/_synth_batch_{start}.json .. "
          f"_{start + len(batches) - 1}.json")
    print("Sonraki: her parçayı bir Claude Code alt-ajanına ver (bkz. script docstring'i), "
          "çıktıyı idiom_data/_synth_raw_<i>.json'a (AYNI i indeksi) yazdır, sonra --build "
          "çalıştır (önceki turların _synth_raw_*.json'ları da otomatik dahil edilir).")


def _stem_match(a: str, b: str) -> bool:
    """Gevşek eşleşme: eşit VEYA biri diğerinin öneki (Turkish snowball stemmer bazı çekim
    eklerini — özellikle '-yor' şimdiki zaman ekini — atmıyor, bkz. proje notu "aorist hâlâ
    kaçıyor"; serbest üretilen cümlelerde bu, katı eşit-stem eşleşmesini %17'ye düşürdü,
    önek toleransıyla %87'ye çıktı). Kısa token'larda (len<3, "su"/"mu" gibi) yanlış-pozitif
    önek eşleşmesini önlemek için önek kuralı yalnız her iki taraf da ≥3 karakterse geçerli."""
    if a == b:
        return True
    if len(a) < 3 or len(b) < 3:
        return False
    return a.startswith(b) or b.startswith(a)


def find_span_lenient(idiom_seq: list[str], sent_stems: list[str], max_gap: int = 3) -> tuple[int, int] | None:
    """`data/prepare_tdk_idiom_examples.py::find_span`'ın gevşek-eşleşme varyantı — yalnız bu
    scriptte (sentetik, serbest üretilmiş cümleler) kullanılır; paylaşılan TDK/derlem boru
    hattına dokunulmaz (`find_span` değişmedi)."""
    n = len(idiom_seq)
    if n == 0:
        return None
    L = len(sent_stems)
    for start in range(L):
        if not _stem_match(sent_stems[start], idiom_seq[0]):
            continue
        pos, matched, gaps = start, 1, 0
        while matched < n and pos + 1 < L:
            pos += 1
            if _stem_match(sent_stems[pos], idiom_seq[matched]):
                matched += 1
            else:
                gaps += 1
                if gaps > max_gap:
                    break
        if matched == n:
            return start, pos + 1
    return None


def report() -> None:
    """Salt-okunur çift-kapsama teşhisi — hiçbir dosya yazmaz.

    Deney Z'nin tüm önermesi "dengeli D+L minimal çift" (GLU kılavuzu: minimal çiftler
    Aşama-3'ün TEK sinyali). 2026-09-22'de ölçüldüğünde havuzun bu önermeyi tutmadığı
    görüldü: deyimlerin %44'ünün hiç L cümlesi yok ve ölçekleme turu eğriliği ARTIRDI
    (2.64:1 → 4.30:1). Span doğrulayıcı de L'yi D'den sert eliyor (yeniden sözcüklenen
    literal cümleler sıralı eşleşmeyi bozuyor), yani filtre havuzu kendisi D'ye eğiyor.
    Bu rapor o eğriliğin bir daha sessizce büyümemesi için kalıcıdır."""
    from data.prepare_tdk_idiom_examples import idiom_stems, stem

    rows = []
    for rp in sorted(DATA.glob("_synth_raw_*.json")):
        for it in json.loads(rp.read_text(encoding="utf-8")):
            seq = idiom_stems(it["idiom"])
            kept = {"D": 0, "L": 0}
            made = {"D": len(it.get("D", [])), "L": len(it.get("L", []))}
            for label, sents in (("D", it.get("D", [])), ("L", it.get("L", []))):
                for s in sents:
                    w = s.split()
                    if len(w) >= 2 and seq and find_span_lenient(seq, [stem(x.lower()) for x in w]):
                        kept[label] += 1
            rows.append((rp.name, it["idiom"], made, kept))

    n = len(rows)
    zero_l = sum(1 for *_, kept in rows if kept["L"] == 0)
    md, ml = sum(r[2]["D"] for r in rows), sum(r[2]["L"] for r in rows)
    kd, kl = sum(r[3]["D"] for r in rows), sum(r[3]["L"] for r in rows)
    print(f"deyim girdisi: {n}")
    print(f"  kabul sonrası L'si HİÇ olmayan (çift OLUŞTURMAYAN): {zero_l} (%{100*zero_l/n:.0f})")
    print(f"  D: üretilen {md} → kabul {kd} (eleme %{100*(md-kd)/max(md,1):.1f})")
    print(f"  L: üretilen {ml} → kabul {kl} (eleme %{100*(ml-kl)/max(ml,1):.1f})"
          f"   ← L daha sert elenirse havuz D'ye eğilir")
    print(f"  kabul edilen D:L = {kd/max(kl,1):.2f} : 1")
    print("  çift oluşturan deyim havuzu (--build --only-paired) "
          f"→ {n - zero_l} deyim, {sum(r[3]['D'] for r in rows if r[3]['L'])} D / {kl} L")


def build(only_paired: bool = False, raw_max_index: int | None = None) -> None:
    import re as _re

    from data.prepare_tdk_idiom_examples import idiom_stems, stem

    raw_files = sorted(DATA.glob("_synth_raw_*.json"))
    if not raw_files:
        sys.exit("idiom_data/_synth_raw_*.json yok — önce --select ve alt-ajan üretim turu.")
    # --raw-max-index: geçmiş bir havuzu BİREBİR geri kurmak için (ör. v8'in 650-deyimlik
    # havuzu = parça 0-25; parça 26-35 Deney AA'nın +250'si, parça 100 terim-negatifleri).
    # Kıyas dosyalarını dondurma disiplini: kanonik havuz sonradan üzerine yazıldığı için
    # eski sürümlerin reçetesi ancak böyle yeniden üretilebiliyor.
    if raw_max_index is not None:
        raw_files = [f for f in raw_files
                     if int(_re.search(r"_synth_raw_(\d+)", f.name).group(1)) <= raw_max_index]
        print(f"--raw-max-index {raw_max_index}: {len(raw_files)} ham parça kullanılacak")

    # --only-paired (Deney AB-1): yalnız GERÇEKTEN çift oluşturan deyimler (en az bir L
    # cümlesi span doğrulamasını geçmiş). Saf SEÇİM — yeni cümle üretilmez. Kanonik havuzu
    # ezmemek için AYRI dosyaya yazar.
    out_recs, out_labels = OUT_RECS, OUT_LABELS
    if raw_max_index is not None:
        out_recs = DATA / f"_synthetic_stage2_upto{raw_max_index}_records.jsonl"
        out_labels = DATA / f"_synthetic_stage2_upto{raw_max_index}_labels.tsv"
    paired: set[str] | None = None
    if only_paired:
        out_recs = DATA / "_synthetic_stage2_paired_records.jsonl"
        out_labels = DATA / "_synthetic_stage2_paired_labels.tsv"
        paired = set()
        for rp in raw_files:
            for it in json.loads(rp.read_text(encoding="utf-8")):
                seq = idiom_stems(it["idiom"])
                if not seq:
                    continue
                for s in it.get("L", []):
                    w = s.split()
                    if len(w) >= 2 and find_span_lenient(seq, [stem(x.lower()) for x in w]):
                        paired.add(it["idiom"])
                        break
        print(f"--only-paired: {len(paired)} deyim çift oluşturuyor, gerisi atlanacak")

    idx = IDX_BASE
    n_recs, n_dropped, dist = 0, 0, {"D": 0, "L": 0}
    with out_recs.open("w", encoding="utf-8") as rf, out_labels.open("w", encoding="utf-8") as lf:
        for rp in raw_files:
            items = json.loads(rp.read_text(encoding="utf-8"))
            for it in items:
                idiom = it["idiom"]
                seq = idiom_stems(idiom)
                if not seq:
                    continue
                if paired is not None and idiom not in paired:
                    continue
                for label, sentences in (("D", it.get("D", [])), ("L", it.get("L", []))):
                    for sent in sentences:
                        words = sent.split()
                        if len(words) < 2:
                            n_dropped += 1
                            continue
                        rng_ = find_span_lenient(seq, [stem(w.lower()) for w in words])
                        if rng_ is None:
                            n_dropped += 1
                            continue
                        s, e = rng_
                        # Not: tags HER ZAMAN aday span'in KONUMUNU işaretler (D/L farketmez —
                        # `_corpus_sample_records.jsonl`'deki doğal-derlem kaydıyla aynı kural,
                        # stage-2 eğitimi bu konumdan pooling yapıyor; asıl D/L kararı ayrı
                        # labels.tsv'de). Yalnız D'de gerçek BIO span sayılır, L'de aynı konum
                        # yalnız pooling referansı.
                        tags = ["O"] * len(words)
                        tags[s] = "B-VID"
                        for k in range(s + 1, e):
                            tags[k] = "I-VID"
                        rf.write(json.dumps({"idx": idx, "words": words, "tags": tags,
                                            "idiom": idiom, "span": " ".join(words[s:e])},
                                            ensure_ascii=False) + "\n")
                        lf.write(f"{idx}\t{label}\n")
                        idx += 1
                        n_recs += 1
                        dist[label] += 1
    print(f"{len(raw_files)} ham dosya işlendi: {n_recs} kayıt kabul edildi "
          f"({dist['D']} D / {dist['L']} L, D:L = {dist['D']/max(dist['L'],1):.2f}:1), "
          f"{n_dropped} cümle span bulunamadığı için atlandı")
    print(f"→ {out_recs.name} + {out_labels.name}")
    print("Sonraki: python training/train_idiomaticity_clf.py --freeze 8 --dropout 0.3 "
          "--weight-decay 0.05 --epochs 14 --synthetic-file idiom_data/_synthetic_stage2_records.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="salt-okunur çift-kapsama teşhisi (hiçbir dosya yazmaz)")
    ap.add_argument("--raw-max-index", type=int, default=None,
                    help="--build ile: yalnız _synth_raw_<=N parçalarını kullan (geçmiş havuzu "
                         "birebir geri kurmak için; v8'in 650-deyimlik havuzu = 25). AYRI dosyaya yazar.")
    ap.add_argument("--only-paired", action="store_true",
                    help="--build ile: yalnız gerçekten D+L çifti oluşturan deyimler; "
                         "AYRI dosyaya yazar (_synthetic_stage2_paired_*), kanonik havuz ezilmez")
    ap.add_argument("--n-idioms", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.report:
        report()
    elif args.select:
        select(args.n_idioms, args.batch_size, args.seed)
    elif args.build:
        build(only_paired=args.only_paired, raw_max_index=args.raw_max_index)
    else:
        ap.error("--select, --build veya --report gerekli")


if __name__ == "__main__":
    main()
