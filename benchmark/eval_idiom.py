#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom değerlendirmesi — PARSEME test-split span-F1 + nokta-atışı deyim/literal seti.

`eval_ambiguity.py`'nin idiom karşılığı: held-out span-F1 genel doğruluğu ölçer ama
deyim/literal minimal-çiftlerini seyrek örnekler; CASES bunu hedefler.

Kullanım:
    python benchmark/eval_idiom.py                                    # PARSEME test + CASES (HF)
    python benchmark/eval_idiom.py --local --checkpoint idiom_data/best_idiom_tagger.pt
    python benchmark/eval_idiom.py --mode cases                       # yalnız nokta-atışı
    python benchmark/eval_idiom.py --mode neural                      # yalnız PARSEME test
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))


# ═══════════════════════════════════════════════════════════════════════
#  Nokta-atışı vaka seti — (kategori, cümle, beklenen_öbek|None, beklenen_etiket|None)
#  beklenen_öbek=None → cümlede hiç VID/LVC span'i beklenmez (serbest birleşim / ilgisiz).
# ═══════════════════════════════════════════════════════════════════════
CASES: list[tuple[str, str, str | None, str | None]] = [
    # ── VID (deyim, figüratif) ──
    ("deyim", "Yalanları ortaya çıkınca patron gözden düştü .", "gözden düştü", "VID"),
    ("deyim", "Komisyon bu konuyu ele aldı .", "ele aldı", "VID"),
    ("deyim", "Kurul , uyarıları göz ardı etti .", "göz ardı etti", "VID"),
    ("deyim", "Bu işten sonra resmen kafayı yedi .", "kafayı yedi", "VID"),
    ("deyim", "Öğrenci öğretmenine kafa tuttu .", "kafa tuttu", "VID"),

    # ── LVC (eşdizim / yardımcı-fiil birleşimi, yarı-birleşimsel) ──
    ("eşdizim", "Aileyle konuşup karar verdi .", "karar verdi", "LVC"),
    ("eşdizim", "Komşusuna yardım etti .", "yardım etti", "LVC"),
    ("eşdizim", "Bana asla yalan söylemeyeceğine söz verdi .", "söz verdi", "LVC"),

    # ── serbest birleşim / ilgisiz (span beklenmez) ──
    ("serbest", "Çocuk okula gitti .", None, None),
    ("serbest", "Doktor gözünü muayene etti .", None, None),
    ("serbest", "Kafasını yastığa dayadı .", None, None),
    ("serbest", "Eline kalemi aldı .", None, None),

    # ── isim/sıfat deyim (fiilsiz — PARSEME kapsamıyor, TDK verisi ekliyor) ──
    ("isim/sıfat", "Adam gerçekten eli açık biridir .", "eli açık", "VID"),
    ("isim/sıfat", "Çocuk küçük yaştan beri dili uzundur .", "dili uzun", "VID"),
    ("isim/sıfat", "Haberi duyunca ağzı kulaklarına vardı .", "ağzı kulaklarına", "VID"),
    ("isim/sıfat", "Bu aralar başı dertte .", "başı dertte", "VID"),
]


def _tokenize(sent: str) -> list[str]:
    return sent.split()


def _check(spans: list[dict], phrase: str | None, tag: str | None) -> str:
    """→ 'ok' | 'fail'."""
    if phrase is None:
        return "ok" if not spans else "fail"
    # Alt-küme kontrolü: beklenen öbeğin TÜM kelimeleri tahmin edilen span'de olmalı
    # (yalnız "herhangi bir kelime çakışıyor" değil — bu, "kafa" eksik "yalnız tuttu"
    # gibi sınır hatalarını "ok" diye işaretleyip gizlerdi). Fazladan kelime (ör. ışık
    # fiili "vardı") tolere edilir — beklenen öbek genelde çekirdeği işaret eder.
    want_words = set(w.lower() for w in phrase.split())
    for sp in spans:
        got_words = set(w.lower() for w in sp["text"].split())
        if want_words <= got_words and sp["category"] == tag:
            return "ok"
    return "fail"


# ═══════════════════════════════════════════════════════════════════════
#  Model yükleyiciler
# ═══════════════════════════════════════════════════════════════════════
def make_predictor(local: bool, checkpoint: str | None, hf_repo: str):
    """→ fn(words) -> list[dict] (predict_spans çıktısı)."""
    if local:
        import torch
        from transformers import AutoTokenizer

        from train_idiom_bert import IdiomLabelSpace, IdiomTagger, MAX_LEN
        from dizgebert_idiom.modeling_dizgebert_idiom import align_words, decode_bigappy_spans, viterbi_decode

        ck = torch.load(checkpoint, map_location="cpu")
        ls = IdiomLabelSpace(ck["label_space"])
        tok = AutoTokenizer.from_pretrained(ls.encoder_model)
        model = IdiomTagger(ls, ls.encoder_model).eval()
        model.load_state_dict(ck["model"])

        @torch.no_grad()
        def pred_local(words):
            enc, kept, fp, lp = align_words(tok, words, MAX_LEN)
            out = model(enc["input_ids"], enc["attention_mask"], fp, lp)
            tags1 = viterbi_decode(out["tags"][0], ls.tags)
            tags2 = viterbi_decode(out["tags2"][0], ls.tags2)
            spans = []
            for span in decode_bigappy_spans(tags1, tags2):
                if len(span) == 3:
                    s, e, cat = span
                    spans.append({"text": " ".join(words[i] for i in range(s, e)),
                                  "start": s, "end": e, "category": cat, "gappy": False})
                else:
                    s1, e1, s2, e2, cat = span
                    text = " ".join(words[s1:e1]) + " ... " + " ".join(words[s2:e2])
                    spans.append({"text": text, "start": s1, "end": e1, "start2": s2,
                                  "end2": e2, "category": cat, "gappy": True})
            return spans
        return pred_local

    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(hf_repo, trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(hf_repo)
    return lambda words: m.predict_spans(words, tokenizer=tok)


def wrap_stage2(base_predict, clf_ckpt: str, thresh: float = 0.5):
    """Fikir 3: aşama-1 span'lerini idyomatiklik sınıflandırıcısından geçir, literal olanı ELE.
    Yalnız VID'e uygulanır; span YALNIZCA p(literal) > thresh ise elenir (thresh↑ → recall↑)."""
    import torch
    from transformers import AutoTokenizer
    from train_idiomaticity_clf import IdiomaticityClf

    ck = torch.load(clf_ckpt, map_location="cpu")
    enc_name = ck.get("encoder", "dbmdz/electra-base-turkish-cased-discriminator")
    tok = AutoTokenizer.from_pretrained(enc_name)
    clf = IdiomaticityClf(enc_name).eval()
    clf.load_state_dict(ck["model"])

    @torch.no_grad()
    def _idiomatic(words, s, e) -> bool:
        enc = tok(words, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=128)
        wid = enc.word_ids()
        first = {}; last = {}
        for i, w in enumerate(wid):
            if w is None:
                continue
            first.setdefault(w, i); last[w] = i
        if s not in first or (e - 1) not in last:
            return True  # span kırpıldı → dokunma
        logit = clf(enc["input_ids"], enc["attention_mask"],
                    torch.tensor([first[s]]), torch.tensor([last[e - 1]]))
        p_literal = float(torch.softmax(logit, -1)[0, 0])
        return p_literal <= thresh  # yalnız GÜVENLİ literal elenir

    def predict(words):
        out = []
        for sp in base_predict(words):
            # stage-2 YALNIZ VID'e: LVC.full (karar vermek, yardım etmek) tanımı gereği
            # yarı-birleşimsel, idyomatik/literal ayrımı yok — filtreye sokulmaz. Gap'li de geçer.
            if sp.get("gappy") or sp.get("category") != "VID":
                out.append(sp); continue
            if _idiomatic(words, sp["start"], sp["end"]):
                out.append(sp)
        return out
    return predict


# ═══════════════════════════════════════════════════════════════════════
#  Mod 1: nokta-atışı CASES
# ═══════════════════════════════════════════════════════════════════════
def run_cases(predict) -> None:
    print("\n=== nokta-atışı deyim/literal seti ===")
    n_ok = n_fail = 0
    cur = None
    for cat, sent, phrase, tag in CASES:
        if cat != cur:
            print(f"\n  ── {cat} ──"); cur = cat
        ws = _tokenize(sent)
        spans = predict(ws)
        status = _check(spans, phrase, tag)
        n_ok += status == "ok"
        n_fail += status == "fail"
        mark = "✓" if status == "ok" else "✗"
        got = ", ".join(f"{s['text']}:{s['category']}" for s in spans) or "(span yok)"
        print(f"  {mark} {sent[:45]:45s}  bulundu: {got:35s}  bkln: {phrase or '(yok)'}")
    print(f"\n  skor: {n_ok}/{n_ok + n_fail}")


# ═══════════════════════════════════════════════════════════════════════
#  Mod 3: dış bağımsız kaynak — Çavuşoğlu & Çöltekin (MWE 2026) 201 deyim,
#  gerçek idyomatik-kullanım + literal-kullanım cümle çifti (eğitim verimizde YOK).
#  CASES'in (16 elle-seçilmiş vaka) aksine büyük ölçekli, bağımsız bir bağlam-
#  bağımlılık (idyomatik/literal ayrımı) testi.
# ═══════════════════════════════════════════════════════════════════════
def run_external(predict) -> None:
    import csv

    tsv_path = _PROJECT / "idiom_data" / "raw" / "turkish_idioms_benchmark.tsv"
    if not tsv_path.exists():
        print(f"\nUYARI: {tsv_path} yok — önce `python fetch_turkish_idioms_benchmark.py`. Atlanıyor.")
        return

    rows = [r for r in csv.DictReader(tsv_path.open(encoding="utf-8"), delimiter="\t")
            if r.get("sample", "").strip() and r.get("literal", "").strip()]
    print(f"\n=== Dış kaynak: Çavuşoğlu & Çöltekin (MWE 2026), {len(rows)} deyim çifti ===")

    n = sample_hit = literal_hit = both_correct = 0
    for r in rows:
        sw, lw = r["sample"].split(), r["literal"].split()
        if len(sw) < 2 or len(lw) < 2:
            continue
        sh = bool(predict(sw))
        lh = bool(predict(lw))
        n += 1
        sample_hit += sh
        literal_hit += lh
        both_correct += sh and not lh

    print(f"  işlenen: {n}")
    if n == 0:
        print("  UYARI: uygun satır yok (hepsi tek-kelimelik filtreye takıldı).")
        return
    print(f"  idyomatik cümlede span işaretledi (duyarlılık): {sample_hit}/{n} = %{100*sample_hit/n:.1f}")
    print(f"  literal cümlede YANLIŞ span işaretledi (yanlış-pozitif): {literal_hit}/{n} = %{100*literal_hit/n:.1f}")
    print(f"  ikisini de doğru ayırt etti: {both_correct}/{n} = %{100*both_correct/n:.1f}")


# ═══════════════════════════════════════════════════════════════════════
#  Mod 2: PARSEME test.cupt span-F1
# ═══════════════════════════════════════════════════════════════════════
def run_neural(predict) -> None:
    from prepare_idiom_data import iter_sentences, sentence_to_record
    from dizgebert_idiom.modeling_dizgebert_idiom import decode_bigappy_spans

    test_path = _PROJECT / "idiom_data" / "raw" / "test.cupt"
    if not test_path.exists():
        print(f"\nUYARI: {test_path} yok — önce `python fetch_parseme_tr.py`. PARSEME testi atlanıyor.")
        return

    print("\n=== PARSEME test.cupt — span-düzeyi (exact-match) P/R/F1 ===")
    stats: Counter = Counter()
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    n_sent = 0
    for toks in iter_sentences(test_path):
        rec = sentence_to_record(toks, stats)
        n_sent += 1
        words = rec["words"]
        gold_spans = set(decode_bigappy_spans(rec["tags"], rec["tags2"]))
        pred = predict(words)
        pred_spans = set()
        for sp in pred:
            if sp.get("gappy"):
                pred_spans.add((sp["start"], sp["end"], sp["start2"], sp["end2"], sp["category"]))
            else:
                pred_spans.add((sp["start"], sp["end"], sp["category"]))
        for s in gold_spans & pred_spans:
            tp[s[-1]] += 1; tp["ALL"] += 1
            if len(s) == 5:
                tp["GAPLI"] += 1
        for s in pred_spans - gold_spans:
            fp[s[-1]] += 1; fp["ALL"] += 1
            if len(s) == 5:
                fp["GAPLI"] += 1
        for s in gold_spans - pred_spans:
            fn[s[-1]] += 1; fn["ALL"] += 1
            if len(s) == 5:
                fn["GAPLI"] += 1

    def f1(c):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    print(f"  {n_sent} cümle")
    real_cats = sorted((set(tp) | set(fp) | set(fn)) - {"ALL", "GAPLI"})
    for c in real_cats + ["GAPLI", "ALL"]:
        p, r, f = f1(c)
        print(f"  {c:6s}  P={100*p:5.2f}  R={100*r:5.2f}  F1={100*f:5.2f}  "
              f"(tp={tp[c]} fp={fp[c]} fn={fn[c]})")


# ═══════════════════════════════════════════════════════════════════════
#  Mod 4: GLU kılavuzu — idyomatik/literal minimal çiftler + hard-negative tanı seti
#  (Gülsün Leylâ Uzun etiketleme kılavuzu; bkz. prepare_glu_examples.py,
#  .claude/skills/idiom/glu_karar_cercevesi.md). CASES'in prensipli, bağlam-odaklı hâli.
# ═══════════════════════════════════════════════════════════════════════
def run_glu(predict) -> None:
    from prepare_glu_examples import glu_diagnostic_cases

    cases = glu_diagnostic_cases()
    print(f"\n=== GLU kılavuzu tanı seti ({len(cases)} vaka) ===")
    n_ok = n_fail = 0
    # minimal çift muhasebesi: aynı yapının deyim + literal cümlesi ardışık
    pair_idiom_hit = pair_literal_hit = pair_total = 0
    cur = None
    prev_was_idiom_ok = None
    for cat, sent, phrase, tag in cases:
        if cat != cur:
            print(f"\n  ── {cat} ──"); cur = cat
        ws = _tokenize(sent)
        spans = predict(ws)
        status = _check(spans, phrase, tag)
        n_ok += status == "ok"; n_fail += status == "fail"
        mark = "✓" if status == "ok" else "✗"
        got = ", ".join(f"{s['text']}:{s['category']}" for s in spans) or "(span yok)"
        print(f"  {mark} {sent[:52]:52s}  bulundu: {got:32s}  bkln: {phrase or '(yok)'}")
        if cat == "glu-deyim":
            prev_was_idiom_ok = bool(spans)
        elif cat == "glu-literal" and prev_was_idiom_ok is not None:
            pair_total += 1
            pair_idiom_hit += prev_was_idiom_ok
            pair_literal_hit += bool(spans)
            prev_was_idiom_ok = None

    print(f"\n  vaka skoru: {n_ok}/{n_ok + n_fail}")
    if pair_total:
        both = pair_idiom_hit - pair_literal_hit  # kaba: deyimde işaretledi, literalde işaretlemedi
        print(f"  minimal çift ({pair_total}): deyimde span %{100*pair_idiom_hit/pair_total:.0f}, "
              f"literalde YANLIŞ span %{100*pair_literal_hit/pair_total:.0f}, "
              f"net doğru-ayırt ~%{100*max(both,0)/pair_total:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "neural", "cases", "external", "glu"], default="all")
    ap.add_argument("--local", action="store_true", help="HF yerine yerel .pt")
    ap.add_argument("--checkpoint", default=str(_PROJECT / "idiom_data" / "best_idiom_tagger.pt"))
    ap.add_argument("--hf-repo", default="iatagun/DizgeBERT-Idiom")
    ap.add_argument("--stage2", default=None,
                    help="idyomatiklik sınıflandırıcı checkpoint'i (Fikir 3 iki-aşama) — "
                         "aşama-1 VID span'leri bundan geçirilip literal olanlar elenir")
    ap.add_argument("--stage2-thresh", type=float, default=0.5,
                    help="span yalnız p(literal) > bu değer ise elenir (yüksek → recall korunur)")
    args = ap.parse_args()

    predict = make_predictor(args.local, args.checkpoint, args.hf_repo)
    if args.stage2:
        predict = wrap_stage2(predict, args.stage2, args.stage2_thresh)
        print(f"[iki-aşama] stage-2 filtresi aktif: {args.stage2} (thresh {args.stage2_thresh})")

    if args.mode in ("all", "neural"):
        run_neural(predict)
    if args.mode in ("all", "cases"):
        run_cases(predict)
    if args.mode in ("all", "external"):
        run_external(predict)
    if args.mode in ("all", "glu"):
        run_glu(predict)


if __name__ == "__main__":
    main()
