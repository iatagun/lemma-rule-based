#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deney C — TDK sözlük-biçimi deyim listesiyle aday-span üretimi (stage-1'in nöral adaylarına EK).

`predict_idiom.py`/`eval_idiom.py`'nin bugüne kadarki hattı saf nöral: viterbi decode → bigappy
birleştir → (opsiyonel) stage-2 filtrele. Filtre yalnız span SİLER, hiç EKLEMEZ — stage-1
görülmemiş bir deyimin adayını hiç öne süremezse, stage-2'nin onu kurtarma şansı yok.

Bu modül aynı katı gövde-eşleştiriciyi (`data/prepare_tdk_idiom_examples.find_span`/`stem`)
TDK sözlüğünün 11k+ deyim METNİNE karşı çalıştırıp nöral tagger'ın KAÇIRDIĞI adayları ekler.
Dış-benchmark'ın (Çavuşoğlu&Çöltekin) kendi deyim listesi lexicon olarak KULLANILMAZ — döngüsel/
sızıntı olurdu; yalnız TDK sözlüğü (eğitim verisinin de kaynağı, ayrı bir liste).

Eklenen adaylar mevcut nöral adaylarla ÇAKIŞMIYORSA eklenir (aynı bölgeyi iki kez etiketleme).
`wrap_stage2` bitişik VID span'lerini kaynağından bağımsız aynı kuralla filtreler — lexicon
adayları da (nöral adaylar gibi) idyomatiklik sınıflandırıcısından geçer, güvenlik ağı bozulmaz.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TDK_CSV = PROJECT_ROOT / "idiom_data" / "raw" / "tdk_atasozu_deyim.csv"


def build_lexicon(csv_path: Path = TDK_CSV) -> list[tuple[str, tuple[str, ...], frozenset[str]]]:
    """→ [(deyim_metni, gövde_dizisi, gövde_kümesi)]. Tek-kelimelik "deyimler" atlanır
    (her cümlede rastgele eşleşip gürültü üretirler)."""
    from data.prepare_tdk_idiom_examples import idiom_stems

    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} yok — önce: node data/fetch_tdk_deyim.mjs")
    rows = csv.DictReader(csv_path.open(encoding="utf-8"))
    lex = []
    for r in rows:
        if r.get("kind") != "idiom":
            continue
        seq = tuple(idiom_stems(r["text"]))
        if len(seq) >= 2:
            lex.append((r["text"], seq, frozenset(seq)))
    return lex


def lexicon_spans(words: list[str], lexicon, stem_fn, find_span_fn) -> list[dict]:
    """`words` içinde lexicon'daki deyimleri ara. Nöral çıktıyla AYNI span sözlük biçimi
    (`{"text","start","end","category":"VID","gappy":False}`) — `spans_from_bigappy`'nin
    kısaltılmış biçimi, `wrap_stage2`'nin beklediği alanların hepsi mevcut."""
    sent_stems = [stem_fn(w.lower()) for w in words]
    sent_set = set(sent_stems)
    out = []
    for text, seq, seq_set in lexicon:
        if not seq_set <= sent_set:      # ucuz ön-eleme: gövdelerin hepsi cümlede yoksa atla
            continue
        span = find_span_fn(list(seq), sent_stems)
        if span is None:
            continue
        s, e = span
        out.append({"text": " ".join(words[s:e]), "start": s, "end": e,
                     "category": "VID", "gappy": False, "source": "lexicon", "lexicon_idiom": text})
    return out


def merge_candidates(neural_spans: list[dict], lex_spans: list[dict]) -> list[dict]:
    """Lexicon adaylarını nöral adaylarla birleştir; mevcut bir span'la (bitişik veya gap'li,
    her iki parçası) ÇAKIŞAN lexicon adayı atlanır — çakışmayanlar eklenir (saf recall artışı)."""
    occupied: list[tuple[int, int]] = []
    for sp in neural_spans:
        occupied.append((sp["start"], sp["end"]))
        if sp.get("gappy"):
            occupied.append((sp["start2"], sp["end2"]))
    out = list(neural_spans)
    for sp in lex_spans:
        s, e = sp["start"], sp["end"]
        if any(s < oe and os_ < e for os_, oe in occupied):
            continue
        out.append(sp)
        occupied.append((s, e))
    return out


def make_lexicon_predict(base_predict, lexicon=None, only_if_empty: bool = False):
    """`base_predict(words)->spans` fonksiyonunu lexicon-aday-birleşimiyle sarmalar.

    `only_if_empty=True`: lexicon adayları YALNIZ nöral tagger cümlede HİÇ span bulamadıysa
    eklenir (muhafazakâr varyant — PARSEME'de precision çöküşü gözlemlendi, bkz. Deney C
    notları; bu, gürültü yüzeyini stage-1'in zaten sessiz kaldığı cümlelerle sınırlar)."""
    from data.prepare_tdk_idiom_examples import stem, find_span

    lex = lexicon if lexicon is not None else build_lexicon()

    def predict(words):
        neural = base_predict(words)
        if only_if_empty and neural:
            return neural
        lex_sp = lexicon_spans(words, lex, stem, find_span)
        return merge_candidates(neural, lex_sp)

    predict.lexicon_size = len(lex)  # type: ignore[attr-defined]
    return predict
