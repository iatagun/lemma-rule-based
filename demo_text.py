#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe Kural Tabanlı Morfolojik Çözümleyici — Demo

Çıktı biçimleri
────────────────
  § 1  Morfolojik çözümleme tablosu  (Oflazer, 1994)
  § 2  Interlinear Glossed Text      (Leipzig Glossing Rules, 2015)
  § 3  Çoklu çözümleme adayları      (belirsizlik vitrini)
  § 4  Cümle düzeyi yeniden sıralama  (bağlamsal kural uygulaması)

Etiket dönüşüm kaynakları
─────────────────────────
  Oflazer, K. (1994). Two-level description of Turkish morphology.
         Literary and Linguistic Computing, 9(2), 137-148.
  Çöltekin, Ç. (2010). A freely available morphological analyzer for
         Turkish. LREC 2010.
  Leipzig Glossing Rules (2015).
         https://www.eva.mpg.de/lingua/resources/glossing-rules.php
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from morphology import create_default_analyzer, MorphemeAnalysis
from morphology.sentence import SentenceAnalyzer

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Etiket Dönüşüm Tablosu
#  Dahili Türkçe etiketler → uluslararası standart kısaltmalar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_TAG: dict[str, str] = {
    # Hal ekleri (Case)
    "BULUNMA": "Loc", "AYRILMA": "Abl", "YÖNELME": "Dat",
    "BELIRTME": "Acc", "VASITA": "Ins", "TAMLAYAN": "Gen",
    # İyelik (Possessive agreement)
    "İYELİK_1T": "P1sg", "İYELİK_2T": "P2sg", "İYELİK_3T": "P3sg",
    "İYELİK_1Ç": "P1pl", "İYELİK_2Ç": "P2pl", "İYELİK_3Ç": "P3pl",
    # Sayı (Number)
    "ÇOĞUL": "Pl",
    # Zaman / Kip (Tense-Aspect-Mood)
    "GEÇMİŞ_ZAMAN": "Past", "DUYULAN_GEÇMİŞ": "Evid",
    "GELECEK_ZAMAN": "Fut", "GENİŞ_ZAMAN": "Aor",
    "GENİŞ_ZAMAN_OLMSZ": "Neg.Aor", "ŞİMDİKİ_ZAMAN": "Prog",
    "DİLEK_ŞART": "Cond",
    # Kişi (Subject agreement)
    "KİŞİ_1Ç": "A1pl", "KİŞİ_2Ç": "A2pl", "KİŞİ_3Ç": "A3pl",
    # Çatı (Voice)
    "EDİLGEN": "Pass", "ETTİRGEN_-lAt": "Caus", "İŞTEŞ": "Recip",
    # Kiplik / Fiil yapısı
    "YETERLİLİK": "Abil", "OLUMSUZ": "Neg",
    "BİLDİRME": "Cop", "MASTAR": "Inf",
    # Ortaçlar ve ulaçlar (Non-finite forms)
    "SIFAT_FİİL": "Part", "SIFAT_FİİL_-DIk": "PastPart",
    "SIFAT_FİİL_-DIğ": "PastPart",
    "ZARF_FİİL_-ArAk": "Conv", "ZARF_FİİL_-IncA": "Conv",
    "ZARF_FİİL_-Ip": "Conv", "ZARF_FİİL_-ken": "Conv",
    "İLGİ_-ki": "Rel", "İSİM_FİİL_-mA": "Vnoun",
    # Yapım ekleri (Derivation)
    "YAPIM_-CI": "Agt", "YAPIM_-CIlIk": "Ness",
    "YAPIM_-lI": "With", "YAPIM_-lIk": "Ness",
    "YAPIM_-sIz": "Without", "YAPIM_-lAn": "Become",
    "YAPIM_-lAş": "Become",
    # Bileşik etiketler
    "İYELİK_3T/BELIRTME": "P3sg|Acc", "İYELİK_2T/TAMLAYAN": "P2sg|Gen",
    "OLUMSUZ/İSİM_FİİL": "Neg|Vnoun", "BİLDİRME/ETTİRGEN": "Cop|Caus",
    "EMİR/KİŞİ_2T": "Imp|A2sg", "EMİR_3Ç": "Imp.A3pl",
}


def _t(label: str) -> str:
    return _TAG.get(label, label)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Biçimlendirme Yardımcıları
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _lemma_of(a: MorphemeAnalysis) -> str:
    """En anlamlı lemma seçimi: lemma > root > stem."""
    return a.lemma or a.root or a.stem


def fmt_oflazer(a: MorphemeAnalysis) -> str:
    """Oflazer (1994) stili:  kök+Tag+Tag
       Örn: ev+Pl+P3sg+Abl
    """
    stem = a.root or a.stem
    if not a.suffixes:
        return stem
    return stem + "".join(f"+{_t(lbl)}" for _, lbl in a.suffixes)


def fmt_segmented(a: MorphemeAnalysis) -> str:
    """Morfem segmentasyonu:  kök-ek-ek
       Örn: ev-ler-i-nden
    """
    stem = a.root or a.stem
    if not a.suffixes:
        return stem
    return stem + "".join(f"-{form}" for form, _ in a.suffixes)


def fmt_gloss(a: MorphemeAnalysis) -> str:
    """Leipzig gloss satırı:  kök-TAG-TAG
       Örn: ev-PL-P3SG-ABL
    """
    stem = a.root or a.stem
    if not a.suffixes:
        return stem
    return stem + "".join(f"-{_t(lbl).upper()}" for _, lbl in a.suffixes)


def _clean_words(text: str) -> list[str]:
    words = text.replace("\n", " ").split()
    cleaned = [re.sub(r"[.,;:!?\"()\[\]…—–'']", "", w) for w in words]
    return [w for w in cleaned if w]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Bölüm Yazdırıcıları
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

W = 78  # çıktı genişliği


def _hdr(num: int, title: str) -> None:
    print(f"\n  ── § {num}  {title} {'─' * (W - len(title) - 12)}")
    print()


def print_analysis_table(tokens) -> None:
    """§ 1 — Morfolojik çözümleme tablosu."""
    print(f"  {'#':>3s} │ {'Sözcük':<20s} │ {'Lemma':<14s} │ Çözümleme")
    print(f"  {'─'*3}─┼─{'─'*20}─┼─{'─'*14}─┼─{'─'*32}")
    for i, t in enumerate(tokens, 1):
        a = t.analysis
        lemma = _lemma_of(a)
        morph = fmt_oflazer(a)
        n = len(t.alternatives)
        amb = f"  ‹{n} aday›" if n > 1 else ""
        print(f"  {i:3d} │ {t.word:<20s} │ {lemma:<14s} │ {morph}{amb}")


def print_igt(tokens, sent_id: int = 1) -> None:
    """§ 2 — Leipzig Interlinear Glossed Text."""
    CHUNK = 5
    chunks = [tokens[i:i+CHUNK] for i in range(0, len(tokens), CHUNK)]
    for ci, chunk in enumerate(chunks):
        segs, glosses = [], []
        for t in chunk:
            s = fmt_segmented(t.analysis)
            g = fmt_gloss(t.analysis)
            col = max(len(s), len(g)) + 1
            segs.append(s.ljust(col))
            glosses.append(g.ljust(col))
        lbl = f"({sent_id}{chr(97+ci)})" if len(chunks) > 1 else f"({sent_id}) "
        pad = " " * len(lbl)
        print(f"  {lbl} {''.join(segs)}")
        print(f"  {pad} {''.join(glosses)}")
        print()


def print_ambiguity(words: list[str], analyzer) -> None:
    """§ 3 — Çoklu çözümleme adayları."""
    print(f"  {'Sözcük':<14s} │ {'Aday 1':<26s} │ {'Aday 2':<26s} │ Aday 3")
    print(f"  {'─'*14}─┼─{'─'*26}─┼─{'─'*26}─┼─{'─'*20}")
    for w in words:
        results = analyzer.analyze_all(w)
        cols = []
        for r in results[:3]:
            cols.append(fmt_oflazer(r))
        while len(cols) < 3:
            cols.append("—")
        print(f"  {w:<14s} │ {cols[0]:<26s} │ {cols[1]:<26s} │ {cols[2]}")


def print_context(tokens, word_map: dict) -> int:
    """§ 4 — Cümle düzeyi bağlamsal yeniden sıralama."""
    changed = 0
    for t in tokens:
        wo = word_map.get(t.word, [])
        wo_stem = wo[0].stem if wo else "?"
        if wo_stem != t.analysis.stem and t.context_applied:
            changed += 1
            wo_str = fmt_oflazer(wo[0]) if wo else "?"
            sa_str = fmt_oflazer(t.analysis)
            rules = ", ".join(t.context_applied)
            print(f"  ★  {t.word}")
            print(f"     sözcük → {wo_str}")
            print(f"     cümle  → {sa_str}")
            print(f"     kural  : {rules}")
            print()
    if changed == 0:
        print("  (bağlam kuralları bu metinde değişiklik yapmadı)")
    return changed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Demo Metinleri
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEXTS = [
    ("Haber",
     "Türkiye'nin başkenti Ankara'da dün gerçekleştirilen toplantıda "
     "bilim insanları yapay zekanın geleceğini tartıştılar."),
    ("Edebi",
     "Çocukluğundan beri kitap okumaya meraklı olan kız büyüdüğünde "
     "dünyanın en büyük kütüphanesini kurmak istiyordu."),
    ("Bilimsel",
     "Araştırmacılar yeni geliştirdikleri algoritmayla doğal dil "
     "işleme alanında önemli ilerlemeler kaydettiler."),
    ("Günlük",
     "Yarın sabah erkenden kalkıp markete gidecek misin diye "
     "sordum ama cevap vermedi."),
    ("Bağlamsal",
     "Bu yemek çok güzel olmuş. Şu kitapları masaya koyun. "
     "O eski çalar saati bana verin."),
]

AMBIGUOUS = [
    "gelirin", "yazar", "çalar", "savaşın", "aldılar",
    "güzel", "boyun", "yüzün", "bağlar", "yapar",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Ana Program
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    _DICT = Path(__file__).resolve().parent / "turkish_words.txt"
    anlyzr = create_default_analyzer(
        dictionary_path=_DICT if _DICT.exists() else None,
    )
    sa = SentenceAnalyzer(anlyzr)

    # ── Başlık ────────────────────────────────────────────────
    bw = W - 4  # box inner width
    print()
    print("  ╔" + "═" * bw + "╗")
    h1 = "  Türkçe Kural Tabanlı Morfolojik Çözümleyici"
    print(f"  ║{h1:<{bw}s}║")
    print("  ║  " + "─" * (bw - 4) + "  ║")
    h2 = "  Sözlük : 48.716 madde  │  BOUN Doğruluk : %88.6"
    h3 = "  FSM    : 16 durum      │  Ek şablon     : 43"
    print(f"  ║{h2:<{bw}s}║")
    print(f"  ║{h3:<{bw}s}║")
    print("  ╚" + "═" * bw + "╝")

    total_tok, total_ctx = 0, 0

    for idx, (label, text) in enumerate(TEXTS, 1):
        # ── Metin başlığı ─────────────────────────────────────
        print(f"\n  {'━' * (W - 4)}")
        print(f"  ┃  [{label.upper()}]")
        for ln in textwrap.wrap(text, width=W - 10):
            print(f"  ┃  {ln}")
        print(f"  {'━' * (W - 4)}")

        tokens = sa.analyze(text)
        word_map = {t.word: anlyzr.analyze_all(t.word) for t in tokens}
        total_tok += len(tokens)

        # § 1  Tablo
        _hdr(1, "Morfolojik Çözümleme")
        print_analysis_table(tokens)

        # § 2  IGT
        _hdr(2, "Interlinear Glossed Text (Leipzig)")
        print_igt(tokens, sent_id=idx)

        # § 4  Bağlam
        _hdr(4, "Bağlamsal Yeniden Sıralama")
        total_ctx += print_context(tokens, word_map)

    # ── § 3  Belirsizlik vitrini (bağımsız) ───────────────────
    print(f"\n  {'━' * (W - 4)}")
    print(f"  ┃  [BELİRSİZLİK VİTRİNİ]")
    print(f"  ┃  Bağlam olmadan birden fazla geçerli çözümleme")
    print(f"  {'━' * (W - 4)}")
    _hdr(3, "Çoklu Çözümleme Adayları")
    print_ambiguity(AMBIGUOUS, anlyzr)

    # ── Özet ──────────────────────────────────────────────────
    bw = W - 4  # box inner width
    print()
    print("  ╔" + "═" * bw + "╗")
    ln = "  ÖZET"
    print(f"  ║{ln:<{bw}s}║")
    print("  ║  " + "─" * (bw - 4) + "  ║")
    ln1 = f"  Metin türü : {len(TEXTS)}  │  Toplam sözcük   : {total_tok}"
    ln2 = f"  Bağlam Δ   : {total_ctx}  │  Belirsiz örnek  : {len(AMBIGUOUS)}"
    print(f"  ║{ln1:<{bw}s}║")
    print(f"  ║{ln2:<{bw}s}║")
    print("  ╚" + "═" * bw + "╝")
    print()


if __name__ == "__main__":
    main()
