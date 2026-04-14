#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cümle düzeyinde bağlamsal çözümleme demo.

Sözcük bazlı (analyze_all) ve cümle bazlı (SentenceAnalyzer)
sonuçlarını yan yana karşılaştırır.  Bağlam kuralının sırayı
değiştirdiği sözcükler ★ ile işaretlenir.

Referans: Oflazer 1994, "Two-level Description of Turkish Morphology"
"""

import sys, textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from morphology import create_default_analyzer
from morphology.sentence import SentenceAnalyzer

# ── Tag haritası (demo_text.py ile uyumlu) ────────────────────
_TAG = {
    "ÇOĞUL": "Pl", "İYELİK_1T": "P1sg", "İYELİK_2T": "P2sg",
    "İYELİK_3T": "P3sg", "İYELİK_1Ç": "P1pl", "İYELİK_2Ç": "P2pl",
    "İYELİK_3Ç": "P3pl", "BELIRTME": "Acc", "YÖNELME": "Dat",
    "BULUNMA": "Loc", "AYRILMA": "Abl", "TAMLAYAN": "Gen",
    "VASITA": "Ins", "GEÇMİŞ_ZAMAN": "Past", "DUYULAN_GEÇMİŞ": "Evid",
    "GENİŞ_ZAMAN": "Aor", "ŞİMDİKİ_ZAMAN": "Prog",
    "GELECEK_ZAMAN": "Fut", "OLUMSUZLUK": "Neg", "SIFAT_FİİL": "Part",
    "ZARF_FİİL": "Conv", "İSİM_FİİL": "Vnoun", "İSİM_FİİL_mAk": "Inf",
    "KİŞİ_1T": "A1sg", "KİŞİ_2T": "A2sg", "KİŞİ_3T": "A3sg",
    "KİŞİ_1Ç": "A1pl", "KİŞİ_2Ç": "A2pl", "KİŞİ_3Ç": "A3pl",
    "EDİLGEN": "Pass", "ETTİRGEN": "Caus", "ETTİRGEN_-lAt": "Caus",
    "DÖNÜŞLÜ": "Refl", "İŞTEŞ": "Recip", "YETERLİLİK": "Abil",
    "BİLDİRME": "Cop", "İLGİ_-ki": "Rel", "SORU": "Q",
    "İYELİK_3T/BELIRTME": "P3sg|Acc",
    "İYELİK_2T/TAMLAYAN": "P2sg|Gen",
    "İYELİK_3T/TAMLAYAN": "P3sg|Gen",
    "SIFAT_FİİL_-dIk": "PastPart", "SIFAT_FİİL_-AcAk": "FutPart",
}

W = 78  # terminal width

# ── Yardımcı ─────────────────────────────────────────────────

def _fmt_oflazer(a):
    """kök+Tag+Tag biçimi."""
    if not a.suffixes:
        return a.stem
    tags = "+".join(_TAG.get(lbl, lbl) for _, lbl in a.suffixes)
    return f"{a.stem}+{tags}"


# ── Kurulum ──────────────────────────────────────────────────
_DICT = Path(__file__).resolve().parent / "turkish_words.txt"
analyzer = create_default_analyzer(
    dictionary_path=_DICT if _DICT.exists() else None,
)
sa = SentenceAnalyzer(analyzer)

TEXTS = [
    ("Haber",
     "Türkiye'nin başkenti Ankara'da dün gerçekleştirilen toplantıda "
     "bilim insanları yapay zekanın geleceğini tartıştılar."),
    ("Bağlamsal",
     "Bu yemek çok güzel olmuş. Şu kitapları masaya koyun. "
     "O eski çalar saati bana verin."),
    ("Karmaşık",
     "Katılımcılardan biri konuşmasında teknolojinin hayatımızdaki "
     "değişimleri anlattı. Öğrenciler de bu gelişmelerden oldukça "
     "etkilenmişlerdi."),
]


def main():
    bw = W - 4
    print()
    print("  ╔" + "═" * bw + "╗")
    h = "  Sözcük Düzeyi  ↔  Cümle Düzeyi  Karşılaştırma"
    print(f"  ║{h:<{bw}s}║")
    print("  ╚" + "═" * bw + "╝")

    total_words = 0
    total_ctx = 0

    for label, text in TEXTS:
        # cümle bazlı
        tokens = sa.analyze(text)
        # sözcük bazlı
        word_only = {t.word: analyzer.analyze_all(t.word) for t in tokens}

        # ── Metin başlığı ────────────────────────────────
        print()
        print("  " + "━" * (W - 2))
        wrapped = textwrap.fill(text, width=W - 6)
        print(f"  ┃  [{label.upper()}]")
        for ln in wrapped.splitlines():
            print(f"  ┃  {ln}")
        print("  " + "━" * (W - 2))
        print()

        # ── Tablo ────────────────────────────────────────
        print(f"    # │ {'Sözcük':<20s}│ {'Sözcük Bazlı':<26s}│ {'Cümle Bazlı':<26s}│ Kural")
        print("  ────┼" + "─" * 20 + "┼" + "─" * 26 + "┼" + "─" * 26 + "┼" + "─" * 15)

        changed = 0
        for i, t in enumerate(tokens, 1):
            total_words += 1
            wo = word_only[t.word]
            wo_fmt = _fmt_oflazer(wo[0]) if wo else "?"
            sa_fmt = _fmt_oflazer(t.analysis)

            wo_stem = wo[0].stem if wo else "?"
            sa_stem = t.analysis.stem
            is_diff = wo_stem != sa_stem
            mark = "★" if is_diff else " "
            rules = ", ".join(t.context_applied) if t.context_applied else "—"

            print(f"  {i:3d} │{mark}{t.word:<19s}│ {wo_fmt:<25s}│ {sa_fmt:<25s}│ {rules}")
            if is_diff:
                changed += 1

        total_ctx += changed
        print()
        if changed:
            print(f"  → {changed} sözcükte bağlam kuralı etkili")
        else:
            print("  → bağlam kuralı bu metinde değişiklik yapmadı")

    # ── Özet ─────────────────────────────────────────────
    print()
    print("  ╔" + "═" * bw + "╗")
    s1 = f"  Toplam sözcük : {total_words}  │  Bağlam değişimi : {total_ctx}"
    print(f"  ║{s1:<{bw}s}║")
    print("  ╚" + "═" * bw + "╝")
    print()


if __name__ == "__main__":
    main()
