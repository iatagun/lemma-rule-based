#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe Kural Tabanlı Kök-Ek Ayırıcı
=====================================
Büyük ve küçük ünlü uyumu kurallarına dayalı morfolojik çözümleme.
Dizge projesinden (https://github.com/dizge/dizge) esinlenilmiştir.

Kullanım:
    python find_lemma.py                    # Yerleşik test sözcükleri
    python find_lemma.py evlerinden         # Tek sözcük çözümle
    python find_lemma.py -i                 # Etkileşimli mod
"""

import sys
from pathlib import Path

from morphology import AnalysisFormatter, MorphologicalAnalyzer, create_default_analyzer

# ── Sözlük dosyası tespiti ────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_DICT_PATH = _SCRIPT_DIR / "turkish_words.txt"

# ── Varsayılan çözümleyici ve biçimlendirici (modül düzeyinde) ──

_analyzer: MorphologicalAnalyzer = create_default_analyzer(
    dictionary_path=_DICT_PATH if _DICT_PATH.exists() else None,
)
_formatter: AnalysisFormatter = AnalysisFormatter()


# ── Geriye dönük uyumluluk ────────────────────────────────────


def find_morphemes(word: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Sözcüğü kök ve eklerine ayırır (geriye dönük uyumlu API).

    Returns:
        tuple: (kök, [(ek_biçimi, etiket), ...])
    """
    result = _analyzer.analyze(word)
    return result.stem, result.suffixes


# ── CLI İşlevleri ─────────────────────────────────────────────


def analyze(word: str, verbose: bool = True) -> tuple[str, list[tuple[str, str]]]:
    """Sözcüğü çözümleyip sonucu yazdırır."""
    result = _analyzer.analyze(word)

    if verbose:
        all_results = _analyzer.analyze_all(word)
        if len(all_results) > 1:
            print(_formatter.format_multi_analysis(word, all_results))
        else:
            print(_formatter.format_analysis(word, result))

    return result.stem, result.suffixes


def interactive_mode() -> None:
    """Etkileşimli mod: kullanıcıdan sözcük alıp çözümler."""
    print("=" * 55)
    print("  Türkçe Kural Tabanlı Kök-Ek Ayırıcı")
    print("  Büyük & Küçük Ünlü Uyumu Analizi")
    print("=" * 55)
    print("  Çıkmak için 'q' yazın.\n")

    while True:
        try:
            word = input("  Sözcük > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not word or word.lower() == "q":
            break
        print()
        analyze(word)


# ── Ana Giriş Noktası ─────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "-i":
            interactive_mode()
        else:
            for arg in sys.argv[1:]:
                analyze(arg)
    else:
        test_words = [
            # Temel testler
            "evlerinden", "kitaplarımızda", "güzelliklerini",
            "okullardan", "çocuklarımız", "gelmişler",
            "öğretmenlerimizden", "yürüyorlar", "bakacaklarmış",
            "gidilmez", "sevdiklerimiz", "okumuşsunuz",
            "başarısızlık", "mutluluklarından", "geliyor", "yazılmış",
            # Edge case: ünsüz yumuşaması
            "kitabından", "rengini", "ağacın",
            # Edge case: ünlü düşmesi
            "burnumda", "oğlumuz", "gönlümüzden",
            # Edge case: ünlü daralması
            "diyor", "yiyorlar",
            # Edge case: tek heceli kökler
            "gözlüklerin", "elden", "içinden",
            # Edge case: alıntı sözcükler
            "saatlerinde", "otobüslerden", "televizyonlarından",
            # Edge case: tampon ünsüz
            "suyunda", "arabasından",
            # Edge case: ünsüz benzeşmesi
            "gittiğimiz", "baktıklarımız",
            # Edge case: uzun zincir / yapım ekleri
            "güzelleştirilemez", "çalışkanlıklarından",
            # Edge case: edilgen / ettirgen
            "yaptırılmış", "okutturulmuş",
            # Edge case: yuvarlak ünlü uyumu
            "görüşünüzden", "gördüğümüzden", "bölüşülmüştür",
            # Edge case: istisna ekler (-ki, -ken, -yor)
            "evdekilerden", "çalışırken", "koşuyormuş",
            # Edge case: yeterlilik + aşırı agglütinasyon
            "yazdırılabileceklerdenmişsiniz",
            # Edge case: İşteş (leksikalleşmiş vs üretken)
            "dövüşebilirlermiş", "konuşulabilecek", "çalıştırılmış",
            # Edge case: ünlü düşmesi + uzun zincir
            "gönlünüzdekilerden", "oğullarımızdan",
            # Edge case: sıfat-fiil + isim ekleri
            "gördüklerimizden", "yapılacaklardanmış",
            # Edge case: özel isim + apostrof
            "Ankara'dakilerden", "İstanbul'undaki",
            # Edge case: çatı zinciri (ettirgen + edilgen)
            "sevdirilmişlerdir", "atılabilir", "içilebilirmiş",
            # Edge case: ünlü daralması + yeterlilik
            "diyebileceklermiş", "yiyebilirsiniz",
            # Edge case: tampon ünsüz + iyelik + tamlayan
            "kapısının",
        ]

        print("=" * 55)
        print("  Türkçe Kural Tabanlı Kök-Ek Ayırıcı")
        print("  Büyük & Küçük Ünlü Uyumu Analizi")
        print("=" * 55)
        print()

        for w in test_words:
            analyze(w)
