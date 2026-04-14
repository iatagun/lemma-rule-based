"""
Türkçe fonoloji sabitleri ve yardımcı fonksiyonlar.

Dizge projesinden (https://github.com/dizge/dizge) esinlenilmiştir.

SOLID – SRP: Yalnızca fonetik veri ve ünlü/ünsüz sınıflandırma işlemlerinden
             sorumludur.
"""

from __future__ import annotations

# ── Ünlü Grupları ──────────────────────────────────────────────

VOWELS = frozenset("aeıioöuüâîû")

BACK_VOWELS = frozenset("aıouâû")  # Kalın ünlüler (â≈a, û≈u)
FRONT_VOWELS = frozenset("eiöüî")  # İnce ünlüler (î≈i)

UNROUNDED = frozenset("aeıiâî")  # Düz ünlüler
ROUNDED = frozenset("oöuüû")  # Yuvarlak ünlüler

OPEN_VOWELS = frozenset("aeoöâ")  # Geniş ünlüler
CLOSE_VOWELS = frozenset("ıiuüîû")  # Dar ünlüler

# ── Ünsüz Grupları ─────────────────────────────────────────────

CONSONANTS = frozenset("bcçdfgğhjklmnprsştvyz")
VOICELESS = frozenset("çfhkpsşt")  # Sert (ötümsüz) ünsüzler

# Kök sonu ünsüz kümesi kontrolü: akıcı/sızıcı + ğ (oğl-, değl- gibi)
SONORANT_SIBILANT = frozenset("lmnryşszğ")


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────


def get_vowels(word: str) -> list[str]:
    """Sözcükteki ünlüleri sırasıyla döndürür."""
    return [ch for ch in word if ch in VOWELS]


def last_vowel(word: str) -> str | None:
    """Sözcükteki son ünlüyü döndürür."""
    for ch in reversed(word):
        if ch in VOWELS:
            return ch
    return None


def turkish_lower(text: str) -> str:
    """Türkçe kurallarına uygun küçük harf dönüşümü.

    Python'un str.lower() fonksiyonu Türkçe İ/I harflerini yanlış dönüştürür:
      - İ → i̇ (dotted, combining dot above) — olması gereken: i
      - I → ı (Python bunu doğru yapmaz — i verir)

    Bu fonksiyon Türkçe locale kurallarını uygular:
      İ → i,  I → ı
    """
    result: list[str] = []
    for ch in text:
        if ch == "\u0130":      # İ (Latin capital I with dot above)
            result.append("i")
        elif ch == "I":         # I (Latin capital I without dot)
            result.append("\u0131")  # ı
        else:
            result.append(ch.lower())
    return "".join(result)


# ── Heceleme (Syllabification) ────────────────────────────────


def syllabify(word: str) -> list[str]:
    """Türkçe sözcüğü hecelere ayırır.

    Her hecede tam bir ünlü (nucleus) bulunur. Temel kurallar:
      - İki ünlü arasında tek ünsüz → sonraki heceye  (V-CV)
      - İki ünlü arasında 2+ ünsüz → son ünsüz sonraki heceye (VC-CV)
      - Bitişik ünlüler ayrı hecelere                   (V-V)

    Dizge projesindeki (dizge/tools/phonology.py) heceleme yaklaşımından
    esinlenilmiştir.

    Returns:
        Hece listesi. Örn: syllabify("evlerinden") → ["ev","le","rin","den"]
    """
    word = word.lower()

    vowel_positions = [i for i, ch in enumerate(word) if ch in VOWELS]

    if len(vowel_positions) <= 1:
        return [word]

    # Ardışık ünlü çiftleri arasındaki sınırları belirle
    splits: list[int] = []
    for k in range(len(vowel_positions) - 1):
        v1 = vowel_positions[k]
        v2 = vowel_positions[k + 1]
        gap = v2 - v1 - 1          # aradaki ünsüz sayısı

        if gap == 0:
            # Bitişik ünlüler: aralarından böl
            splits.append(v2)
        else:
            # Bir veya daha fazla ünsüz: son ünsüz sonraki heceye
            splits.append(v2 - 1)

    parts: list[str] = []
    prev = 0
    for sp in splits:
        parts.append(word[prev:sp])
        prev = sp
    parts.append(word[prev:])

    return parts


def get_syllable_nuclei(word: str) -> list[str]:
    """Sözcükteki her hecenin çekirdeğini (nucleus = ünlü) döndürür.

    Heceleme sonrası her hecedeki ilk ünlüyü seçer.

    Returns:
        Nucleus listesi. Örn: get_syllable_nuclei("evlerinden") → ["e","e","i","e"]
    """
    nuclei: list[str] = []
    for syl in syllabify(word):
        for ch in syl:
            if ch in VOWELS:
                nuclei.append(ch)
                break
    return nuclei


def is_loanword_candidate(word: str) -> bool:
    """Alıntı sözcük adayı tespiti.

    Türkçe kökenli sözcüklerde yuvarlak geniş ünlüler (o, ö) yalnızca
    ilk hecede bulunur. İkinci veya sonraki hecelerde o/ö varsa sözcük
    büyük olasılıkla alıntıdır (doktor, kontrol, otobüs, profesör).
    """
    nuclei = get_syllable_nuclei(word)
    return any(v in ("o", "ö") for v in nuclei[1:])
