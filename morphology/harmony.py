"""
Ünlü ve ünsüz uyumu kontrol modülü.

SOLID:
  SRP – Yalnızca ses uyumu kurallarından sorumludur.
  OCP – Yeni uyum stratejileri (RelaxedHarmonyChecker vb.) mevcut kodu
        değiştirmeden eklenebilir.
  DIP – HarmonyChecker protokolü aracılığıyla soyutlama sağlar;
        çözümleyici somut sınıflara değil bu protokole bağımlıdır.
"""

from __future__ import annotations

from typing import Protocol

from .phonology import (
    BACK_VOWELS,
    CLOSE_VOWELS,
    CONSONANTS,
    ROUNDED,
    UNROUNDED,
    VOICELESS,
    get_syllable_nuclei,
    get_vowels,
    last_vowel,
)


# ── Temel Uyum Fonksiyonları (modül düzeyinde, paylaşılır) ────


def check_major_harmony(v_stem: str | None, v_suffix: str | None) -> bool:
    """
    Büyük Ünlü Uyumu (Kalınlık-İncelik Uyumu):
      Kalın (a,ı,o,u) → kalın  |  İnce (e,i,ö,ü) → ince
    """
    if v_stem is None or v_suffix is None:
        return True
    return (v_stem in BACK_VOWELS) == (v_suffix in BACK_VOWELS)


def check_minor_harmony(v_stem: str | None, v_suffix: str | None) -> bool:
    """
    Küçük Ünlü Uyumu (Düzlük-Yuvarlaklık Uyumu):
      Düz   (a,e,ı,i) → düz
      Yuvarlak (o,ö,u,ü) → dar yuvarlak (u,ü) veya geniş düz (a,e)
    """
    if v_stem is None or v_suffix is None:
        return True
    if v_stem in UNROUNDED:
        return v_suffix in UNROUNDED
    if v_suffix in CLOSE_VOWELS:
        return v_suffix in ROUNDED
    return v_suffix in UNROUNDED


def check_vowel_harmony(stem: str, suffix: str) -> bool:
    """Kök ile ek arasındaki tam ünlü uyumunu kontrol eder (BÜU + KÜU)."""
    sv = last_vowel(stem)
    sfx_vowels = get_vowels(suffix)
    if not sfx_vowels or sv is None:
        return True
    fv = sfx_vowels[0]
    return check_major_harmony(sv, fv) and check_minor_harmony(sv, fv)


def check_consonant_harmony(stem: str, suffix: str) -> bool:
    """
    Ünsüz Benzeşmesi:
      Sert ünsüzden (ç,f,h,k,p,s,ş,t) sonra d→t, c→ç olur.
    """
    if not stem or not suffix:
        return True
    last_ch = stem[-1]
    first_ch = suffix[0]
    if first_ch == "d" and last_ch in VOICELESS:
        return False
    if first_ch == "t" and last_ch not in VOICELESS:
        return False
    if first_ch == "c" and last_ch in VOICELESS:
        return False
    if first_ch == "ç" and last_ch not in VOICELESS:
        return False
    return True


def check_word_internal_harmony(word: str) -> dict:
    """Sözcüğün iç hece uyumunu kontrol eder (BÜU + KÜU).

    Heceleme yoluyla nucleusları (ünlüleri) çıkarır, ardışık her çift
    arasında hem büyük hem küçük ünlü uyumunu denetler.

    Returns:
        {
            "buu_ok": bool,     # Tüm çiftler BÜU'ya uyuyor mu
            "kuu_ok": bool,     # Tüm çiftler KÜU'ya uyuyor mu
            "full_ok": bool,    # Her iki uyum da sağlanıyor mu
            "nuclei": list,     # Hece çekirdekleri
            "violations": list, # İhlal detayları
        }
    """
    nuclei = get_syllable_nuclei(word)

    violations: list[dict] = []
    all_buu = True
    all_kuu = True

    for i in range(len(nuclei) - 1):
        v1, v2 = nuclei[i], nuclei[i + 1]
        buu = check_major_harmony(v1, v2)
        kuu = check_minor_harmony(v1, v2)

        if not buu:
            all_buu = False
        if not kuu:
            all_kuu = False
        if not buu or not kuu:
            violations.append(
                {"pos": i, "v1": v1, "v2": v2, "buu": buu, "kuu": kuu}
            )

    return {
        "buu_ok": all_buu,
        "kuu_ok": all_kuu,
        "full_ok": all_buu and all_kuu,
        "nuclei": nuclei,
        "violations": violations,
    }


# ── HarmonyChecker Protokolü (DIP) ───────────────────────────


class HarmonyChecker(Protocol):
    """Uyum kontrol arayüzü — çözümleyici bu protokole bağımlıdır."""

    def check_vowel_harmony(self, stem: str, suffix: str) -> bool: ...
    def check_consonant_harmony(self, stem: str, suffix: str) -> bool: ...


# ── Somut Uyum Stratejileri (OCP – yeni strateji = yeni sınıf) ──


class StrictHarmonyChecker:
    """Tam uyum: büyük + küçük ünlü uyumu + ünsüz benzeşmesi."""

    def check_vowel_harmony(self, stem: str, suffix: str) -> bool:
        sv = last_vowel(stem)
        sfx_vowels = get_vowels(suffix)
        if not sfx_vowels or sv is None:
            return True
        fv = sfx_vowels[0]
        return check_major_harmony(sv, fv) and check_minor_harmony(sv, fv)

    def check_consonant_harmony(self, stem: str, suffix: str) -> bool:
        return check_consonant_harmony(stem, suffix)


class RelaxedHarmonyChecker:
    """Alıntı sözcükler için gevşek uyum: yalnızca küçük ünlü uyumu + ünsüz."""

    def check_vowel_harmony(self, stem: str, suffix: str) -> bool:
        sv = last_vowel(stem)
        sfx_vowels = get_vowels(suffix)
        if not sfx_vowels or sv is None:
            return True
        return check_minor_harmony(sv, sfx_vowels[0])

    def check_consonant_harmony(self, stem: str, suffix: str) -> bool:
        return check_consonant_harmony(stem, suffix)
