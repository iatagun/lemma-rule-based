"""
Ek tanımları, şablon açılım sistemi ve ek kaydı (registry).

SOLID:
  SRP – Yalnızca ek verisi ve şablon açılımıyla ilgilenir.
  OCP – register() ile yeni ekler eklenebilir; mevcut kod değiştirilmez.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuffixDefinition:
    """Şablon tabanlı ek tanımı (girdi formatı)."""

    template: str
    label: str
    harmony_exempt: bool = False
    min_stem_length: int = 2


@dataclass(frozen=True)
class SuffixForm:
    """Açılmış (somut) ek biçimi."""

    form: str
    label: str
    harmony_exempt: bool
    min_stem_length: int


class SuffixRegistry:
    """
    Ek kaydı ve yönetimi.

    register() / register_many() ile ek kümeleri genişletilebilir.
    create_default() varsayılan Türkçe ekleri yükler.
    """

    TEMPLATE_VARS: dict[str, list[str]] = {
        "{A}": ["a", "e"],
        "{I}": ["ı", "i", "u", "ü"],
        "{D}": ["d", "t"],
        "{C}": ["c", "ç"],
    }

    def __init__(self) -> None:
        self._suffixes: list[SuffixForm] = []
        self._sorted = False

    def register(self, definition: SuffixDefinition) -> None:
        """Tek bir şablon tanımını açarak kayda ekler."""
        for form in self._expand(definition.template):
            self._suffixes.append(
                SuffixForm(
                    form=form,
                    label=definition.label,
                    harmony_exempt=definition.harmony_exempt,
                    min_stem_length=definition.min_stem_length,
                )
            )
        self._sorted = False

    def register_many(self, definitions: list[SuffixDefinition]) -> None:
        """Birden çok tanımı toplu kayda ekler."""
        for d in definitions:
            self.register(d)

    @property
    def suffixes(self) -> list[SuffixForm]:
        """Uzunluğa göre sıralı ek listesi (uzun ekler önce)."""
        if not self._sorted:
            self._suffixes.sort(key=lambda s: (-len(s.form), s.form))
            self._sorted = True
        return self._suffixes

    @classmethod
    def _expand(cls, template: str) -> list[str]:
        """Şablondaki yer tutucuları tüm olası değerlere açar."""
        if "{" not in template:
            return [template]

        results = [template]
        changed = True
        while changed:
            changed = False
            for placeholder, options in cls.TEMPLATE_VARS.items():
                new_results = []
                for r in results:
                    if placeholder in r:
                        changed = True
                        for opt in options:
                            new_results.append(r.replace(placeholder, opt, 1))
                    else:
                        new_results.append(r)
                results = new_results
        return results

    @classmethod
    def create_default(cls) -> SuffixRegistry:
        """Varsayılan Türkçe ek kümesini yükleyerek yeni bir kayıt oluşturur."""
        registry = cls()
        registry.register_many(DEFAULT_SUFFIX_DEFINITIONS)
        return registry


# ════════════════════════════════════════════════════════════════
# Varsayılan Türkçe Ek Tanımları
# ════════════════════════════════════════════════════════════════

_D = SuffixDefinition

DEFAULT_SUFFIX_DEFINITIONS: list[SuffixDefinition] = [
    # ── ÇOK HECELİ EKLER (5+ karakter) ────────────────────────
    _D("s{I}nl{A}r", "EMİR_3Ç"),
    _D("{C}{I}l{I}k", "YAPIM_-CIlIk"),
    _D("s{I}n{I}z", "KİŞİ_2Ç"),
    _D("{I}m{I}z", "İYELİK_1Ç"),
    _D("{I}n{I}z", "İYELİK_2Ç"),
    _D("y{A}r{A}k", "ZARF_FİİL_-ArAk"),
    _D("{A}r{A}k", "ZARF_FİİL_-ArAk"),
    _D("y{I}nc{A}", "ZARF_FİİL_-IncA"),
    _D("{I}nc{A}", "ZARF_FİİL_-IncA"),
    _D("{A}c{A}k", "GELECEK_ZAMAN"),
    _D("l{A}r{I}", "İYELİK_3Ç"),
    _D("n{D}{A}n", "AYRILMA"),
    # ── 4 HARFLİ EKLER ────────────────────────────────────────
    _D("{I}yor", "ŞİMDİKİ_ZAMAN"),
    _D("{D}{I}k", "SIFAT_FİİL_-DIk"),
    _D("{A}bil", "YETERLİLİK"),
    # ── 3 HARFLİ EKLER ────────────────────────────────────────
    _D("l{A}r", "ÇOĞUL"),
    _D("{D}{I}ğ", "SIFAT_FİİL_-DIğ"),
    _D("{D}{A}n", "AYRILMA"),
    _D("n{I}n", "TAMLAYAN"),
    _D("l{I}k", "YAPIM_-lIk"),
    _D("s{I}z", "YAPIM_-sIz"),
    _D("m{I}ş", "DUYULAN_GEÇMİŞ"),
    _D("m{A}k", "MASTAR"),
    _D("m{A}z", "GENİŞ_ZAMAN_OLMSZ"),
    _D("{D}{I}r", "BİLDİRME/ETTİRGEN"),
    _D("l{A}ş", "YAPIM_-lAş"),
    _D("l{A}n", "YAPIM_-lAn"),
    _D("l{A}t", "ETTİRGEN_-lAt"),
    _D("y{I}p", "ZARF_FİİL_-Ip"),
    _D("s{I}n", "EMİR/KİŞİ_2T"),
    _D("yor", "ŞİMDİKİ_ZAMAN", harmony_exempt=True),
    # ── 2 HARFLİ EKLER ────────────────────────────────────────
    _D("{D}{A}", "BULUNMA"),
    _D("y{A}", "YÖNELME"),
    _D("n{A}", "YÖNELME"),
    _D("{D}{I}", "GEÇMİŞ_ZAMAN"),
    _D("l{A}", "VASITA"),
    _D("y{I}", "BELIRTME"),
    _D("n{I}", "BELIRTME"),
    _D("{I}m", "İYELİK_1T", min_stem_length=3),
    _D("{I}n", "İYELİK_2T/TAMLAYAN", min_stem_length=3),
    _D("s{I}", "İYELİK_3T"),
    _D("l{I}", "YAPIM_-lI", min_stem_length=3),
    _D("{C}{I}", "YAPIM_-CI"),
    _D("s{A}", "DİLEK_ŞART"),
    _D("m{A}", "OLUMSUZ/İSİM_FİİL"),
    _D("{I}l", "EDİLGEN", min_stem_length=3),
    _D("{I}ş", "İŞTEŞ", min_stem_length=3),
    _D("{A}n", "SIFAT_FİİL", min_stem_length=3),
    _D("{I}r", "GENİŞ_ZAMAN", min_stem_length=3),
    _D("{A}r", "GENİŞ_ZAMAN", min_stem_length=3),
    _D("{I}z", "KİŞİ_1Ç", min_stem_length=3),
    _D("{I}p", "ZARF_FİİL_-Ip"),
    _D("ken", "ZARF_FİİL_-ken", harmony_exempt=True),
    _D("ki", "İLGİ_-ki", harmony_exempt=True),
    # ── 1 HARFLİ EKLER ────────────────────────────────────────
    _D("{I}", "İYELİK_3T/BELIRTME", min_stem_length=3),
    _D("{A}", "YÖNELME", min_stem_length=3),
    # Yalın kişi ekleri (fiil zaman çekimi sonrası):
    # gördü+m, yaptı+m, geldi+m → KİŞİ_1T
    _D("m", "KİŞİ_1T", min_stem_length=4),
]
