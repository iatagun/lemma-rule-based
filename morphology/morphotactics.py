"""
Türkçe morfotaktik sonlu durum makinesi (FSM).

BFS ek soyma sırasında geçerli ek dizilimlerini enforce eder.
Dıştan içe soyma yönünde çalışır.

İsim yolu (içten dışa): KÖK → [YAPIM]* → [ÇOĞUL] → [İYELİK] → [HAL] → [İLGİ_-ki]
Fiil yolu (içten dışa): KÖK → [ÇATI]* → [OLUMSUZ] → [YETERLİLİK] → [ZAMAN/KİP] → [KİŞİ] → [BİLDİRME]
Köprüler: SIFAT_FİİL, MASTAR, İSİM_FİİL → isim yoluna geçiş
"""

from __future__ import annotations


# ── Slot tanımları ────────────────────────────────────────────

SLOT_RELATIONAL = "N_RELATIONAL"   # İLGİ_-ki
SLOT_CASE = "N_CASE"               # HAL ekleri
SLOT_POSSESSIVE = "N_POSSESSIVE"   # İYELİK ekleri
SLOT_PLURAL = "N_PLURAL"           # ÇOĞUL
SLOT_N_DERIVATION = "N_DERIVATION" # İsim YAPIM ekleri
SLOT_COPULA = "V_COPULA"           # BİLDİRME
SLOT_PERSON = "V_PERSON"           # KİŞİ ekleri
SLOT_TENSE = "V_TENSE"            # ZAMAN/KİP ekleri
SLOT_ABILITY = "V_ABILITY"         # YETERLİLİK
SLOT_NEGATION = "V_NEGATION"       # OLUMSUZ
SLOT_VOICE = "V_VOICE"            # ÇATI ekleri
SLOT_PARTICIPLE = "PARTICIPLE"     # SIFAT_FİİL (köprü: isim→fiil)
SLOT_CONVERB = "CONVERB"           # ZARF_FİİL
SLOT_INFINITIVE = "INFINITIVE"     # MASTAR
SLOT_NOMINAL_INF = "NOMINAL_INF"   # İSİM_FİİL

# ── Etiket → Slot eşlemesi ────────────────────────────────────
# Multi-label etiketler "/" ile ayrılmış alt etiketlere sahiptir.
# Her alt etiket bağımsız olarak bir slot'a eşlenir.

_LABEL_TO_SLOT: dict[str, str] = {
    # İsim slotları
    "İLGİ_-ki": SLOT_RELATIONAL,
    "BULUNMA": SLOT_CASE,
    "AYRILMA": SLOT_CASE,
    "YÖNELME": SLOT_CASE,
    "BELIRTME": SLOT_CASE,
    "VASITA": SLOT_CASE,
    "TAMLAYAN": SLOT_CASE,
    "İYELİK_3T": SLOT_POSSESSIVE,
    "İYELİK_3Ç": SLOT_POSSESSIVE,
    # Yapım ekleri (isim)
    "YAPIM_-CI": SLOT_N_DERIVATION,
    "YAPIM_-CIlIk": SLOT_N_DERIVATION,
    "YAPIM_-lAn": SLOT_N_DERIVATION,
    "YAPIM_-lAş": SLOT_N_DERIVATION,
    "YAPIM_-lI": SLOT_N_DERIVATION,
    "YAPIM_-lIk": SLOT_N_DERIVATION,
    "YAPIM_-sIz": SLOT_N_DERIVATION,
    # Fiil slotları
    "BİLDİRME": SLOT_COPULA,
    "KİŞİ_2Ç": SLOT_PERSON,
    "EMİR": SLOT_PERSON,
    "EMİR_3Ç": SLOT_PERSON,
    "KİŞİ_1T": SLOT_PERSON,
    "KİŞİ_2T": SLOT_PERSON,
    "KİŞİ_3Ç": SLOT_PERSON,
    "GEÇMİŞ_ZAMAN": SLOT_TENSE,
    "DUYULAN_GEÇMİŞ": SLOT_TENSE,
    "GELECEK_ZAMAN": SLOT_TENSE,
    "GENİŞ_ZAMAN": SLOT_TENSE,
    "GENİŞ_ZAMAN_OLMSZ": SLOT_TENSE,
    "ŞİMDİKİ_ZAMAN": SLOT_TENSE,
    "DİLEK_ŞART": SLOT_TENSE,
    "YETERLİLİK": SLOT_ABILITY,
    "OLUMSUZ": SLOT_NEGATION,
    "EDİLGEN": SLOT_VOICE,
    "ETTİRGEN_-lAt": SLOT_VOICE,
    "ETTİRGEN": SLOT_VOICE,
    # Köprüler
    "SIFAT_FİİL": SLOT_PARTICIPLE,
    "SIFAT_FİİL_-DIk": SLOT_PARTICIPLE,
    "SIFAT_FİİL_-DIğ": SLOT_PARTICIPLE,
    "ZARF_FİİL_-ArAk": SLOT_CONVERB,
    "ZARF_FİİL_-IncA": SLOT_CONVERB,
    "ZARF_FİİL_-Ip": SLOT_CONVERB,
    "ZARF_FİİL_-ken": SLOT_CONVERB,
    "MASTAR": SLOT_INFINITIVE,
    "İSİM_FİİL_-mA": SLOT_NOMINAL_INF,
}

# Dual-purpose etiketler: hem isim hem fiil bağlamında kullanılan ekler.
# Örn. -{I}m hem İYELİK_1T (evim) hem KİŞİ_1T (geldim).
# -lAr hem ÇOĞUL (evler) hem KİŞİ_3Ç (geldiler).
_MULTI_LABEL_SLOTS: dict[str, list[tuple[str, str]]] = {
    "BİLDİRME/ETTİRGEN": [
        ("BİLDİRME", SLOT_COPULA),
        ("ETTİRGEN", SLOT_VOICE),
    ],
    "OLUMSUZ/İSİM_FİİL": [
        ("OLUMSUZ", SLOT_NEGATION),
        ("İSİM_FİİL_-mA", SLOT_NOMINAL_INF),
    ],
    "İYELİK_3T/BELIRTME": [
        ("İYELİK_3T", SLOT_POSSESSIVE),
        ("BELIRTME", SLOT_CASE),
    ],
    "İYELİK_2T/TAMLAYAN": [
        ("İYELİK_2T", SLOT_POSSESSIVE),
        ("TAMLAYAN", SLOT_CASE),
        ("KİŞİ_2T", SLOT_PERSON),
    ],
    "EMİR/KİŞİ_2T": [
        ("EMİR", SLOT_PERSON),
        ("KİŞİ_2T", SLOT_PERSON),
    ],
    # -{I}z: KİŞİ_1Ç (geldik→iz) VEYA possessive in buffer-n contexts (bölge+n+iz)
    "KİŞİ_1Ç": [
        ("KİŞİ_1Ç", SLOT_PERSON),
        ("İYELİK_2T_buffer", SLOT_POSSESSIVE),
    ],
    # -{I}m: İYELİK_1T (evim) VEYA KİŞİ_1T (geldim, yaptım)
    "İYELİK_1T": [
        ("İYELİK_1T", SLOT_POSSESSIVE),
        ("KİŞİ_1T", SLOT_PERSON),
    ],
    # -lAr: ÇOĞUL (evler) VEYA KİŞİ_3Ç (geldiler)
    "ÇOĞUL": [
        ("ÇOĞUL", SLOT_PLURAL),
        ("KİŞİ_3Ç", SLOT_PERSON),
    ],
    # -{I}n{I}z: İYELİK_2Ç (eviniz) VEYA KİŞİ_2Ç (geldiniz)
    "İYELİK_2Ç": [
        ("İYELİK_2Ç", SLOT_POSSESSIVE),
        ("KİŞİ_2Ç", SLOT_PERSON),
    ],
    # -{I}m{I}z: İYELİK_1Ç (evimiz) VEYA KİŞİ_1Ç (yaptığımız → bağlamda)
    "İYELİK_1Ç": [
        ("İYELİK_1Ç", SLOT_POSSESSIVE),
        ("KİŞİ_1Ç", SLOT_PERSON),
    ],
    # -{I}ş: İŞTEŞ (vuruşmak, reciprocal voice) VEYA deverbal noun (bakış, görüş)
    "İŞTEŞ": [
        ("İŞTEŞ_voice", SLOT_VOICE),
        ("İŞTEŞ_deriv", SLOT_N_DERIVATION),
    ],
}


def _resolve_slots(suffix_label: str) -> list[str]:
    """Bir ek etiketini olası slot'lara çözümler.

    Multi-label etiketler birden fazla slot döndürebilir.
    """
    if suffix_label in _MULTI_LABEL_SLOTS:
        return list({slot for _, slot in _MULTI_LABEL_SLOTS[suffix_label]})
    slot = _LABEL_TO_SLOT.get(suffix_label)
    if slot is not None:
        return [slot]
    # Bilinmeyen etiket → muhafazakâr davran, engelleme
    return []


# ── Geçiş tablosu (dıştan içe) ───────────────────────────────
# Her durumdan hangi slot'lara geçiş yapılabilir?
# Dıştan içe soyma: START en dışta, iç katmanlara doğru gidilir.

_TRANSITIONS: dict[str, frozenset[str]] = {
    # START: ilk ek herhangi bir slot olabilir
    "START": frozenset({
        SLOT_RELATIONAL, SLOT_CASE, SLOT_POSSESSIVE, SLOT_PLURAL,
        SLOT_N_DERIVATION, SLOT_COPULA, SLOT_PERSON, SLOT_TENSE,
        SLOT_ABILITY, SLOT_NEGATION, SLOT_VOICE, SLOT_PARTICIPLE,
        SLOT_CONVERB, SLOT_INFINITIVE, SLOT_NOMINAL_INF,
    }),

    # ── İsim yolu (dıştan içe) ────────────────────────────────
    SLOT_RELATIONAL: frozenset({
        SLOT_CASE, SLOT_POSSESSIVE, SLOT_PLURAL, SLOT_N_DERIVATION,
        SLOT_PARTICIPLE, SLOT_NOMINAL_INF, SLOT_INFINITIVE,
        SLOT_CONVERB,
    }),

    SLOT_CASE: frozenset({
        SLOT_POSSESSIVE, SLOT_PLURAL, SLOT_N_DERIVATION,
        SLOT_PARTICIPLE, SLOT_NOMINAL_INF, SLOT_INFINITIVE,
        SLOT_CONVERB,
        SLOT_TENSE,       # nominalized tense: yapılacakta (BULUNMA→GELECEK_ZAMAN)
        SLOT_RELATIONAL,  # -ki reset: evdekinden (AYRILMA→İLGİ_-ki→BULUNMA)
    }),

    SLOT_POSSESSIVE: frozenset({
        SLOT_PLURAL, SLOT_N_DERIVATION,
        SLOT_PARTICIPLE, SLOT_NOMINAL_INF, SLOT_INFINITIVE,
        SLOT_CONVERB,
        SLOT_TENSE,       # nominalized tense: yapacağı (İYELİK→GELECEK_ZAMAN)
        SLOT_RELATIONAL,  # -ki reset
    }),

    SLOT_PLURAL: frozenset({
        SLOT_N_DERIVATION,
        SLOT_PARTICIPLE, SLOT_NOMINAL_INF, SLOT_INFINITIVE,
        SLOT_CONVERB,
        SLOT_TENSE,       # nominalized tense: olacakları (ÇOĞUL→GELECEK_ZAMAN)
        SLOT_RELATIONAL,  # -ki reset: çevresindekilere (ÇOĞUL→İLGİ_-ki→BULUNMA)
    }),

    # YAPIM soyuldu → YAPIM tekrar, köprüler, fiil slotları, isim slotları (stray vowel)
    SLOT_N_DERIVATION: frozenset({
        SLOT_N_DERIVATION,
        SLOT_PARTICIPLE, SLOT_NOMINAL_INF, SLOT_INFINITIVE,
        SLOT_VOICE, SLOT_TENSE, SLOT_NEGATION, SLOT_ABILITY,
        SLOT_CONVERB,
        SLOT_POSSESSIVE, SLOT_CASE,  # partial suffix: kullanıcılar → (ı+cı+lar)
    }),

    # ── Fiil yolu (dıştan içe, köprü sonrası) ────────────────
    SLOT_PARTICIPLE: frozenset({
        SLOT_TENSE, SLOT_ABILITY, SLOT_NEGATION, SLOT_VOICE,
        SLOT_N_DERIVATION,
    }),

    SLOT_CONVERB: frozenset({
        SLOT_ABILITY, SLOT_NEGATION, SLOT_VOICE, SLOT_TENSE,
        SLOT_N_DERIVATION,
        SLOT_PARTICIPLE,  # compound converb: -DIğIncA = PARTICIPLE + CONVERB
        SLOT_PERSON,      # finite verb + -ken: otururlarken (KİŞİ→CONVERB)
    }),

    SLOT_INFINITIVE: frozenset({
        SLOT_NEGATION, SLOT_VOICE, SLOT_ABILITY,
        SLOT_N_DERIVATION,
    }),

    SLOT_NOMINAL_INF: frozenset({
        SLOT_VOICE, SLOT_NEGATION, SLOT_ABILITY,
        SLOT_N_DERIVATION,
    }),

    # BİLDİRME: hem fiilsel (geliyordur) hem isimsel (güzeldir, yapılmaktadır)
    SLOT_COPULA: frozenset({
        SLOT_PERSON, SLOT_TENSE, SLOT_ABILITY, SLOT_NEGATION, SLOT_VOICE,
        SLOT_N_DERIVATION,
        SLOT_CASE, SLOT_POSSESSIVE, SLOT_PLURAL,
        SLOT_PARTICIPLE, SLOT_NOMINAL_INF, SLOT_INFINITIVE,
        SLOT_CONVERB,
    }),

    # KİŞİ soyuldu → ZAMAN, YETERLİLİK, OLUMSUZ, ÇATI, YAPIM, BİLDİRME
    SLOT_PERSON: frozenset({
        SLOT_TENSE, SLOT_ABILITY, SLOT_NEGATION, SLOT_VOICE,
        SLOT_N_DERIVATION,
        SLOT_COPULA,  # kişi sonrası kopula: gelmişlerdir (KİŞİ→BİLDİRME)
    }),

    # ZAMAN soyuldu → YETERLİLİK, OLUMSUZ, ÇATI, YAPIM
    # V_PERSON: bileşik zamanlarda ZAMAN2 → KİŞİ → ZAMAN1
    # V_TENSE: bileşik zamanlar (olurdu=-rdı, olursa=-rsa, diyormuş)
    # Nominal slots: isimleşmiş zaman (uğraşmaktansa: DİLEK_ŞART→AYRILMA→MASTAR)
    SLOT_TENSE: frozenset({
        SLOT_ABILITY, SLOT_NEGATION, SLOT_VOICE,
        SLOT_N_DERIVATION, SLOT_PERSON,
        SLOT_TENSE, SLOT_COPULA,
        SLOT_CASE, SLOT_POSSESSIVE, SLOT_PLURAL,
        SLOT_PARTICIPLE, SLOT_NOMINAL_INF, SLOT_INFINITIVE,
        SLOT_CONVERB,
    }),

    # YETERLİLİK soyuldu → OLUMSUZ, ÇATI, YAPIM
    SLOT_ABILITY: frozenset({
        SLOT_NEGATION, SLOT_VOICE,
        SLOT_N_DERIVATION,
    }),

    # OLUMSUZ soyuldu → ÇATI, YAPIM, HAL (connecting vowel: -eme/-ama pattern)
    SLOT_NEGATION: frozenset({
        SLOT_VOICE,
        SLOT_N_DERIVATION,
        SLOT_CASE,  # connecting vowel in negative ability: üretileme+diği → me+e(YÖNELME)
    }),

    # ÇATI soyuldu → ÇATI (tekrar), YAPIM (güzel+leş+tir)
    SLOT_VOICE: frozenset({
        SLOT_VOICE,
        SLOT_N_DERIVATION,
    }),
}


class MorphotacticFSM:
    """Türkçe morfotaktik durum makinesi.

    BFS ek soyma sırasında geçerli ek dizilimlerini enforce eder.
    Dıştan içe soyma yönünde çalışır.
    """

    def __init__(self) -> None:
        self._transitions = _TRANSITIONS

    def initial_state(self) -> str:
        """Başlangıç durumu (henüz ek soyulmamış)."""
        return "START"

    def transition(self, current_state: str, suffix_label: str) -> list[str]:
        """Verilen durumdan suffix_label soyulabilir mi?

        Multi-label etiketler birden fazla geçerli duruma yol açabilir.

        Returns:
            Geçerli yeni durumların listesi (boş liste = geçersiz geçiş).
        """
        allowed_slots = self._transitions.get(current_state)
        if allowed_slots is None:
            return []

        slots = _resolve_slots(suffix_label)

        # Bilinmeyen etiket → muhafazakâr: geçişe izin ver, mevcut durumda kal
        if not slots:
            return [current_state]

        valid_states: list[str] = []
        for slot in slots:
            if slot in allowed_slots:
                # İLGİ_-ki sonrası reset: isim slotları yeniden başlar
                if slot == SLOT_RELATIONAL:
                    valid_states.append(SLOT_RELATIONAL)
                else:
                    valid_states.append(slot)

        return valid_states

    def is_valid_sequence(self, suffix_labels: list[str]) -> bool:
        """Dıştan içe sıralanmış ek dizisinin geçerli olup olmadığını kontrol eder."""
        states = [self.initial_state()]
        for label in suffix_labels:
            next_states: list[str] = []
            for state in states:
                next_states.extend(self.transition(state, label))
            if not next_states:
                return False
            # Deduplicate
            states = list(dict.fromkeys(next_states))
        return True
