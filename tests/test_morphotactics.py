"""MorphotacticFSM birim testleri."""

import pytest

from morphology.morphotactics import MorphotacticFSM


@pytest.fixture
def fsm():
    return MorphotacticFSM()


# ── İsim yolu testleri ────────────────────────────────────────


class TestNominalPath:
    """İsim ek sıralaması (dıştan içe)."""

    def test_single_case(self, fsm):
        # ev-de
        assert fsm.is_valid_sequence(["BULUNMA"])

    def test_case_after_plural(self, fsm):
        # çocuk-lar-ı → BFS: BELIRTME → ÇOĞUL
        assert fsm.is_valid_sequence(["BELIRTME", "ÇOĞUL"])

    def test_possessive_after_plural(self, fsm):
        # çocuk-lar-ımız → BFS: İYELİK_1Ç → ÇOĞUL
        assert fsm.is_valid_sequence(["İYELİK_1Ç", "ÇOĞUL"])

    def test_case_possessive_plural(self, fsm):
        # ev-ler-imiz-den → BFS: AYRILMA → İYELİK_1Ç → ÇOĞUL
        assert fsm.is_valid_sequence(["AYRILMA", "İYELİK_1Ç", "ÇOĞUL"])

    def test_relational_ki_reset(self, fsm):
        # evdeki-ler-den → BFS: AYRILMA → ÇOĞUL → İLGİ_-ki → BULUNMA
        assert fsm.is_valid_sequence(
            ["AYRILMA", "ÇOĞUL", "İLGİ_-ki", "BULUNMA"]
        )

    def test_derivation_plural(self, fsm):
        # güzel-lik-ler → BFS: ÇOĞUL → YAPIM_-lIk
        assert fsm.is_valid_sequence(["ÇOĞUL", "YAPIM_-lIk"])


# ── Fiil yolu testleri ────────────────────────────────────────


class TestVerbalPath:
    """Fiil ek sıralaması (dıştan içe)."""

    def test_past_tense_passive(self, fsm):
        # yapıl-dı → BFS: GEÇMİŞ_ZAMAN → EDİLGEN
        assert fsm.is_valid_sequence(["GEÇMİŞ_ZAMAN", "EDİLGEN"])

    def test_person_tense(self, fsm):
        # gel-di-m → BFS: İYELİK_1T → GEÇMİŞ_ZAMAN
        assert fsm.is_valid_sequence(["İYELİK_1T", "GEÇMİŞ_ZAMAN"])

    def test_compound_tense(self, fsm):
        # gel-iyor-du → BFS: GEÇMİŞ_ZAMAN → ŞİMDİKİ_ZAMAN
        assert fsm.is_valid_sequence(["GEÇMİŞ_ZAMAN", "ŞİMDİKİ_ZAMAN"])

    def test_compound_tense_conditional(self, fsm):
        # ol-ur-sa → BFS: DİLEK_ŞART → GENİŞ_ZAMAN
        assert fsm.is_valid_sequence(["DİLEK_ŞART", "GENİŞ_ZAMAN"])

    def test_ability_negation_voice(self, fsm):
        # yap-tır-ıl-ama-z → complex chain
        assert fsm.is_valid_sequence(
            ["GENİŞ_ZAMAN_OLMSZ", "OLUMSUZ/İSİM_FİİL", "EDİLGEN"]
        )

    def test_copula_after_person(self, fsm):
        # gel-miş-ler-dir → BFS: BİLDİRME/ETTİRGEN → ÇOĞUL → DUYULAN_GEÇMİŞ
        assert fsm.is_valid_sequence(
            ["BİLDİRME/ETTİRGEN", "ÇOĞUL", "DUYULAN_GEÇMİŞ"]
        )


# ── Köprü testleri ────────────────────────────────────────────


class TestBridgePaths:
    """Fiil→İsim geçiş köprüleri."""

    def test_participle_bridge(self, fsm):
        # yap-ıl-an-lar → BFS: ÇOĞUL → SIFAT_FİİL → EDİLGEN
        assert fsm.is_valid_sequence(["ÇOĞUL", "SIFAT_FİİL", "EDİLGEN"])

    def test_infinitive_bridge(self, fsm):
        # yap-mak-tan → BFS: AYRILMA → MASTAR
        assert fsm.is_valid_sequence(["AYRILMA", "MASTAR"])

    def test_nominalized_tense(self, fsm):
        # ol-acak-lar-ı → BFS: İYELİK_3T/BELIRTME → ÇOĞUL → GELECEK_ZAMAN
        assert fsm.is_valid_sequence(
            ["İYELİK_3T/BELIRTME", "ÇOĞUL", "GELECEK_ZAMAN"]
        )

    def test_converb_with_person(self, fsm):
        # otur-ur-lar-ken → BFS: ZARF_FİİL_-ken → ÇOĞUL → GENİŞ_ZAMAN
        assert fsm.is_valid_sequence(
            ["ZARF_FİİL_-ken", "ÇOĞUL", "GENİŞ_ZAMAN"]
        )

    def test_is_deverbal_noun(self, fsm):
        # bağır-ış-lar-ı → BFS: İYELİK_3T/BELIRTME → ÇOĞUL → İŞTEŞ
        assert fsm.is_valid_sequence(
            ["İYELİK_3T/BELIRTME", "ÇOĞUL", "İŞTEŞ"]
        )


# ── Geçersiz dizilim testleri ─────────────────────────────────


class TestInvalidSequences:
    """Kesinlikle geçersiz ek dizilimleri."""

    def test_double_case(self, fsm):
        # Çift hal eki olamaz: -de-den
        assert not fsm.is_valid_sequence(["AYRILMA", "BULUNMA"])

    def test_double_possessive(self, fsm):
        # Çift iyelik olamaz: -im-in
        assert not fsm.is_valid_sequence(["İYELİK_3T", "İYELİK_3Ç"])

    def test_plural_after_voice(self, fsm):
        # ÇOĞUL maps to both N_PLURAL and V_PERSON (KİŞİ_3Ç),
        # so ÇOĞUL → EDİLGEN is valid via V_PERSON path (yapıldılar)
        assert fsm.is_valid_sequence(["ÇOĞUL", "EDİLGEN"])

    def test_voice_after_case(self, fsm):
        # HAL → ÇATI imkansız
        assert not fsm.is_valid_sequence(["EDİLGEN", "BULUNMA"])

    def test_relational_after_voice(self, fsm):
        # İLGİ_-ki → ÇATI imkansız
        assert not fsm.is_valid_sequence(["İLGİ_-ki", "EDİLGEN"])


# ── Multi-label etiket testleri ───────────────────────────────


class TestMultiLabel:
    """Çok anlamlı etiketlerin doğru işlenmesi."""

    def test_bildirme_ettirgen_dual(self, fsm):
        results = fsm.transition("START", "BİLDİRME/ETTİRGEN")
        assert len(results) >= 2  # Both copula and voice paths

    def test_iyelik_belirtme_dual(self, fsm):
        results = fsm.transition("START", "İYELİK_3T/BELIRTME")
        assert len(results) >= 2

    def test_cogul_as_person(self, fsm):
        results = fsm.transition("START", "ÇOĞUL")
        assert len(results) >= 2

    def test_unknown_label_conservative(self, fsm):
        # Bilinmeyen etiket → engellenmemeli (muhafazakâr)
        results = fsm.transition("START", "BILINMEYEN_ETIKET")
        assert results  # Should not be empty
