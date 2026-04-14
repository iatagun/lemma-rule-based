"""Cümle düzeyinde bağlamsal morfolojik çözümleme.

Mevcut sözcük düzeyindeki analyze_all() çıktısını komşu sözcük
bağlamını kullanarak yeniden sıralar.  Mevcut API'ye dokunmaz.

Kullanım:
    from morphology.sentence import SentenceAnalyzer

    sa = SentenceAnalyzer(analyzer)
    tokens = sa.analyze("Yapay zekanın geleceğini tartıştılar.")
    for t in tokens:
        print(t.word, t.analysis.stem, t.context_applied)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .analyzer import MorphologicalAnalyzer, MorphemeAnalysis
from .phonology import turkish_lower


# ── Veri Modeli ──────────────────────────────────────────────────


@dataclass
class SentenceToken:
    """Cümle içindeki bir sözcüğün bağlamsal çözümleme sonucu."""

    word: str
    analysis: MorphemeAnalysis
    alternatives: list[MorphemeAnalysis] = field(default_factory=list)
    context_applied: list[str] = field(default_factory=list)


# ── Cümle Çözümleyici ───────────────────────────────────────────


class SentenceAnalyzer:
    """Cümle düzeyinde bağlamsal morfolojik çözümleme.

    Sözcük düzeyindeki analyze_all() çıktısını komşu sözcük
    bağlamıyla yeniden sıralar.  Kurallar:

      Sol-bağlam (left context):
        1. BELİRLEYİCİ → İSİM : Belirleyiciden sonra isimsel okuma tercih
        2. SIFAT → İSİM        : Sıfattan sonra isimsel okuma tercih
        3. ARDIŞIK_FİİL ↓      : Ardışık çekimli fiil nadir → isim tercih
        4. TAMLAYAN → İYELİK   : Tamlayan ekinden sonra iyelik eki bekle

      Sağ-bağlam (right context):
        A1. X + EDAT          : Edattan önce nominal okuma tercih
        A2. X + SORU_EDATI    : Soru edatından önce fiilsel okuma tercih

      Genişletilmiş sol-bağlam:
        B3. SAYI + X          : Sayıdan sonra isimsel okuma tercih
        B5. ZARF_FİİL + X    : Zarf-fiilden sonra fiilsel okuma tercih
        B6. İSİM_FİİL + X    : İsim-fiilden sonra isimsel okuma tercih
        B7. YÖNELME + X       : Yönelme ekinden sonra fiil beklentisi

      Deyimsel/kalıp:
        C8. "ele" + "al-"     : "ele almak" deyimi → el+YÖNELME
        C9. "göz" + "önün/ardı": göz deyimleri → nominal

      Morfo-sözdizimsel uyum:
        D10. BELIRTME + X     : Belirtme ekinden sonra fiil beklentisi
        D11. İYELİK zinciri   : Peş peşe iyelik → isim tamlaması

      Cümle pozisyonu:
        E12. Cümle sonu       : Fiilsel okuma tercih (SOV)
        E13. Cümle başı       : İsimsel/zarf hafif tercih
    """

    # ── Sözcük sınıfı kümeleri ───────────────────────────────────

    _DETERMINERS: frozenset[str] = frozenset({
        "bir", "bu", "şu", "o", "her", "bazı", "birçok",
        "tüm", "bütün", "hiçbir", "birkaç", "kimi", "öbür",
        "öteki", "hangi", "kaç",
    })

    _CONJUNCTIONS: frozenset[str] = frozenset({
        "ve", "ama", "fakat", "ancak", "veya", "ya",
        "ile", "ki", "de", "da",
    })

    _COMMON_ADJECTIVES: frozenset[str] = frozenset({
        "yeni", "eski", "büyük", "küçük", "güzel", "iyi", "kötü",
        "doğal", "yapay", "ilk", "son", "önemli", "farklı",
        "uzun", "kısa", "genç", "yaşlı", "ağır", "hafif",
    })

    # ── Dilbilgisel etiket kümeleri ──────────────────────────────

    _VERB_FINAL_LABELS: frozenset[str] = frozenset({
        "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
        "ŞİMDİKİ_ZAMAN", "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
        "DİLEK_ŞART",
        "KİŞİ_1T", "KİŞİ_2T", "KİŞİ_1Ç", "KİŞİ_2Ç", "KİŞİ_3Ç",
        "EMİR", "EMİR_3Ç",
        "BİLDİRME",
        "MASTAR",
    })

    _IYELIK_LABELS: frozenset[str] = frozenset({
        "İYELİK_1T", "İYELİK_2T", "İYELİK_3T",
        "İYELİK_1Ç", "İYELİK_2Ç", "İYELİK_3Ç",
    })

    # ── Sağ-bağlam ve genişletilmiş kümeleri ────────────────────

    _POSTPOSITIONS: frozenset[str] = frozenset({
        "için", "gibi", "kadar", "göre", "karşı", "rağmen", "dair",
        "üzere", "doğru", "dolayı", "itibaren", "beri", "hakkında",
        "ait", "ilişkin", "karşın", "boyunca", "önce", "sonra",
    })

    _QUESTION_PARTICLES: frozenset[str] = frozenset({
        "mi", "mı", "mu", "mü",
    })

    _NUMERALS: frozenset[str] = frozenset({
        "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz",
        "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş",
        "seksen", "doksan", "yüz", "bin", "milyon", "milyar",
    })

    _CONVERB_LABELS: frozenset[str] = frozenset({
        "ZARF_FİİL_-ArAk", "ZARF_FİİL_-IncA",
        "ZARF_FİİL_-Ip", "ZARF_FİİL_-ken",
    })

    _PARTICIPLE_LABELS: frozenset[str] = frozenset({
        "SIFAT_FİİL", "SIFAT_FİİL_-DIk", "SIFAT_FİİL_-DIğ",
    })

    # ── Kurucu ───────────────────────────────────────────────────

    def __init__(self, analyzer: MorphologicalAnalyzer) -> None:
        self._analyzer = analyzer

    # ── Ana API ──────────────────────────────────────────────────

    def analyze(
        self,
        text: str,
        max_per_word: int = 5,
    ) -> list[SentenceToken]:
        """Cümleyi tokenize edip bağlamsal çözümleme yapar.

        Adımlar:
          1. Tokenize
          2. Her sözcük için analyze_all() → aday listesi
          3. Sol bağlam kurallarıyla yeniden sıralama
          4. SentenceToken listesi döndür
        """
        words = self._tokenize(text)
        if not words:
            return []

        # Faz 1: Sözcük düzeyinde tüm çözümlemeler
        all_analyses: list[list[MorphemeAnalysis]] = [
            self._analyzer.analyze_all(w, max_results=max_per_word * 2)
            for w in words
        ]

        # Faz 2: Bağlamsal yeniden sıralama (soldan sağa tek geçiş)
        results: list[SentenceToken] = []
        for i, (word, analyses) in enumerate(zip(words, all_analyses)):
            if not analyses:
                results.append(SentenceToken(
                    word=word,
                    analysis=MorphemeAnalysis(stem=turkish_lower(word)),
                ))
                continue

            left = results[i - 1] if i > 0 else None
            reranked, rules = self._rerank(analyses, left, words, i)

            results.append(SentenceToken(
                word=word,
                analysis=reranked[0],
                alternatives=reranked[1:max_per_word],
                context_applied=rules,
            ))

        return results

    # ── Tokenizer ────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Basit tokenizer: sözcük ve apostrof koruyarak ayırır."""
        tokens = re.findall(r"[\w'\u2019]+", text, re.UNICODE)
        return [t for t in tokens if t.strip()]

    # ── Bağlamsal Yeniden Sıralama ───────────────────────────────

    def _rerank(
        self,
        analyses: list[MorphemeAnalysis],
        left: SentenceToken | None,
        words: list[str],
        idx: int,
    ) -> tuple[list[MorphemeAnalysis], list[str]]:
        """Bağlamsal kurallara göre çözümlemeleri yeniden sıralar.

        Her çözümlemeye bağlamsal skor eklenir. Orijinal sıra
        (morfolojik puanlama) korunur, bağlam yalnızca güçlü
        kanıtla sırayı değiştirir.
        """
        if len(analyses) <= 1:
            return analyses, []

        n = len(analyses)
        # Orijinal sıra bonusu: [1] = 0, [2] = -1, [3] = -2, ...
        scores = [-i * 1.0 for i in range(n)]
        applied: list[str] = []

        left_word = turkish_lower(words[idx - 1]) if idx > 0 else ""
        right_word = turkish_lower(words[idx + 1]) if idx + 1 < len(words) else ""
        cur_lower = turkish_lower(words[idx])

        # ── Kural 1: BELİRLEYİCİ + X → X isimsel ───────────────
        if left_word in self._DETERMINERS:
            for j, a in enumerate(analyses):
                if self._is_verb_final(a):
                    scores[j] -= 4.0
                else:
                    scores[j] += 3.0
            applied.append("BELİRLEYİCİ→İSİM")

        # ── Kural 2: SIFAT + X → X isimsel ──────────────────────
        # Bonus düşük: sıfat zarf işlevinde de olabilir (güzel yazar→
        # fiil okuma da geçerli). Yalnızca nazik tercih, güçlü baskı değil.
        elif left_word in self._COMMON_ADJECTIVES:
            for j, a in enumerate(analyses):
                if not self._is_verb_final(a):
                    scores[j] += 0.5
            applied.append("SIFAT→İSİM")

        # ── Kural 2 (genişletilmiş): Sol sözcüğün analizi sıfatsal
        elif left and self._is_adjectival(left.analysis):
            for j, a in enumerate(analyses):
                if not self._is_verb_final(a):
                    scores[j] += 0.3
            applied.append("SIFATSAL→İSİM")

        # ── Kural 3: Ardışık çekimli fiil nadir ─────────────────
        if left and self._is_verb_final(left.analysis):
            is_after_conj = (
                idx >= 2
                and turkish_lower(words[idx - 1]) in self._CONJUNCTIONS
            )
            if not is_after_conj:
                for j, a in enumerate(analyses):
                    if self._is_verb_final(a):
                        scores[j] -= 3.0
                applied.append("ARDIŞIK_FİİL↓")

        # ── Kural 4: TAMLAYAN + İYELİK uyumu ────────────────────
        if left and self._has_tamlayan(left.analysis):
            for j, a in enumerate(analyses):
                if self._has_iyelik(a):
                    scores[j] += 3.0
            applied.append("TAMLAYAN→İYELİK")

        # ══════════════════════════════════════════════════════════
        # Sağ-bağlam kuralları (right context)
        # ══════════════════════════════════════════════════════════

        # ── Kural A1: X + EDAT → X isimsel ──────────────────────
        # Edattan önce isimsel okuma tercih edilir.
        # Türkçe'de edatlar isim gruplarını yönetir:
        # "ev için", "okul gibi", "akşama kadar"
        if right_word in self._POSTPOSITIONS:
            for j, a in enumerate(analyses):
                if self._is_verb_final(a):
                    scores[j] -= 1.5
                else:
                    scores[j] += 2.0
            applied.append("X→EDAT:İSİM")

        # ── Kural A2: X + SORU_EDATI → X fiilsel ────────────────
        # Soru edatı ("mi/mı/mu/mü") öncesinde çekimli fiil beklenir:
        # "gelecek mi?", "biliyor musun?"
        if right_word in self._QUESTION_PARTICLES:
            for j, a in enumerate(analyses):
                if self._is_verb_final(a):
                    scores[j] += 2.0
            applied.append("X→SORU:FİİL")

        # ══════════════════════════════════════════════════════════
        # Genişletilmiş sol-bağlam kuralları
        # ══════════════════════════════════════════════════════════

        # ── Kural B3: SAYI + X → X isimsel ──────────────────────
        # Sayılardan sonra isimsel okuma tercih edilir.
        # "beş kişi", "on yıl", "iki gün"
        # ("bir" zaten BELİRLEYİCİ kümesinde — burada tekrar yok)
        if left_word in self._NUMERALS:
            for j, a in enumerate(analyses):
                if self._is_verb_final(a):
                    scores[j] -= 1.5
                else:
                    scores[j] += 2.0
            applied.append("SAYI→İSİM")

        # ── Kural B5: ZARF_FİİL + X → X fiilsel ────────────────
        # Zarf-fiilden (-arak, -ıp, -ınca, -ken) sonra ana fiil
        # veya yardımcı fiil beklenir:
        # "koşarak geldi", "alıp verdi", "görünce anladım"
        if left and self._has_converb(left.analysis):
            for j, a in enumerate(analyses):
                if self._is_verb_final(a):
                    scores[j] += 2.0
            applied.append("ZARF_FİİL→FİİL")

        # ── Kural B6: İSİM_FİİL + X → X isimsel ────────────────
        # İsim-fiilden (-ma/-me) sonra isim beklenir (isim tamlaması):
        # "okuma parçası", "yazma becerisi", "dil işleme alanında"
        if left and self._has_isim_fiil(left.analysis):
            for j, a in enumerate(analyses):
                if not self._is_verb_final(a):
                    scores[j] += 1.5
            applied.append("İSİM_FİİL→İSİM")

        # ── Kural B7: YÖNELME + X → X fiilsel ───────────────────
        # Yönelme (dative) ekinden sonra fiil beklenir (SOV düzeni):
        # "eve gitti", "okula başladı", "ona baktı"
        # Nazik tercih: yönelme her zaman hemen fiilden önce olmayabilir.
        if left and self._has_yonelme(left.analysis):
            for j, a in enumerate(analyses):
                if self._is_verb_final(a):
                    scores[j] += 1.0
            applied.append("YÖNELME→FİİL")

        # ══════════════════════════════════════════════════════════
        # Deyimsel / kalıp kuralları
        # ══════════════════════════════════════════════════════════

        # ── Kural C8: "ele" + "al-" deyimi ──────────────────────
        # "ele almak" kalıbında "ele" → "el+e" (yönelme) olmalı.
        if cur_lower == "ele" and right_word.startswith("al"):
            for j, a in enumerate(analyses):
                if a.stem == "el" and self._has_suffix_label(a, "YÖNELME"):
                    scores[j] += 3.0
            applied.append("DEYİM:ele_al")

        # ── Kural C9: "göz" + "önün/ardı" deyimi ────────────────
        # "göz önünde bulundurmak", "gözardı etmek" kalıplarında
        # "göz" isimsel okunmalı.
        if cur_lower == "göz" and (
            right_word.startswith("önün") or right_word.startswith("ardı")
        ):
            for j, a in enumerate(analyses):
                if not self._is_verb_final(a):
                    scores[j] += 2.0
            applied.append("DEYİM:göz_kalıp")

        # ══════════════════════════════════════════════════════════
        # Morfo-sözdizimsel uyum kuralları
        # ══════════════════════════════════════════════════════════

        # ── Kural D10: BELIRTME + X → X fiilsel ─────────────────
        # Belirtme (accusative) ekinden sonra geçişli fiil beklenir:
        # "kitabı okudu", "evi gördüm"
        if left and self._has_belirtme(left.analysis):
            for j, a in enumerate(analyses):
                if self._is_verb_final(a):
                    scores[j] += 1.5
            applied.append("BELİRTME→FİİL")

        # ── Kural D11: İYELİK zinciri ───────────────────────────
        # Peş peşe iyelik → isim tamlaması zinciri devam beklentisi.
        # "okulun müdürünün arabası" — iyelikten sonra tamlayan/iyelik
        if left and self._has_iyelik(left.analysis):
            for j, a in enumerate(analyses):
                if self._has_iyelik(a) or self._has_tamlayan(a):
                    scores[j] += 1.0
            applied.append("İYELİK_ZİNCİR")

        # ══════════════════════════════════════════════════════════
        # Cümle pozisyonu kuralları
        # ══════════════════════════════════════════════════════════

        # ── Kural E12: Cümle sonu → fiilsel okuma ───────────────
        # Türkçe SOV dil: cümle sonunda yüklem (fiil) beklenir.
        # Sadece dizinin son sözcüğü için uygulanır (basitleştirilmiş).
        if idx == len(words) - 1:
            for j, a in enumerate(analyses):
                if self._is_verb_final(a):
                    scores[j] += 1.0
            applied.append("CÜMLE_SONU→FİİL")

        # ── Kural E13: Cümle başı → isimsel hafif tercih ────────
        # Türkçe'de cümle başında genellikle özne (isim) veya zarf
        # bulunur. Çok hafif tercih — güçlü baskı değil.
        if idx == 0:
            for j, a in enumerate(analyses):
                if not self._is_verb_final(a):
                    scores[j] += 0.5
            applied.append("CÜMLE_BAŞI→İSİM")

        if not applied:
            return analyses, []

        # Skor sıralaması
        indexed = sorted(
            range(n), key=lambda j: scores[j], reverse=True,
        )
        reranked = [analyses[j] for j in indexed]
        return reranked, applied

    # ── Etiket Sorguları ─────────────────────────────────────────

    def _is_verb_final(self, a: MorphemeAnalysis) -> bool:
        """Çözümleme çekimli fiil-sonuyla mı bitiyor?"""
        if not a.suffixes:
            return False
        for _, label in a.suffixes[-2:]:
            for sub in label.split("/"):
                if sub in self._VERB_FINAL_LABELS:
                    return True
        return False

    def _is_adjectival(self, a: MorphemeAnalysis) -> bool:
        """Çözümleme sıfatsal mı? (son ek SIFAT_FİİL veya ek yok)"""
        if not a.suffixes:
            return False
        last = a.suffixes[-1][1]
        return "SIFAT_FİİL" in last

    def _has_tamlayan(self, a: MorphemeAnalysis) -> bool:
        """Son ek TAMLAYAN içeriyor mu?"""
        if not a.suffixes:
            return False
        return "TAMLAYAN" in a.suffixes[-1][1]

    def _has_iyelik(self, a: MorphemeAnalysis) -> bool:
        """Çözümlemede iyelik eki var mı?"""
        for _, label in a.suffixes:
            for sub in label.split("/"):
                if sub in self._IYELIK_LABELS:
                    return True
        return False

    # ── Yeni Etiket Sorguları (genişletilmiş kurallar için) ──────

    def _has_any_label(
        self, a: MorphemeAnalysis, labels: frozenset[str],
    ) -> bool:
        """Çözümlemedeki herhangi bir ek etiketi verilen kümede mi?"""
        for _, label in a.suffixes:
            for sub in label.split("/"):
                if sub in labels:
                    return True
        return False

    def _has_suffix_label(self, a: MorphemeAnalysis, target: str) -> bool:
        """Çözümlemede belirli bir ek etiketi var mı?"""
        for _, label in a.suffixes:
            for sub in label.split("/"):
                if sub == target:
                    return True
        return False

    def _has_converb(self, a: MorphemeAnalysis) -> bool:
        """Çözümlemede zarf-fiil eki var mı? (-arak, -ıp, -ınca, -ken)"""
        return self._has_any_label(a, self._CONVERB_LABELS)

    def _has_isim_fiil(self, a: MorphemeAnalysis) -> bool:
        """Çözümlemede isim-fiil eki var mı? (-ma/-me)"""
        return self._has_suffix_label(a, "İSİM_FİİL")

    def _has_yonelme(self, a: MorphemeAnalysis) -> bool:
        """Çözümlemede yönelme (dative) eki var mı?"""
        return self._has_suffix_label(a, "YÖNELME")

    def _has_belirtme(self, a: MorphemeAnalysis) -> bool:
        """Çözümlemede belirtme (accusative) eki var mı?"""
        return self._has_suffix_label(a, "BELIRTME")
