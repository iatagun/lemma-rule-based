"""
Morfolojik çözümleme motoru.

SOLID:
  SRP – Yalnızca çözümleme algoritmasından sorumlu.
  OCP – Yeni uyum stratejileri veya kök doğrulayıcıları eklenebilir.
  DIP – HarmonyChecker protokolüne bağımlı, somut sınıflara değil.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .harmony import HarmonyChecker, StrictHarmonyChecker, check_vowel_harmony
from .phonology import (
    CLOSE_VOWELS,
    CONSONANTS,
    SONORANT_SIBILANT,
    VOWELS,
    get_vowels,
    syllabify,
    turkish_lower,
)
from .morphotactics import MorphotacticFSM
from .suffix import SuffixForm, SuffixRegistry

if TYPE_CHECKING:
    from .dictionary import TurkishDictionary


# ── Veri Modeli ────────────────────────────────────────────────


@dataclass(frozen=True)
class MorphemeAnalysis:
    """Bir sözcüğün morfolojik çözümleme sonucu."""

    stem: str
    suffixes: list[tuple[str, str]] = field(default_factory=list)
    root: str | None = None   # Temel kök (yaz←yazdır, et←edil, kitap←kitab)
    lemma: str | None = None  # Gövde/lemma (yazdır, edil, gelmek)

    @property
    def parts(self) -> list[str]:
        return [self.stem] + [s[0] for s in self.suffixes]


# ── Kök Doğrulayıcı ──────────────────────────────────────────


class StemValidator:
    """
    Kök adayının geçerli bir Türkçe kök gibi görünüp görünmediğini kontrol eder.
    - En az 1 ünlü içermeli
    - Sondaki ünsüz kümesi geçerli olmalı
    """

    def is_valid(self, stem: str) -> bool:
        if not get_vowels(stem):
            return False

        trailing = 0
        for ch in reversed(stem):
            if ch in CONSONANTS:
                trailing += 1
            else:
                break

        if trailing >= 3:
            return False

        if trailing == 2:
            first_of_cluster = stem[-trailing]
            if first_of_cluster not in SONORANT_SIBILANT:
                return False

        return True


class DictionaryStemValidator:
    """
    Sözlük destekli kök doğrulayıcı.

    Hem sezgisel kontrolü (StemValidator) hem de sözlük eşleşmesini
    gerektirir. Morfofonemik dönüşümleri (ünsüz yumuşaması, ünlü
    düşmesi, fiil kökü) de dikkate alır.
    """

    def __init__(self, dictionary: TurkishDictionary) -> None:
        self._dictionary = dictionary
        self._heuristic = StemValidator()

    def is_valid(self, stem: str) -> bool:
        if not self._heuristic.is_valid(stem):
            return False
        return self._dictionary.find_root(stem) is not None


# ── Uyum Stratejisi ──────────────────────────────────────────


@dataclass(frozen=True)
class HarmonyStrategy:
    """
    Uyum stratejisi: bir HarmonyChecker ile opsiyonel minimum kök
    uzunluğu eşiğini ve opsiyonel kök doğrulayıcı aşırı yüklemeyi
    birleştirir.
    """

    checker: HarmonyChecker
    min_stem_override: int | None = None
    stem_validator: StemValidator | DictionaryStemValidator | None = None


# ── Ana Çözümleyici ──────────────────────────────────────────


class MorphologicalAnalyzer:
    """
    Türkçe sözcükleri kök ve eklerine ayıran kural-tabanlı çözümleyici.

    Strateji:
      1. Sağdan sola, en uzun eşleşen eki bul (tam uyum)
      2. Bulunamazsa, sıradaki stratejiyle tekrar dene
      3. Kök adayının geçerliliğini kontrol et
      4. Ek geçerliyse soy, kalan kökle tekrarla

    Sözlük varsa:
      – Sözlük destekli stratejiler önce denenir (tercih)
      – Sözlükte bilinen bir sözcüğü bilinmeyen parçalara ayırma engellenir
      – Analiz sonunda kök çözümlemesi yapılır (kitab → kitap)
    """

    MAX_ITERATIONS = 10

    # Düzensiz zamir çekimleri: kural tabanlı analiz bunları doğru
    # çözemez (tampon ünsüz belirsizliği, tümleyen değişimi vb.)
    # Anahtar: yüzey biçimi (küçük harf), değer: kök/lemma.
    _IRREGULAR_PRONOUNS: dict[str, str] = {
        # ben (I) – tümleyen değişimi (suppletion)
        "bana": "ben", "beni": "ben", "bende": "ben",
        "bendeki": "ben",
        # o (he/she/it) – tampon-n, "on" (10) ile karışır
        "onu": "o", "ona": "o", "onda": "o", "ondan": "o",
        "onun": "o", "onunla": "o",
        "onlar": "o", "onları": "o", "onlara": "o", "onlarda": "o",
        "onlardan": "o", "onların": "o", "onlarla": "o",
        # bu (this) – tampon-n
        "bunu": "bu", "buna": "bu", "bunda": "bu", "bundan": "bu",
        "bunun": "bu", "bununla": "bu", "bunla": "bu",
        "bunlar": "bu", "bunları": "bu", "bunlara": "bu",
        "bunlarda": "bu", "bunlardan": "bu", "bunların": "bu",
        # şu (that) – tampon-n
        "şunu": "şu", "şuna": "şu", "şunda": "şu", "şundan": "şu",
        "şunun": "şu", "şunlar": "şu", "şunları": "şu",
        # biz (we) / siz (you pl.) – ayrıştırılamaz çekim
        "bize": "biz", "size": "siz", "sizce": "siz", "sizden": "siz",
        # ora / bura – gösterme yer zamirleri
        "burada": "bura", "buradan": "bura", "buraya": "bura",
        "buradaki": "bura",
        "orada": "ora", "oradan": "ora", "oraya": "ora",
        "oradaki": "ora",
        # kendi (self)
        "kendimi": "kendi", "kendini": "kendi", "kendisi": "kendi",
        "kendimize": "kendi", "kendinize": "kendi",
        "kendilerine": "kendi", "kendinden": "kendi",
        "kendisine": "kendi", "kendisini": "kendi",
        "kendisiyle": "kendi", "kendinizi": "kendi",
        "kendimce": "kendi",
        # hep (all)
        "hepsi": "hep", "hepimiz": "hep", "hepsini": "hep",
        # birbiri
        "birbirine": "birbiri", "birbirini": "birbiri",
        "birbirimize": "birbiri", "birbirlerinden": "birbiri",
        # herkes
        "herkesin": "herkes", "herkese": "herkes", "herkesi": "herkes",
        # biri
        "birisi": "biri", "birileri": "biri",
    }

    # AUX (yardımcı fiil / klitik) eşleşme tablosu.
    # BOUN Treebank'te çok sözcüklü tokenlardan ayrılan klitik biçimler
    # (güzeldir → güzel + dir) bağımsız token olarak gelir.
    # POS bilgisi AUX olduğunda doğrudan eşleştirilir.
    # Eşleşme bulunamazsa normal analiz yapılır (olan→ol gibi).
    _AUX_COPULA_TABLE: dict[str, str] = {
        # ── imek (i-) bildirme: -DIr ──
        "dir": "i", "dır": "i", "tir": "i", "tır": "i",
        "lardır": "i",
        # ── imek (i-) koşul: -IsA ──
        "ise": "i", "sa": "i", "se": "i",
        # ── imek (i-) geçmiş (idi): -ydI ──
        "ydı": "i", "ydü": "i",
        "ydik": "i", "ydim": "i",
        "idi": "i",
        # ── imek (i-) rivayet (imiş) ──
        "mış": "i", "mışsın": "i",
        # ── imek (i-) zarf (-yken) ──
        "ken": "i",
        # ── imek (i-) kişi ──
        "yim": "i",
        # ── imek (i-) belirsiz: dı → majority y, ama i de var ──
        "dik": "i",
        # ── y- (görülen geçmiş zaman yardımcısı): -DI ──
        "di": "y", "dı": "y", "du": "y", "dü": "y",
        "ti": "y", "tı": "y", "tu": "y", "tü": "y",
        "dum": "y", "dım": "y", "dim": "y", "düm": "y",
        "tum": "y", "tım": "y", "tim": "y", "tüm": "y",
        "dun": "y", "dın": "y", "din": "y", "dün": "y",
        "tun": "y", "tın": "y", "tin": "y", "tün": "y",
        "duk": "y", "dık": "y",
        "tuk": "y", "tık": "y",
        "sin": "y", "diniz": "y",
        "dur": "y", "tur": "y", "dür": "y", "tür": "y",
        # ── y- zaman (rivayet) ──
        "ydi": "y", "ydu": "y", "ydum": "y",
        "miş": "y", "muş": "y", "ymış": "y",
        "iken": "y",
        # ── mi (soru eki) ──
        "mı": "mi", "mi": "mi", "mu": "mi", "mü": "mi",
        "mısın": "mi", "misin": "mi", "musun": "mi",
        "misiniz": "mi", "musunuz": "mi",
        "mıydı": "mi",
        # ── değil (olumsuz) ──
        "değil": "değil", "değildir": "değil",
        "değildi": "değil", "değildik": "değil",
        "değilim": "değil", "değiliz": "değil",
        "değilmiş": "değil", "değilse": "değil",
        "değilsin": "değil", "değilsiniz": "değil",
        "değilken": "değil",
    }

    # Sıra sayıları (ordinal numbers).
    # Sözlükte bağımsız girdi olarak bulunduklarından dict-protection
    # ayrıştırmayı engeller.  POS=NUM ile tetiklenen ön-eşleme tablosu.
    _ORDINAL_NUMBERS: dict[str, str] = {
        "birinci": "bir", "ikinci": "iki", "üçüncü": "üç",
        "dördüncü": "dört", "beşinci": "beş", "altıncı": "altı",
        "yedinci": "yedi", "sekizinci": "sekiz", "dokuzuncu": "dokuz",
        "onuncu": "on", "yirminci": "yirmi", "otuzuncu": "otuz",
        "kırkıncı": "kırk", "ellinci": "elli", "altmışıncı": "altmış",
        "yetmişinci": "yetmiş", "sekseninci": "seksen",
        "doksanıncı": "doksan", "yüzüncü": "yüz", "bininci": "bin",
        "milyonuncu": "milyon", "sonuncu": "son",
    }

    # İlgeç-adlar (postpositional nouns).
    # Türkçe'de mekânsal/ilişkisel adlar (iç, üst, alt, yan, yer, …)
    # çekimli biçimleriyle TDK sözlüğünde ayrı madde başı olarak yer alır
    # (içinde, üzerinde, yerine …).  Dict-protection bu sözcüklerin
    # ayrıştırılmasını engeller.  POS=NOUN / NUM olduğunda tablo eşleşmesi
    # yapılarak doğru lemma döndürülür.
    _POSTPOSITIONAL_NOUNS: dict[str, str] = {
        # iç (inside)
        "içinde": "iç", "içine": "iç", "içinden": "iç",
        "içini": "iç", "içinin": "iç", "içindeki": "iç",
        # üzer (upon/over)
        "üzerinde": "üzer", "üzerine": "üzer", "üzerinden": "üzer",
        "üzerindeki": "üzer",
        # yer (place) – yalnızca ilgeç kullanımı
        "yerine": "yer", "yerinde": "yer", "yerinden": "yer",
        "yerindeki": "yer",
        # yan (side)
        "yanında": "yan", "yanına": "yan", "yanından": "yan",
        "yanındaki": "yan",
        # alt (bottom)
        "altında": "alt", "altına": "alt", "altından": "alt",
        "altındaki": "alt",
        # üst (top)
        "üstünde": "üst", "üstüne": "üst", "üstünden": "üst",
        "üstündeki": "üst",
        # baş (head/beginning)
        "başında": "baş", "başına": "baş", "başından": "baş",
        "başındaki": "baş",
        # hak (right/claim) – geminasyon: hak→hakk
        "hakkında": "hak", "hakkını": "hak", "hakkındaki": "hak",
        "hakkının": "hak", "hakkına": "hak",
        # yüz (face/surface/hundred)
        "yüzde": "yüz", "yüzden": "yüz", "yüzünden": "yüz",
        "yüzüne": "yüz",
        # ön (front)
        "önünde": "ön", "önüne": "ön", "önünden": "ön",
        "önündeki": "ön",
        # arka (back)
        "arkasında": "arka", "arkasına": "arka", "arkasından": "arka",
        "arkasındaki": "arka",
        # ara (between/gap)
        "arasında": "ara", "arasına": "ara", "arasından": "ara",
        "arasındaki": "ara",
    }

    # Fiil ekleri: sözlük koruması altındaki sözcüklerin ayrıştırılmasına
    # yalnızca fiil kökü + fiil eki kombinasyonunda izin verilir.
    # Bu, "için → iç+in(TAMLAYAN)" gibi yanlış ayrışmaları engeller
    # ama "olur → ol+ur(GENİŞ_ZAMAN)" gibi doğru ayrışmalara izin verir.
    _VERBAL_SUFFIX_LABELS: frozenset[str] = frozenset({
        "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
        "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
        "ŞİMDİKİ_ZAMAN",
        "SIFAT_FİİL", "SIFAT_FİİL_-DIk", "SIFAT_FİİL_-DIğ",
        "ZARF_FİİL_-ArAk", "ZARF_FİİL_-IncA", "ZARF_FİİL_-Ip",
        "ZARF_FİİL_-ken",
        "EDİLGEN", "İŞTEŞ", "ETTİRGEN_-lAt",
        "YETERLİLİK", "OLUMSUZ/İSİM_FİİL",
        "DİLEK_ŞART", "BİLDİRME/ETTİRGEN",
        "EMİR/KİŞİ_2T", "EMİR_3Ç", "KİŞİ_2Ç", "KİŞİ_1Ç",
        "MASTAR",
    })

    # ── Ek Hiyerarşisi (Slot-tabanlı sıralama) ─────────────────
    # Türkçe'de ekler kesin bir sıra izler:
    #   İSİM: KÖK → Yapım → Çoğul → İyelik → Hal → -ki
    #   FİİL: KÖK → Çatı → Olumsuz → Yeterlilik → Zaman → Kişi → Bildirme
    # Nominalizasyon (sıfat/isim fiil) sonrası isim slotları yeniden başlar.
    #
    # _FORBIDDEN_SUFFIX_BIGRAMS: (iç_ek, dış_ek) çiftleri.
    # Kök→dış sırasında bu geçişler dilbilgisel olarak imkânsızdır
    # ve benchmark verisinde doğru çözümlemelerde HİÇ görülmez.
    # Ek sıyırma sırasında bu çiftler engellendiğinde analizör
    # daha kısa/doğru bir ek eşleşmesi arar.
    _FORBIDDEN_SUFFIX_BIGRAMS: frozenset[tuple[str, str]] = frozenset({
        # SIFAT_FİİL (-An) + kişi eki: ad-fiil sonrası kişi gelmez
        ("SIFAT_FİİL", "KİŞİ_1Ç"),
        ("SIFAT_FİİL", "KİŞİ_2Ç"),
        # SIFAT_FİİL (-An) + iyelik/belirtme: bakan→bak+an+ı yanlış
        ("SIFAT_FİİL", "İYELİK_3T/BELIRTME"),
        ("SIFAT_FİİL", "BULUNMA"),
        # GENİŞ_ZAMAN + hal ekleri: doğrudan zaman→hal dizilimi imkânsız
        ("GENİŞ_ZAMAN", "YÖNELME"),
        ("GENİŞ_ZAMAN", "AYRILMA"),
        ("GENİŞ_ZAMAN", "İYELİK_3Ç"),
        ("GENİŞ_ZAMAN", "İYELİK_3T/BELIRTME"),
        ("GENİŞ_ZAMAN", "İYELİK_1Ç"),
        # GEÇMİŞ_ZAMAN + hal ekleri
        ("GEÇMİŞ_ZAMAN", "AYRILMA"),
        ("GEÇMİŞ_ZAMAN", "YÖNELME"),
        # Kişi + iyelik/tamlayan dizilimi
        ("KİŞİ_1Ç", "İYELİK_2T/TAMLAYAN"),
        # EDİLGEN + iyelik çoğul: fiil çatısı sonrası doğrudan İYELİK_3Ç gelmez
        ("EDİLGEN", "İYELİK_3Ç"),
        ("EDİLGEN", "KİŞİ_1Ç"),
        ("EDİLGEN", "ÇOĞUL"),
        # YAPIM_-lI + zaman eki: yapım sonrası doğrudan zaman gelmez
        ("YAPIM_-lI", "ŞİMDİKİ_ZAMAN"),
        # YAPIM_-CI + hal ekleri
        ("YAPIM_-CI", "AYRILMA"),
        # YÖNELME + yapım eki: hal sonrası yapım gelmez
        ("YÖNELME", "YAPIM_-lI"),
        # ZARF_FİİL_-IncA + çoğul: zarf fiil sonrası çoğul gelmez
        ("ZARF_FİİL_-IncA", "ÇOĞUL"),
    })

    # Düzensiz fiil gövdeleri: demek / yemek.
    # Türkçe'nin yalnızca 2 düzensiz fiili var.  Kök değişimleri:
    #   demek: de-, di-, diy-, ded-, den-, deni-, denil-, dem-
    #   yemek: ye-, yi-, yiy-, yed-, yen-, yem-
    # Ek sıyırma sonrası kalan gövde bu tablodaysa lemma düzeltilir.
    _IRREGULAR_VERB_STEMS: dict[str, str] = {
        # demek — görülen geçmiş
        "ded": "de",
        # demek — şimdiki zaman / gelecek / istek
        "di": "de", "diy": "de", "diye": "de", "diyel": "de",
        # demek — geniş zaman
        "der": "de",
        # demek — edilgen
        "den": "de", "deni": "de", "denil": "de",
        # demek — olumsuz yeterlilik
        "demem": "de",
        # yemek — şimdiki zaman / sıfat fiil
        "yi": "ye", "yiy": "ye", "yiye": "ye", "yiyel": "ye",
        # yemek — olumsuz yeterlilik
        "yemem": "ye", "yememem": "ye", "yemeyel": "ye",
        # etmek — ünsüz yumuşaması: t→d
        "ed": "et", "eder": "et", "edi": "et", "edil": "et",
        "eden": "et", "edebil": "et",
        # gitmek — ünsüz yumuşaması: t→d
        "gid": "git", "gider": "git", "gidi": "git", "gidil": "git",
    }

    def __init__(
        self,
        registry: SuffixRegistry,
        strategies: list[HarmonyStrategy],
        validator: StemValidator | None = None,
        dictionary: TurkishDictionary | None = None,
    ) -> None:
        self._registry = registry
        self._strategies = strategies
        self._validator = validator or StemValidator()
        self._dictionary = dictionary
        self._fsm = MorphotacticFSM()

    def analyze(self, word: str, upos: str | None = None) -> MorphemeAnalysis:
        """Sözcüğü morfolojik bileşenlerine ayırır.

        Dahili olarak ``analyze_all()`` çağrılır ve BFS + sıralama
        sonucu en yüksek puanlı çözümleme döndürülür.

        Args:
            word: Çözümlenecek sözcük.
            upos: İsteğe bağlı Universal POS etiketi. AUX verildiğinde
                  klitik kopula tablosu öncelikli olarak kullanılır.
        """
        results = self.analyze_all(word, upos=upos, max_results=5)
        if results:
            return results[0]
        return MorphemeAnalysis(stem=turkish_lower(word.strip()))

    def _strip_suffixes(
        self, word: str,
    ) -> tuple[str, list[tuple[str, str]]]:
        """Sözcükten ekleri soyarak kök ve ek listesi döndürür.

        Ek hiyerarşisi kontrolü: her yeni ek, bir önceki (daha dıştaki) ek
        ile yasaklı çift oluşturuyorsa atlanır.  Bu, ``bakan→bak+an+ı``
        veya ``sinir→sin+ir+den`` gibi yanlış-pozitif zincirleri engeller.
        """
        found_suffixes: list[tuple[str, str]] = []
        current = word

        for _ in range(self.MAX_ITERATIONS):
            # Son bulunan (dıştaki) ekin etiketine göre yasaklı iç ekleri belirle
            forbidden: frozenset[str] = frozenset()
            if found_suffixes:
                outer_label = found_suffixes[0][1]
                forbidden = frozenset(
                    inner
                    for inner, outer in self._FORBIDDEN_SUFFIX_BIGRAMS
                    if outer == outer_label
                )

            result = self._find_suffix_match(current, forbidden)
            if result is None:
                break

            stem_candidate, sfx_form, sfx_label = result
            found_suffixes.insert(0, (sfx_form, sfx_label))
            current = stem_candidate

        return current, found_suffixes

    # ── Çoklu Çözümleme (Belirsizlik Desteği) ────────────────

    def analyze_all(
        self,
        word: str,
        upos: str | None = None,
        max_results: int = 5,
    ) -> list[MorphemeAnalysis]:
        """Sözcüğün tüm olası morfolojik çözümlemelerini döndürür.

        Türkçe'de birçok sözcük bağlam olmadan birden fazla geçerli
        çözümlemeye sahiptir.  Örneğin ``gelirin`` hem *gelir+in*
        (İYELİK_2T) hem *gel+ir+in* (GENİŞ_ZAMAN+İYELİK_2T) olarak
        ayrıştırılabilir.  Doğru çözümleme cümledeki bağlamdan (POS,
        sözdizimsel konum vb.) belirlenir.

        Args:
            word: Çözümlenecek sözcük.
            upos: İsteğe bağlı POS etiketi (verilirse filtreleme yapılır).
            max_results: En fazla döndürülecek çözümleme sayısı.

        Returns:
            Güvenilirlik sırasına göre sıralı ``MorphemeAnalysis`` listesi.
        """
        original = word.strip()
        word = turkish_lower(original)

        # ── Tek-çözümlemeli özel durumlar ──────────────────────
        if upos == "AUX" and word in self._AUX_COPULA_TABLE:
            return [MorphemeAnalysis(
                stem=word, root=self._AUX_COPULA_TABLE[word],
            )]

        if upos == "VERB" and word in self._AUX_COPULA_TABLE:
            lemma = self._AUX_COPULA_TABLE[word]
            if lemma == "değil":
                return [MorphemeAnalysis(stem=word, root=lemma)]

        if upos in ("CCONJ", "INTJ"):
            return [MorphemeAnalysis(stem=word)]

        if upos in ("NUM", "ADJ") and word in self._ORDINAL_NUMBERS:
            return [MorphemeAnalysis(
                stem=word, root=self._ORDINAL_NUMBERS[word],
            )]

        if upos in ("NOUN", "NUM") and word in self._POSTPOSITIONAL_NOUNS:
            return [MorphemeAnalysis(
                stem=word, root=self._POSTPOSITIONAL_NOUNS[word],
            )]

        if word in self._IRREGULAR_PRONOUNS:
            return [MorphemeAnalysis(
                stem=word, root=self._IRREGULAR_PRONOUNS[word],
            )]

        # ── Apostrof işleme ────────────────────────────────────
        for apos in ("\u2019", "'"):
            if apos in original:
                raw_base, _, raw_tail = original.partition(apos)
                base = turkish_lower(raw_base)
                tail = turkish_lower(raw_tail)
                if base and tail:
                    is_proper = raw_base[0].isupper() if raw_base else False
                    if is_proper:
                        tail_sfxs: list[tuple[str, str]] = []
                        remaining = tail
                        for sfx in self._registry.suffixes:
                            if remaining == sfx.form:
                                tail_sfxs.append((sfx.form, sfx.label))
                                remaining = ""
                                break
                        if remaining:
                            decomps = self._strip_suffixes_all(
                                base + tail, max_results, upos=upos,
                            )
                        else:
                            decomps = [(base, tail_sfxs)]
                    else:
                        decomps = self._strip_suffixes_all(
                            base + tail, max_results, upos=upos,
                        )

                    return self._build_multi_results(
                        decomps, upos, max_results,
                    ) or [MorphemeAnalysis(stem=base)]

                word = word.replace(apos, "")
                break

        # ── Çok-yollu ek sıyırma ──────────────────────────────
        decompositions = self._strip_suffixes_all(word, max_results, upos=upos)

        return self._build_multi_results(
            decompositions, upos, max_results,
        ) or [MorphemeAnalysis(stem=word)]

    # ── Çoklu-yol yardımcıları ────────────────────────────────

    def _build_multi_results(
        self,
        decompositions: list[tuple[str, list[tuple[str, str]]]],
        upos: str | None,
        max_results: int,
    ) -> list[MorphemeAnalysis]:
        """Ayrıştırma yollarını MorphemeAnalysis listesine dönüştürür."""
        results: list[MorphemeAnalysis] = []
        seen: set[tuple[str | None, str | None, tuple[str, ...]]] = set()

        for stem, sfxs in decompositions:
            stem, sfxs = self._handle_yor_connector(stem, list(sfxs))
            sfxs = self._disambiguate_suffixes(sfxs)
            root, lemma = self._resolve_root(stem, sfxs, upos=upos)

            if upos == "VERB":
                key_root = root if root else stem
                if key_root in self._IRREGULAR_VERB_STEMS:
                    root = self._IRREGULAR_VERB_STEMS[key_root]

            # Aynı (kök, lemma, etiket-dizisi) → tekrarsız
            label_seq = tuple(s[1] for s in sfxs)
            key = (root or stem, lemma, label_seq)
            if key in seen:
                continue
            seen.add(key)

            results.append(MorphemeAnalysis(
                stem=stem, suffixes=sfxs, root=root, lemma=lemma,
            ))

            if len(results) >= max_results:
                break

        return results

    def _strip_suffixes_all(
        self,
        word: str,
        max_results: int = 5,
        upos: str | None = None,
    ) -> list[tuple[str, list[tuple[str, str]]]]:
        """Tüm geçerli ayrıştırma yollarını keşfeder (BFS).

        Her adımda tüm olası ek eşleşmelerini toplar ve dallanır.
        Morfotaktik FSM ile ek sıralama kısıtları uygulanır.
        Sonuçlar kalite puanına göre sıralanır.
        """
        from collections import deque

        _MAX_QUEUE = 200

        fsm = self._fsm
        init_state = fsm.initial_state()

        # BFS durumu: (kök-adayı, ekler[iç→dış], derinlik, fsm_states)
        QueueItem = tuple[str, list[tuple[str, str]], int, list[str]]
        queue: deque[QueueItem] = deque()
        queue.append((word, [], 0, [init_state]))

        raw: list[tuple[str, list[tuple[str, str]]]] = []
        seen_states: set[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]]] = set()
        seen_results: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

        while queue and len(raw) < max_results * 5:
            if len(queue) > _MAX_QUEUE:
                break

            current, found_sfxs, depth, fsm_states = queue.popleft()

            state_key = (current, tuple(found_sfxs), tuple(sorted(set(fsm_states))))
            if state_key in seen_states:
                continue
            seen_states.add(state_key)

            if depth >= self.MAX_ITERATIONS:
                rk = (current, tuple(found_sfxs))
                if rk not in seen_results:
                    seen_results.add(rk)
                    raw.append((current, list(found_sfxs)))
                continue

            # Yasaklı iç-ek etiketleri (forbidden bigrams korunuyor)
            forbidden: frozenset[str] = frozenset()
            if found_sfxs:
                outer_label = found_sfxs[0][1]
                forbidden = frozenset(
                    inner
                    for inner, outer in self._FORBIDDEN_SUFFIX_BIGRAMS
                    if outer == outer_label
                )

            matches = self._find_all_suffix_matches(current, forbidden)

            if not matches:
                rk = (current, tuple(found_sfxs))
                if rk not in seen_results:
                    seen_results.add(rk)
                    raw.append((current, list(found_sfxs)))
            else:
                # Alternatif: burada durmak (en az 1 ek varsa)
                if found_sfxs:
                    rk = (current, tuple(found_sfxs))
                    if rk not in seen_results:
                        seen_results.add(rk)
                        raw.append((current, list(found_sfxs)))

                # Derinlik 0: sözcük sözlükte → yalın kök alternatifi
                # "yazar" = yaz+ar (fiil) VEYA yazar (isim)
                if (
                    depth == 0
                    and not found_sfxs
                    and self._dictionary is not None
                    and self._dictionary.contains(current)
                ):
                    rk = (current, ())
                    if rk not in seen_results:
                        seen_results.add(rk)
                        raw.append((current, []))

                for stem_cand, sfx_form, sfx_label in matches:
                    # FSM geçiş kontrolü: her mevcut FSM durumu için dene
                    new_fsm_states: list[str] = []
                    for fs in fsm_states:
                        new_fsm_states.extend(fsm.transition(fs, sfx_label))
                    if not new_fsm_states:
                        continue  # Morfotaktik olarak geçersiz

                    # Deduplicate FSM states
                    new_fsm_states = list(dict.fromkeys(new_fsm_states))

                    new_sfxs = [(sfx_form, sfx_label)] + list(found_sfxs)
                    if len(new_sfxs) <= self.MAX_ITERATIONS:
                        queue.append((stem_cand, new_sfxs, depth + 1, new_fsm_states))

                        # Tampon-n dallanması: BULUNMA/AYRILMA soyulduktan
                        # sonra kalan "n" iyelik tampon ünsüzü olabilir.
                        # konuşmasın+da → konuşması (n kaldırılır)
                        if (
                            sfx_label in ("BULUNMA", "AYRILMA")
                            and len(stem_cand) > 2
                            and stem_cand.endswith("n")
                        ):
                            debuffered = stem_cand[:-1]
                            queue.append((
                                debuffered, new_sfxs, depth + 1,
                                new_fsm_states,
                            ))

                        # Ünlü daralması dallanması: -yor soyulduktan sonra
                        # kalan kökteki dar ünlü genişletilir.
                        # istiyor → isti → iste (i→e)
                        # anlıyor → anlı → anla (ı→a)
                        # söylüyor → söylü → söyle (ü→e, yuvarlak bağlam)
                        # oynuyor → oynu → oyna (u→a, yuvarlak bağlam)
                        # Yuvarlak ünlülerde 2 aday olabilir (u→o/a, ü→ö/e)
                        _NARROWING_WIDEN: dict[str, list[str]] = {
                            "ı": ["a"],
                            "i": ["e"],
                            "u": ["o", "a"],
                            "ü": ["ö", "e"],
                        }
                        if (
                            sfx_label == "ŞİMDİKİ_ZAMAN"
                            and len(stem_cand) >= 2
                            and stem_cand[-1] in _NARROWING_WIDEN
                        ):
                            # Bağlayıcı ünlü yolu zaten geçerliyse daralma atla.
                            # bulu+yor → bul+uyor (bulmak var) → daralmaya gerek yok
                            pre_conn = stem_cand[:-1]
                            conn_valid = (
                                len(pre_conn) >= 2
                                and pre_conn[-1] in CONSONANTS
                                and self._dictionary is not None
                                and (
                                    self._dictionary.contains(pre_conn + "mak")
                                    or self._dictionary.contains(pre_conn + "mek")
                                )
                            )
                            if not conn_valid:
                                for wide_v in _NARROWING_WIDEN[stem_cand[-1]]:
                                    widened = stem_cand[:-1] + wide_v
                                    queue.append((
                                        widened, new_sfxs, depth + 1,
                                        new_fsm_states,
                                    ))

        ranked = self._rank_analyses(word, raw, upos=upos)

        # Kalite filtresi: sözlükte desteklenmeyen kökleri ele
        if self._dictionary is not None:
            filtered = [
                item for item in ranked
                if self._is_plausible_stem(item[0])
            ]
            return filtered[:max_results] if filtered else ranked[:1]

        return ranked[:max_results]

    def _is_plausible_stem(self, stem: str) -> bool:
        """Kök adayının sözlük tarafından desteklenip desteklenmediğini kontrol eder."""
        if self._dictionary is None:
            return True
        if self._dictionary.contains(stem):
            return True
        if (
            self._dictionary.contains(stem + "mak")
            or self._dictionary.contains(stem + "mek")
        ):
            return True
        if self._dictionary.find_root(stem) is not None:
            return True
        # Düzensiz fiil gövdeleri: diy→de, yiy→ye, gid→git, ed→et
        if stem in self._IRREGULAR_VERB_STEMS:
            base = self._IRREGULAR_VERB_STEMS[stem]
            if any(
                self._dictionary.contains(base + inf)
                for inf in ("mak", "mek")
            ):
                return True
        return False

    def _find_all_suffix_matches(
        self,
        current: str,
        forbidden_labels: frozenset[str] = frozenset(),
    ) -> list[tuple[str, str, str]]:
        """Mevcut konumda tüm geçerli (kök, ek, etiket) üçlülerini toplar."""
        seen: set[tuple[str, str, str]] = set()
        results: list[tuple[str, str, str]] = []

        if self._dictionary:
            is_direct = self._dictionary.contains(current)
            is_resolvable = (
                is_direct
                or self._dictionary.find_root(current) is not None
            )

            if is_resolvable:
                is_verb_root = (
                    self._dictionary.contains(current + "mak")
                    or self._dictionary.contains(current + "mek")
                )

                for strategy in self._strategies:
                    if strategy.stem_validator is not None:
                        for match in self._try_all_strategy_matches(
                            current, strategy, forbidden_labels,
                        ):
                            sc, sf, sl = match
                            if is_direct or is_verb_root:
                                is_sv = (
                                    self._dictionary.contains(sc + "mak")
                                    or self._dictionary.contains(sc + "mek")
                                )
                                # Düzensiz fiil gövdeleri: diy→de, yiy→ye
                                if not is_sv and sc in self._IRREGULAR_VERB_STEMS:
                                    base = self._IRREGULAR_VERB_STEMS[sc]
                                    is_sv = (
                                        self._dictionary.contains(base + "mak")
                                        or self._dictionary.contains(
                                            base + "mek",
                                        )
                                    )
                                # İzin: (1) fiil kök+fiil eki
                                #        (2) sözlükteki isim kök+herhangi ek
                                if is_sv and sl in self._VERBAL_SUFFIX_LABELS:
                                    pass  # fiil çözümlemesi OK
                                elif self._dictionary.contains(sc):
                                    pass  # isimsel çözümleme OK
                                else:
                                    continue
                            if match not in seen:
                                seen.add(match)
                                results.append(match)

                # Sözlük doğrulayıcı düzensiz fiil gövdelerini (diy, yiy)
                # tanıyamayabilir — sezgisel stratejilerle yeniden dene.
                # Yalnızca fiil eki + düzensiz gövde eşleşmelerine izin ver.
                if not results and (is_direct or is_verb_root):
                    for strategy in self._strategies:
                        if strategy.stem_validator is None:
                            for match in self._try_all_strategy_matches(
                                current, strategy, forbidden_labels,
                            ):
                                sc, sf, sl = match
                                if sc not in self._IRREGULAR_VERB_STEMS:
                                    continue
                                if sl not in self._VERBAL_SUFFIX_LABELS:
                                    continue
                                base = self._IRREGULAR_VERB_STEMS[sc]
                                if not any(
                                    self._dictionary.contains(base + inf)
                                    for inf in ("mak", "mek")
                                ):
                                    continue
                                if match not in seen:
                                    seen.add(match)
                                    results.append(match)

                return results

        for strategy in self._strategies:
            for match in self._try_all_strategy_matches(
                current, strategy, forbidden_labels,
            ):
                if match not in seen:
                    seen.add(match)
                    results.append(match)
        return results

    def _try_all_strategy_matches(
        self,
        current: str,
        strategy: HarmonyStrategy,
        forbidden_labels: frozenset[str] = frozenset(),
    ) -> list[tuple[str, str, str]]:
        """Tek bir strateji için tüm geçerli ek eşleşmelerini toplar."""
        results: list[tuple[str, str, str]] = []
        validator = strategy.stem_validator or self._validator

        for sfx in self._registry.suffixes:
            if sfx.label in forbidden_labels:
                continue
            if len(sfx.form) >= len(current):
                continue
            if not current.endswith(sfx.form):
                continue

            stem_candidate = current[: -len(sfx.form)]

            if strategy.stem_validator is not None:
                min_stem = 2
            elif strategy.min_stem_override is not None:
                min_stem = max(
                    sfx.min_stem_length, strategy.min_stem_override,
                )
            else:
                min_stem = sfx.min_stem_length

            if len(stem_candidate) < min_stem:
                continue
            if not validator.is_valid(stem_candidate):
                continue

            if not sfx.harmony_exempt:
                if not strategy.checker.check_vowel_harmony(
                    stem_candidate, sfx.form,
                ):
                    continue

            if not strategy.checker.check_consonant_harmony(
                stem_candidate, sfx.form,
            ):
                continue

            # Tampon-n: her iki alternatifi de sun
            if (
                sfx.form.startswith("n")
                and len(sfx.form) > 1
                and sfx.label in self._BUFFER_N_LABELS
                and self._dictionary is not None
            ):
                longer_stem = stem_candidate + "n"
                non_buffer = sfx.form[1:]
                if (
                    self._dictionary.contains(longer_stem)
                    and current.endswith(non_buffer)
                ):
                    for sfx2 in self._registry.suffixes:
                        if sfx2.form == non_buffer:
                            results.append(
                                (longer_stem, sfx2.form, sfx2.label),
                            )
                            break
                    results.append(
                        (stem_candidate, sfx.form, sfx.label),
                    )
                    continue

            results.append((stem_candidate, sfx.form, sfx.label))

        return results

    # ── Fiil-sonu etiketleri (sıralama için) ─────────────────────
    # Sözcüğün çekimli fiil olduğunu gösteren dışsal ek etiketleri.
    # Son 2 ekte bunlardan biri varsa fiil bonusu uygulanır.
    # Sıfat fiil, isim fiil, zarf fiil gibi türetme ekleri
    # sözcüğü isimsel/sıfatsal yapar → fiil bonusu uygulanmaz.
    _VERB_FINAL_LABELS: frozenset[str] = frozenset({
        # Zaman ekleri
        "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
        "ŞİMDİKİ_ZAMAN", "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
        # Kip ekleri
        "DİLEK_ŞART",
        # Kişi ekleri (zaman ekinden sonra gelir)
        "KİŞİ_1T", "KİŞİ_2T", "KİŞİ_1Ç", "KİŞİ_2Ç", "KİŞİ_3Ç",
        # Emir
        "EMİR", "EMİR_3Ç",
        # Bildirme (kopula)
        "BİLDİRME",
        # Mastar
        "MASTAR",
    })

    def _rank_analyses(
        self,
        word: str,
        analyses: list[tuple[str, list[tuple[str, str]]]],
        upos: str | None = None,
    ) -> list[tuple[str, list[tuple[str, str]]]]:
        """Çözümleme adaylarını kalite puanına göre sıralar.

        Puanlama:
          +15  kök fiilse VE (POS=VERB VEYA çekimli fiil-sonuyla bitiyorsa)
          +10  kök sözlükte
          +8   kök morfofonemik çözümlenebilir
          +0.5 kök uzunluğu başına (uzun kök tercih)
          −0.3 ek sayısı başına (basit çözümleme tercih)
          −10  kök 2 karakterden kısa

        POS=VERB verildiğinde fiil bonusu koşulsuz uygulanır; böylece
        "olması" → ol (+15) > olma (bonus yok) gibi fiil kökü tercih
        edilir.  POS bilgisi yoksa veya başka POS'taysa ek-sonuna bakılır.
        """

        def _is_verb_final(sfxs: list[tuple[str, str]]) -> bool:
            """Son 2 ekte çekimli fiil etiketi var mı?"""
            if not sfxs:
                return False
            for _, label in sfxs[-2:]:
                for sub in label.split("/"):
                    if sub in self._VERB_FINAL_LABELS:
                        return True
            return False

        def _score(item: tuple[str, list[tuple[str, str]]]) -> float:
            stem, sfxs = item
            s = 0.0
            if self._dictionary:
                in_dict = self._dictionary.contains(stem)
                is_verb = (
                    self._dictionary.contains(stem + "mak")
                    or self._dictionary.contains(stem + "mek")
                )

                # Sözlük bonusu: kök sözlükteyse VEYA fiil kökü sözlükteyse
                # ("gel" sözlükte yok ama "gelmek" var → gel de sözlüksel kök)
                if in_dict or is_verb:
                    s += 10

                if is_verb:
                    # POS=VERB → koşulsuz fiil bonusu (kısa fiil kökü tercih)
                    # POS=NOUN → fiil bonusu yok (parmak≠par+mak, yemek≠ye+mek)
                    # POS=ADP → fiil bonusu yok (dair≠da+ir)
                    # Diğer POS / POS yok → yalnızca çekimli fiil-sonuysa
                    _NO_VERB_BONUS = {"NOUN", "ADJ", "ADP", "CCONJ", "SCONJ", "INTJ"}
                    if upos == "VERB":
                        s += 15
                    elif upos not in _NO_VERB_BONUS and _is_verb_final(sfxs):
                        s += 15

                # Morfofonemik bonus: sözlükte veya fiil olarak bulunamadıysa
                if not in_dict and not is_verb:
                    resolved = self._dictionary.find_root(stem)
                    if resolved is not None:
                        # Çözümlenen kök sözlükteyse tam kredi
                        # (sonuc→sonuç, ağac→ağaç gibi ünsüz yumuşaması)
                        if self._dictionary.contains(resolved):
                            s += 10
                        else:
                            s += 8

            s += len(stem) * 0.5
            if len(stem) < 2:
                s -= 10
            s -= len(sfxs) * 0.3

            # Kaynaştırma ünsüzü belirsizliği (yalnızca y-tamponu):
            # Kök ünlü+y ile bitiyorsa VE kök[:-1] de sözlükteyse
            # → muhtemelen kaynaştırma y'si gövdeye yapışmış.
            # ortay+a yerine orta+ya, ney+i yerine ne+yi tercih et.
            # n-tamponu çok riskli (yan, alan, başkan hepsi n-sonlu).
            if (
                self._dictionary
                and self._dictionary.contains(stem)
                and len(stem) >= 3
                and stem[-1] == "y"
                and stem[-2] in VOWELS
                and self._dictionary.contains(stem[:-1])
            ):
                s -= 2

            # EMİR + hal eki dizilimi dilbilgisel olarak nadir;
            # İYELİK + hal çok daha yaygın. Ranking'de ayır.
            _HAL_LABELS = {"BULUNMA", "AYRILMA", "YÖNELME", "BELIRTME"}
            for i in range(len(sfxs) - 1):
                inner_lbl = sfxs[i][1]
                outer_lbl = sfxs[i + 1][1]
                if inner_lbl == "EMİR/KİŞİ_2T" and outer_lbl in _HAL_LABELS:
                    s -= 3

            return s

        return sorted(analyses, key=_score, reverse=True)

    # ── Dahili Yardımcılar ────────────────────────────────────

    def _find_suffix_match(
        self,
        current: str,
        forbidden_labels: frozenset[str] = frozenset(),
    ) -> tuple[str, str, str] | None:
        """
        Tüm stratejileri sırayla deneyerek en uzun ek eşleşmesini bulur.

        Args:
            current: Ek sıyrılacak sözcük parçası.
            forbidden_labels: Ek hiyerarşisi gereği yasaklanan ek etiketleri.
                Bu etiketler eşleşme aşamasında atlanır.

        Sözlük koruması (üç seviyeli):
          1. Fiil gövdesi koruması: sözcük + mak/mek sözlükte → atomik
          2. Sözlük sözcüğü koruması: sözcük sözlükte → yalnızca fiil
             köklerine ayrıştırılabilir (neden→ne ✗, okumak→oku ✓)
          3. Morfofonemik koruma: find_root ile çözümlenebilir sözcükler
        """
        if self._dictionary:
            is_direct = self._dictionary.contains(current)
            is_resolvable = (
                is_direct or self._dictionary.find_root(current) is not None
            )

            if is_resolvable:
                if (
                    self._dictionary.contains(current + "mak")
                    or self._dictionary.contains(current + "mek")
                ):
                    return None

                result = self._find_dict_backed_match(
                    current, forbidden_labels,
                )
                if result is not None:
                    stem_candidate, _, suffix_label = result

                    if is_direct:
                        is_stem_verb = (
                            self._dictionary.contains(
                                stem_candidate + "mak"
                            )
                            or self._dictionary.contains(
                                stem_candidate + "mek"
                            )
                        )
                        is_verbal = (
                            suffix_label in self._VERBAL_SUFFIX_LABELS
                        )
                        if not (is_stem_verb and is_verbal):
                            return None
                    return result
                return None

        # Normal akış: tüm stratejileri sırayla dene
        for strategy in self._strategies:
            result = self._try_strategy(
                current, strategy, forbidden_labels,
            )
            if result is not None:
                return result
        return None

    def _find_dict_backed_match(
        self,
        current: str,
        forbidden_labels: frozenset[str] = frozenset(),
    ) -> tuple[str, str, str] | None:
        """Yalnızca sözlük destekli stratejileri dener."""
        for strategy in self._strategies:
            if strategy.stem_validator is not None:
                result = self._try_strategy(
                    current, strategy, forbidden_labels,
                )
                if result is not None:
                    return result
        return None

    # Tampon-n içeren ek etiketleri: ndan→dan, nın→ın, na→a, nı→ı
    # Bu eklerin tampon-n'siz karşılıkları tercih edilir (kök sözlükteyse).
    _BUFFER_N_LABELS: frozenset[str] = frozenset({
        "AYRILMA", "TAMLAYAN", "YÖNELME", "BELIRTME",
    })

    def _try_strategy(
        self,
        current: str,
        strategy: HarmonyStrategy,
        forbidden_labels: frozenset[str] = frozenset(),
    ) -> tuple[str, str, str] | None:
        """Tek bir strateji ile ek eşleşmesi dener."""
        validator = strategy.stem_validator or self._validator

        for sfx in self._registry.suffixes:
            if sfx.label in forbidden_labels:
                continue
            if len(sfx.form) >= len(current):
                continue
            if not current.endswith(sfx.form):
                continue

            stem_candidate = current[: -len(sfx.form)]

            # Sözlük destekli stratejilerde min_stem=2 yeterli:
            # sözlük doğrulaması kökün geçerliliğini garanti eder.
            # Türkçe'de yalın kökler genellikle 1-2 hecelidir (ol, iç, et, el, ev, ön...)
            if strategy.stem_validator is not None:
                min_stem = 2
            elif strategy.min_stem_override is not None:
                min_stem = max(sfx.min_stem_length, strategy.min_stem_override)
            else:
                min_stem = sfx.min_stem_length

            if len(stem_candidate) < min_stem:
                continue
            if not validator.is_valid(stem_candidate):
                continue

            if not sfx.harmony_exempt:
                if not strategy.checker.check_vowel_harmony(
                    stem_candidate, sfx.form
                ):
                    continue

            if not strategy.checker.check_consonant_harmony(
                stem_candidate, sfx.form
            ):
                continue

            # Tampon-n tercihi: ek "n" ile başlıyorsa ve kök+n sözlükteyse,
            # tampon-n'siz formu tercih et. yandan: ya+ndan → yan+dan
            if (
                sfx.form.startswith("n")
                and len(sfx.form) > 1
                and sfx.label in self._BUFFER_N_LABELS
                and self._dictionary is not None
            ):
                longer_stem = stem_candidate + "n"
                non_buffer = sfx.form[1:]
                if (
                    self._dictionary.contains(longer_stem)
                    and current.endswith(non_buffer)
                ):
                    # Tampon-n'siz karşılık eki bul
                    for sfx2 in self._registry.suffixes:
                        if sfx2.form == non_buffer:
                            return longer_stem, sfx2.form, sfx2.label
                    # Karşılık bulunamazsa orijinal eşleşmeyi kullan

            return stem_candidate, sfx.form, sfx.label

        return None

    # ── Morfolojik Birlikte Belirleme (Suffix Co-determination) ─
    #
    # Türkçe'de aynı biçimbirim, sözcük içindeki konumuna ve komşu
    # eklere göre farklı dilbilgisel görev alır.  Bu motor tüm çok
    # etiketli ekleri bağlamlarına bakarak tek etikete çözümler.
    #
    # Yön kuralı:
    #   İLERİ  (→) : sonraki ek, şimdikinin görevini belirler
    #   GERİ  (←) : önceki ek, şimdikinin görevini belirler

    # ── -mA : OLUMSUZ / İSİM_FİİL ──────────────────────────────
    # İleri bağlam: sonraki ek isimsel → İSİM_FİİL, fiilsel → OLUMSUZ

    _NOMINAL_SUFFIXES_AFTER_MA: frozenset[str] = frozenset({
        "İYELİK_1T", "İYELİK_2T", "TAMLAYAN", "İYELİK_3T",
        "İYELİK_3T/BELIRTME", "İYELİK_2T/TAMLAYAN",
        "İYELİK_1Ç", "İYELİK_2Ç", "İYELİK_3Ç",
        "ÇOĞUL", "KİŞİ_3Ç",
        "BULUNMA", "AYRILMA", "YÖNELME", "BELIRTME", "TAMLAYAN", "VASITA",
        "İLGİ_-ki",
        "YAPIM_-lI", "YAPIM_-sIz", "YAPIM_-lIk",
    })

    _VERBAL_SUFFIXES_AFTER_MA: frozenset[str] = frozenset({
        "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
        "ŞİMDİKİ_ZAMAN", "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
        "SIFAT_FİİL", "SIFAT_FİİL_-DIk", "SIFAT_FİİL_-DIğ",
        "ZARF_FİİL_-ArAk", "ZARF_FİİL_-IncA", "ZARF_FİİL_-Ip",
        "EDİLGEN", "İŞTEŞ", "ETTİRGEN_-lAt",
        "YETERLİLİK", "DİLEK_ŞART",
        "BİLDİRME", "ETTİRGEN", "BİLDİRME/ETTİRGEN",
        "KİŞİ_2Ç", "KİŞİ_1Ç",
        "EMİR", "KİŞİ_2T", "EMİR/KİŞİ_2T", "EMİR_3Ç",
    })

    # ── -lAr : ÇOĞUL / KİŞİ_3Ç ─────────────────────────────────
    # Geri bağlam: önceki ek fiilsel → KİŞİ_3Ç, değilse → ÇOĞUL

    _TENSE_MOOD_LABELS: frozenset[str] = frozenset({
        "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
        "ŞİMDİKİ_ZAMAN", "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
        "DİLEK_ŞART", "YETERLİLİK",
        "BİLDİRME", "ETTİRGEN", "BİLDİRME/ETTİRGEN",
        "OLUMSUZ",
    })

    # ── -sIn : EMİR / KİŞİ_2T ───────────────────────────────────
    # Geri bağlam: önceki ek zaman/kip → KİŞİ_2T, değilse → EMİR

    _TENSE_LABELS_BEFORE_SIN: frozenset[str] = frozenset({
        "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
        "GELECEK_ZAMAN", "ŞİMDİKİ_ZAMAN",
        "DUYULAN_GEÇMİŞ",
    })

    # ── -DIr : BİLDİRME / ETTİRGEN ──────────────────────────────
    # Geri bağlam: önceki ek zaman/kip veya isimsel → BİLDİRME
    #              önceki ek yoksa (kök sonrası) → ETTİRGEN

    _LABELS_BEFORE_DIR_BILDIRME: frozenset[str] = frozenset({
        "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
        "ŞİMDİKİ_ZAMAN", "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
        "DİLEK_ŞART", "YETERLİLİK",
        "ÇOĞUL", "KİŞİ_3Ç",
        "İYELİK_3T", "İYELİK_3T/BELIRTME",
        "İYELİK_1T", "İYELİK_2T/TAMLAYAN",
        "İYELİK_1Ç", "İYELİK_2Ç", "İYELİK_3Ç",
        "OLUMSUZ",
    })

    # ── -(I) : İYELİK_3T / BELIRTME ──────────────────────────────
    # Geri bağlam: önceki ek iyelik → BELIRTME (çift iyelik olamaz)
    #              sonraki ek hal → İYELİK_3T (iyelik sonrası hal gelir)

    _IYELIK_LABELS: frozenset[str] = frozenset({
        "İYELİK_1T", "İYELİK_2T", "İYELİK_3T",
        "İYELİK_2T/TAMLAYAN", "İYELİK_3T/BELIRTME",
        "İYELİK_1Ç", "İYELİK_2Ç", "İYELİK_3Ç",
    })

    _HAL_LABELS_FOR_IYELIK: frozenset[str] = frozenset({
        "BULUNMA", "AYRILMA", "YÖNELME", "BELIRTME", "TAMLAYAN", "VASITA",
        "İLGİ_-ki",
    })

    def _disambiguate_suffixes(
        self,
        suffixes: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """
        Tüm çok etiketli eklerin bağlama göre tek etikete çözümlenmesi.

        Ek listesi kök→yüzey sırasındadır: suffixes[0] = en iç ek.
        Her ek için komşu eklere (ileri/geri bağlam) bakılarak
        dilbilgisel görev belirlenir.
        """
        result = list(suffixes)
        n = len(result)

        for i, (form, label) in enumerate(result):
            prev_label = result[i - 1][1] if i > 0 else None
            next_label = result[i + 1][1] if i + 1 < n else None

            # ── -mA: ileri bağlam ────────────────────────────────
            if label == "OLUMSUZ/İSİM_FİİL":
                if next_label in self._NOMINAL_SUFFIXES_AFTER_MA:
                    result[i] = (form, "İSİM_FİİL_-mA")
                elif next_label in self._VERBAL_SUFFIXES_AFTER_MA:
                    result[i] = (form, "OLUMSUZ")

            # ── -lAr: geri bağlam ────────────────────────────────
            elif label == "ÇOĞUL":
                if prev_label in self._TENSE_MOOD_LABELS:
                    result[i] = (form, "KİŞİ_3Ç")

            # ── -sIn: geri bağlam ────────────────────────────────
            elif label == "EMİR/KİŞİ_2T":
                if prev_label in self._TENSE_LABELS_BEFORE_SIN:
                    result[i] = (form, "KİŞİ_2T")
                elif prev_label is None or prev_label in (
                    "OLUMSUZ", "OLUMSUZ/İSİM_FİİL", "İSİM_FİİL_-mA",
                    "YETERLİLİK",
                ):
                    result[i] = (form, "EMİR")

            # ── -DIr: geri + ileri bağlam ─────────────────────────
            elif label == "BİLDİRME/ETTİRGEN":
                if prev_label in self._LABELS_BEFORE_DIR_BILDIRME:
                    result[i] = (form, "BİLDİRME")
                elif prev_label is None and next_label is None:
                    # Tek başına kök+DIr: büyük olasılıkla bildirme (güzeldir)
                    result[i] = (form, "BİLDİRME")
                elif prev_label is None and next_label is not None:
                    # Kök+DIr+başka ek: ettirgen çatısı (yaptırdı)
                    result[i] = (form, "ETTİRGEN")

            # ── -(I): çift yönlü bağlam ──────────────────────────
            elif label == "İYELİK_3T/BELIRTME":
                if prev_label in self._IYELIK_LABELS:
                    result[i] = (form, "BELIRTME")
                elif next_label in self._HAL_LABELS_FOR_IYELIK:
                    result[i] = (form, "İYELİK_3T")

            # ── -(I)n: çift yönlü bağlam ─────────────────────────
            elif label == "İYELİK_2T/TAMLAYAN":
                if prev_label in self._IYELIK_LABELS:
                    result[i] = (form, "TAMLAYAN")
                elif next_label in self._HAL_LABELS_FOR_IYELIK:
                    result[i] = (form, "İYELİK_2T")

        return result

    def _handle_yor_connector(
        self,
        stem: str,
        suffixes: list[tuple[str, str]],
    ) -> tuple[str, list[tuple[str, str]]]:
        """
        Şimdiki zaman "-yor" eki öncesindeki bağlayıcı ünlüyü işler.
        gel+i+yor → kök "gel", ek "-iyor"
        görm+üyor → kök "gör", ek "-müyor"
        """
        if not suffixes:
            return stem, suffixes

        first_form, first_label = suffixes[0]
        if "ŞİMDİKİ_ZAMAN" not in first_label:
            return stem, suffixes

        # "yor" veya "{I}yor" genişletilmiş formu (üyor, iyor, ıyor, uyor)
        if first_form == "yor":
            # Bağlayıcı ünlü kök sonunda: gel+i → geli + yor
            if not stem or stem[-1] not in CLOSE_VOWELS:
                return stem, suffixes

            # Kök zaten geçerli bir fiil kökü mü? (okumak, yürümek vb.)
            # → Son ünlü bağlayıcı değil, köke ait — dönüştürme.
            if self._dictionary and (
                self._dictionary.contains(stem + "mak")
                or self._dictionary.contains(stem + "mek")
            ):
                return stem, suffixes

            remaining = stem[:-1]
            connector = stem[-1]

            if len(remaining) < 2 or not get_vowels(remaining):
                return stem, suffixes
            if remaining[-1] not in CONSONANTS:
                return stem, suffixes

            if check_vowel_harmony(remaining, connector):
                suffixes[0] = (connector + "yor", "ŞİMDİKİ_ZAMAN")
                return remaining, suffixes

        elif first_form.endswith("yor") and len(first_form) == 4:
            # Genişletilmiş form: üyor, iyor vb.
            connector = first_form[0]
            if connector not in CLOSE_VOWELS:
                return stem, suffixes

            # Kök zaten geçerli bir fiil kökü mü? (kork+uyor gibi)
            # → Ünsüz kümesindeki son harf köke ait, taşıma.
            if self._dictionary and (
                self._dictionary.contains(stem + "mak")
                or self._dictionary.contains(stem + "mek")
            ):
                return stem, suffixes

            # Kök sonu ünsüz kümesiyse (görm, bakm gibi), son ünsüz
            # aslında ekin parçası — köke geri ver: görm+üyor → gör+müyor
            if (
                len(stem) >= 3
                and stem[-1] in CONSONANTS
                and stem[-2] in CONSONANTS
            ):
                new_stem = stem[:-1]
                moved = stem[-1]
                if get_vowels(new_stem):
                    suffixes[0] = (
                        moved + connector + "yor",
                        "ŞİMDİKİ_ZAMAN",
                    )
                    return new_stem, suffixes

        return stem, suffixes

    # ── Türetim Eki Çözümleme ─────────────────────────────────

    # Fiil türetim ekleri (en uzundan kısaya)
    # Yalnızca belirgin, yanlış-pozitif riski düşük ekler
    _DERIVATIONAL_VERB_SUFFIXES: tuple[str, ...] = (
        # Ettirgen (Causative): yazdır→yaz, bildirmek→bilmek
        "dır", "dir", "dur", "dür",
        "tır", "tir", "tur", "tür",
        # Edilgen (Passive): yapıl→yap, veril→ver, açıl→aç
        "ıl", "il", "ul", "ül",
        # İşteş (Reciprocal): bölüş→böl, vuruş→vur
        "ış", "iş", "uş", "üş",
        # Kısa ettirgen: belirt→belir, başlat→başla
        "t",
        # Kısa edilgen: yaşan→yaşa, etkilen→etkile
        "n",
    )

    # Leksikalleşmiş İşteş — morfolojik olarak kök+İŞTEŞ gibi görünür
    # ama anlam olarak bağımsız leksemlerdir, soyulmamalıdır.
    _LEXICALIZED_RECIPROCAL: frozenset[str] = frozenset({
        "çalış", "konuş", "oluş", "değiş", "karış", "görüş",
        "dönüş", "yetiş", "alış", "buluş", "geliş", "çatış",
        "tartış", "bakış", "sıkış", "yakış", "iniş", "içiş",
        "öpüş", "gülüş", "uçuş", "kızış", "doluş", "akış",
        "savaş", "barış", "tanış", "yarış",
    })

    # Leksikalleşmiş Türetilmiş Fiiller — morfolojik olarak kök+ettirgen/edilgen
    # gibi görünür ama anlam olarak bağımsız leksemlerdir, soyulmamalıdır.
    _LEXICALIZED_DERIVED_VERBS: frozenset[str] = frozenset({
        # Kısa ettirgen -t: bağımsız fiil, soyma
        "anlat",    # anlatmak (anlatmak) ≠ anlamak
        "yat",      # yatmak (uzanmak) — yamak ile ilgisi yok
        "yarat",    # yaratmak (oluşturmak) ≠ yaramak
        "eksilt",   # eksiltmek ≠ eksilmek
        "yut",      # yutmak (yutkunmak) — yumak ile ilgisi yok
        "yücelt",   # yüceltmek ≠ yücelmek
        "yoğurt",   # yoğurtmak ≠ yoğurmak
        # Kısa edilgen -n: bağımsız fiil, soyma
        "dinlen",   # dinlenmek (istirahat) ≠ dinlemek
        "dayan",    # dayanmak (tahammül) ≠ dayamak
        "sun",      # sunmak (takdim) — sumak (bitki) ile ilgisi yok
        "seslen",   # seslenmek ≠ seslemek
        "bürün",    # bürünmek ≠ bürümek
        "dokun",    # dokunmak (temas) ≠ dokumak (bez dokumak)
        "eğlen",    # eğlenmek ≠ eğlemek
        "ilgilen",  # ilgilenmek — ilgilemek yok
        "diren",    # direnmek ≠ diremek
        "değerlen", # değerlenmek — zincirde ara leksem
        "sonuçlan", # sonuçlanmak — tekil leksem
        # Uzun ettirgen/edilgen: bağımsız fiil, soyma
        "kaldır",   # kaldırmak ≠ kalmak
        "takıl",    # takılmak ≠ takmak (farklı anlam)
        "bayıl",    # bayılmak ≠ baymak
        "yanıl",    # yanılmak ≠ yanmak (farklı anlam)
        "yatır",    # yatırmak — kısa ettirgen zinciri
        "soruştur", # soruşturmak ≠ sormak
    })

    def _find_bare_verb_root(self, verb_stem: str) -> str | None:
        """
        Türetilmiş fiil gövdesinden temel kökü bulur.

        yazdır → yaz (ettirgen -dır kaldır)
        edil → ed → et (edilgen -il kaldır + ünsüz yumuşaması)
        okuttur → okut (bir katman, -t kısa ettirgen dahil değil)

        Her adımda sözlük doğrulaması yapılır: sonuç kök + mak/mek
        sözlükte olmalıdır.
        """
        if self._dictionary is None:
            return None

        current = verb_stem
        for _ in range(5):  # Çoklu türetim katmanları
            # Leksikalleşmiş türetilmiş fiil gövdesi → daha fazla soyma
            if current in self._LEXICALIZED_DERIVED_VERBS:
                break
            found = False
            for sfx in self._DERIVATIONAL_VERB_SUFFIXES:
                if not current.endswith(sfx):
                    continue
                # İşteş eki ise ve leksikalleşmişse, soyma
                if sfx in ("ış", "iş", "uş", "üş") and current in self._LEXICALIZED_RECIPROCAL:
                    continue
                candidate = current[: -len(sfx)]
                if len(candidate) < 2:
                    continue

                resolved = self._dictionary.find_root(candidate)
                if resolved is None:
                    continue

                # Çözümlenen biçimin fiil kökü olduğunu doğrula
                if any(
                    self._dictionary.contains(resolved + inf)
                    for inf in ("mak", "mek")
                ):
                    current = resolved
                    found = True
                    break

            if not found:
                break

        return current if current != verb_stem else None

    def _resolve_root(
        self, stem: str, suffixes: list[tuple[str, str]],
        upos: str | None = None,
    ) -> tuple[str | None, str | None]:
        """
        Sözlük kullanarak kökün temel biçimini ve lemmasını çözümler.

        Kök (root): Temel/ilkel kök biçimi.
            Morfofonemik: kitab→kitap, gid→git, burn→burun.
            Türetim: yazdır→yaz, edil→et, yapıl→yap.

        Lemma: Türetilmiş gövde veya mastar biçimi.
            Türetilmiş fiiller: yazdır, edil (gövde biçimi).
            Basit fiiller: gelmek, çalışmak (mastar biçimi).
            İsimler: None (kök ile aynı).

        Returns:
            (root, lemma) çifti. Değişmeyen alanlar None olur.
        """
        if self._dictionary is None:
            return None, None

        root = self._dictionary.find_root(stem)

        # Ünlü daralması: di→de, yi→ye (yalnızca -yor eki önünde)
        if root is None and suffixes:
            first_form = suffixes[0][0]
            if "yor" in first_form:
                narrowed = self._dictionary.find_root_with_narrowing(stem)
                if narrowed is not None and narrowed != stem:
                    root = narrowed

        # Düzensiz fiil gövdeleri: diy→de, yiy→ye, gid→git, ed→et vb.
        # -yor dışındaki eklerde de (diy+ebil, yiy+ebil) çözümleme yapabilmek
        # için _IRREGULAR_VERB_STEMS tablosuna başvurulur.
        if root is None and stem in self._IRREGULAR_VERB_STEMS:
            root = self._IRREGULAR_VERB_STEMS[stem]

        base = root if root is not None else stem

        # ── Türetilmiş fiil gövdesi çözümlemesi ──
        # base + mak/mek sözlükte → fiil gövdesi → temel kökü ara
        is_verb = any(
            self._dictionary.contains(base + inf) for inf in ("mak", "mek")
        )

        if is_verb:
            # Türetilmiş fiil kökü soyma: yalnızca VERB POS'ta
            # NOUN bağlamında "karar→kar", "yan→ya" gibi yanlış
            # soymaları önlemek için upos kontrolü yapılır.
            if upos == "VERB":
                bare = self._find_bare_verb_root(base)
                if bare is not None:
                    # root = temel kök (yaz), lemma = türetilmiş gövde (yazdır)
                    return bare, base
            # Basit fiil: kök = morfofonemik biçim, lemma = mastar
            effective_root = root if (root is not None and root != stem) else None
            lemma = None
            # Düzensiz fiil gövdelerinde (diy, di, yiy, gid, ed vb.)
            # base sözlükte müstakil sözcük olarak bulunsa bile
            # mastar biçimi (demek, yemek) lemma olarak verilmelidir.
            is_irregular = stem in self._IRREGULAR_VERB_STEMS
            if not self._dictionary.contains(base) or is_irregular:
                for inf in ("mak", "mek"):
                    if self._dictionary.contains(base + inf):
                        lemma = base + inf
                        break
            return effective_root, lemma

        # ── İsim/sıfat: yalnızca morfofonemik çözümleme ──
        effective_root = root if (root is not None and root != stem) else None
        return effective_root, None
