"""Kural-tabanlı bağımlılık çözümleme (dependency parsing).

Morfolojik çözümleme çıktısını kullanarak cümle içindeki
sözcükler arası baş-bağımlı (head-dependent) ilişkilerini belirler.

SOLID:
  SRP – Her kural sınıfı tek bir sözdizimsel olguyu ele alır.
  OCP – Yeni kurallar DependencyRule alt sınıfıyla eklenir; parser değişmez.
  LSP – Tüm kurallar aynı arayüzü uygular, birbiri yerine geçebilir.
  ISP – Kural arayüzü tek metotludur (apply).
  DIP – Parser soyut DependencyRule'a bağımlıdır, somut kurallara değil.

Kullanım:
    from morphology.sentence import SentenceAnalyzer
    from morphology.dependency import DependencyParser

    sa = SentenceAnalyzer(analyzer)
    dp = DependencyParser()

    tokens = sa.analyze("Ali kitabı okudu")
    dep_tokens = dp.parse(tokens)
    print(dp.to_conllu(dep_tokens, text="Ali kitabı okudu"))
    print(dp.to_tree(dep_tokens))
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from .analyzer import MorphemeAnalysis
from .phonology import turkish_lower
from .sentence import SentenceToken

logger = logging.getLogger("morphology.dependency")


# ═══════════════════════════════════════════════════════════════════
#  Sabitler ve Etiket Kümeleri
# ═══════════════════════════════════════════════════════════════════

_CASE_LABEL_TO_UD: dict[str, str] = {
    "BELIRTME": "Acc",
    "YÖNELME": "Dat",
    "BULUNMA": "Loc",
    "AYRILMA": "Abl",
    "TAMLAYAN": "Gen",
    "ARAÇ": "Ins",
}

_TENSE_LABEL_TO_UD: dict[str, str] = {
    "GEÇMİŞ_ZAMAN": "Past",
    "DUYULAN_GEÇMİŞ": "Evi",
    "ŞİMDİKİ_ZAMAN": "Pres",
    "GELECEK_ZAMAN": "Fut",
    "GENİŞ_ZAMAN": "Aor",
    "GENİŞ_ZAMAN_OLMSZ": "Aor",
}

_PERSON_LABEL_TO_UD: dict[str, tuple[str, str]] = {
    "KİŞİ_1T": ("1", "Sing"),
    "KİŞİ_2T": ("2", "Sing"),
    "KİŞİ_1Ç": ("1", "Plur"),
    "KİŞİ_2Ç": ("2", "Plur"),
    "KİŞİ_3Ç": ("3", "Plur"),
}

VERB_TENSE_LABELS: frozenset[str] = frozenset({
    "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
    "ŞİMDİKİ_ZAMAN", "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
    "DİLEK_ŞART", "EMİR", "EMİR_3Ç",
})

VERB_FINAL_LABELS: frozenset[str] = frozenset({
    *VERB_TENSE_LABELS,
    "KİŞİ_1T", "KİŞİ_2T", "KİŞİ_1Ç", "KİŞİ_2Ç", "KİŞİ_3Ç",
    "BİLDİRME",
})

PARTICIPLE_LABELS: frozenset[str] = frozenset({
    "SIFAT_FİİL", "SIFAT_FİİL_-DIk", "SIFAT_FİİL_-DIğ",
})

CONVERB_LABELS: frozenset[str] = frozenset({
    "ZARF_FİİL_-ArAk", "ZARF_FİİL_-IncA",
    "ZARF_FİİL_-Ip", "ZARF_FİİL_-ken",
})

# Fiilden türeme etiketleri — _infer_upos'ta VERB tespiti için
VERBAL_NOUN_LABELS: frozenset[str] = frozenset({
    "MASTAR", "İSİM_FİİL", "İŞTEŞ", "EDİLGEN", "ETTİRGEN",
})

# İsimleşmiş fiil formu tespiti — İYELİK_3T/BELIRTME altında gizli sıfat-fiil/isim-fiil
# -DIk/-DIğ participle, -EcEk/-EcEğ future participle, -mA/-mAs gerund
_NOMINALIZED_VERB_RE: re.Pattern[str] = re.compile(
    r"(?:"
    r"[dt][ıiuü][ğk]"         # -DIk/-DIğ participle
    r"|[ea]c[ea][ğk]"         # -EcEk/-EcEğ future participle
    r"|m[ea]s[ıiuü]"          # -mAsI gerund (etmesini, yapması)
    r"|m[ea]y[ıiuü]"          # -mAyI infinitive+acc (etmeyi, yapmayı)
    r"|m[ea]l[ıiuü]"          # -mAlI necessity (aldanmamalı)
    r"|[ıiuü]labil"           # -Ilabil- possibility passive
    r")", re.IGNORECASE
)

CASE_LABELS: frozenset[str] = frozenset(_CASE_LABEL_TO_UD.keys())

IYELIK_LABELS: frozenset[str] = frozenset({
    "İYELİK_1T", "İYELİK_2T", "İYELİK_3T",
    "İYELİK_1Ç", "İYELİK_2Ç", "İYELİK_3Ç",
})

DETERMINERS: frozenset[str] = frozenset({
    "bir", "bu", "şu", "o", "her", "bazı", "birçok",
    "tüm", "bütün", "hiçbir", "birkaç", "kimi", "öbür",
    "öteki", "hangi", "kaç",
})

CONJUNCTIONS: frozenset[str] = frozenset({
    "ve", "veya", "ya", "ama", "fakat", "ancak", "ile", "ne",
})

POSTPOSITIONS: frozenset[str] = frozenset({
    "için", "gibi", "kadar", "göre", "karşı", "rağmen", "dair",
    "üzere", "doğru", "dolayı", "itibaren", "beri", "hakkında",
    "ait", "ilişkin", "karşın", "boyunca", "önce", "sonra",
})

# Bilinen sıfatlar — sözlük/POS etiketi olmadığında UPOS çıkarımı için
COMMON_ADJECTIVES: frozenset[str] = frozenset({
    # Temel nitelik
    "güzel", "iyi", "kötü", "büyük", "küçük", "yeni", "eski",
    "uzun", "kısa", "genç", "yaşlı", "doğal", "yapay",
    "önemli", "farklı", "ağır", "hafif", "sıcak", "soğuk",
    "parlak", "koyu", "açık", "kapalı", "zengin", "fakir",
    "mutlu", "mutsuz", "hızlı", "yavaş", "güçlü", "zayıf",
    "temiz", "kirli", "derin", "sığ", "geniş", "dar",
    "ilk", "son", "tek", "diğer",
    # Renk
    "kırmızı", "mavi", "sarı", "yeşil", "siyah", "beyaz",
    "gri", "mor", "turuncu", "pembe", "lacivert", "bordo",
    # Boyut / konum
    "yakın", "uzak", "dış", "iç", "ön", "arka", "orta", "alt", "üst",
    "yüksek", "alçak", "ince", "kalın", "geniş", "dar",
    # Durum / değerlendirme
    "mümkün", "imkansız", "gerekli", "zorunlu", "lazım", "uygun",
    "mükemmel", "kusursuz", "eksik", "tam", "bütün", "boş", "dolu",
    "kolay", "zor", "basit", "karmaşık", "net", "belirsiz",
    "doğru", "yanlış", "haklı", "haksız", "kesin", "olası",
    # Kişilik / sosyal
    "akıllı", "aptal", "cesur", "korkak", "dürüst", "sahte",
    "kibar", "kaba", "terbiyeli", "sakin", "sinirli", "ciddi",
    "meşhur", "ünlü", "tanınmış", "bilinmeyen",
    # Fiziksel
    "yumuşak", "sert", "katı", "tatlı", "tuzlu", "ekşi", "acı",
    "sessiz", "gürültülü", "pürüzsüz", "düzgün", "bozuk",
    # Zaman / modernlik
    "modern", "antik", "klasik", "çağdaş", "ilkel", "gelişmiş",
    # Genel sıfatlar
    "hazır", "meşgul", "serbest", "özgür", "bağımsız", "ortak",
    "özel", "genel", "resmi", "sivil", "yerel", "ulusal",
    "yabancı", "yerli", "karşı", "ayrı", "beraber", "benzer",
    "aynı", "başka", "asıl", "gerçek", "sahici", "normal",
    "garip", "tuhaf", "ilginç", "sıradan", "olağan",
    # BOUN benchmark sık karşılaşılan (v6 ekleme)
    "adlı", "bol", "bilimsel", "kişisel", "hoşnut", "etnik",
    "teknik", "değişik", "uygar", "dini", "askeri", "siyasi",
    "ekonomik", "toplumsal", "sosyal", "kültürel", "tarihsel",
    "bireysel", "hukuki", "yasal", "mali", "milli",
    "kapsamlı", "belli", "belirli", "temel",
    "esas", "somut", "soyut", "potansiyel", "aktif",
    "pasif", "olumlu", "pozitif", "negatif",
    "etkili", "verimli", "yaratıcı", "yoğun", "sabit",
    "değerli", "önceki", "sonraki", "mevcut", "güncel",
})

# Bilinen zarflar — eksiz kullanımda UPOS=ADV çıkarımı için
COMMON_ADVERBS: frozenset[str] = frozenset({
    # Derece / miktar
    "çok", "az", "daha", "en", "pek", "hiç", "hep", "artık",
    "biraz", "epey", "fazla", "aşırı", "iyice", "hayli",
    "oldukça", "gayet", "son derece",
    # Zaman
    "dün", "bugün", "yarın", "şimdi", "sonra", "önce",
    "hemen", "yine", "tekrar", "bazen", "sık", "geç", "erken",
    "henüz", "hâlâ", "hala", "zaten", "artık", "sürekli", "daima",
    "devamlı", "nihayet", "sonunda", "derhal", "önceden",
    # Tarz / biçim
    "böyle", "şöyle", "öyle", "birlikte", "beraber",
    "doğrudan", "açıkça", "işte",
    # Değerlendirme / kesinlik
    "gerçekten", "aslında", "genellikle", "mutlaka",
    "kesinlikle", "muhakkak", "elbette", "tabii",
    "neredeyse", "yaklaşık", "herhalde", "yeterince",
    # Odaklama / sınırlama
    "bile", "sadece", "yalnız", "ancak",
    "özellikle", "yalnızca",
    # Soru / belirsizlik
    "belki", "acaba",
    # Olumsuzluk
    "asla",
    # Yön
    "içeri", "dışarı", "ileri", "geri",
    # Üstelik / ekleme
    "üstelik", "ayrıca", "dahası",
})

# Sık fiil-olarak-yanlış-çözümlenen isimler — _infer_upos'ta VERB→NOUN düzeltme
# Morfolojik çözümleyici bu sözcükleri fiil+ek olarak parse eder:
#   zaman → zam+an (GENİŞ_ZAMAN), bilim → bil+im (MASTAR) vb.
# Tam sözcük eşleşmesi olduğunda NOUN olarak kabul et.
COMMON_NOUNS: frozenset[str] = frozenset({
    # -An sonlu (GENİŞ_ZAMAN/SIFAT_FİİL ile çakışan)
    "zaman", "başkan", "ozan", "divan", "meydan", "kazan",
    "düzen", "neden", "güven", "yüzen",
    # -Im/-Um sonlu (MASTAR ile çakışan)
    "bilim", "eğitim", "geçim", "toplum", "hücum", "önlem",
    "eylem", "devam", "teslim", "temsil", "yatırım", "ikram",
    "yaşam",
    # -Ar/-Er sonlu (GENİŞ_ZAMAN ile çakışan)
    "karar", "pazar", "sınır", "şeker", "kültür",
    # -Iş sonlu (İŞTEŞ ile çakışan)
    "artış", "bakış", "çıkış", "giriş", "dönüş", "yürüyüş",
    # -İp/-Up sonlu
    "takip", "sahip",
    # Diğer yanlış-çözümlemeler
    "akşam", "parmak", "albüm", "helikopter", "deniz",
    "politika", "yardım", "bölge", "dahil", "insan",
    "adam", "tahmin", "talep", "taraf",
})

# Sayı sözcükleri — UPOS=NUM, deprel=nummod
NUMERALS: frozenset[str] = frozenset({
    "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz",
    "dokuz", "on", "yirmi", "otuz", "kırk", "elli",
    "altmış", "yetmiş", "seksen", "doksan", "yüz", "bin",
    "milyon", "milyar", "trilyon", "sıfır",
})

# Zaman isimleri — yalın kullanımda obl:tmod
TEMPORAL_NOUNS: frozenset[str] = frozenset({
    "sabah", "öğle", "öğlen", "akşam", "gece", "gündüz",
    "hafta", "ay", "yıl", "sene", "mevsim",
    "pazar", "pazartesi", "salı", "çarşamba",
    "perşembe", "cuma", "cumartesi",
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
    "bahar", "yaz", "sonbahar", "kış",
})

# Hafif fiiller — compound:lvc yapılarının fiil bileşeni
LIGHT_VERBS: frozenset[str] = frozenset({
    "et", "yap", "ol", "kıl", "buyur", "eyle",
})

# Zamirler — UPOS=PRON (ben, sen, o, … + soru zamirleri + dönüşlü)
PRONOUNS: frozenset[str] = frozenset({
    "ben", "sen", "biz", "siz",
    "beni", "bana", "bende", "benden", "benim", "benimle",
    "seni", "sana", "sende", "senden", "senin", "seninle",
    "onu", "ona", "onda", "ondan", "onun", "onunla",
    "bizi", "bize", "bizde", "bizden", "bizim", "bizimle",
    "sizi", "size", "sizde", "sizden", "sizin", "sizinle",
    "onları", "onlara", "onlarda", "onlardan", "onların",
    "kendi", "kendisi", "kendim", "kendin", "kendimiz", "kendiniz",
    "kim", "kimi", "kime", "kimde", "kimden", "kimin",
    "ne", "neyi", "neye", "nede", "neden", "neyin",
    "nere", "nereye", "nerede", "nereden", "nereyi",
    "buraya", "burada", "buradan", "burayı",
    "şuraya", "şurada", "şuradan",
    "oraya", "orada", "oradan", "orayı",
    "herkes", "hepsi", "hepimiz", "hepiniz",
    "hiçbiri", "birisi", "birbirimiz", "birbiri",
    "hangisi", "hangileri",
})

# Zamir kökleri — çekimli formları tanımak için
PRONOUN_STEMS: frozenset[str] = frozenset({
    "ben", "sen", "o", "biz", "siz", "onlar",
    "kendi", "kim", "ne", "nere",
    "bura", "şura", "ora",
    "herkes", "hep", "hiçbir", "biri", "birbir",
    "hangi",
})

# Soru partikülleri — UPOS=PART veya AUX
QUESTION_PARTICLES: frozenset[str] = frozenset({
    "mi", "mı", "mu", "mü",
    "mısın", "misin", "musun", "müsün",
    "mıyız", "miyiz", "muyuz", "müyüz",
})

# Odaklama / pekiştirme edatları — advmod:emph
EMPHASIS_PARTICLES: frozenset[str] = frozenset({
    "de", "da", "bile", "dahi", "sadece", "yalnızca",
    "özellikle", "hatta", "yalnız",
})

# Ünlemler — UPOS=INTJ
INTERJECTIONS: frozenset[str] = frozenset({
    "evet", "hayır", "yok", "tamam", "peki", "hay",
    "eyvah", "oh", "ah", "vah", "bravo", "aman",
    "maalesef", "lütfen", "merhaba", "güle",
})

# Bağımlama edatları — UPOS=SCONJ
SUBORDINATORS: frozenset[str] = frozenset({
    "ki", "diye", "çünkü", "madem", "mademki",
    "eğer", "şayet", "halbuki", "oysa", "oysaki",
})

# Hafif fiil + isimleştirme kalıbı (etmeyi, yapması, olduğu, …)
_LVC_NOM_RE = re.compile(
    r"^(et|yap|ol|kıl|buyur|eyle)(me|ma|iş|ış|uş|üş)",
    re.IGNORECASE,
)

# Pro-drop: 1. ve 2. kişi ekleri → özne düşmüş
PRODROP_PERSON_LABELS: frozenset[str] = frozenset({
    "KİŞİ_1T", "KİŞİ_2T", "KİŞİ_1Ç", "KİŞİ_2Ç",
})

# Scope kırıcı etiketler — sıfat-fiil argüman scope sınırı
_SCOPE_BREAKERS: frozenset[str] = VERB_FINAL_LABELS | PARTICIPLE_LABELS | CONVERB_LABELS


# ═══════════════════════════════════════════════════════════════════
#  Veri Modeli
# ═══════════════════════════════════════════════════════════════════


@dataclass
class DepToken:
    """CoNLL-U uyumlu bağımlılık belirteci.

    UD (Universal Dependencies) formatına doğrudan eşlenebilir:
      ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
    """

    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: dict[str, str]
    head: int
    deprel: str
    deps: str
    misc: str

    # İç kullanım (CoNLL-U çıktısında yer almaz)
    _suffixes: list[tuple[str, str]] = field(
        default_factory=list, repr=False,
    )
    _analysis: MorphemeAnalysis | None = field(default=None, repr=False)
    _label_cache: frozenset[str] | None = field(
        default=None, repr=False, init=False,
    )

    # ── Fabrika Metodu ────────────────────────────────────────────

    @classmethod
    def from_sentence_token(cls, st: SentenceToken, idx: int) -> DepToken:
        """SentenceToken'dan DepToken oluşturur."""
        a = st.analysis
        feats = _extract_feats(a) if a else {}
        upos = _infer_upos(st, feats)

        return cls(
            id=idx,
            form=st.word,
            lemma=a.stem if a else turkish_lower(st.word),
            upos=upos,
            xpos="_",
            feats=feats,
            head=0,
            deprel="_",
            deps="_",
            misc="_",
            _suffixes=list(a.suffixes) if a else [],
            _analysis=a,
        )

    # ── Özellik Sorgulama ─────────────────────────────────────────

    @property
    def feats_str(self) -> str:
        """UD özellik dizgisi: Key1=Val1|Key2=Val2"""
        if not self.feats:
            return "_"
        return "|".join(f"{k}={v}" for k, v in sorted(self.feats.items()))

    @property
    def is_assigned(self) -> bool:
        """Bağımlılık ilişkisi atanmış mı?"""
        return self.deprel not in ("_", "dep")

    @property
    def labels(self) -> frozenset[str]:
        """Tüm ek etiketlerinin kümesi (lazy cache)."""
        if self._label_cache is None:
            lset: set[str] = set()
            for _, lbl in self._suffixes:
                for sub in lbl.split("/"):
                    lset.add(sub)
            object.__setattr__(self, "_label_cache", frozenset(lset))
        return self._label_cache  # type: ignore[return-value]

    def has_label(self, label: str) -> bool:
        """Ek etiketleri arasında belirli bir etiket var mı?"""
        return label in self.labels

    def has_any_label(self, target_labels: frozenset[str]) -> bool:
        """Ek etiketlerinden herhangi biri verilen kümede mi?"""
        return bool(self.labels & target_labels)

    @property
    def has_case(self) -> bool:
        """Herhangi bir hal eki taşıyor mu?"""
        return self.has_any_label(CASE_LABELS)

    @property
    def has_iyelik(self) -> bool:
        """İyelik eki taşıyor mu?"""
        return self.has_any_label(IYELIK_LABELS)

    @property
    def is_bare_nominal(self) -> bool:
        """Eksiz (yalın) isimsel öge mi?"""
        return self.upos in ("NOUN", "PROPN") and not self._suffixes

    @property
    def is_nominal_head(self) -> bool:
        """İsim başı veya nominal yüklem mi? (det/amod bağlanabilir)"""
        if self.upos in ("NOUN", "PROPN"):
            return True
        # BİLDİRME ekli fiiller özünde nominal yüklemdir (ülkedir, güzeldir)
        return self.upos == "VERB" and self.has_label("BİLDİRME")


# ═══════════════════════════════════════════════════════════════════
#  Yardımcı: Özellik Çıkarımı
# ═══════════════════════════════════════════════════════════════════


def _extract_feats(a: MorphemeAnalysis) -> dict[str, str]:
    """Morfolojik çözümlemeden UD özelliklerini çıkarır."""
    feats: dict[str, str] = {}
    for _, label in a.suffixes:
        for sub in label.split("/"):
            if sub in _CASE_LABEL_TO_UD:
                feats["Case"] = _CASE_LABEL_TO_UD[sub]
            elif sub in _TENSE_LABEL_TO_UD:
                feats["Tense"] = _TENSE_LABEL_TO_UD[sub]
            elif sub in _PERSON_LABEL_TO_UD:
                person, number = _PERSON_LABEL_TO_UD[sub]
                feats["Person"] = person
                feats["Number"] = number
            elif sub == "ÇOĞUL":
                feats["Number"] = "Plur"
            elif sub == "OLUMSUZLUK":
                feats["Polarity"] = "Neg"
            elif sub in PARTICIPLE_LABELS:
                feats["VerbForm"] = "Part"
            elif sub in CONVERB_LABELS:
                feats["VerbForm"] = "Conv"
    return feats


def _infer_upos(st: SentenceToken, feats: dict[str, str]) -> str:
    """Morfolojik çözümleme + bağlam ipuçlarından UPOS tahmin eder."""
    w = turkish_lower(st.word)
    a = st.analysis

    # Zamir — çekimli formlar dahil
    if w in PRONOUNS:
        return "PRON"
    # Zamir kökü + ek (onu, onun, benim, seni, …)
    if a and a.stem in PRONOUN_STEMS and a.suffixes:
        return "PRON"

    # Soru partikülü
    if w in QUESTION_PARTICLES:
        return "PART"

    # Bağımlama edatı
    if w in SUBORDINATORS and not (a and a.suffixes):
        return "SCONJ"

    # Ünlem
    if w in INTERJECTIONS and not (a and a.suffixes):
        return "INTJ"

    if w in DETERMINERS and not (a and a.suffixes):
        return "DET"
    if w in CONJUNCTIONS and not (a and a.suffixes):
        return "CCONJ"
    if w in POSTPOSITIONS and not (a and a.suffixes):
        return "ADP"
    if (w in NUMERALS or re.match(r"^\d+$", st.word)) and not (a and a.suffixes):
        return "NUM"

    # Zarf ve sıfat: tam sözcük eşleşmesi — morfolojik çözümleme hatalı
    # olabilir ("daha" → dah+a/YÖNELME gibi). Ek koşulu kaldırıldı.
    if w in COMMON_ADVERBS:
        return "ADV"
    if w in COMMON_ADJECTIVES and not (a and a.suffixes):
        return "ADJ"

    # Fiil olarak yanlış çözümlenen isimler: tam sözcük veya kök eşleşmesi
    # ("zaman"→zam+an/GENİŞ_ZAMAN, "bilim"→bil+im/MASTAR,
    #  "insanlar"→insan+lar ama başka parse'da fiil gibi görünüyor vb.)
    if w in COMMON_NOUNS:
        return "NOUN"
    if a and a.stem and a.stem.lower() in COMMON_NOUNS:
        return "NOUN"

    if not a or not a.suffixes:
        return "NOUN"

    for _, label in a.suffixes:
        for sub in label.split("/"):
            if sub in VERB_FINAL_LABELS:
                return "VERB"
            if sub in PARTICIPLE_LABELS:
                return "VERB"
            if sub in CONVERB_LABELS:
                return "VERB"
            if sub in VERBAL_NOUN_LABELS:
                return "VERB"

    # Form-tabanlı isimleşmiş fiil tespiti:
    # İYELİK_3T/BELIRTME altında gizli -DIk, -EcEk, -mA yapıları
    has_iyelik_belirtme = False
    for _, label in a.suffixes:
        subs = label.split("/")
        if "İYELİK_3T" in subs or "BELIRTME" in subs:
            has_iyelik_belirtme = True
            break
    if has_iyelik_belirtme and _NOMINALIZED_VERB_RE.search(w):
        return "VERB"

    return "NOUN"


# ═══════════════════════════════════════════════════════════════════
#  Kural Arayüzü (SRP + ISP + LSP)
# ═══════════════════════════════════════════════════════════════════


class DependencyRule(ABC):
    """Bağımlılık kuralı soyut arayüzü.

    Her somut kural tek bir sözdizimsel olguyu ele alır (SRP).
    Yeni kurallar bu sınıftan türetilir; parser değişmez (OCP).
    Tüm alt sınıflar aynı arayüzü destekler (LSP).
    """

    @abstractmethod
    def apply(self, tokens: list[DepToken]) -> list[str]:
        """Belirteçlere bağımlılık ata, uygulanan kural adlarını döndür.

        Kurallar *tokens* listesini yerinde (in-place) değiştirir:
        ``token.head`` ve ``token.deprel`` alanlarını ayarlar.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
#  Phase A — Temel S-O-V Yapısı
# ═══════════════════════════════════════════════════════════════════


class PredicateRule(DependencyRule):
    """Cümlenin ana yüklemini (root) bulur.

    Strateji: Sağdan-sola tara, ilk çekimli fiili ``root`` yap.
    Türkçe SOV dil olduğu için yüklem genellikle cümle sonundadır.
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        for t in reversed(tokens):
            if t.has_any_label(VERB_FINAL_LABELS):
                t.head = 0
                t.deprel = "root"
                return ["YÜKLEM_BUL"]
        return []


class NominalPredicateRule(DependencyRule):
    """Fiilsiz cümlede nominal yüklemi (root) atar.

    Yalnızca ``PredicateRule`` root bulamadığında devreye girer.
    Strateji: Son sözcüğü nominal root olarak işaretle.
    Örnek: 'Hava güzel' → güzel = root
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        if any(t.deprel == "root" for t in tokens):
            return []
        if not tokens:
            return []
        tokens[-1].head = 0
        tokens[-1].deprel = "root"
        return ["NOMİNAL_YÜKLEM"]


class CaseRoleRule(DependencyRule):
    """Hal eklerine göre sözdizimsel görev atar.

    Eşleme:
        BELIRTME (-ı/-i)  → obj   (belirtili nesne)
        YÖNELME  (-e/-a)  → obl   (dolaylı tümleç)
        BULUNMA  (-de/-da)→ obl   (yer tümleyicisi)
        AYRILMA  (-den)   → obl   (ayrılma noktası)
        ARAÇ     (-le)    → obl   (araç)
        Yalın    (∅)      → nsubj / obj  (pro-drop heuristiği)

    Pro-drop: Fiilde 1./2. kişi eki varsa özne düşmüştür;
    yalın isimler obj olarak atanır.

    İyelik başı: İyelik tamlamasının başı olan sözcükte BELIRTME
    eki aslında iyelik fonksiyonundadır → nsubj olarak değerlendirilir.
    """

    _CASE_TO_DEPREL: dict[str, str] = {
        "BELIRTME": "obj",
        "YÖNELME": "obl",
        "BULUNMA": "obl",
        "AYRILMA": "obl",
        "ARAÇ": "obl",
    }

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []

        applied: list[str] = []
        # Önceki kurallar zaten nsubj atamış mı kontrol et
        nsubj_assigned = any(t.deprel == "nsubj" for t in tokens)

        # Pro-drop tespiti: fiilde 1./2. kişi eki varsa özne düşmüş
        root_tok = next((t for t in tokens if t.id == root_id), None)
        is_prodrop = (
            root_tok is not None
            and root_tok.has_any_label(PRODROP_PERSON_LABELS)
        )

        # İyelik başı olan token'ları bul (nmod:poss bağımlısı var)
        poss_heads = frozenset(
            t.head for t in tokens if t.deprel == "nmod:poss"
        )

        # Edat bağımlısı olan isimler
        has_case_child = self._tokens_with_case_child(tokens)

        for t in tokens:
            if t.is_assigned or t.id == root_id:
                continue

            # 1) İyelik başı → BELIRTME eki iyelik fonksiyonunda
            #    Hal eki olarak sayma, yalın gibi değerlendir
            if t.id in poss_heads and t.has_iyelik:
                role = self._detect_case_role_skip_belirtme(t)
                if role:
                    t.head = root_id
                    t.deprel = role
                    applied.append(f"HAL→{role.upper()}")
                    continue
                # BELIRTME dışında hal eki yok → yalın gibi davran
                if not nsubj_assigned and not is_prodrop:
                    t.head = root_id
                    t.deprel = "nsubj"
                    nsubj_assigned = True
                    applied.append("İYELİK_BAŞI→NSUBJ")
                else:
                    t.head = root_id
                    t.deprel = "obj" if is_prodrop else "nsubj"
                    if not is_prodrop:
                        nsubj_assigned = True
                    applied.append("İYELİK_BAŞI→OBJ" if is_prodrop else "İYELİK_BAŞI→NSUBJ")
                continue

            # 2) Hal eki → doğrudan görev eşleme
            #    İsimleşmiş fiil: VERB + BELIRTME → ccomp (tümleç yan cümlesi)
            #    geldiğini biliyorum → geldiğini = ccomp
            role = self._detect_case_role(t)
            if role:
                if role == "obj" and t.upos == "VERB":
                    t.head = root_id
                    t.deprel = "ccomp"
                    applied.append("FİİL_BELIRTME→CCOMP")
                else:
                    t.head = root_id
                    t.deprel = role
                    applied.append(f"HAL→{role.upper()}")
                continue

            # 3) Edat bağımlısı var → obl ("ev için" → ev=obl)
            if t.id in has_case_child:
                t.head = root_id
                t.deprel = "obl"
                applied.append("EDAT_BAĞIMLI→OBL")
                continue

            # 4) Yalın isim/zamir → belirlilik hiyerarşisi
            #    Türkçe'de yalın ortak isim = belirtisiz nesne (kitap okudu)
            #    Özel isim / zamir = belirli → özne adayı
            if t.upos in ("NOUN", "PROPN", "PRON") and not t.has_case:
                is_definite = t.upos in ("PROPN", "PRON") or t.has_iyelik
                if is_prodrop:
                    # Pro-drop: belirli → nsubj (Onu gördüm ama Ali geldi)
                    #           belirsiz → obj (kitap okudum)
                    if is_definite and not nsubj_assigned:
                        t.head = root_id
                        t.deprel = "nsubj"
                        nsubj_assigned = True
                        applied.append("BELİRLİ_PRODROP→NSUBJ")
                    else:
                        t.head = root_id
                        t.deprel = "obj"
                        applied.append("PRODROP→OBJ")
                elif not nsubj_assigned:
                    # İlk yalın: belirli veya ortak isim → nsubj
                    t.head = root_id
                    t.deprel = "nsubj"
                    nsubj_assigned = True
                    applied.append("YALIN→NSUBJ")
                else:
                    # nsubj zaten var, ikinci yalın
                    if is_definite:
                        t.head = root_id
                        t.deprel = "obj"
                        applied.append("BELİRLİ→OBJ")
                    else:
                        # Yalın ortak isim, ikinci → belirtisiz nesne
                        t.head = root_id
                        t.deprel = "obj"
                        applied.append("BELİRTİSİZ→OBJ")

        return applied

    @staticmethod
    def _tokens_with_case_child(tokens: list[DepToken]) -> frozenset[int]:
        """Edat (case) bağımlısı olan token id'lerini döndürür."""
        return frozenset(
            t.head for t in tokens if t.deprel == "case"
        )

    def _detect_case_role(self, t: DepToken) -> str | None:
        """Hal eki varsa karşılık gelen bağımlılık ilişkisini döndürür."""
        for _, label in t._suffixes:
            for sub in label.split("/"):
                if sub in self._CASE_TO_DEPREL:
                    return self._CASE_TO_DEPREL[sub]
        return None

    def _detect_case_role_skip_belirtme(self, t: DepToken) -> str | None:
        """BELIRTME hariç hal eki arar (iyelik başı için)."""
        for _, label in t._suffixes:
            for sub in label.split("/"):
                if sub != "BELIRTME" and sub in self._CASE_TO_DEPREL:
                    return self._CASE_TO_DEPREL[sub]
        return None


# ═══════════════════════════════════════════════════════════════════
#  Phase B — İsim Öbeği İç Yapısı
# ═══════════════════════════════════════════════════════════════════


class PossessiveRule(DependencyRule):
    """İyelik tamlaması (genitif-posesif) zincirini çözer.

    Belirtili: TAMLAYAN + İYELİK → nmod:poss
    Belirtisiz (3. tekil): yalın + İYELİK_3T → CompoundNounRule'a bırak

    Strateji: TAMLAYAN ekli sözcüğü sağdaki İYELİK ekli sözcüğe bağla.
    Arama kapsamı genişletilmiş: sıfat, belirleyici, sayı, başka
    tamlayan sözcükleri atlayarak İYELİK taşıyan baş aranır.
    Örnek: 'ülkenin geleceği' → ülkenin ──nmod:poss──▶ geleceği
           'çocuğun güzel kitabı' → çocuğun ──nmod:poss──▶ kitabı
    """

    # Arama sırasında atlanabilecek UPOS türleri
    _SKIP_UPOS: frozenset[str] = frozenset({
        "ADJ", "DET", "NUM", "NOUN", "PROPN", "ADV",
    })

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned or not t.has_label("TAMLAYAN"):
                continue
            # Sağdaki ilk İYELİK-ekli sözcüğü bul (geniş arama)
            for j in range(i + 1, len(tokens)):
                candidate = tokens[j]
                if candidate.has_iyelik:
                    t.head = candidate.id
                    t.deprel = "nmod:poss"
                    applied.append("TAMLAYAN→İYELİK")
                    break
                if candidate.upos not in self._SKIP_UPOS:
                    break
                # Başka bir TAMLAYAN tokenı → iç içe tamlama, atla
                if candidate.has_label("TAMLAYAN"):
                    continue
        return applied


class DeterminerRule(DependencyRule):
    """Belirleyicileri sağdaki ilk isme bağlar.

    Örnek: 'bu kitap' → bu ──det──▶ kitap
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.upos != "DET" or t.is_assigned:
                continue
            head = self._find_right_nominal(tokens, i)
            if head:
                t.head = head.id
                t.deprel = "det"
                applied.append("BELİRLEYİCİ→DET")
        return applied

    @staticmethod
    def _find_right_nominal(
        tokens: list[DepToken], start: int,
    ) -> DepToken | None:
        """Sağdaki ilk isimsel token'ı bulur (sıfat/det/zarf atlayarak)."""
        for j in range(start + 1, len(tokens)):
            t = tokens[j]
            if t.is_nominal_head:
                return t
            if t.upos not in ("ADJ", "DET", "NUM", "ADV"):
                break
        return None


class AdjectiveRule(DependencyRule):
    """Sıfatları sağdaki isme amod olarak bağlar.

    Heuristik:
    1. Eksiz NOUN/ADJ + sağında ekli isim → amod (orijinal)
    2. ADJ UPOS + sağında herhangi NOUN/PROPN → amod (genişletilmiş)
    Örnek: 'güzel kitabı' → güzel ──amod──▶ kitabı
           'büyük şehir' → büyük ──amod──▶ şehir
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned or t._suffixes:
                continue
            if t.upos not in ("NOUN", "ADJ"):
                continue
            # Strateji 1: ekli isim (orijinal — her NOUN/ADJ için)
            head = self._find_right_inflected_noun(tokens, i)
            if head:
                t.head = head.id
                t.deprel = "amod"
                t.upos = "ADJ"
                applied.append("SIFAT→AMOD")
                continue
            # Strateji 2: ADJ UPOS + sağda bare/inflected NOUN
            if t.upos == "ADJ":
                head = self._find_right_any_noun(tokens, i)
                if head:
                    t.head = head.id
                    t.deprel = "amod"
                    applied.append("ADJ→AMOD_BARE")
        return applied

    @staticmethod
    def _find_right_inflected_noun(
        tokens: list[DepToken], start: int,
    ) -> DepToken | None:
        """Sağdaki ilk ekli ismi bulur; araya sıfat/det girebilir."""
        for j in range(start + 1, len(tokens)):
            t = tokens[j]
            if t.is_nominal_head and t._suffixes:
                return t
            if t.upos in ("VERB", "ADP", "CCONJ"):
                # BİLDİRME ekli fiiller nominal baş olarak kabul
                if t.is_nominal_head:
                    return t
                break
        return None

    @staticmethod
    def _find_right_any_noun(
        tokens: list[DepToken], start: int,
    ) -> DepToken | None:
        """Sağdaki ilk ismi bulur (ekli/eksiz); araya ADJ/DET/NUM girebilir."""
        for j in range(start + 1, len(tokens)):
            t = tokens[j]
            if t.upos in ("NOUN", "PROPN") and not t.is_assigned:
                return t
            if t.upos in ("ADJ", "DET", "NUM"):
                continue
            break
        return None


class ConverbRule(DependencyRule):
    """Zarf-fiilleri ana yükleme advcl olarak bağlar.

    Örnek: 'koşarak geldi' → koşarak ──advcl──▶ geldi
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []
        applied: list[str] = []
        for t in tokens:
            if t.is_assigned or t.id == root_id:
                continue
            if t.has_any_label(CONVERB_LABELS):
                t.head = root_id
                t.deprel = "advcl"
                applied.append("ZARF_FİİL→ADVCL")
        return applied


class ParticipleRule(DependencyRule):
    """Sıfat-fiilleri sağdaki isme acl olarak bağlar (scope-aware).

    İki aşama:
      1. Sıfat-fiili sağdaki isme acl olarak bağla.
      2. Sıfat-fiilin solundaki atanmamış argümanları scope dahilinde
         sıfat-fiilin kendisine bağla (ilgi cümlesi iç yapısı).

    Scope sınırı: çekimli fiil, başka sıfat/zarf-fiil, bağlaç, cümle başı.
    Örnek: 'Okula giden çocuk güldü'
           → giden ──acl──▶ çocuk,  Okula ──obl──▶ giden
    """

    _CASE_TO_ROLE: dict[str, str] = {
        "BELIRTME": "obj",
        "YÖNELME": "obl",
        "BULUNMA": "obl",
        "AYRILMA": "obl",
        "ARAÇ": "obl",
    }

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned:
                continue
            if not t.has_any_label(PARTICIPLE_LABELS):
                continue

            # Aşama 1: Sağdaki isme acl bağla
            acl_head = None
            for j in range(i + 1, len(tokens)):
                candidate = tokens[j]
                if candidate.upos in ("NOUN", "PROPN"):
                    acl_head = candidate
                    t.head = candidate.id
                    t.deprel = "acl"
                    applied.append("SIFAT_FİİL→ACL")
                    break
                if candidate.upos not in ("ADJ", "DET", "NUM"):
                    break

            if not acl_head:
                continue

            # Aşama 2: Scope — soldan argümanları sıfat-fiiline bağla
            scope_start = self._find_scope_start(tokens, i)
            for k in range(scope_start, i):
                arg = tokens[k]
                if arg.is_assigned:
                    continue
                role = self._detect_arg_role(arg)
                if role:
                    arg.head = t.id
                    arg.deprel = role
                    applied.append(f"RC_SCOPE→{role.upper()}")

        return applied

    @staticmethod
    def _find_scope_start(tokens: list[DepToken], part_idx: int) -> int:
        """Sıfat-fiilin scope başlangıcını bulur (sola doğru)."""
        for k in range(part_idx - 1, -1, -1):
            tok = tokens[k]
            if tok.has_any_label(_SCOPE_BREAKERS):
                return k + 1
            if tok.upos == "CCONJ":
                return k + 1
        return 0

    @classmethod
    def _detect_arg_role(cls, t: DepToken) -> str | None:
        """Token'ın hal ekine göre scope-içi rolünü belirler."""
        for sub in t.labels:
            if sub in cls._CASE_TO_ROLE:
                return cls._CASE_TO_ROLE[sub]
        if t.upos in ("NOUN", "PROPN") and not t.has_case:
            return "nsubj"
        if t.upos == "ADV":
            return "advmod"
        return None


class MultiPredicateRule(DependencyRule):
    """Çok yüklemli cümlelerde ikincil çekimli fiilleri conj olarak bağlar.

    PredicateRule en sağdaki çekimli fiili root yapar.
    ConverbRule zarf-fiilleri advcl yapar.
    ParticipleRule sıfat-fiilleri acl yapar.
    Geriye kalan indicative fiiller (geçmiş, şimdiki, gelecek, geniş) → conj(root).

    DİLEK_ŞART ve EMİR hariç tutulur — bunlar tipik olarak advcl'dir.

    Örnek: 'Geldi, gördü, kazandı' → gördü──conj──▶kazandı, geldi──conj──▶kazandı
    """

    _INDICATIVE_TENSES: frozenset[str] = frozenset({
        "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
        "ŞİMDİKİ_ZAMAN", "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
    })

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []
        applied: list[str] = []
        for t in tokens:
            if t.is_assigned or t.id == root_id:
                continue
            if t.upos != "VERB":
                continue
            # Sadece indicative (haber kipi) fiiller → conj
            if not t.has_any_label(self._INDICATIVE_TENSES):
                continue
            # Zarf-fiil veya sıfat-fiil değilse → conj
            if t.has_any_label(CONVERB_LABELS | PARTICIPLE_LABELS):
                continue
            t.head = root_id
            t.deprel = "conj"
            applied.append("ÇOK_YÜKLEM→CONJ")
        return applied


class PostpositionRule(DependencyRule):
    """Edatları solundaki sözcüğe case olarak bağlar.

    Örnek: 'okul için' → için ──case──▶ okul
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            w = turkish_lower(t.form)
            if w not in POSTPOSITIONS or t.is_assigned:
                continue
            if i > 0:
                left = tokens[i - 1]
                t.head = left.id
                t.deprel = "case"
                t.upos = "ADP"
                applied.append("EDAT→CASE")
        return applied


class AdvmodRule(DependencyRule):
    """Zarfları (ADV) yükleme veya komşu sıfat/zarfa advmod olarak bağlar.

    Strateji:
      1. Sağ komşu ADJ/ADV ise → o tokene bağla ("çok güzel" → çok→güzel)
      2. Aksi halde → yükleme bağla ("dün geldim" → dün→geldim)
    Örnek: 'çok güzel kitap' → çok ──advmod──▶ güzel
           'daha iyi oldu'  → daha ──advmod──▶ iyi
           'hemen geldim'   → hemen ──advmod──▶ geldim
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned or t.id == root_id:
                continue
            if t.upos == "ADV":
                target = self._find_target(tokens, i, root_id)
                t.head = target
                t.deprel = "advmod"
                applied.append("ZARF→ADVMOD")
        return applied

    @staticmethod
    def _find_target(tokens: list[DepToken], idx: int, root_id: int) -> int:
        """Zarfın bağlanacağı hedefi belirle: sağ komşu ADJ/ADV → o; aksi → root."""
        if idx + 1 < len(tokens):
            right = tokens[idx + 1]
            if right.upos in ("ADJ", "ADV"):
                return right.id
        return root_id


class CoordinationRule(DependencyRule):
    """Eşgüdüm yapısını (ve/veya/ama) ve fiil koordinasyonunu çözer.

    Stratejiler:
      1. Bağlaç + isimsel: bağlaç→cc, sağ→conj(sol)
      2. Bağlaç + fiil: bağlaç→cc, fiil→conj(root)
      3. Çoklu yüklem: son fiil root, önceki fiiller conj
    Örnek: 'Ali ve Ayşe geldi'    → ve──cc──▶Ayşe, Ayşe──conj──▶Ali
           'geldi ve gitti'       → ve──cc──▶gitti, gitti──conj──▶geldi
           'hem geldi hem gitti'  → hem→cc, gitti→conj(geldi)
    """

    _CORRELATIVES: frozenset[str] = frozenset({
        "hem", "ne", "ya", "ister", "gerek", "olsun",
    })

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []

        # Faz 1: Explicit bağlaçlar (ve, veya, ama, fakat)
        for i, t in enumerate(tokens):
            if t.upos != "CCONJ" or t.is_assigned:
                continue
            left = self._find_left_conjunct(tokens, i)
            right = self._find_right_conjunct(tokens, i)
            if left and right:
                t.head = right.id
                t.deprel = "cc"
                if not right.is_assigned:
                    right.head = left.id
                    right.deprel = "conj"
                applied.append("BAĞLAÇ→CC+CONJ")

        # Faz 2: İlişkili bağlaçlar (hem...hem, ne...ne, ya...ya)
        applied.extend(self._handle_correlatives(tokens))

        return applied

    @staticmethod
    def _find_left_conjunct(tokens: list[DepToken], conj_idx: int) -> DepToken | None:
        """Bağlacın solundaki ilk uygun ögeyi bul."""
        for j in range(conj_idx - 1, -1, -1):
            t = tokens[j]
            if t.upos in ("CCONJ", "DET", "ADP"):
                continue
            return t
        return None

    @staticmethod
    def _find_right_conjunct(tokens: list[DepToken], conj_idx: int) -> DepToken | None:
        """Bağlacın sağındaki ilk uygun ögeyi bul."""
        for j in range(conj_idx + 1, len(tokens)):
            t = tokens[j]
            if t.upos in ("CCONJ", "DET"):
                continue
            return t
        return None

    def _handle_correlatives(self, tokens: list[DepToken]) -> list[str]:
        """hem...hem, ne...ne, ya...ya kalıplarını işle."""
        applied: list[str] = []
        w_lower = [turkish_lower(t.form) for t in tokens]
        i = 0
        while i < len(tokens) - 2:
            if w_lower[i] in self._CORRELATIVES and not tokens[i].is_assigned:
                corr = w_lower[i]
                # İkinci correlative'i bul
                for j in range(i + 2, len(tokens)):
                    if w_lower[j] == corr and not tokens[j].is_assigned:
                        # İlk corr → cc (sağdaki ögeye)
                        right1 = tokens[i + 1] if i + 1 < len(tokens) else None
                        right2 = tokens[j + 1] if j + 1 < len(tokens) else None
                        if right1 and right2:
                            tokens[i].head = right1.id
                            tokens[i].deprel = "cc"
                            tokens[j].head = right2.id
                            tokens[j].deprel = "cc"
                            if not right2.is_assigned:
                                right2.head = right1.id
                                right2.deprel = "conj"
                            applied.append("İLİŞKİLİ→CC+CONJ")
                        break
            i += 1
        return applied


# ═══════════════════════════════════════════════════════════════════
#  Phase D — İleri Yapılar
# ═══════════════════════════════════════════════════════════════════


class LightVerbRule(DependencyRule):
    """Hafif fiil yapılarını compound:lvc olarak bağlar.

    Yalın isim + hafif fiil (et, yap, ol, kıl, buyur, eyle)
    birleşimini tespit eder. Hafif fiilin çekimli (etti) veya
    isimleşmiş (etmeyi, yapması) formları da desteklenir.
    Örnek: 'yardım etti'     → yardım ──compound:lvc──▶ etti
           'dans etmeyi'     → dans ──compound:lvc──▶ etmeyi
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned or t.upos not in ("NOUN", "ADJ") or t._suffixes:
                continue
            if turkish_lower(t.form) in TEMPORAL_NOUNS:
                continue
            if i + 1 >= len(tokens):
                continue
            right = tokens[i + 1]
            if self._is_light_verb(right):
                t.head = right.id
                t.deprel = "compound:lvc"
                applied.append("HAFİF_FİİL→COMPOUND_LVC")
        return applied

    @staticmethod
    def _is_light_verb(token: DepToken) -> bool:
        """Token'ın hafif fiil (çekimli veya isimleşmiş) olup olmadığını kontrol eder."""
        if token.upos == "VERB" and token.lemma in LIGHT_VERBS:
            return True
        # İsimleşmiş hafif fiil: etmeyi, yapması, oluşu, …
        if _LVC_NOM_RE.match(turkish_lower(token.form)):
            return True
        return False


class NummodRule(DependencyRule):
    """Sayıları sağdaki isme nummod olarak bağlar.

    'bir' hariç (DET olarak kalır).
    Örnek: 'üç kitap' → üç ──nummod──▶ kitap
           '100 kişi' → 100 ──nummod──▶ kişi
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned:
                continue
            w = turkish_lower(t.form)
            is_num = w in NUMERALS or bool(re.match(r"^\d+$", t.form))
            if not is_num:
                continue
            for j in range(i + 1, len(tokens)):
                candidate = tokens[j]
                if candidate.upos in ("NOUN", "PROPN"):
                    t.head = candidate.id
                    t.deprel = "nummod"
                    t.upos = "NUM"
                    applied.append("SAYI→NUMMOD")
                    break
                if candidate.upos not in ("ADJ", "NUM"):
                    break
        return applied


class FlatNameRule(DependencyRule):
    """Ardışık özel isimleri flat ile bağlar.

    BOUN Treebank kuralı: özel isim zincirleri ``flat`` olarak etiketlenir.
    Heuristik: Ardışık büyük harfle başlayan sözcükler.
    Suffix kontrolü yapılmaz — morfolojik çözümleyici özel isimleri
    yanlış çözümleyebilir (ör: Mustafa → mustaf+a).
    Örnek: 'Mustafa Kemal' → Kemal ──flat──▶ Mustafa
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.form[0].isupper() and not t.is_assigned:
                chain_start = i
                j = i + 1
                while j < len(tokens):
                    nxt = tokens[j]
                    if (nxt.form[0].isupper()
                            and not nxt.is_assigned
                            and nxt.upos in ("NOUN", "PROPN")):
                        j += 1
                    else:
                        break
                if j > chain_start + 1:
                    head_tok = tokens[chain_start]
                    head_tok.upos = "PROPN"
                    object.__setattr__(head_tok, "_suffixes", ())
                    object.__setattr__(head_tok, "_label_cache", None)
                    for k in range(chain_start + 1, j):
                        tokens[k].head = head_tok.id
                        tokens[k].deprel = "flat"
                        tokens[k].upos = "PROPN"
                        applied.append("FLAT")
                    i = j
                    continue
            i += 1
        return applied


class TemporalAdvmodRule(DependencyRule):
    """Yalın zaman isimlerini obl:tmod olarak bağlar.

    Zaman isimleri (akşam, sabah, gece, gün adları, ay adları…)
    yalın kullanıldığında nsubj değil obl:tmod olarak atanmalıdır.
    Örnek: 'akşam geldim' → akşam ──obl:tmod──▶ geldim
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []
        applied: list[str] = []
        for t in tokens:
            if t.is_assigned or t.id == root_id:
                continue
            w = turkish_lower(t.form)
            if w in TEMPORAL_NOUNS and not t._suffixes:
                t.head = root_id
                t.deprel = "obl:tmod"
                applied.append("ZAMAN→OBL_TMOD")
        return applied


class AdvmodEmphRule(DependencyRule):
    """Odaklama/pekiştirme edatlarını advmod:emph olarak bağlar.

    de/da, bile, dahi, sadece gibi edatlar cümle içinde önceki
    sözcüğe advmod:emph (emphasis modifier) olarak bağlanır.
    Örnek: 'Ali de geldi'  → de ──advmod:emph──▶ Ali
           'O bile bilir'  → bile ──advmod:emph──▶ O
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned:
                continue
            w = turkish_lower(t.form)
            if w not in EMPHASIS_PARTICLES:
                continue
            # Solundaki sözcüğe bağla
            if i > 0:
                t.head = tokens[i - 1].id
                t.deprel = "advmod:emph"
                applied.append("ODAKLAMA→ADVMOD_EMPH")
        return applied


class CopulaRule(DependencyRule):
    """Kopula (bağ fiil) yapısını çözer.

    Türkçede kopula iki biçimde bulunur:
      1. BİLDİRME eki (-dır/-dir/-tır/-tir) → nominal yüklemin
         kendisine bağlı cop
      2. 'değil' sözcüğü → olumsuz kopula

    Strateji: BİLDİRME ekli yüklemde, sözcüğün kendisi nominal root
    kalır; kopula bilgisi feats olarak işlenir (BOUN böyle yapmaz,
    ama biz değil/ise gibi bağımsız sözcükleri cop olarak atariz).

    Örnek: 'Bu öğrenci değil' → değil ──cop──▶ öğrenci
    """

    _COP_WORDS: frozenset[str] = frozenset({
        "değil", "değildir", "değildi", "değilmiş",
        "idi", "imiş", "ise",
    })

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned:
                continue
            w = turkish_lower(t.form)
            if w not in self._COP_WORDS:
                continue

            # Sola doğru nominal baş ara
            head_tok = None
            for k in range(i - 1, -1, -1):
                cand = tokens[k]
                if cand.upos in ("NOUN", "PROPN", "ADJ", "PRON", "NUM"):
                    head_tok = cand
                    break
                if cand.upos in ("VERB", "CCONJ"):
                    break

            if head_tok:
                t.head = head_tok.id
                t.deprel = "cop"
                t.upos = "AUX"
                applied.append("KOPULA→COP")
            else:
                # Nominal baş bulunamadı → root'a bağla
                root_id = _find_root_id(tokens)
                if root_id:
                    t.head = root_id
                    t.deprel = "cop"
                    t.upos = "AUX"
                    applied.append("KOPULA→COP_ROOT")

        return applied


class CompoundNounRule(DependencyRule):
    """Belirtisiz isim tamlaması (N + İYELİK_3T) yapısını çözer.

    Türkçede belirtisiz tamlama: ön isim yalın + baş isim İYELİK_3T.
    BOUN Treebank'ta bu yapı ``nmod:poss`` olarak etiketlenir
    (gerçek compound yalnızca leksikalize bileşikler için kullanılır).
    Örnek: 'okul kitabı'   → okul ──nmod:poss──▶ kitabı
           'masa örtüsü'   → masa ──nmod:poss──▶ örtüsü
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned:
                continue
            if t.upos not in ("NOUN", "PROPN"):
                continue
            if t.has_case or t.has_label("TAMLAYAN"):
                continue
            if i + 1 >= len(tokens):
                continue
            right = tokens[i + 1]
            if right.upos not in ("NOUN", "PROPN"):
                continue
            if not right.has_label("İYELİK_3T"):
                continue
            # Hal eki kontrolü (İYELİK_3T/BELIRTME ambiguitesi)
            if right.has_case:
                non_belirtme = right.labels & (CASE_LABELS - {"BELIRTME"})
                if non_belirtme:
                    # YÖNELME/BULUNMA/AYRILMA gibi açık hal eki → bağımsız isim
                    continue
                # Sadece BELIRTME → bağlam kontrolü:
                # Sağında doğrudan fiil varsa → muhtemelen accusative nesne
                if i + 2 < len(tokens):
                    nxt = tokens[i + 2]
                    if nxt.upos == "VERB" or nxt.has_any_label(VERB_FINAL_LABELS):
                        continue
            # Tamlayan zaten bağlıysa atla (belirtili tamlama)
            if any(
                tok.deprel == "nmod:poss" and tok.head == right.id
                for tok in tokens
            ):
                continue
            t.head = right.id
            t.deprel = "nmod:poss"
            applied.append("BELİRTİSİZ_TAMLAMA→NMOD_POSS")
        return applied


class AdjAdvDisambiguationRule(DependencyRule):
    """Sıfat-zarf belirsizliğini bağlamla çözer.

    COMMON_ADJECTIVES kümesindeki yalın sözcükler için:
      - Sağında isim → amod (sıfat)
      - Sağında fiil → advmod (zarf)
      - Sağında DET/NUM + fiil (NP yapısı) → amod (nominal yüklem)
    Örnek: 'hızlı koştu'            → hızlı ──advmod──▶ koştu
           'hızlı araba'            → hızlı ──amod──▶ araba
           'büyük bir ülkedir'      → büyük ──amod──▶ ülkedir
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned or t._suffixes:
                continue
            if t.upos != "ADJ":
                continue

            right, det_skipped = self._find_right_content(tokens, i)
            if right is None:
                continue

            if right.upos in ("NOUN", "PROPN"):
                t.head = right.id
                t.deprel = "amod"
                applied.append("SIFAT_ZARF→AMOD")
            elif right.upos == "VERB":
                if det_skipped:
                    # DET/NUM atlayarak fiil bulduk → NP yapısı (nominal yüklem)
                    t.head = right.id
                    t.deprel = "amod"
                    applied.append("SIFAT_ZARF→AMOD_NOM")
                else:
                    root_id = _find_root_id(tokens)
                    target = root_id if root_id != 0 else right.id
                    t.head = target
                    t.deprel = "advmod"
                    t.upos = "ADV"
                    applied.append("SIFAT_ZARF→ADVMOD")

        return applied

    @staticmethod
    def _find_right_content(
        tokens: list[DepToken], start: int,
    ) -> tuple[DepToken | None, bool]:
        """Sağdaki ilk içerik sözcüğünü bulur (DET/NUM atlayarak).

        Returns:
            (token, det_skipped): Bulunan token ve arada DET/NUM atlanıp atlanmadığı.
        """
        det_skipped = False
        for j in range(start + 1, len(tokens)):
            t = tokens[j]
            if t.upos in ("DET", "NUM", "ADJ"):
                if t.upos in ("DET", "NUM"):
                    det_skipped = True
                continue
            return t, det_skipped
        return None, False


class FallbackRule(DependencyRule):
    """Atanmamış token'ları UPOS-aware olarak bağlar.

    Strateji:
      - CCONJ → cc (root'a)
      - DET → sağdaki en yakın NOUN/PROPN'a det
      - Geri kalan → root'a dep (graceful degradation)
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned or t.id == root_id:
                continue

            # CCONJ: bağlaç → cc olarak root'a bağla
            if t.upos == "CCONJ":
                t.head = root_id
                t.deprel = "cc"
                applied.append("FALLBACK→CC")
                continue

            # DET: sağdaki en yakın NOUN/PROPN'a det bağla
            if t.upos == "DET":
                target = self._find_right_noun(tokens, i)
                if target:
                    t.head = target.id
                    t.deprel = "det"
                    applied.append("FALLBACK→DET")
                    continue

            t.head = root_id
            t.deprel = "dep"
            applied.append("FALLBACK→DEP")
        return applied

    @staticmethod
    def _find_right_noun(tokens: list[DepToken], start: int) -> DepToken | None:
        """start'ın sağındaki en yakın NOUN/PROPN tokenini bul."""
        for j in range(start + 1, min(start + 5, len(tokens))):
            if tokens[j].upos in ("NOUN", "PROPN"):
                return tokens[j]
            if tokens[j].upos not in ("ADJ", "NUM", "DET", "ADV"):
                break
        return None



# ═══════════════════════════════════════════════════════════════════
#  Yardımcılar
# ═══════════════════════════════════════════════════════════════════


def _find_root_id(tokens: list[DepToken]) -> int:
    """Root belirtecinin id'sini döndürür (bulunamazsa 0)."""
    for t in tokens:
        if t.deprel == "root":
            return t.id
    return 0


def _build_tree_lines(
    node: DepToken,
    children: dict[int, list[DepToken]],
    lines: list[str],
    prefix: str,
    is_last: bool,
) -> None:
    """Özyinelemeli ASCII ağaç oluşturucu."""
    connector = "└── " if is_last else "├── "
    label = f"{node.form} [{node.deprel}]"
    lines.append(f"{prefix}{connector}{label}")

    child_list = sorted(children.get(node.id, []), key=lambda c: c.id)
    new_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(child_list):
        _build_tree_lines(
            child, children, lines, new_prefix, i == len(child_list) - 1,
        )


# ═══════════════════════════════════════════════════════════════════
#  Parser Orkestratörü (DIP — soyut kurallara bağımlı)
# ═══════════════════════════════════════════════════════════════════


class DependencyParser:
    """Kural-tabanlı bağımlılık çözümleyici (v5 — 20 kural).

    ``SentenceAnalyzer`` çıktısını tüketir, konfigüre edilebilir
    kural zincirini sırayla uygular. Kurallar dışarıdan enjekte
    edilebilir (DIP); varsayılan zincir tüm fazları kapsar.

    v5 kural sırası:
      1. PredicateRule          — root
      2. NominalPredicateRule   — nominal root
      3. PostpositionRule       — case
      4. PossessiveRule         — nmod:poss (genişletilmiş arama)
      5. DeterminerRule         — det
      6. NummodRule             — nummod
      7. FlatNameRule           — flat (BOUN uyumlu)
      8. CoordinationRule       — cc + conj
      9. LightVerbRule          — compound:lvc
     10. CompoundNounRule       — compound (belirtisiz tamlama)
     11. ConverbRule            — advcl
     12. ParticipleRule         — acl (scope-aware)
     13. TemporalAdvmodRule     — obl:tmod
     14. AdvmodEmphRule         — advmod:emph (de/da/bile)
     15. CopulaRule             — cop (değil/idi/imiş)
     16. AdjAdvDisambiguationRule — amod/advmod
     17. AdvmodRule             — advmod
     18. CaseRoleRule           — nsubj/obj/obl
     19. AdjectiveRule          — amod
     20. FallbackRule           — dep

    MultiPredicateRule tanımlı fakat zincirde devre dışı — root
    çakışması riski nedeniyle (gelecek sürüm için hazır).
    """

    def __init__(
        self,
        rules: Sequence[DependencyRule] | None = None,
    ) -> None:
        self._rules: list[DependencyRule] = (
            list(rules) if rules else self._default_rules()
        )
        self._last_trace: list[dict] = []

    @staticmethod
    def _default_rules() -> list[DependencyRule]:
        """Varsayılan kural zincirini üretir (v5 — 20 kural)."""
        return [
            # 1-2: Yüklem (en yüksek öncelik)
            PredicateRule(),
            NominalPredicateRule(),
            # 3: Edatlar (hal ilişkisi — NP'den önce)
            PostpositionRule(),
            # 4-7: İsim öbeği iç yapısı
            PossessiveRule(),
            DeterminerRule(),
            NummodRule(),
            FlatNameRule(),
            # 8: Koordinasyon
            CoordinationRule(),
            # 9-10: Bileşik yapılar
            LightVerbRule(),
            CompoundNounRule(),
            # 11-12: Yan cümleler
            ConverbRule(),
            ParticipleRule(),
            # 13-15: Özel yapılar (CaseRoleRule'dan önce)
            TemporalAdvmodRule(),
            AdvmodEmphRule(),
            CopulaRule(),
            # 16-19: Genel bağımlılıklar
            AdjAdvDisambiguationRule(),
            AdvmodRule(),
            CaseRoleRule(),
            AdjectiveRule(),
            # 20: Fallback
            FallbackRule(),
        ]

    # ── Ana API ───────────────────────────────────────────────────

    def parse(
        self,
        sentence_tokens: list[SentenceToken],
        *,
        trace: bool = False,
    ) -> list[DepToken]:
        """SentenceToken listesini bağımlılık ağacına dönüştürür.

        Args:
            sentence_tokens: Morfolojik çözümleme çıktısı.
            trace: True ise kural uygulamalarını loglar, self._last_trace'e yazar.
        """
        if not sentence_tokens:
            self._last_trace = []
            return []

        dep_tokens = [
            DepToken.from_sentence_token(st, i + 1)
            for i, st in enumerate(sentence_tokens)
        ]

        trace_log: list[dict] = []

        for rule in self._rules:
            if trace:
                before = [(t.head, t.deprel) for t in dep_tokens]

            applied = rule.apply(dep_tokens)

            if trace:
                after = [(t.head, t.deprel) for t in dep_tokens]
                changes = []
                for idx, (b, a2) in enumerate(zip(before, after)):
                    if b != a2:
                        changes.append({
                            "token": dep_tokens[idx].form,
                            "before": {"head": b[0], "deprel": b[1]},
                            "after": {"head": a2[0], "deprel": a2[1]},
                        })
                entry = {
                    "rule": rule.__class__.__name__,
                    "applied": applied,
                    "changes": changes,
                }
                trace_log.append(entry)
                if changes:
                    logger.debug(
                        "%s: %s",
                        rule.__class__.__name__,
                        ", ".join(a for a in applied),
                    )

        if trace:
            self._last_trace = trace_log

        return dep_tokens

    # ── Çıktı Formatları ──────────────────────────────────────────

    @staticmethod
    def to_conllu(dep_tokens: list[DepToken], text: str = "") -> str:
        """CoNLL-U formatında çıktı üretir."""
        lines: list[str] = []
        if text:
            lines.append(f"# text = {text}")
        for t in dep_tokens:
            fields = [
                str(t.id), t.form, t.lemma, t.upos, t.xpos,
                t.feats_str, str(t.head), t.deprel, t.deps, t.misc,
            ]
            lines.append("\t".join(fields))
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_tree(dep_tokens: list[DepToken]) -> str:
        """ASCII ağaç gösterimi üretir."""
        if not dep_tokens:
            return "(boş)"

        root = None
        children: dict[int, list[DepToken]] = {}
        for t in dep_tokens:
            children.setdefault(t.head, []).append(t)
            if t.deprel == "root":
                root = t

        if not root:
            return "(root bulunamadı)"

        lines: list[str] = []
        _build_tree_lines(root, children, lines, prefix="", is_last=True)
        return "\n".join(lines)
