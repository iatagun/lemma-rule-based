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

# Çekimli fiil ekleri — FallbackRule'da conj tespiti için
_FINITE_TENSE_LABELS: frozenset[str] = frozenset({
    "GEÇMİŞ_ZAMAN", "DUYULAN_GEÇMİŞ", "GELECEK_ZAMAN",
    "ŞİMDİKİ_ZAMAN", "GENİŞ_ZAMAN", "GENİŞ_ZAMAN_OLMSZ",
    "KİŞİ_1T", "KİŞİ_2T", "KİŞİ_1Ç", "KİŞİ_2Ç", "KİŞİ_3Ç",
})

PARTICIPLE_LABELS: frozenset[str] = frozenset({
    "SIFAT_FİİL", "SIFAT_FİİL_-DIk", "SIFAT_FİİL_-DIğ",
})

CONVERB_LABELS: frozenset[str] = frozenset({
    "ZARF_FİİL_-ArAk", "ZARF_FİİL_-IncA",
    "ZARF_FİİL_-Ip", "ZARF_FİİL_-ken",
})

# Form-tabanlı zarf-fiil tespiti (morph analyzer bu ekleri tanımayabiliyor)
_CONVERB_FORM_RE: re.Pattern[str] = re.compile(
    r"(?:"
    r"[aeıioöuü]rken$"         # -Irken/-Arken (yaparken, gülerken, olurken)
    r"|[ıiuü]nc[ae]$"          # -IncA (ölünce, gelince, yapınca)
    r"|m[ae]d[ae]n$"            # -mAdAn (yemeden, gitmeden)
    r"|[dt][ıiuü]k[çc][ae]$"   # -DIkçA (yaklaştıkça, geldikçe)
    r")", re.IGNORECASE
)

# ── Türetim eki tabanlı UPOS tespiti ──────────────────────────────
# Uzun et al. (1992) "Türkiye Türkçesinin Türetim Ekleri" referansı.
# Sıfat (ADJ) türeten en sık türetim ekleri:
#   -lI (Ç39 İ7C): güçlü, mutlu, tuzlu, anlamlı, yetkili (1259+ örnek)
#   -sIz (İ72): güçsüz, evsiz, sessiz, anlamsız (150+ örnek)
#   -sAl (İ70): ulusal, bilimsel, evrensel, toplumsal (80+ örnek)
#   -CI (Ç39): yolcu, dışçı (1259 — çoğunlukla AD ama ADJ kullanımı da var)
#   -(I)msI: yeşilimsi, mavimsi (nadir)
_DERIV_ADJ_RE: re.Pattern[str] = re.compile(
    r"(?:"
    r".{3,}[lL][ıiuü]$"        # -lI: güçlü, mutlu, anlamlı (taban min 3 harf)
    r"|.{2,}s[ıiuü]z$"         # -sIz: sessiz, evsiz (taban min 2 harf)
    r"|.{2,}s[ae]l$"            # -sAl: ulusal, bilimsel (taban min 2 harf)
    r"|.{3,}[ıiuü]ms[ıiuü]$"  # -(I)msI: yeşilimsi, mavimsi (taban min 3 harf)
    r")", re.IGNORECASE
)

# Lokasyon-ilgi sıfatı: BULUNMA(-DA) + ki → sıfat işlevi
# arasındaki, altındaki, önündeki, bendeki, evindeki vb.
_DAKI_ADJ_RE: re.Pattern[str] = re.compile(
    r".{2,}[dt][ae]ki$", re.IGNORECASE
)

# Bileşik geçmiş zaman formu tespiti — İYELİK_3T/BELIRTME altında gizli fiil
# Morfolojik çözümleyici bu yapıları tanımadığında yalın isim gibi görünür
_HIDDEN_PAST_VERB_RE: re.Pattern[str] = re.compile(
    r"(?:"
    r".{3,}[yY]ord[uü]$"                   # -yordu (şimdiki+hikâye: diyordu)
    r"|.{3,}m[ıiuü]ş[tT][ıiuü]$"          # -mIştI (duyulan+hikâye: gerekmişti)
    r"|.{3,}[aeıioöuü]r[dt][ıiuü]$"        # -VrdI (geniş+hikâye: olurdu, severdi)
    r"|.{4,}[yY][dt][ıiuü]$"               # -ydI (kopula+hikâye: Çiftçiydi)
    r"|.{3,}[aeıioöuü]lm[ıiuü]ş$"         # -AlmIş (yeterlk+duyulan: görebilmiş)
    r")", re.IGNORECASE
)

# -mIş formu tespiti — İŞTEŞ olarak yanlış etiketlenen duyulan geçmiş/sıfat-fiil
# kurulmuş, yazılmış, açılmış, demiş, gelmiş vb.
_MIS_VERB_RE: re.Pattern[str] = re.compile(
    r".{2,}m[ıiuü]ş$", re.IGNORECASE
)

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


def _has_only_derivational(token) -> bool:
    """Token yalnızca yapım eki taşıyor mu (çekim eki yok)?"""
    if not token._suffixes:
        return False
    return all(
        label.startswith("YAPIM_") for _, label in token._suffixes
    )


DETERMINERS: frozenset[str] = frozenset({
    "bir", "bu", "şu", "o", "her", "bazı", "birçok",
    "tüm", "bütün", "hiçbir", "birkaç", "kimi", "öbür",
    "öteki", "hangi", "kaç",
})

CONJUNCTIONS: frozenset[str] = frozenset({
    "ve", "veya", "ya", "ama", "fakat", "ancak", "ile", "ne",
    "hem", "oysa", "hâlbuki", "halbuki", "gerek",
    "yani", "örneğin", "hatta", "eğer",
})

# Morfolojik çözümleme ne derse desin daima CCONJ olan sözcükler.
# "hatta"→hat+ta, "yani"→yan+i, "veya"→ve+ya gibi sahte çözümlemeleri override eder.
_ALWAYS_CCONJ: frozenset[str] = frozenset({
    "veya", "ama", "fakat", "ancak", "oysa",
    "hâlbuki", "halbuki", "yani", "hatta",
})

POSTPOSITIONS: frozenset[str] = frozenset({
    "için", "gibi", "kadar", "göre", "karşı", "rağmen", "dair",
    "üzere", "doğru", "dolayı", "itibaren", "beri", "hakkında",
    "ait", "ilişkin", "karşın", "boyunca", "önce", "sonra",
    "olarak", "diye",
})

# Bilinen sıfatlar — sözlük/POS etiketi olmadığında UPOS çıkarımı için
COMMON_ADJECTIVES: frozenset[str] = frozenset({
    # Temel nitelik
    "güzel", "iyi", "kötü", "büyük", "küçük", "yeni", "eski",
    "uzun", "kısa", "genç", "yaşlı", "doğal", "yapay",
    "önemli", "farklı", "ağır", "hafif", "sıcak", "soğuk",
    "parlak", "koyu", "açık", "kapalı", "zengin", "fakir",
    "mutlu", "mutsuz", "hızlı", "güçlü", "zayıf",
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
    # v7 ekleme — amod→obj hata analizinden
    "türlü", "başarılı", "büyülü", "sinsi", "faydalı",
    "kuvvetli", "politik", "kronik", "ideolojik", "feodal",
    "bembeyaz", "sivri", "milli", "yerleşik", "organik",
    "stratejik", "diplomatik", "sistematik", "otoriter",
    "birinci", "ikinci", "üçüncü",
    # v9 ekleme — BOUN frekans analizi + türetim eki kitabı
    "rahatsız", "altın", "yarım", "çeşitli", "olumsuz",
    "gizli", "sözde", "organize", "karışık", "birtakım",
    "kocaman", "kara", "taze", "kritik", "dönük", "katlı",
    "korkunç", "sayın", "yakışıklı", "muhtemel", "evrensel",
    "sanal", "geçen", "şeffaf", "yoğunlaşmış", "sürdürülebilir",
    "somutlaşmış", "mevzii", "vicdani", "uluslararası",
    # v13 ekleme — BOUN amod frekans analizi
    "türk", "siyasi", "toplumsal",
    "müthiş", "anonim", "engin", "ölümcül", "yetersiz",
    "durgun", "doygun", "çekingen", "nadir", "zarif",
    "alışılmış", "tanıdık", "kararlı", "bağımsız",
    "fransız", "alman", "arap", "kürt", "rum", "ermeni",
    # v16 ekleme — UPOS/deprel hata analizi
    "favori", "mahalli", "çekinik", "yıllık", "yardımcı",
    "ilgili", "etik", "yepyeni", "bağlı", "parasız",
    "nazik", "rasyonel", "emin", "estetik", "kurulu",
    "yaklaşık", "dengeli", "görüntülü", "zanlı", "güleryüzlü",
    "ölümlü", "şekerli", "huzurlu", "yetkili", "coşkulu",
    "küresel", "katı", "yoğun", "acil",
    "kalıcı", "geçici", "günlük", "aylık", "haftalık",
    "yüzlük", "sınırlı", "bilinçli", "sağlıklı", "doğal",
    "ulvi", "zahiri", "batıni", "müstakil", "münferit",
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
    "neredeyse", "herhalde", "yeterince",
    # Odaklama / sınırlama
    "bile", "sadece", "yalnız", "ancak",
    "özellikle", "yalnızca",
    # Soru / belirsizlik
    "belki", "acaba", "nasıl",
    # Olumsuzluk
    "asla",
    # Yön
    "içeri", "dışarı", "ileri", "geri",
    # Yer (demonstratif zarflar)
    "burada", "orada", "şurada", "buraya", "oraya", "şuraya",
    "buradan", "oradan", "şuradan", "nerede", "nereye", "nereden",
    # Üstelik / ekleme
    "üstelik", "ayrıca", "dahası",
    # v11 ekleme — BOUN frekans analizi
    "böylece", "tamamen", "yeniden", "sanki", "ardından",
    "sonradan", "baştan", "henüz", "meselâ", "mesela",
    # v16.13 ekleme — BOUN gold=ADV frekans analizi
    "yıllarca", "birdenbire", "öylece", "aheste", "usul",
    "yavaş", "peki", "sizce", "nedense", "meğerse",
    "yoktu", "çabuk",
})

# Sık fiil-olarak-yanlış-çözümlenen isimler — _infer_upos'ta VERB→NOUN düzeltme
# Morfolojik çözümleyici bu sözcükleri fiil+ek olarak parse eder:
#   zaman → zam+an (GENİŞ_ZAMAN), bilim → bil+im (MASTAR) vb.
# Tam sözcük eşleşmesi olduğunda NOUN olarak kabul et.
COMMON_NOUNS: frozenset[str] = frozenset({
    # -An sonlu (GENİŞ_ZAMAN/SIFAT_FİİL ile çakışan)
    "zaman", "başkan", "ozan", "divan", "meydan", "kazan",
    "düzen", "neden", "güven", "yüzen", "başbakan", "hayran",
    # -Im/-Um sonlu (MASTAR ile çakışan)
    "bilim", "eğitim", "geçim", "toplum", "hücum", "önlem",
    "eylem", "devam", "teslim", "temsil", "yatırım", "ikram",
    "yaşam", "üretim", "işlem", "program", "dilim",
    # -Ar/-Er sonlu (GENİŞ_ZAMAN ile çakışan)
    "karar", "pazar", "sınır", "şeker", "kültür", "değer",
    "hayır", "asır",
    # -Iş sonlu (İŞTEŞ ile çakışan)
    "artış", "bakış", "çıkış", "giriş", "dönüş", "yürüyüş",
    "buluş",
    # -İp/-Up sonlu
    "takip", "sahip",
    # -mAk sonlu (MASTAR ile çakışan)
    "yemek",
    # Postpozisyonel isimler
    "tarafından", "bakımından", "yüzünden", "yönünden",
    # var/yok ailesi (BOUN: NOUN olarak etiketler)
    "vardır", "yoktur",
    # Diğer yanlış-çözümlemeler
    "akşam", "parmak", "albüm", "helikopter", "deniz",
    "politika", "yardım", "bölge", "dahil", "insan",
    "adam", "tahmin", "talep", "taraf", "kereviz",
    "satanizm", "kader",
    # v17 ekleme — UPOS VERB→NOUN düzeltme (base-form, sözlükte, 0 break)
    "yüzden", "aslan", "şehir", "bayram", "zafer", "kasım",
    "sultan", "hanım", "kilise", "giyecek", "astım", "esmer",
    "baldır", "ayran", "düşünce", "reform", "deprem", "biber",
    "potasyum", "duman", "kibir", "katılım", "leopar", "sonbahar",
    "kalorifer", "huzur", "lüfer", "karabiber", "kırmızıbiber",
    "sekreter", "önder", "sevecen", "karakteristik", "cezasız",
    # Büyük harfli kurum/unvan isimleri — PROPN false-positive azaltmak
    "devlet", "güney", "mayıs", "mart", "meclis", "genelkurmay",
    "cumhurbaşkan", "savcı", "dışişleri", "bakan",
    # EDİLGEN false-positive: -il/-ul sonlu isimler (gön+ül→EDİLGEN hatası)
    "gönül", "kurul", "tatil", "varil", "tahıl", "cahil", "temsil",
    # Yaygın kök isimler (çekimli formları VERB oluyor)
    "yan", "gün", "yıl", "su", "ateş", "kapı", "yer",
    "sıra", "otel", "anne", "baba", "kız",
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
    "onlar", "onları", "onlara", "onlarda", "onlardan", "onların",
    # İşaret zamiri çekimleri (bu/şu/o → bun-/şun-)
    "bunu", "bunun", "buna", "bundan", "bunla", "bununla",
    "bunlar", "bunları", "bunların", "bunlara", "bunlarda", "bunlardan",
    "şunu", "şunun", "şuna", "şundan", "şununla",
    "şunlar", "şunları", "şunların", "şunlara", "şunlarda",
    # Dönüşlü zamir (kendi) genişletme
    "kendi", "kendisi", "kendim", "kendin", "kendimiz", "kendiniz",
    "kendisine", "kendisini", "kendisinin", "kendisinden", "kendisiyle",
    "kendime", "kendimi", "kendimin", "kendimden", "kendimle",
    "kendine", "kendini", "kendinin", "kendinden", "kendinle",
    "kendileri", "kendilerine", "kendilerini", "kendilerinin",
    # Soru zamirleri
    "kim", "kimi", "kime", "kimde", "kimden", "kimin", "kimse",
    "ne", "neyi", "neye", "nede", "neden", "neyin",
    "neler", "neleri", "nelere", "nelerden", "nelerin",
    "nere", "nereye", "nerede", "nereden", "nereyi", "nerenin",
    # Yer zamirleri
    "buraya", "burada", "buradan", "burayı", "buranın",
    "şuraya", "şurada", "şuradan",
    "oraya", "orada", "oradan", "orayı", "oranın",
    # Belgisiz zamirler
    "herkes", "herkesi", "herkese", "herkeste", "herkesten", "herkesin",
    "hepsi", "hepsini", "hepsine", "hepsinden", "hepsinin",
    "hepimiz", "hepiniz",
    "hiçbiri", "hiçbirini", "hiçbirine", "hiçbirinden", "hiçbirinin",
    "birisi", "birisine", "birisini", "birisinin",
    "biri", "birini", "birine", "birinde", "birinden", "birinin",
    "birbirimiz", "birbiri", "birbirine", "birbirini", "birbirinin",
    "birbirinden", "birbirlerini", "birbirlerine",
    "hangisi", "hangileri", "hangisine", "hangisini",
})

# Zamir kökleri — çekimli formları tanımak için
PRONOUN_STEMS: frozenset[str] = frozenset({
    "ben", "sen", "o", "biz", "siz", "onlar",
    "kendi", "kim", "ne", "nere",
    "bura", "şura", "ora",
    "bun", "şun",  # bu/şu → bun-/şun- çekim kökü
    "herkes", "hep", "hiçbir", "biri", "birbir",
    "hangi",
})

# Soru partikülleri — UPOS=PART veya AUX
QUESTION_PARTICLES: frozenset[str] = frozenset({
    "mi", "mı", "mu", "mü",
    "mısın", "misin", "musun", "müsün",
    "mıyız", "miyiz", "muyuz", "müyüz",
    "mısınız", "misiniz", "musunuz", "müsünüz",
    "midir", "mıdır",
    "mıydı", "miydi", "muydu", "müydü",
    "mıymış", "miymiş", "muymuş", "müymüş",
})

# Odaklama partikülü — BOUN'da PART olarak etiketlenen de/da/ki
_FOCUS_PARTICLES: frozenset[str] = frozenset({"de", "da", "ki"})

# Odaklama / pekiştirme edatları — advmod:emph
EMPHASIS_PARTICLES: frozenset[str] = frozenset({
    "de", "da", "bile", "dahi", "sadece", "yalnızca",
    "özellikle", "yalnız",
})

# Ünlemler — UPOS=INTJ
INTERJECTIONS: frozenset[str] = frozenset({
    "evet", "hayır", "tamam", "peki", "hay",
    "eyvah", "oh", "ah", "vah", "bravo", "aman",
    "maalesef", "lütfen", "merhaba", "güle",
    "haydi", "hadi", "hey", "ey", "sakın", "hooop",
    "tey", "ha", "aaa", "tabi",
})

# Bağımlama edatları — UPOS=SCONJ
SUBORDINATORS: frozenset[str] = frozenset({
    "ki", "diye", "çünkü", "madem", "mademki",
    "şayet", "oysaki",
})

# Hafif fiil + isimleştirme kalıbı (etmeyi, yapması, olduğu, …)
_LVC_NOM_RE = re.compile(
    r"^(et|yap|ol|kıl|buyur|eyle)(me|ma|iş|ış|uş|üş)",
    re.IGNORECASE,
)

# "ol" ile başlayan ama hafif fiil olmayan sözcükler
_OL_BLACKLIST: frozenset[str] = frozenset({
    "olay", "olağan", "olağanüstü", "olası", "olasılık", "oluşum",
    "olgu", "oluk", "olumlu", "olumsuz", "olarak",
})

# Hafif fiil prefix eşlemesinden hariç tutulan formlar
_LVC_FORM_BLACKLIST: frozenset[str] = frozenset({
    "olarak",   # ADP/case — "X olarak" = "as X"
    "yapısı", "yapısını", "yapısının", "yapısında",  # yapı (structure)
    "yapılan", "yapılır", "yapılmış",  # yapıl- (passive of yap) — genellikle acl
    "etik", "etiket",  # bağımsız sözcükler
})

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
    # Virgül bilgisi: bu tokendan sonra virgül var mı?
    has_comma_after: bool = field(default=False, repr=False)

    # ── Fabrika Metodu ────────────────────────────────────────────

    @classmethod
    def from_sentence_token(cls, st: SentenceToken, idx: int,
                            is_first: bool = False) -> DepToken:
        """SentenceToken'dan DepToken oluşturur."""
        a = st.analysis
        feats = _extract_feats(a) if a else {}
        upos = _infer_upos(st, feats, is_first=is_first)

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


def _infer_upos(st: SentenceToken, feats: dict[str, str],
                is_first: bool = False) -> str:
    """Morfolojik çözümleme + bağlam ipuçlarından UPOS tahmin eder."""
    w = turkish_lower(st.word)
    a = st.analysis

    # Zamir — çekimli formlar dahil
    if w in PRONOUNS:
        return "PRON"
    # Zamir kökü + ek (onu, onun, benim, seni, …)
    if a and a.stem in PRONOUN_STEMS and a.suffixes:
        return "PRON"

    # Soru partikülü — BOUN tutarlı olarak AUX + discourse:q etiketler
    if w in QUESTION_PARTICLES:
        return "AUX"

    # Odaklama partikülü (de/da/ki) — BOUN'da PART olarak etiketlenir
    if w in _FOCUS_PARTICLES and not (a and a.suffixes):
        return "PART"

    # Bağımlama edatı
    if w in SUBORDINATORS and not (a and a.suffixes):
        return "SCONJ"

    # Ünlem
    if w in INTERJECTIONS and not (a and a.suffixes):
        return "INTJ"

    if w in DETERMINERS and not (a and a.suffixes):
        return "DET"
    if w in _ALWAYS_CCONJ:
        return "CCONJ"
    if w in CONJUNCTIONS and not (a and a.suffixes):
        return "CCONJ"
    if w in POSTPOSITIONS and not (a and a.suffixes):
        return "ADP"
    if (w in NUMERALS or re.match(r"^\d+$", st.word)) and not (a and a.suffixes):
        return "NUM"

    # Zarf ve sıfat: tam sözcük eşleşmesi — morfolojik çözümleme hatalı
    # olabilir ("daha" → dah+a/YÖNELME, "yeni" → yen+i/İYELİK gibi).
    # Ek koşulu kaldırıldı: tam eşleşme morfolojik hatayı override eder.
    if w in COMMON_ADVERBS:
        return "ADV"
    if w in COMMON_ADJECTIVES:
        return "ADJ"

    # Fiil olarak yanlış çözümlenen isimler: tam sözcük veya kök eşleşmesi
    # ("zaman"→zam+an/GENİŞ_ZAMAN, "bilim"→bil+im/MASTAR,
    #  "insanlar"→insan+lar ama başka parse'da fiil gibi görünüyor vb.)
    if w in COMMON_NOUNS:
        return "NOUN"
    if a and a.stem and a.stem.lower() in COMMON_NOUNS:
        # Fiil çekim eki (sıfat-fiil, zarf-fiil, zaman/kip) varsa NOUN zorlama
        _verb_use = VERB_FINAL_LABELS | PARTICIPLE_LABELS | CONVERB_LABELS
        has_verb_sfx = a.suffixes and any(
            sub in _verb_use
            for _, lbl in a.suffixes for sub in lbl.split("/")
        )
        if not has_verb_sfx:
            return "NOUN"

    # Çoğul isim tespiti: -lar/-ler soneki + ÇOĞUL etiketi → NOUN
    # "insanlar", "adamlar" gibi formlar fiil olarak yanlış çözümlenir
    if a and a.suffixes:
        has_cogul = any("ÇOĞUL" in lbl for _, lbl in a.suffixes)
        if has_cogul and not any(
            sub in VERB_FINAL_LABELS
            for _, lbl in a.suffixes
            for sub in lbl.split("/")
            if sub != "ÇOĞUL"
        ):
            return "NOUN"

    # ── Büyük harf tabanlı PROPN tespiti (erken) ────────────────
    # Apostrof + büyük harf → PROPN (Türkiye'nin, İstanbul'a, Atatürk'ün)
    has_apostrophe = "'" in st.word or "\u2019" in st.word
    if has_apostrophe and st.word and st.word[0].isupper():
        return "PROPN"
    # Cümle-içi büyük harf + ek yok → PROPN (Yugoslav, MGK, AB, Cook)
    if not is_first and st.word and st.word[0].isupper():
        if not a or not a.suffixes:
            return "PROPN"

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

    # Bileşik geçmiş zaman tespiti:
    # İYELİK_3T/BELIRTME altında gizli hikâye bileşik zamanları
    # diyordu, gerekmişti, olurdu, Çiftçiydi vb.
    if has_iyelik_belirtme and _HIDDEN_PAST_VERB_RE.search(w):
        return "VERB"

    # Form-tabanlı zarf-fiil tespiti: -ken, -mAdAn, -DIkçA vb.
    if _CONVERB_FORM_RE.search(w):
        return "VERB"

    # ── Lokasyon-ilgi sıfatı: -DAki ──────────────────────────────
    # arasındaki, altındaki, önündeki vb. → ADJ
    # BULUNMA(-de/-da) + ki yapısı → sıfat işlevi
    if _DAKI_ADJ_RE.search(w):
        return "ADJ"

    # ── Türetim eki tabanlı UPOS tespiti ──────────────────────────
    # Uzun et al. (1992) "Türkiye Türkçesinin Türetim Ekleri" referansıyla
    # en sık ADJ türeten eklere bakarak UPOS belirleme.
    # KOŞUL: Sözcüğün morfolojik çözümlemesinde hal/iyelik eki OLMAMALI.
    # Yoksa "yılı"(yıl+ı/İYELİK), "hali"(hal+i/İYELİK) gibi formlar
    # yanlışlıkla ADJ olarak etiketlenir.
    has_case_or_poss = False
    for _, label in a.suffixes:
        subs = label.split("/")
        for s in subs:
            if s in CASE_LABELS or s in IYELIK_LABELS or s == "ÇOĞUL":
                has_case_or_poss = True
                break
        if has_case_or_poss:
            break
    if not has_case_or_poss and _DERIV_ADJ_RE.search(w):
        return "ADJ"

    # ── Büyük harf tabanlı PROPN tespiti (ekli formlar) ──────────
    # Cümle-içi büyük harf: hal/iyelik ekli olsa bile PROPN
    # (Türkiye+YÖNELME, Wonka+YÖNELME, Meclisi+İYELİK vb.)
    # Apostrof kontrolü yukarıda yapıldı (erken çıkış).
    if not is_first and st.word and st.word[0].isupper():
        return "PROPN"

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
        "VASITA": "obl",
    }

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []

        applied: list[str] = []
        # Her yüklem için nsubj atanmış mı takip et
        nsubj_per_pred: set[int] = {
            t.head for t in tokens if t.deprel == "nsubj"
        }

        # Yüklem→pro-drop haritası (1./2. kişi eki varsa özne düşmüş)
        _prodrop_preds: set[int] = set()
        for t in tokens:
            if t.deprel == "root" or t.id == root_id:
                if t.has_any_label(PRODROP_PERSON_LABELS):
                    _prodrop_preds.add(t.id)
            elif t.deprel in ("advcl", "ccomp", "acl", "csubj"):
                if t.has_any_label(PRODROP_PERSON_LABELS):
                    _prodrop_preds.add(t.id)

        # İyelik başı olan token'ları bul (nmod:poss bağımlısı var)
        poss_heads = frozenset(
            t.head for t in tokens if t.deprel == "nmod:poss"
        )

        # Edat bağımlısı olan isimler
        has_case_child = self._tokens_with_case_child(tokens)

        for t in tokens:
            if t.is_assigned or t.id == root_id:
                continue

            # Yerel yüklem: çok-yüklemli cümlelerde en yakın sağ yüklemi bul
            local_pred = _find_local_predicate(tokens, t.id, root_id)

            # 1) İyelik başı → BELIRTME eki iyelik fonksiyonunda
            #    Hal eki olarak sayma, yalın gibi değerlendir
            if t.id in poss_heads and t.has_iyelik:
                role = self._detect_case_role_skip_belirtme(t)
                if role:
                    t.head = local_pred
                    t.deprel = role
                    applied.append(f"HAL→{role.upper()}")
                    continue
                # BELIRTME dışında hal eki yok → yalın gibi davran
                is_prodrop = local_pred in _prodrop_preds
                if local_pred not in nsubj_per_pred and not is_prodrop:
                    t.head = local_pred
                    t.deprel = "nsubj"
                    nsubj_per_pred.add(local_pred)
                    applied.append("İYELİK_BAŞI→NSUBJ")
                else:
                    t.head = local_pred
                    t.deprel = "obj" if is_prodrop else "nsubj"
                    if not is_prodrop:
                        nsubj_per_pred.add(local_pred)
                    applied.append("İYELİK_BAŞI→OBJ" if is_prodrop else "İYELİK_BAŞI→NSUBJ")
                continue

            # 2) Hal eki → doğrudan görev eşleme
            #    İsimleşmiş fiil: VERB + BELIRTME → ccomp (tümleç yan cümlesi)
            #    geldiğini biliyorum → geldiğini = ccomp
            #    ADJ/ADV/DET: hal eki morph artifact olabilir → atla
            #    (AdjectiveRule / AdvmodRule bunları sonra yakalar)
            if t.upos in ("ADJ", "ADV", "DET"):
                continue

            # Bilinen sıfatlar + sağda isim → sıfat tamlayıcısı olma ihtimali
            # CaseRoleRule'un obj/nsubj atamasından muaf tut
            if t.form.lower() in COMMON_ADJECTIVES:
                right = self._right_content_word(tokens, t.id)
                if right and right.upos in ("NOUN", "PROPN"):
                    continue

            role = self._detect_case_role(t)
            if role:
                if role == "obj" and t.upos == "VERB":
                    t.head = local_pred
                    t.deprel = "ccomp"
                    applied.append("FİİL_BELIRTME→CCOMP")
                else:
                    t.head = local_pred
                    t.deprel = role
                    applied.append(f"HAL→{role.upper()}")
                continue

            # 3) Edat bağımlısı var → obl ("ev için" → ev=obl)
            if t.id in has_case_child:
                t.head = local_pred
                t.deprel = "obl"
                applied.append("EDAT_BAĞIMLI→OBL")
                continue

            # 4) Yalın isim/zamir → belirlilik hiyerarşisi
            #    Türkçe'de yalın ortak isim = belirtisiz nesne (kitap okudu)
            #    Özel isim / zamir = belirli → özne adayı
            #    Per-predicate nsubj: her yüklem kendi öznesini alabilir
            if t.upos in ("NOUN", "PROPN", "PRON") and not t.has_case:
                is_definite = t.upos in ("PROPN", "PRON") or t.has_iyelik
                is_prodrop = local_pred in _prodrop_preds
                pred_has_nsubj = local_pred in nsubj_per_pred
                if is_prodrop:
                    if is_definite and not pred_has_nsubj:
                        t.head = local_pred
                        t.deprel = "nsubj"
                        nsubj_per_pred.add(local_pred)
                        applied.append("BELİRLİ_PRODROP→NSUBJ")
                    else:
                        t.head = local_pred
                        t.deprel = "obj"
                        applied.append("PRODROP→OBJ")
                elif not pred_has_nsubj:
                    t.head = local_pred
                    t.deprel = "nsubj"
                    nsubj_per_pred.add(local_pred)
                    applied.append("YALIN→NSUBJ")
                else:
                    if is_definite:
                        t.head = local_pred
                        t.deprel = "obj"
                        applied.append("BELİRLİ→OBJ")
                    else:
                        t.head = local_pred
                        t.deprel = "obj"
                        applied.append("BELİRTİSİZ→OBJ")

        return applied

    @staticmethod
    def _tokens_with_case_child(tokens: list[DepToken]) -> frozenset[int]:
        """Edat (case) bağımlısı olan token id'lerini döndürür."""
        return frozenset(
            t.head for t in tokens if t.deprel == "case"
        )

    @staticmethod
    def _right_content_word(tokens: list[DepToken], tid: int) -> DepToken | None:
        """Token'ın sağındaki ilk içerik sözcüğünü bulur (DET/NUM atlayarak)."""
        found = False
        for tk in tokens:
            if tk.id <= tid:
                continue
            if tk.upos in ("DET", "NUM", "ADJ"):
                continue
            return tk
        return None

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
    Zamir tamlayan: benim/senin/onun/bizim/sizin + İYELİK → nmod:poss

    Strateji: TAMLAYAN ekli veya tamlayan zamiri olan sözcüğü
    sağdaki İYELİK ekli sözcüğe bağla.
    Örnek: 'ülkenin geleceği' → ülkenin ──nmod:poss──▶ geleceği
           'benim kitabım'   → benim ──nmod:poss──▶ kitabım
    """

    # Tamlayan zamirleri — morph analyzer TAMLAYAN etiketi üretmeyebilir
    _GEN_PRONOUNS: frozenset[str] = frozenset({
        "benim", "senin", "onun", "bizim", "sizin", "onların",
        "bunun", "şunun", "bunların", "şunların",
        "kendi", "kendinin", "kendisinin",
    })

    # Arama sırasında atlanabilecek UPOS türleri
    _SKIP_UPOS: frozenset[str] = frozenset({
        "ADJ", "DET", "NUM", "NOUN", "PROPN", "ADV",
        "CCONJ", "PART",
    })

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned:
                continue
            is_gen = (t.has_label("TAMLAYAN")
                      or t.form.lower() in self._GEN_PRONOUNS)
            if not is_gen:
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
            if t.is_assigned:
                continue
            # Sadece ADJ-UPOS tokenlar → güvenli amod ataması
            if t.upos != "ADJ":
                continue
            # Strateji 1: ekli isim sağda → amod
            head = self._find_right_inflected_noun(tokens, i)
            if head:
                t.head = head.id
                t.deprel = "amod"
                applied.append("ADJ→AMOD")
                continue
            # Strateji 2: sağda bare/inflected NOUN
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
        """Sağdaki ilk ekli ismi bulur; araya sıfat/det/CCONJ girebilir."""
        for j in range(start + 1, len(tokens)):
            t = tokens[j]
            if t.is_nominal_head and t._suffixes:
                return t
            if t.upos == "CCONJ":
                # Sıfat koordinasyonunda CCONJ atla (ADJ CCONJ ADJ NOUN)
                if j + 1 < len(tokens) and tokens[j + 1].upos in ("ADJ", "ADV"):
                    continue
                break
            if t.upos in ("VERB", "ADP"):
                # BİLDİRME ekli fiiller nominal baş olarak kabul
                if t.is_nominal_head:
                    return t
                break
        return None

    @staticmethod
    def _find_right_any_noun(
        tokens: list[DepToken], start: int,
    ) -> DepToken | None:
        """Sağdaki ilk ismi bulur (ekli/eksiz); araya ADJ/DET/NUM/CCONJ/ADV girebilir.

        "Nazik ve sıcak bir özür" → Nazik ──amod──▶ özür
        CCONJ yalnızca sonrasında ADJ/ADV geliyorsa atlanır (sıfat koordinasyonu).
        ADV yalnızca derece zarfları (en, çok, daha) için atlanır.

        Not: is_assigned kontrolü yapılmaz — bir isim hem head (obj/nsubj)
        hem de amod target olabilir (UD'de amod bir modifier ilişkisidir).
        """
        for j in range(start + 1, len(tokens)):
            t = tokens[j]
            if t.upos in ("NOUN", "PROPN"):
                return t
            if t.upos in ("ADJ", "DET", "NUM"):
                continue
            # CCONJ atlama: sıfat koordinasyonunda (ADJ CCONJ ADJ NOUN)
            if t.upos == "CCONJ":
                if j + 1 < len(tokens) and tokens[j + 1].upos in ("ADJ", "ADV"):
                    continue
                break
            # ADV atlama: derece zarfları (en güzel, çok büyük)
            if t.upos == "ADV":
                continue
            break
        return None


class ConverbRule(DependencyRule):
    """Zarf-fiilleri ana yükleme advcl olarak bağlar.

    İki sinyal:
      1. Morfolojik etiket: CONVERB_LABELS (ZARF_FİİL_-ArAk vb.)
      2. Form-tabanlı: _CONVERB_FORM_RE (-ken, -mAdAn, -DIkçA vb.)
    Örnek: 'koşarak geldi' → koşarak ──advcl──▶ geldi
           'gülerken düştü' → gülerken ──advcl──▶ düştü
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []
        applied: list[str] = []
        for t in tokens:
            if t.is_assigned or t.id == root_id:
                continue
            is_converb = (t.has_any_label(CONVERB_LABELS)
                          or _CONVERB_FORM_RE.search(t.form.lower()))
            # DİLEK_ŞART: koşul fiilleri → advcl (gelse, yapılsa, dokunulsa)
            if not is_converb and t.has_label("DİLEK_ŞART") and t.upos == "VERB":
                is_converb = True
            if is_converb:
                t.head = root_id
                t.deprel = "advcl"
                applied.append("ZARF_FİİL→ADVCL")
        return applied


class InfinitiveRule(DependencyRule):
    """Mastar fiilleri (-mAk) yerel yükleme csubj olarak bağlar.

    Türkçede mastar fiiller cümle öznesi (csubj) veya
    açık tümleç (xcomp) olabilir.
    Örnek: 'Abartmak gerekiyor' → abartmak ──csubj──▶ gerekiyor
           'Olmak istiyorum'    → olmak ──csubj──▶ istiyorum
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        root_id = _find_root_id(tokens)
        if root_id == 0:
            return []
        applied: list[str] = []
        for t in tokens:
            if t.is_assigned or t.id == root_id:
                continue
            if not t.has_label("MASTAR"):
                continue
            if t.upos != "VERB":
                continue
            local_pred = _find_local_predicate(tokens, t.id, root_id)
            t.head = local_pred
            t.deprel = "csubj"
            applied.append("MASTAR→CSUBJ")
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
            # ADV'ler sıfat-fiil değildir — SIFAT_FİİL etiketi false positive
            if t.upos == "ADV":
                continue
            # Morfolojik etiket VEYA form-tabanlı -mIş tespiti
            is_participle = t.has_any_label(PARTICIPLE_LABELS)
            if not is_participle:
                # -mIş formu: İŞTEŞ olarak yanlış etiketlenmiş olabilir
                if t.upos == "VERB" and _MIS_VERB_RE.search(t.form):
                    is_participle = True
            if not is_participle:
                continue

            # Aşama 1: Sağdaki isme acl bağla
            acl_head = None
            for j in range(i + 1, len(tokens)):
                candidate = tokens[j]
                if candidate.upos in ("NOUN", "PROPN", "PRON"):
                    acl_head = candidate
                    t.head = candidate.id
                    t.deprel = "acl"
                    applied.append("SIFAT_FİİL→ACL")
                    break
                if candidate.upos not in ("ADJ", "DET", "NUM", "ADV"):
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
        # ADV her zaman advmod — hal eki etiketi false positive olabilir
        if t.upos == "ADV":
            return "advmod"
        for sub in t.labels:
            if sub in cls._CASE_TO_ROLE:
                return cls._CASE_TO_ROLE[sub]
        if t.upos in ("NOUN", "PROPN") and not t.has_case:
            return "nsubj"
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
            is_ile = w == "ile"
            if w not in POSTPOSITIONS and not is_ile:
                continue
            if t.is_assigned:
                continue
            if i > 0:
                left = tokens[i - 1]
                t.head = left.id
                t.deprel = "case"
                if not is_ile:
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
        """Zarfın bağlanacağı hedefi belirle: sağ komşu ADJ/ADV → o; aksi → yerel yüklem."""
        if idx + 1 < len(tokens):
            right = tokens[idx + 1]
            if right.upos in ("ADJ", "ADV"):
                return right.id
        return _find_local_predicate(tokens, tokens[idx].id, root_id)


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

    # SCONJ words that function as cc (coordinating) in BOUN
    _SCONJ_AS_CC: frozenset[str] = frozenset({"çünkü"})

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []

        # Faz 1: İlişkili bağlaçlar (hem...hem, ne...ne, ya...ya da)
        # Phase 1'de çiftler doğru eşleşir; sonra Phase 2 tekli CCONJ'ları işler.
        applied.extend(self._handle_correlatives(tokens))

        # Faz 2: Explicit bağlaçlar (ve, veya, ama, fakat)
        for i, t in enumerate(tokens):
            is_cconj = t.upos == "CCONJ"
            is_sconj_cc = (
                t.upos == "SCONJ"
                and turkish_lower(t.form) in self._SCONJ_AS_CC
            )
            if (not is_cconj and not is_sconj_cc) or t.is_assigned:
                continue

            # "ya da" kalıbı: "da"yı atla, fixed olarak bağla
            search_from = i
            da_idx: int | None = None
            if (turkish_lower(t.form) == "ya"
                    and i + 1 < len(tokens)
                    and turkish_lower(tokens[i + 1].form) == "da"
                    and not tokens[i + 1].is_assigned):
                da_idx = i + 1
                search_from = i + 1

            left = self._find_left_conjunct(tokens, i)
            right = self._find_right_conjunct(tokens, search_from, left=left)
            if left and right:
                t.head = right.id
                t.deprel = "cc"
                if da_idx is not None:
                    tokens[da_idx].head = t.id
                    tokens[da_idx].deprel = "fixed"
                # Explicit CCONJ → conj (BOUN: first conjunct = head)
                if right.deprel == "root":
                    # Second conjunct was root → promote first conjunct
                    if not left.is_assigned:
                        left.head = 0
                        left.deprel = "root"
                        right.head = left.id
                        right.deprel = "conj"
                    # else: left already assigned, can't move root safely
                else:
                    right.head = left.id
                    right.deprel = "conj"
                applied.append("BAĞLAÇ→CC+CONJ")

        # Faz 3: Virgüllü koordinasyon (asindeton)
        # "Ali, Ayşe, Mehmet" → Ayşe→conj→Ali, Mehmet→conj→Ali
        applied.extend(self._handle_comma_coordination(tokens))

        return applied

    # Left conjunct: UPOS → which UPOS to seek on the right
    _CONJUNCT_HEAD_UPOS: dict[str, frozenset[str]] = {
        "NOUN": frozenset({"NOUN", "PROPN"}),
        "PROPN": frozenset({"NOUN", "PROPN"}),
        "VERB": frozenset({"VERB"}),
    }

    @staticmethod
    def _find_left_conjunct(tokens: list[DepToken], conj_idx: int) -> DepToken | None:
        """Bağlacın solundaki ilk uygun ögeyi bul."""
        for j in range(conj_idx - 1, -1, -1):
            t = tokens[j]
            if t.upos in ("CCONJ", "DET", "ADP"):
                continue
            return t
        return None

    @classmethod
    def _find_right_conjunct(
        cls,
        tokens: list[DepToken],
        conj_idx: int,
        left: DepToken | None = None,
    ) -> DepToken | None:
        """Bağlacın sağındaki uygun ögeyi bul (UPOS-farkında).

        Türkçe baş-sonda (head-final) dil olduğundan, CCONJ'dan sonra
        önce niteleyiciler (ADJ, NUM, ADV), sonra başlık sözcüğü gelir.
        Sol conjunct NOUN/PROPN veya VERB ise, sağda eşleşen UPOS aranır.
        """
        seek = cls._CONJUNCT_HEAD_UPOS.get(left.upos) if left else None
        first_content: DepToken | None = None
        window_end = min(conj_idx + 6, len(tokens))

        for j in range(conj_idx + 1, window_end):
            t = tokens[j]
            if t.upos in ("CCONJ", "DET"):
                continue
            if first_content is None:
                first_content = t
            if seek and t.upos in seek:
                return t

        return first_content

    def _handle_correlatives(self, tokens: list[DepToken]) -> list[str]:
        """hem...hem, ne...ne, ya...ya (da) kalıplarını işle."""
        applied: list[str] = []
        w_lower = [turkish_lower(t.form) for t in tokens]
        i = 0
        while i < len(tokens) - 2:
            if w_lower[i] in self._CORRELATIVES and not tokens[i].is_assigned:
                corr = w_lower[i]
                # İkinci correlative'i bul
                for j in range(i + 2, len(tokens)):
                    if w_lower[j] == corr and not tokens[j].is_assigned:
                        # "ya...ya da" kalıbı: "da"yı atla, fixed olarak bağla
                        da_idx = None
                        search_start = j
                        if corr == "ya" and j + 1 < len(tokens) and w_lower[j + 1] == "da":
                            da_idx = j + 1
                            search_start = j + 1

                        right1 = self._find_right_conjunct(tokens, i)
                        right2 = self._find_right_conjunct(
                            tokens, search_start, left=right1
                        )
                        if right1 and right2:
                            tokens[i].head = right1.id
                            tokens[i].deprel = "cc:preconj"
                            tokens[j].head = right2.id
                            tokens[j].deprel = "cc:preconj"
                            if da_idx is not None and not tokens[da_idx].is_assigned:
                                tokens[da_idx].head = tokens[j].id
                                tokens[da_idx].deprel = "fixed"
                            if not right2.is_assigned or right2.deprel not in ("root",):
                                right2.head = right1.id
                                right2.deprel = "conj"
                            applied.append("İLİŞKİLİ→CC+CONJ")
                        break
            i += 1
        return applied

    # UPOS grupları: aynı grup içindeki tokenlar koordine olabilir
    _COORD_COMPAT: dict[str, frozenset[str]] = {
        "NOUN": frozenset({"NOUN", "PROPN", "NUM"}),
        "PROPN": frozenset({"NOUN", "PROPN"}),
        "VERB": frozenset({"VERB"}),
        "ADJ": frozenset({"ADJ"}),
        "ADV": frozenset({"ADV"}),
        "NUM": frozenset({"NOUN", "NUM"}),
    }

    @staticmethod
    def _handle_comma_coordination(tokens: list[DepToken]) -> list[str]:
        """Virgüllü asindeton koordinasyonu: A, B, C → B→conj→A, C→conj→A.

        Strateji: has_comma_after işareti olan tokenın sağındaki aynı
        UPOS grubundan token'ı conj olarak bağla. İlk öge baş, sonrakiler conj.
        """
        applied: list[str] = []
        n = len(tokens)

        for i in range(n - 1):
            t = tokens[i]
            # Virgül yoksa atla
            if not t.has_comma_after:
                continue
            # Sol token content olmalı
            if t.upos in ("PUNCT", "CCONJ", "DET", "ADP", "PART"):
                continue

            # Sağdaki content token'ı bul (DET/ADP atlayarak)
            right = None
            for j in range(i + 1, n):
                cand = tokens[j]
                if cand.upos in ("PUNCT", "CCONJ", "DET", "ADP", "PART"):
                    continue
                right = cand
                break

            if not right or right.is_assigned:
                continue

            # UPOS uyumluluk kontrolü
            compat = CoordinationRule._COORD_COMPAT.get(t.upos)
            if not compat or right.upos not in compat:
                continue

            # Zincir başını bul (t zaten conj ise, onun head'ine bağla)
            head = t
            while head.deprel == "conj" and head.head != 0:
                prev = next((tok for tok in tokens if tok.id == head.head), None)
                if prev is None:
                    break
                head = prev

            right.head = head.id
            right.deprel = "conj"
            applied.append("VİRGÜL→CONJ")

        return applied


# ═══════════════════════════════════════════════════════════════════
#  Phase D — İleri Yapılar
# ═══════════════════════════════════════════════════════════════════


class LightVerbRule(DependencyRule):
    """Hafif fiil yapılarını compound:lvc olarak bağlar (BOUN konvansiyonu).

    BOUN Treebank'ta hafif fiil yapısında isim baş, fiil bağımlıdır:
      söz + edecek → edecek ──compound:lvc──▶ söz
      devam + ediyor → ediyor ──compound:lvc──▶ devam
      neden + olan → olan ──compound:lvc──▶ neden

    Strateji: Hafif fiil tespit edildiğinde, fiilin solundaki yalın
    isme compound:lvc olarak bağla. İsim, fiilin aldığı sözdizimsel
    rolü (root/advcl/obj vb.) üstlenir.
    """

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i, t in enumerate(tokens):
            if t.is_assigned:
                continue
            # Hafif fiil tespiti
            if not self._is_light_verb(t):
                continue
            # Solundaki yalın isme bağla
            left = self._find_left_noun(tokens, i)
            if left:
                t.head = left.id
                t.deprel = "compound:lvc"
                applied.append("HAFİF_FİİL→COMPOUND_LVC")
        return applied

    @staticmethod
    def _is_light_verb(token: DepToken) -> bool:
        """Token'ın hafif fiil olup olmadığını kontrol eder.

        Lemma-tabanlı + form-tabanlı hibrit yaklaşım.
        Morfolojik çözümleyici bazı hafif fiillere yanlış lemma atadığı için
        (ederler→ederl, olan→olan) form prefix kontrolü de kullanılır.
        """
        w = turkish_lower(token.form)
        if w in _LVC_FORM_BLACKLIST:
            return False
        if token.lemma in LIGHT_VERBS:
            return True
        if _LVC_NOM_RE.match(w):
            return True
        for stem in ("et", "ed", "ol", "yap", "kıl", "buyur", "eyle"):
            if w == stem:
                return True
            if w.startswith(stem) and len(w) > len(stem):
                if stem == "ol" and w in _OL_BLACKLIST:
                    continue
                return True
        return False

    @staticmethod
    def _find_left_noun(tokens: list[DepToken], verb_idx: int) -> DepToken | None:
        """Hafif fiilin solundaki yalın ismi bul."""
        for j in range(verb_idx - 1, max(verb_idx - 3, -1), -1):
            cand = tokens[j]
            if cand.upos in ("NOUN", "ADJ") and not cand._suffixes:
                if turkish_lower(cand.form) not in TEMPORAL_NOUNS:
                    return cand
            break  # sadece hemen soldaki tokene bak
        return None

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
            # Zincir başı: büyük harfle başlayan, atanmamış, NOUN/PROPN
            if (t.form[0].isupper() and not t.is_assigned
                    and t.upos in ("NOUN", "PROPN")):
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


class ReduplicationRule(DependencyRule):
    """Türkçe ikileme (reduplication) tespiti: compound:redup.

    Üç kalıp:
    1) Tam tekrar — aynı biçim: ``sık sık``, ``yavaş yavaş``
    2) Kök tekrarı — aynı lemma: ``der demez``, ``olursa olsun``
    3) Eşanlamlı çiftler — sözlük tabanlı: ``toz duman``, ``aşağı yukarı``

    BOUN kuralı: ikinci öge (sağdaki) compound:redup olarak birinciye bağlanır.
    """

    _REDUP_PAIRS: frozenset[tuple[str, str]] = frozenset({
        ("paldır", "küldür"), ("konu", "komşu"), ("toz", "duman"),
        ("saç", "sakal"), ("had", "hesap"), ("dolap", "molap"),
        ("kaba", "saba"), ("yer", "gök"), ("aşağı", "yukarı"),
        ("yaşam", "zor"), ("bağır", "çağır"), ("ol", "bit"),
    })

    def apply(self, tokens: list[DepToken]) -> list[str]:
        applied: list[str] = []
        for i in range(1, len(tokens)):
            t = tokens[i]
            if t.is_assigned:
                continue
            # Find previous non-PUNCT token
            prev_idx = i - 1
            if tokens[prev_idx].upos == "PUNCT" and i >= 2:
                prev_idx = i - 2
            prev = tokens[prev_idx]

            t_form = turkish_lower(t.form)
            p_form = turkish_lower(prev.form)
            t_lem = turkish_lower(t.lemma) if t.lemma else t_form
            p_lem = turkish_lower(prev.lemma) if prev.lemma else p_form

            match = False
            if t_form == p_form:
                match = True
            elif t_lem == p_lem and len(t_lem) >= 2:
                match = True
            elif (p_lem, t_lem) in self._REDUP_PAIRS:
                match = True

            if match:
                t.head = prev.id
                t.deprel = "compound:redup"
                applied.append(f"İKİLEME({t.form})")
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
                local_pred = _find_local_predicate(tokens, t.id, root_id)
                t.head = local_pred
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
            if t.is_assigned:
                continue
            # Yapım eki (YAPIM_*) sıfat niteliğini bozmaz
            if t._suffixes and not _has_only_derivational(t):
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
                    # UPOS'u ADJ olarak koru — BOUN, sıfat-zarf sözcükleri
                    # fiil bağlamında bile ADJ olarak etiketler
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

            # SCONJ: bağımlama bağlacı → mark, soldaki en yakın VERB'e bağla
            if t.upos == "SCONJ":
                target = self._find_left_verb(tokens, i)
                if target:
                    t.head = target.id
                    t.deprel = "mark"
                    applied.append("FALLBACK→MARK")
                    continue

            # Soru partikülleri (AUX/PART): mi/mı/mu/mü → discourse:q
            if t.upos in ("AUX", "PART") and turkish_lower(t.form) in QUESTION_PARTICLES:
                target = self._find_left_content(tokens, i)
                if target:
                    t.head = target.id
                    t.deprel = "discourse:q"
                    applied.append("FALLBACK→DISCOURSE_Q")
                    continue

            # PART: "ki" → mark (tümce bağlayıcı/vurgulayıcı)
            # BOUN'da ki genellikle mark (SCONJ) olarak etiketlenir
            if t.upos == "PART" and turkish_lower(t.form) == "ki":
                target = self._find_left_verb(tokens, i)
                if target:
                    t.head = target.id
                    t.deprel = "mark"
                    applied.append("FALLBACK→MARK_KI")
                    continue
                # Soldaki fiil yoksa root'a bağla
                t.head = root_id
                t.deprel = "mark"
                applied.append("FALLBACK→MARK_KI_ROOT")
                continue

            # INTJ: ünlem → discourse, root'a bağla
            if t.upos == "INTJ":
                t.head = root_id
                t.deprel = "discourse"
                applied.append("FALLBACK→DISCOURSE")
                continue

            # ADJ: sıfat → bağlama göre amod / advcl / root
            if t.upos == "ADJ":
                # Pattern 1: ADJ + olarak → advcl (kesin olarak, sıcak olarak)
                if i + 1 < len(tokens) and turkish_lower(tokens[i + 1].form) == "olarak":
                    local_pred = _find_local_predicate(tokens, t.id, root_id)
                    t.head = local_pred
                    t.deprel = "advcl"
                    applied.append("FALLBACK→ADVCL_OLARAK")
                    continue
                # Pattern 2: Sağda NOUN varsa amod (CCONJ atlayarak da)
                target = self._find_right_noun(tokens, i)
                if not target:
                    target = self._find_right_head_for_adj(tokens, i)
                if target:
                    t.head = target.id
                    t.deprel = "amod"
                    applied.append("FALLBACK→AMOD")
                    continue
                # Pattern 3: Cümle sonu ADJ → nominal root adayı
                if i == len(tokens) - 1 or all(
                    tokens[j].upos in ("AUX", "PUNCT", "CCONJ")
                    for j in range(i + 1, len(tokens))
                ):
                    t.head = 0
                    t.deprel = "root"
                    # Mevcut root'u conj'a çevir
                    for other in tokens:
                        if other.id != t.id and other.deprel == "root":
                            other.head = t.id
                            other.deprel = "conj"
                            break
                    applied.append("FALLBACK→ADJ_ROOT")
                    continue

            # DET: belirleyici → sağdaki ilk içerik sözcüğüne det
            if t.upos == "DET":
                target = self._find_right_content(tokens, i)
                if target:
                    t.head = target.id
                    t.deprel = "det"
                    applied.append("FALLBACK→DET")
                    continue

            # NUM: sayı → sağda NOUN varsa nummod, yoksa VERB'e de dene
            if t.upos == "NUM":
                target = self._find_right_noun(tokens, i)
                if not target:
                    target = self._find_right_head_for_adj(tokens, i)
                if target:
                    t.head = target.id
                    t.deprel = "nummod"
                    applied.append("FALLBACK→NUMMOD")
                    continue

            # VERB: fiil → finite ise conj adayı, SIFAT_FİİL ise ccomp
            if t.upos == "VERB":
                if t.has_any_label(_FINITE_TENSE_LABELS):
                    # Çekimli fiil → eşgüdüm (conj) olarak root'a bağla
                    t.head = root_id
                    t.deprel = "conj"
                    applied.append("FALLBACK→CONJ_VERB")
                    continue
                # İSİM_FİİL/İŞTEŞ (-mA/-Iş) + sağda NOUN → nmod:poss
                # içme suyu, geçiş süreci, yaratma potansiyeli vb.
                if t.has_any_label({"İSİM_FİİL", "İŞTEŞ"}) or (
                    t._suffixes and any(
                        "İSİM_FİİL" in lb or "İŞTEŞ" in lb
                        for _, lb in t._suffixes
                    )
                ):
                    target = self._find_right_noun(tokens, i)
                    if target:
                        t.head = target.id
                        t.deprel = "nmod:poss"
                        applied.append("FALLBACK→NMOD_POSS_VN")
                        continue
                # Sıfat-fiil ama ParticipleRule'da sağda isim bulamadı
                if t.has_any_label(PARTICIPLE_LABELS):
                    # Sağda isim varsa acl olarak bağla (daha geniş arama)
                    target = self._find_right_noun_wide(tokens, i)
                    if target:
                        t.head = target.id
                        t.deprel = "acl"
                        applied.append("FALLBACK→ACL")
                        continue
                    # Sağda isim yok → ccomp (tümleç cümlesi) veya nsubj adayı
                    t.head = root_id
                    t.deprel = "ccomp"
                    applied.append("FALLBACK→CCOMP_PART")
                    continue
                # -mIş formu: İŞTEŞ olarak etiketlenmiş ama aslında duyulan geçmiş
                # kurulmuş, yazılmış, demiş, gelmiş vb.
                if _MIS_VERB_RE.search(t.form):
                    target = self._find_right_noun_wide(tokens, i)
                    if target:
                        t.head = target.id
                        t.deprel = "acl"
                        applied.append("FALLBACK→ACL_MIS")
                        continue
                    # Sağda isim yok → finite predicate olarak conj
                    t.head = root_id
                    t.deprel = "conj"
                    applied.append("FALLBACK→CONJ_MIS")
                    continue

            # NOUN/PROPN: TAMLAYAN ekli → nmod:poss olarak sağdaki isme bağla
            if t.upos in ("NOUN", "PROPN"):
                if t.has_label("TAMLAYAN"):
                    target = self._find_right_possessed(tokens, i)
                    if target:
                        t.head = target.id
                        t.deprel = "nmod:poss"
                        applied.append("FALLBACK→NMOD_POSS")
                        continue

            local_pred = _find_local_predicate(tokens, t.id, root_id)
            t.head = local_pred
            t.deprel = "dep"
            applied.append("FALLBACK→DEP")
        return applied

    @staticmethod
    def _find_right_noun(tokens: list[DepToken], start: int) -> DepToken | None:
        """start'ın sağındaki en yakın NOUN/PROPN tokenini bul."""
        for j in range(start + 1, min(start + 6, len(tokens))):
            if tokens[j].upos in ("NOUN", "PROPN"):
                return tokens[j]
            if tokens[j].upos not in ("ADJ", "NUM", "DET", "ADV", "CCONJ", "PART"):
                break
        return None

    @staticmethod
    def _find_left_verb(tokens: list[DepToken], start: int) -> DepToken | None:
        """start'ın solundaki en yakın VERB tokenini bul."""
        for j in range(start - 1, max(start - 6, -1), -1):
            if tokens[j].upos == "VERB":
                return tokens[j]
        return None

    @staticmethod
    def _find_left_content(tokens: list[DepToken], start: int) -> DepToken | None:
        """start'ın solundaki en yakın içerik sözcüğünü bul."""
        for j in range(start - 1, max(start - 4, -1), -1):
            if tokens[j].upos in ("VERB", "NOUN", "ADJ", "PROPN", "ADV"):
                return tokens[j]
        return None

    @staticmethod
    def _find_right_noun_wide(tokens: list[DepToken], start: int) -> DepToken | None:
        """start'ın sağındaki NOUN/PROPN'u daha geniş pencerede bul (8 token)."""
        for j in range(start + 1, min(start + 8, len(tokens))):
            t = tokens[j]
            if t.upos in ("NOUN", "PROPN"):
                return t
            if t.upos in ("VERB",) and not t.has_any_label(PARTICIPLE_LABELS):
                break
        return None

    @staticmethod
    def _find_right_possessed(tokens: list[DepToken], start: int) -> DepToken | None:
        """TAMLAYAN'lı tokenın sağındaki İYELİK'li ismi bul (6 token pencere)."""
        for j in range(start + 1, min(start + 6, len(tokens))):
            t = tokens[j]
            if t.upos in ("NOUN", "PROPN") and t.has_iyelik:
                return t
            if t.upos in ("ADJ", "DET", "NUM"):
                continue
            if t.upos in ("NOUN", "PROPN"):
                return t
            break
        return None

    @staticmethod
    def _find_right_content(tokens: list[DepToken], start: int) -> DepToken | None:
        """start'ın sağındaki en yakın içerik sözcüğünü bul (UPOS bağımsız)."""
        for j in range(start + 1, min(start + 5, len(tokens))):
            if tokens[j].upos in ("NOUN", "PROPN", "VERB", "ADJ", "PRON", "NUM"):
                return tokens[j]
        return None

    @staticmethod
    def _find_right_head_for_adj(tokens: list[DepToken], start: int) -> DepToken | None:
        """ADJ için sağdaki VERB/ADJ hedefini bul (UPOS hatası telafisi).

        Birçok NOUN, UPOS çıkarımında VERB olarak etiketlenir (adamlar,
        kapılar vb.). Bu yardımcı NOUN bulunamazsa en yakın VERB/ADJ'yi döndürür.
        """
        for j in range(start + 1, min(start + 3, len(tokens))):
            t = tokens[j]
            if t.upos in ("VERB", "ADJ"):
                return t
            if t.upos in ("CCONJ", "ADP"):
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


def _find_local_predicate(
    tokens: list[DepToken], position: int, root_id: int,
) -> int:
    """Verilen konumdaki token için en yakın sağdaki yüklemi bulur.

    Türkçe baş-sonu (head-final) yapısına göre argümanlar yüklemlerinden
    ÖNCE gelir. Bu yardımcı, çok-yüklemli cümlelerde her argümanı
    kendi yerel yüklemine bağlamak için kullanılır.

    Yüklem adayları: root + advcl + acl + ccomp deprel'li tokenlar.
    """
    best_id = root_id
    best_pos = len(tokens) + 1
    for t in tokens:
        if t.id <= position:
            continue
        if t.deprel in ("root", "advcl", "acl", "ccomp"):
            idx = next((k for k, tok in enumerate(tokens) if tok.id == t.id), best_pos)
            if idx < best_pos:
                best_pos = idx
                best_id = t.id
    return best_id


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
            # 8: İkileme (compound:redup)
            ReduplicationRule(),
            # 9: Koordinasyon
            CoordinationRule(),
            # 9-10: Bileşik yapılar
            LightVerbRule(),
            CompoundNounRule(),
            # 11-12: Yan cümleler
            ConverbRule(),
            InfinitiveRule(),
            ParticipleRule(),
            # 13-15: Özel yapılar (CaseRoleRule'dan önce)
            TemporalAdvmodRule(),
            AdvmodEmphRule(),
            CopulaRule(),
            # 16-19: Genel bağımlılıklar
            AdjAdvDisambiguationRule(),
            AdvmodRule(),
            AdjectiveRule(),       # CaseRoleRule'dan ÖNCE → ADJ tokenları claim et
            CaseRoleRule(),
            # 20: Fallback
            FallbackRule(),
        ]

    # ── Ana API ───────────────────────────────────────────────────

    def parse(
        self,
        sentence_tokens: list[SentenceToken],
        *,
        text: str = "",
        trace: bool = False,
    ) -> list[DepToken]:
        """SentenceToken listesini bağımlılık ağacına dönüştürür.

        Args:
            sentence_tokens: Morfolojik çözümleme çıktısı.
            text: Orijinal cümle metni (virgül tespiti için).
            trace: True ise kural uygulamalarını loglar, self._last_trace'e yazar.
        """
        if not sentence_tokens:
            self._last_trace = []
            return []

        dep_tokens = [
            DepToken.from_sentence_token(st, i + 1, is_first=(i == 0))
            for i, st in enumerate(sentence_tokens)
        ]

        # Orijinal metinde virgül konumlarını tespit et
        if text:
            self._detect_commas(dep_tokens, text)

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

        # Son-işlem: her yüklem başına en fazla 1 obj tut
        self._limit_obj_per_pred(dep_tokens)

        # Son-işlem: koordinasyon root düzeltmesi
        self._swap_root_for_coordination(dep_tokens)

        return dep_tokens

    @staticmethod
    def _swap_root_for_coordination(tokens: list[DepToken]) -> None:
        """UD konvansiyonu: ilk eşgüdümlü öge root olmalı.

        Koşul: ilk conj VERB + virgüllü + root'tan önce → swap.
        """
        root = next((t for t in tokens if t.deprel == "root"), None)
        if not root:
            return
        # İlk conj (root'a bağlı, root'tan önce, VERB, virgüllü)
        first_conj = None
        for t in tokens:
            if (t.deprel == "conj" and t.head == root.id
                    and t.id < root.id and t.upos == "VERB"
                    and t.has_comma_after):
                first_conj = t
                break
        if not first_conj:
            return
        # Swap: first_conj → root, root → conj
        old_root_id = root.id
        new_root_id = first_conj.id
        first_conj.head = 0
        first_conj.deprel = "root"
        root.head = new_root_id
        root.deprel = "conj"
        # Diğer conj'lar yeni root'a bağlan
        for t in tokens:
            if t.deprel == "conj" and t.head == old_root_id and t.id != new_root_id:
                t.head = new_root_id

    @staticmethod
    def _limit_obj_per_pred(tokens: list[DepToken]) -> None:
        """Her yüklem başına en fazla 1 obj bırakır.

        Fazla obj tokenlarını → nmod:poss (İYELİK varsa) veya obl yapar.
        Yükleme en yakın olan obj korunur.
        """
        from collections import defaultdict
        pred_objs: dict[int, list[DepToken]] = defaultdict(list)
        for t in tokens:
            if t.deprel == "obj":
                pred_objs[t.head].append(t)
        for head_id, objs in pred_objs.items():
            if len(objs) <= 1:
                continue
            # En yakın obj'yi koru (yüklemin hemen solundaki)
            objs.sort(key=lambda t: abs(t.id - head_id))
            for t in objs[1:]:
                if t.has_iyelik:
                    t.deprel = "nmod:poss"
                else:
                    t.deprel = "obl"

    @staticmethod
    def _detect_commas(dep_tokens: list[DepToken], text: str) -> None:
        """Orijinal metinden virgül konumlarını tespit edip tokenlara işaretle."""
        pos = 0
        for i, t in enumerate(dep_tokens):
            idx = text.find(t.form, pos)
            if idx < 0:
                # Case-insensitive fallback
                idx = text.lower().find(t.form.lower(), pos)
            if idx < 0:
                continue
            end = idx + len(t.form)
            # Bir sonraki tokena kadar olan aralıkta virgül var mı?
            if i + 1 < len(dep_tokens):
                next_idx = text.find(dep_tokens[i + 1].form, end)
                if next_idx < 0:
                    next_idx = text.lower().find(dep_tokens[i + 1].form.lower(), end)
                if next_idx >= 0:
                    between = text[end:next_idx]
                    if "," in between:
                        t.has_comma_after = True
            pos = end

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
