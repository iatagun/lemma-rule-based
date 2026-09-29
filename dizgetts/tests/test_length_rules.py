"""Uzun ünlü (`ː`) kuralları (frontend/pronounce.py LengthRules): dizge'nin `ː`'si üç ayrı gerçeği tek atomda birleştiriyor (ğ+ünsüz/son = uzama x1,4;
ünlü arası ğ = geçiş; y yan ünlüsü). Yalnız gerçek uzama `ː` kalır. Kaynak: kullanıcı tarifi + Antalia ölçümü (docs/pronunciation_issues.md).
  python -X utf8 -m dizgetts.tests.test_length_rules"""
import warnings

from dizgetts.engine import Engine
from dizgetts.frontend.phonemize import Phonemizer
from dizgetts.frontend.symbols import tokenize

raw = Phonemizer(bert_fallback=False)
lr = Phonemizer(bert_fallback=False, length_rules=True)
both = Phonemizer(bert_fallback=False, pron_exceptions=True, length_rules=True)
gu = Phonemizer(bert_fallback=False, pron_exceptions=True, length_rules=True, register="gündelik")
gu_only = Phonemizer(bert_fallback=False, length_rules=True, register="gündelik")
A = lambda ph, w: " ".join(tokenize(ph.word(w)))
# sözlük açıkken sıradan ses gösterimi (kullanıcı 2026-09-30): ö/ü -> ø/y, sözcük sonu r ɣ -> ɾ; karşılaştırmalar bunun dışında
NATIVE = lambda ph: ph.translate(str.maketrans({"œ": "ø", "Y": "y", "ɣ": "ɾ"}))

# 0) kapalıyken dizge çıktısı birebir aynı
assert A(raw, "ağır") == "ɑː ɨ ɣ" and A(raw, "ayak") == "ɑː I ɑ k"

# 1) ğ ünsüz önünde / sözcük sonunda: UZAMA kalır (Antalia: 64 ms, kısa ünlü 46)
for w in ("dağ", "sağ", "dağlar", "sağlık", "doğru", "öğle", "öğretmen", "ağzı"):
    assert lr.word(w) == raw.word(w), (w, A(lr, w), A(raw, w))

# 2) ğ farklı iki ünlü arasında: ilk ünlü UZAMIYOR (46 ms), geçiş
for w, want in (("ağır", "ɑ ɨ ɣ"), ("soğuk", "s o U k"), ("yoğurt", "j o U ɾ t"), ("bağır", "b ɑ ɨ ɣ"), ("yapacağım", "j ɑ p ɑ dʒ ɑ ɨ m"), ("diğer", "d I ɛ ɣ")):
    assert A(lr, w) == want, (w, A(lr, w))
# aynı ünlü arasında ğ (olduğunu: u-ğ-u) tek uzun ünlü olarak kalır
assert A(lr, "olduğunu") == A(raw, "olduğunu") and "uː" in A(lr, "olduğunu")

# 3) y yan ünlüsü: `Vː I` -> `V j` (y kendi sesiyle; kullanıcı kör değerlendirmesi 2026-09-30); i + y -> i j (y'yi `j` olarak geri koy, dizge'nin kendi yapıyor/büyük yazımı gibi)
for w, want in (("ayak", "ɑ j ɑ k"), ("şey", "ʃ ɛ j"), ("aynı", "ɑ j n ɨ"), ("kuyu", "kʰ U j U"), ("koyun", "kʰ o j U n"), ("sayı", "s ɑ j ɨ"), ("boyunca", "b o j U n dʒ ɑ"),
                ("iyi", "i j I"), ("geliyor", "ɟ e l i j ɔ ɣ"), ("diye", "d i j ɛ"), ("saniye", "s ɑ n i j ɛ")):
    assert A(lr, w) == want, (w, A(lr, w), "beklenen", want)

# 3b) kayıt: özenli y SESLENİR (iyi -> i j I), gündelik y seslenmez, "ii" (dizge'nin iː'si kalır); y yan ünlüsü (ayak) her iki kayıtta ɑ j
assert A(gu_only, "iyi") == A(raw, "iyi") == "iː I" and A(gu_only, "geliyor") == A(raw, "geliyor") and A(gu_only, "diye") == A(raw, "diye")
assert A(gu_only, "ayak") == "ɑ j ɑ k" and A(gu_only, "ağır") == "ɑ ɨ ɣ"
assert Phonemizer(bert_fallback=False, length_rules=True).word("iyi") != gu_only.word("iyi")

# 3c) diğer: TEK y'leşen ğ (kullanıcı 2026-09-26): diyer. dizge kökte `d Iː ɛ`, çekimlerde `d iː e` verir (iki alternatif satır, ünlü niteliği korunur);
# iğne / öğün / düğün / öğle y'leşmez
for w, want in (("diğer", "d I j ɛ ɾ"), ("diğeri", "d I j ɛ ɾ I"), ("diğerleri", "d i j e ɾ l ɛ ɾ I"), ("diğerine", "d i j e ɾ I n ɛ"), ("diğeriyle", "d i j e ɾ I j l ɛ")):
    assert A(both, w) == want, (w, A(both, w), "beklenen", want)
for w in ("iğne", "öğün", "düğün", "öğle", "eğlence"):
    assert both.word(w) == NATIVE(lr.word(w)), w
assert A(lr, "diğer") == "d I ɛ ɣ"       # yalnız kural (sözlük kapalı): y'leşme yok, geçiş
with warnings.catch_warnings():
    warnings.simplefilter("error")     # alternatif satırlar: biri eşleşince diğeri sahte uyarı vermez
    for w in ("diğer", "diğerleri", "diğerine"):
        Phonemizer(bert_fallback=False, pron_exceptions=True).word(w)

# 3d) eğri: e UZAR (kullanıcı), eğlence y'leşir; aynı e+ğ+ünsüz bağlamı, sözcüğe bağlı -> sözlük satırı. Kural katmanı sözlüğün ɛː'sine dokunmaz
for w, want in (("eğri", "ɛː ɾ I"), ("eğrisi", "ɛː ɾ I s I"), ("eğriyi", "ɛː ɾ I j I"), ("eğrilik", "ɛː ɾ I l I c")):
    assert A(both, w) == want and A(gu, w) == want, (w, A(both, w), "beklenen", want)
assert A(both, "eğlence") == "e j l ɛ n dʒ ɛ"

# 3e) kullanıcı kararları 2026-09-27: değer/eğer UZAMA; değişik/eğitim y'leşme (dizge'nin j'si kalır); -eceğim DİFTONG (ğ'nin j'si düşer, ɛ I bitişik);
# -diği/-liği/-tiği (i-ğ-i) UZAMA: dizge `iː I` (uzun i + fazladan i) verir, tek uzun `iː`'ye birleşir (uğu -> uː gibi)
for w, want in (("değer", "d ɛː ɾ"), ("eğer", "ɛː ɾ"), ("değerli", "d ɛː ɾ l I"), ("değeri", "d ɛː ɾ I"), ("eğerse", "ɛː ɾ s ɛ")):
    assert A(both, w) == want, (w, A(both, w), "beklenen", want)
for w in ("değişik", "eğitim", "değil", "değişim", "eğlenmek"):
    assert both.word(w) == NATIVE(raw.word(w)), (w, A(both, w))                # y'leşme / karar bekleyen: dizge'nin okuması
for w, want in (("göndereceğim", "ɟ ø n d e ɾ ɛ dʒ ɛ I m"), ("edeceğiz", "e d ɛ dʒ ɛ I z̥"), ("ekleyeceğim", "e c l ɛ j ɛ dʒ ɛ I m"), ("olacağım", "ɔ ł ɑ dʒ ɑ ɨ m")):
    assert A(both, w) == want, (w, A(both, w), "beklenen", want)
for w, want in (("gönderdiğim", "ɟ ø n d ɛ ɾ d iː m"), ("dediğin", "d e d iː n"), ("güvenliğiniz", "ɟ y ʋ ɛ n l iː n I z̥"), ("söylendiğinde", "s ø j l e n d iː n d ɛ")):
    assert A(both, w) == want, (w, A(both, w), "beklenen", want)
# hizalama sıkılığı: iyiliğinden = iy (dizge burada j bırakır) + ğ (iː): ğ'nin iː'si iy olayına ATANMAMALI (aksi halde iː -> i j)
assert A(both, "iyiliğinden") == "I j I l iː n d ɛ n", A(both, "iyiliğinden")

# 4) dokunulmayanlar: ğ'siz/y'siz, ı+y ve ü+y (dizge j'yi zaten koruyor), e+ğ (dizge birincil okuması zaten y'leşmiş: eğlence -> e j l ɛ...), â/hâkim
for w in ("kitap", "yapıyor", "büyük", "eğlence", "eğer", "değil", "yeğen", "hâlâ", "kâr", "mahzur", "kral"):
    assert lr.word(w) == raw.word(w), (w, A(lr, w), A(raw, w))
assert A(lr, "eğlence") == "e j l ɛ n dʒ ɛ"
assert A(both, "değil") == "d e j I l" and A(gu, "değil") == "d e j I l"   # değil = deyil (kullanıcı 2026-09-27): y'leşme, dizge'nin okuması

# 5) çoklu kaynak: ağabey = ğ (a-ğ-a aynı ünlü, uzun kalır) + ey (ɛː I -> ɛ j)
assert A(lr, "ağabey") == "ɑː b ɛ j", A(lr, "ağabey")

# 5b) hizalama: her `ː` atomu, ÜNLÜSÜ uyumlu ilk olayla eşlenir (olayların hepsi `ː` üretmez: ö+y'de dizge bazen `j` bırakır, â `ː` üretmez)
for w, want in (("hikâye", "ç I k ɑ j ɛ"), ("şikâyete", "ʃ I k ɑ j e t ɛ"), ("söyleyeyim", "s œ j l e j ɛ j I m"), ("teyzeye", "tʰ e j z ɛ j ɛ"),
                ("düğmeye", "d yː m e j ɛ"), ("çubuğuyla", "tʃ U b uː j ł ɑ")):
    assert A(lr, w) == want, (w, A(lr, w), "beklenen", want)
# dizge sözlüğünden gelen (harf olayı olmayan) uzun ünlüler dokunulmaz: nisan, itibaren, teminat
for w in ("nisan", "itibaren", "teminat"):
    assert lr.word(w) == raw.word(w), w

# 6) sözlükle birlikte (sıra: sözlük -> kural): kağıt = kaeıt (ön a, geçiş; uzunluk YOK), gündelik kaat (a uzun, ı yok); hakim sözlüğün aː'sı korunur
assert A(both, "kağıt") == "cʰ a ɨ t" and A(both, "kağıdı") == "cʰ a ɨ d ɨ" and A(both, "kâğıt") == "cʰ a ɨ t"
assert A(gu, "kağıt") == "cʰ aː t" and A(gu, "kağıdı") == "cʰ aː d ɨ"
assert A(both, "hakim") == "x aː c I m" and A(both, "hâkim") == "x aː c I m"       # ğ/y olayı yok, ː sözlükten: dokunulmaz

# 7) hizalanamayan (olay sayısı ≠ ː sayısı) sözcük DOKUNULMADAN döner ve sayılır (sessiz bozma yok)
p = Phonemizer(bert_fallback=False, length_rules=True)
p.word("hakim")  # ğ/y/â olayı yok ama dizge sözlüğünden bir ː var
assert p.stats["uzun_hizalanamadi"] == 1 and p.word("hakim") == raw.word("hakim")

# 8) engine bayrağı: varsayılan KAPALI; açıkken sürüm dizgesine girer
e0, e1 = Engine(bert_fallback=False), Engine(bert_fallback=False, length_rules=True)
assert e0.frontend("Ağır ayak iyi.").tokens != e1.frontend("Ağır ayak iyi.").tokens
assert e0.frontend("Merhaba dünya.").tokens == e1.frontend("Merhaba dünya.").tokens
assert "uzun" not in e0.versions()["phonemize"] and "uzun" in e1.versions()["phonemize"]
print("OK")
