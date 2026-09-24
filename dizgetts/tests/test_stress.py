"""python -X utf8 -m dizgetts.tests.test_stress"""
from dizgetts.engine import Engine
from dizgetts.frontend.stress import StressRules, to_phone_index
from dizgetts.frontend.symbols import STRESS

R = StressRules()
# varsayılan: son seslem; ek alınca yeni son seslem
assert R.syllable("kitap") == (1, "varsayılan_son") and R.syllable("kitaplar") == (2, "varsayılan_son")
assert R.syllable("evlerinizden") == (4, "varsayılan_son")  # e-e-i-i-e: 5 ünlü
# kök sözlüğü (kullanıcı örnekleri, onaysız): kökte vurgu, ek alınca kökte KALIR
assert R.syllable("şimdi") == (0, "kök") and R.syllable("lokanta") == (1, "kök")
assert R.syllable("lokantalarda") == (1, "kök+ek") and R.syllable("Ordunun") == (0, "kök+ek")
# yer adları yalnız büyük harfle: ordu (ordu=asker) / Ordu (şehir), bebek (bebek) / Bebek (semt)
assert R.syllable("ordu") == (1, "varsayılan_son") and R.syllable("Ordu") == (0, "kök")
assert R.syllable("bebek") == (1, "varsayılan_son") and R.syllable("Bebek") == (0, "kök")
# sahte eşleşme: 'ordu'+'ç' gibi ünsüz-yalnız kalıntı köke ek sayılmaz
assert R.syllable("orduç") == (1, "varsayılan_son")
# clitic ve ünlüsüz
assert R.syllable("da") == (None, "clitic") and R.syllable("ki") == (None, "clitic") and R.syllable("ptt") == (None, "ünlüsüz")
# yazım ünlüsü -> dizge ünlü atomu
assert to_phone_index("kitap", ["kʰ", "I", "tʰ", "ɑ", "p"], 1) == (3, "eşit")
# 'ay' yan ünlüsü: bilgisayar 4 ünlü harf, dizge 5 ünlü atomu (I I ɑː I ɑ); son seslem = son atom
ph = ["b", "I", "l", "ɟ", "I", "s", "ɑː", "I", "ɑ", "ɣ"]
assert to_phone_index("bilgisayar", ph, 3) == (8, "yan_ünlü_atıldı")
assert to_phone_index("bilgisayar", ph, 0) == (1, "yan_ünlü_atıldı")   # ilk seslem = ilk atom (yan ünlü kaydırmaz)

# uçtan uca: vurgu simgesi ünlünün ÖNÜNE; clitic'e konmaz
e = Engine(bert_fallback=False)
u = e.frontend("Şimdi gelirsin de.")
t = u.tokens
assert t[t.index(STRESS) + 1] in ("I",) and t.index(STRESS) > 0 and t[t.index(STRESS) - 1] in ("ʃ",), t   # ʃ ˈI m d I -> ŞİM-di: ilk ünlü
assert u.words[-1].stress is None  # 'de' clitic
assert t.count(STRESS) == 2, t
print("OK")

# ---- M1b: morfolojik katmanlar (upos/feats elle verildi; DizgeBERT-Morph gerekmez) ----
from dizgetts.frontend.stress import TIERS  # noqa: E402

T = TIERS
V = lambda **f: ("VERB", f)
cases = [  # (sözcük, upos, feats, beklenen seslem indeksi BAŞTAN, etiket)
    ("geliyorum", "VERB", {}, 1, "yor_önü"),                # ge-Lİ-yo-rum
    ("görünüyor", "VERB", {}, 2, "yor_önü"),                # gö-rü-NÜ-yor (espeak ile aynı)
    ("gelmedi", "VERB", {"Polarity": "Neg"}, 0, "olumsuzluk_önü"),          # GEL-me-di
    ("olmadığını", "VERB", {"Polarity": "Neg"}, 0, "olumsuzluk_önü"),
    ("gelmeyeceksiniz", "VERB", {"Polarity": "Neg", "Person": "2", "Number": "Plur"}, 0, "olumsuzluk_önü"),
    ("arkadaşımla", "NOUN", {"Case": "Ins"}, 3, "ins_önü"),  # ar-ka-da-ŞIM-la
    ("gelirsiniz", "VERB", {"Person": "2", "Number": "Plur"}, 1, "kişi_eki_önü"),   # ge-LİR-si-niz
    ("geldim", "VERB", {"Person": "1", "Number": "Sing"}, 1, "kişi_eki_önü"),        # gel-DİM
    ("yapsaydı", "VERB", {"Person": "3", "Number": "Sing"}, 1, "koşaç_önü"),          # yap-SAY-dı
    ("yapsaymış", "VERB", {}, 1, "koşaç_önü"),
    ("gelirdi", "VERB", {"Person": "3"}, 1, "koşaç_önü"),                             # ge-LİR-di
    ("geldin", "VERB", {"Person": "2", "Number": "Sing"}, 1, "kişi_eki_önü"),         # gel-DİN
    ("geldik", "VERB", {"Person": "1", "Number": "Plur"}, 1, "kişi_eki_önü"),
    ("geldiniz", "VERB", {"Person": "2", "Number": "Plur"}, 1, "kişi_eki_önü"),      # gel-Dİ-niz
    ("gelirim", "VERB", {"Person": "1", "Number": "Sing"}, 1, "kişi_eki_önü"),       # ge-LİR-im
    ("geleceğim", "VERB", {"Person": "1", "Number": "Sing"}, 2, "kişi_eki_önü"),     # ge-le-CEĞ-im (0. ge, 1. le, 2. ceğ)
    ("geldi", "VERB", {"Person": "3"}, 1, "varsayılan_son"),                          # düz geçmiş -dı vurgulanır (koşaç değil)
    ("kesin", "ADJ", {}, 1, "varsayılan_son"),                                        # -sın diye SOYULMAZ (kişi eki yalnız VERB'de)
    ("okula", "NOUN", {"Case": "Dat"}, 2, "varsayılan_son"),                          # -la diye soyulmaz (Case=Ins değil)
]
for w, up, f, k, tag in cases:
    got = R.syllable(w, up, f, T)
    assert got == (k, tag), (w, got, (k, tag))
# katman kapalıyken (M1a) hiçbiri devreye girmez
assert R.syllable("geliyorum", "VERB", {}, ()) == (3, "varsayılan_son")
print("OK (M1b)")
