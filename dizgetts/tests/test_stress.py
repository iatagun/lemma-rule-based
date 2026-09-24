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
