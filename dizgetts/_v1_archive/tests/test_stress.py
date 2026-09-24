"""python -X utf8 dizgetts/tests/test_stress.py"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frontend.stress import dizge_words, espeak_stress_index, espeak_words, transfer_stress  # noqa: E402
from frontend.symbols import tokenize  # noqa: E402

# --- espeak tarafı: vurgulu ünlünün sözcük içi sırası ve ünlü sayısı
assert espeak_stress_index("ɟœɾynˈyjɔr") == (2, 4)     # görünüyor: -yor öncesi
assert espeak_stress_index("sˈekɪz") == (0, 2)          # sekiz: ilk hece
assert espeak_stress_index("kˈɯrk") == (0, 1)
assert espeak_stress_index("vɛ") == (None, 1)           # vurgusuz sözcük
assert espeak_stress_index("aɫadʒˈaːɯm") == (2, 4)      # uzunluk işareti ünlü sayısını etkilemez
assert espeak_words("a , b ; ! c") == ["a", "b", "c"]  # noktalama parçaları atılır

# --- tam cümle: "görünüyor; sekiz ve"  (dizge: ünlüler œ Y Y ɔ | ɛ I | ɛ)
tok = tokenize("ɟœɾYnYjɔɣ;") + [" "] + tokenize("sɛcIz̥") + [" "] + tokenize("vɛ")
out, st = transfer_stress(tok, "ɟœɾynˈyjɔr ; sˈekɪz vɛ")
assert out == ["ɟ", "œ", "ɾ", "Y", "n", "ˈ", "Y", "j", "ɔ", "ɣ", ";", " ", "s", "ˈ", "ɛ", "c", "I", "z̥", " ", "v", "ɛ"], out
assert st["ünlü_sayısı_eşit"] == 2 and st["espeak_vurgusuz"] == 1 and st["vurgu_sayısı"] == 2, st

# --- sözcük sayısı tutmazsa cümle yedeği: her sözcüğün SON ünlüsü vurgulu
out2, st2 = transfer_stress(tokenize("mɛɾxɑbɑ") + [" "] + tokenize("dYnjɑ"), "mˈɛɾhɑbɑ dˈynja bɛ")
assert st2["cümle_yedek_son_ünlü"] == 1 and out2 == ["m", "ɛ", "ɾ", "x", "ɑ", "b", "ˈ", "ɑ", " ", "d", "Y", "n", "j", "ˈ", "ɑ"], out2

# --- ünlü sayıları eşit (2 = 2): aynı sıradaki ünlü (1.)
out3, st3 = transfer_stress(["b", "ɑ", "l", "ɑ"], "bˈalaj")
assert st3["ünlü_sayısı_eşit"] == 1 and out3 == ["b", "ˈ", "ɑ", "l", "ɑ"], (out3, st3)
# --- ünlü sayıları farklı: espeak 3 ünlü, dizge 2 ünlü
out4, st4 = transfer_stress(["b", "ɑ", "l", "ɑ"], "bˈalaɾɪ")  # espeak 3 ünlü, vurgu 1. (sondan 3.) -> dizge 2 ünlüde sığmaz -> son ünlü
assert st4["yedek_son_ünlü"] == 1 and out4[-2:] == ["ˈ", "ɑ"], (out4, st4)
out5, st5 = transfer_stress(["b", "ɑ", "l", "ɑ"], "balˈaɾɪ")  # vurgu 2. (sondan 2.) -> dizge 2 ünlüde sondan 2. = 1. ünlü
assert st5["sondan_sıra"] == 1 and out5 == ["b", "ˈ", "ɑ", "l", "ɑ"], (out5, st5)

# --- yalnız noktalamadan oluşan parça sözcük sayılmaz
assert len(dizge_words([";", " ", "a", "b", " ", "c"])) == 2
print("OK")
