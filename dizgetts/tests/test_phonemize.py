"""/d/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/tests/test_phonemize.py"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frontend.phonemize import Phonemizer  # noqa: E402
from frontend.symbols import PAUSES, STRESS, SYMBOL_TO_ID, to_ids, tokenize  # noqa: E402

p = Phonemizer(bert_fallback=False)
norm, toks = p("Merhaba, dünya! 3'te İstanbul'da.")
assert norm == "Merhaba , dünya ! üçte İstanbulda .", norm
assert toks[0] == "m" and toks[-1] == "." and "," in toks and "!" in toks, toks
assert all(t in SYMBOL_TO_ID for t in toks)
i = toks.index(",")
assert toks[i - 1] != " " and toks[i + 1] == " "  # noktalama öncesi ayraç yok, sonrası var
assert p.word("İstanbul") == p.word("istanbul")  # Türkçe büyük/küçük harf
assert p.word("xbox") != "bɔ"  # dizge x'i sessizce atıyordu; FOREIGN eşlemesi
assert tokenize("ˈɑ") == [STRESS, "ɑ"] and to_ids([STRESS])  # vurgu işareti geçirilir (rezerv)
assert set(PAUSES) <= set(SYMBOL_TO_ID)
print("OK")
