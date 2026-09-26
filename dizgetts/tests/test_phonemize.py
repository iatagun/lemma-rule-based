"""/d/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/tests/test_phonemize.py"""
import os, sys
from dizgetts.frontend.phonemize import Phonemizer  # noqa: E402
from dizgetts.frontend.symbols import PAUSES, STRESS, SYMBOL_TO_ID, to_ids, tokenize  # noqa: E402

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
# regresyon (2026-09-26 code review): dizge.g2p tanımadığı harfi SESSİZCE atıyordu (café -> dʒɑf, m² -> m); artık eşlenir, eşlenemeyen atılır ve UYARILIR
import warnings

q = Phonemizer(bert_fallback=False)
assert q.word("café") == q.word("cafe") and q.word("Müller") == q.word("müller") and q.word("naïve") == q.word("naive")
assert q.word("straße") == q.word("strasse") and q.word("ölçü") == p.word("ölçü")
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    assert q.word("kжedi") == q.word("kedi") and q.dropped["ж"] == 1
    assert q.word("ж") == "" and q.stats["fonemsiz"] == 1
assert sum(issubclass(w.category, RuntimeWarning) for w in W) == 3, [str(w.message) for w in W]  # atılan harf x2 + fonemsiz sözcük
# önbellek örnek başına (metot üstünde lru_cache self'i sonsuza dek tutuyordu)
assert Phonemizer(bert_fallback=False).stats["dizge"] == 0
print("OK")
