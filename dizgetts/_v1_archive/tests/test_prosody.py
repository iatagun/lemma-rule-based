"""python -X utf8 dizgetts/tests/test_prosody.py"""
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from frontend.prosody import add_features  # noqa: E402

t = ["a", "b", ",", " ", "c", " ", "d", "e", "."]           # "ab, c de."
toks, f = add_features(t)
assert toks == ["a", "b", ",", "c", "d", "e", "."], toks
# ab: a=sözcük başı(4), b=son + noktalama izliyor -> B3(3); c: baş(4)+son B1(1)=5 (noktalama yok); de: d=baş(4), e=son + cümle sonu -> B3(3)
assert f == [4, 3, 0, 5, 4, 3, 0], f
print("OK")
