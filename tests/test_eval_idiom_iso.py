# -*- coding: utf-8 -*-
"""stage2-iso locator — Çavuşoğlu çiftlerinde deyimi bulan fuzzy içerik-sözcük penceresi.

run_external'ın katı gövde-eşleşmesi 198 satırın yalnız 11'ini konumlandırıyordu;
_iso_locate ~179'unu yakalamalı (izole stage-2 ölçümünün güvenilir tabanı).

Çalıştır:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_eval_idiom_iso.py -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.eval_idiom import _iso_locate, _iso_content_stems  # noqa: E402


def _ident(w):  # gövdeleyici yerine kimlik — saf pencere mantığını test et
    return w.lower()


def test_content_stems_drops_parens_and_stopwords():
    # "(bir şeyle) arası hoş (iyi) olmamak" → parantez + kısa işlev sözcük atılır
    got = _iso_content_stems("(bir şeyle) arası hoş (iyi) olmamak", _ident)
    assert "arası" in got and "olmamak" in got
    assert "bir" not in got and "iyi" not in got  # parantez içi


def test_locate_contiguous():
    words = "Abimle babamın arası hoş değil bu aralar".split()
    span = _iso_locate("arası hoş olmamak", words, _ident)
    assert span is not None
    lo, hi = span
    assert words[lo] == "arası" and "hoş" in words[lo:hi]


def test_locate_with_gap():
    # gövdeleyici "aldı"→"al", "almak"→"al" eşler; burada onu sahtele
    def fake_stem(w):
        return {"aldı": "al", "almak": "al"}.get(w.lower(), w.lower())
    words = "Komisyon bu konuyu dün ele nihayet aldı".split()
    span = _iso_locate("ele almak", words, fake_stem)
    assert span == (4, 7)  # ele .. aldı, araya "nihayet"


def test_locate_returns_none_when_absent():
    assert _iso_locate("pabucu dama atılmak", "Bugün hava çok güzel".split(), _ident) is None


def test_locate_rejects_too_scattered():
    # tek içerik sözcüğü eşleşir, diğeri cümlenin çok uzağında → None
    words = "ele aldı".split() + ["x"] * 20 + ["almak"]
    assert _iso_locate("ele geçirmek almak devretmek", words, _ident) is None
