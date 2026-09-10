# -*- coding: utf-8 -*-
"""ingest_gold — iki-katman BIO (süreksiz öbek dahil) + aday-öbek örtüşme skoru.

Çalıştır:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ingest_gold.py -q
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
ig = importlib.import_module("ingest_gold")


def test_bio2_contiguous():
    spans = [{"s": 1, "e": 3, "cat": "VID", "sense": "idio"},
             {"s": 5, "e": 7, "cat": "LVC", "sense": "idio"},
             {"s": 8, "e": 9, "cat": "VID", "sense": "lit"}]      # lit → O
    t1, t2 = ig.bio2(10, spans)
    assert t1 == ["O", "B-VID", "I-VID", "O", "O", "B-LVC", "I-LVC", "O", "O", "O"]
    assert t2 == ["o"] * 10


def test_bio2_gappy():
    # "sahip ... olarak" — 1. parça [1,2), 2. parça [4,6)
    spans = [{"s": 1, "e": 2, "s2": 4, "e2": 6, "cat": "VID", "sense": "idio"}]
    t1, t2 = ig.bio2(7, spans)
    assert t1 == ["O", "B-VID", "O", "O", "O", "O", "O"]
    assert t2 == ["o", "o", "o", "o", "b-VID", "i-VID", "o"]


def test_contig():
    assert ig.contig(5, 1, 3) == ["O", "B-VID", "I-VID", "O", "O"]
    assert ig.contig(5, 0, 2, "LVC") == ["B-LVC", "I-LVC", "O", "O", "O"]


def test_covers():
    cand = {"s": 2, "e": 4}
    assert ig.covers({"s": 2, "e": 4}, cand) == 1.0
    assert ig.covers({"s": 1, "e": 5}, cand) == 1.0
    assert ig.covers({"s": 2, "e": 3}, cand) == 0.5
    assert ig.covers({"s": 6, "e": 8}, cand) == 0.0
