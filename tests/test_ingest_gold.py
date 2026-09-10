# -*- coding: utf-8 -*-
"""ingest_gold — BIO türetme + aday-öbek örtüşme skoru.

Çalıştır:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ingest_gold.py -q
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
ig = importlib.import_module("ingest_gold")


def test_bio_from_idio_spans():
    spans = [{"s": 1, "e": 3, "cat": "VID", "sense": "idio"},
             {"s": 5, "e": 7, "cat": "LVC", "sense": "idio"},
             {"s": 8, "e": 9, "cat": "VID", "sense": "lit"}]      # lit → O
    assert ig.bio(10, spans) == ["O", "B-VID", "I-VID", "O", "O", "B-LVC", "I-LVC", "O", "O", "O"]


def test_bio_empty():
    assert ig.bio(4, []) == ["O", "O", "O", "O"]
    assert ig.bio(4, [{"s": 0, "e": 2, "cat": "VID", "sense": "lit"}]) == ["O", "O", "O", "O"]


def test_covers():
    cand = {"s": 2, "e": 4}
    assert ig.covers({"s": 2, "e": 4}, cand) == 1.0            # tam
    assert ig.covers({"s": 1, "e": 5}, cand) == 1.0            # kapsıyor
    assert ig.covers({"s": 2, "e": 3}, cand) == 0.5            # yarı (IoU 1/2)
    assert ig.covers({"s": 6, "e": 8}, cand) == 0.0            # ayrık
