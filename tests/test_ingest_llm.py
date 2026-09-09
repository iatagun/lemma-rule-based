# -*- coding: utf-8 -*-
"""filter_corpus_idiomaticity.py — --ingest-llm join/dedup + --gate κ testleri.

Riskli mantık: auto-LLM etiketlerini cümle metniyle corpus_examples'a join edip
frozen-deyim / görülmüş-cümle olanları atlayarak append-only idx sürekliliğiyle eklemek.
İlk 2173 satır ASLA dokunulmamalı.

Çalıştır:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ingest_llm.py -q
"""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

fc = importlib.import_module("filter_corpus_idiomaticity")


def _setup(tmp_path, monkeypatch):
    """3 frozen kayıt (idx 0-2) + 1 held-out deyim; auto-LLM label dosyası hazırlanır."""
    recs = tmp_path / "recs.jsonl"
    labs = tmp_path / "labels.tsv"
    tsv = tmp_path / "sample.tsv"
    hold = tmp_path / "holdout.json"
    auto = tmp_path / "auto.jsonl"
    recs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [
        {"idx": 0, "words": ["a", "boy", "attı"], "tags": ["O", "B-VID", "I-VID"], "idiom": "boy atmak", "span": "boy attı"},
        {"idx": 1, "words": ["b"], "tags": ["O"], "idiom": "boy atmak", "span": "x"},
        {"idx": 2, "words": ["c"], "tags": ["O"], "idiom": "göze girmek", "span": "y"},
    ]), encoding="utf-8")
    labs.write_text("idx\tlabel\n0\tD\n1\tL\n2\tD\n", encoding="utf-8")
    tsv.write_text("idx\tspan\tcümle\n", encoding="utf-8")
    hold.write_text(json.dumps(["el atmak"]), encoding="utf-8")   # held-out deyim
    monkeypatch.setattr(fc, "SAMPLE_RECS", recs)
    monkeypatch.setattr(fc, "MANUAL_LABELS", labs)
    monkeypatch.setattr(fc, "SAMPLE_TSV", tsv)
    monkeypatch.setattr(fc, "HOLDOUT_JSON", hold)
    monkeypatch.setattr(fc, "LABELS", auto)
    return recs, labs, auto


def test_ingest_llm_join_dedup_append_only(tmp_path, monkeypatch):
    recs, labs, auto = _setup(tmp_path, monkeypatch)
    frozen_before = recs.read_text(encoding="utf-8")
    labs_before = labs.read_text(encoding="utf-8")

    items = [
        {"words": ["yeni", "cümle", "bir"], "tags": ["O", "B-VID", "O"], "idiom": "kulak vermek", "span": "cümle"},
        {"words": ["baska", "yeni", "cümle"], "tags": ["O", "O", "B-VID"], "idiom": "göz atmak", "span": "cümle"},
        {"words": ["a", "boy", "attı"], "tags": ["O", "B-VID", "I-VID"], "idiom": "boy atmak", "span": "boy attı"},
        {"words": ["held", "out", "cumlesi"], "tags": ["O", "B-VID", "O"], "idiom": "el atmak", "span": "out"},
    ]
    auto.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in [
        {"k": "yeni cümle bir", "label": "D", "idiom": "kulak vermek", "span": "cümle"},
        {"k": "baska yeni cümle", "label": "N", "idiom": "göz atmak", "span": "cümle"},     # N → negatif (L)
        {"k": "yeni cümle bir", "label": "D", "idiom": "kulak vermek", "span": "cümle"},   # tekrar → atla
        {"k": "a boy attı", "label": "N", "idiom": "boy atmak", "span": "boy attı"},        # frozen deyim → atla
        {"k": "held out cumlesi", "label": "D", "idiom": "el atmak", "span": "out"},        # held-out → atla
        {"k": "join yok bu", "label": "D", "idiom": "yok", "span": "x"},                    # join yok → atla
    ]), encoding="utf-8")

    fc.ingest_llm(items)

    # ilk 2173 (burada 3) satır bit-aynı
    assert recs.read_text(encoding="utf-8").startswith(frozen_before)
    assert labs.read_text(encoding="utf-8").startswith(labs_before)
    new_recs = [json.loads(l) for l in recs.read_text(encoding="utf-8").splitlines()][3:]
    assert [r["idx"] for r in new_recs] == [3, 4]                     # idx sürekli
    assert {r["idiom"] for r in new_recs} == {"kulak vermek", "göz atmak"}
    new_labs = labs.read_text(encoding="utf-8").splitlines()[4:]
    assert new_labs == ["3\tD", "4\tL"]


def test_ingest_llm_idempotent(tmp_path, monkeypatch):
    recs, labs, auto = _setup(tmp_path, monkeypatch)
    items = [{"words": ["yeni", "cümle"], "tags": ["O", "B-VID"], "idiom": "kulak vermek", "span": "cümle"}]
    auto.write_text(json.dumps({"k": "yeni cümle", "label": "D", "idiom": "kulak vermek", "span": "cümle"}) + "\n",
                    encoding="utf-8")
    fc.ingest_llm(items)
    after_first = recs.read_text(encoding="utf-8")
    fc.ingest_llm(items)                          # ikinci koşu hiçbir şey eklememeli
    assert recs.read_text(encoding="utf-8") == after_first


def test_cohen_kappa():
    assert fc._cohen_kappa([("D", "D"), ("L", "L")], ("D", "L")) == 1.0
    assert abs(fc._cohen_kappa([("D", "L"), ("L", "D"), ("D", "D"), ("L", "L")], ("D", "L"))) < 1e-9
    assert fc._cohen_kappa([], ("D", "L")) == 0.0
