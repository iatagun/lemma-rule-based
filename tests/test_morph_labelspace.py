# -*- coding: utf-8 -*-
"""DizgeBERT-Morph etiket uzayı round-trip testleri.

Çalıştır:  python -X utf8 -m pytest tests/test_morph_labelspace.py -q
Önkoşul :  python prepare_morph_data_ud.py --build-label-space
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(
    not (ROOT / "morph_data" / "label_space.json").exists(),
    reason="label_space.json yok (önce prepare_morph_data_ud.py --build-label-space)",
)

from train_morph_bert import LabelSpace  # noqa: E402


@pytest.fixture(scope="module")
def ls():
    return LabelSpace.load()


BUNDLES = [
    "Case=Loc|Number=Sing|Number[psor]=Sing|Person=3|Person[psor]=3",
    "Aspect=Prog|Mood=Ind|Number=Sing|Person=2|Polarity=Pos|Tense=Pres|VerbForm=Fin",
    "Evident=Nfh|Number=Plur|Person=3|Polarity=Pos|Tense=Past",
    "Case=Nom|Number=Sing|Person=3",
    "Polarity=Pos|VerbForm=Part",
    "PronType=Prs",
]


@pytest.mark.parametrize("bundle", BUNDLES)
def test_roundtrip_known_bundles(ls, bundle):
    parsed = ls.parse_feats(bundle)
    # yalnızca label_space'te bilinen (name,value) parçaları beklenir
    known = {
        p for p in bundle.split("|")
        if p.split("=")[0] in ls.feat_to_id and p.split("=")[1] in ls.feat_to_id[p.split("=")[0]]
    }
    rebuilt = ls.feats_to_string(parsed)
    assert set(rebuilt.split("|")) == known


def test_underscore_is_class_zero(ls):
    for name in ls.feat_names:
        assert ls.feat_values[name][0] == "_"
        assert ls.feat_to_id[name]["_"] == 0


def test_all_absent_reassembles_to_underscore(ls):
    empty = {n: "_" for n in ls.feat_names}
    assert ls.feats_to_string(empty) == "_"
    assert ls.parse_feats("_") == empty


def test_canonical_ordering(ls):
    # girişteki sıra ne olursa olsun çıktı harf-duyarsız alfabetik
    scrambled = "VerbForm=Fin|Aspect=Prog|Number=Sing"
    out = ls.feats_to_string(ls.parse_feats(scrambled))
    assert out == "Aspect=Prog|Number=Sing|VerbForm=Fin"
