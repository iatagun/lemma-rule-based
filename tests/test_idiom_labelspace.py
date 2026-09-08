# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom etiket uzayı + BIO çözümleme testleri.

Çalıştır:  python -X utf8 -m pytest tests/test_idiom_labelspace.py -q
Önkoşul :  python prepare_idiom_data.py --build-label-space
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dizgebert_idiom.modeling_dizgebert_idiom import (  # noqa: E402
    decode_bigappy_spans,
    decode_bio_spans,
    viterbi_decode,
)

pytestmark = pytest.mark.skipif(
    not (ROOT / "idiom_data" / "label_space.json").exists(),
    reason="label_space.json yok (önce prepare_idiom_data.py --build-label-space)",
)

from training.train_idiom_bert import IdiomLabelSpace  # noqa: E402


@pytest.fixture(scope="module")
def ls():
    return IdiomLabelSpace.load()


def test_label_space_has_o_at_zero(ls):
    assert ls.tags[0] == "O"
    assert ls.tag_to_id["O"] == 0


def test_label_space_bio_pairs(ls):
    # her B-X için karşılık gelen I-X mevcut olmalı
    b_cats = {t[2:] for t in ls.tags if t.startswith("B-")}
    i_cats = {t[2:] for t in ls.tags if t.startswith("I-")}
    assert b_cats == i_cats
    assert b_cats == {"VID", "LVC"}


# ─── decode_bio_spans (etiket uzayından bağımsız, saf birim testler) ───
@pytest.mark.parametrize("tags,expected", [
    (["O", "B-VID", "I-VID", "O"], [(1, 3, "VID")]),
    (["B-LVC", "I-LVC", "B-VID", "O"], [(0, 2, "LVC"), (2, 3, "VID")]),
    (["I-VID", "O"], [(0, 1, "VID")]),          # yetim I- → yeni span
    (["B-VID", "I-LVC"], [(0, 1, "VID"), (1, 2, "LVC")]),  # kategori değişti
    ([], []),
    (["O", "O"], []),
    (["B-VID"], [(0, 1, "VID")]),                # span cümle sonunda kapanır
])
def test_decode_bio_spans(tags, expected):
    assert decode_bio_spans(tags) == expected


# ─── viterbi_decode (geçiş-kısıtlı en-iyi-yol) ───
def test_viterbi_fixes_orphan_i_tag():
    import torch
    TAGS = ["O", "B-VID", "I-VID", "B-LVC", "I-LVC"]
    # argmax "O, I-VID, I-VID, O" üretir (yetim I- başlangıcı) ama B-VID ikinci en olası;
    # Viterbi yapısal olarak geçerli "O, B-VID, I-VID, O" yolunu seçmeli.
    logits = torch.tensor([
        [5.0, 0.0, 0.1, 0.0, 0.0],
        [0.0, 1.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0],
    ])
    assert [TAGS[i] for i in logits.argmax(-1).tolist()] == ["O", "I-VID", "I-VID", "O"]
    assert viterbi_decode(logits, TAGS) == ["O", "B-VID", "I-VID", "O"]


# ─── decode_bigappy_spans (bigappy-unicrossy 2-katman birleştirme) ───
def test_bigappy_gapli_span_merged():
    tags1 = ["O", "O", "B-LVC", "O", "O", "O"]
    tags2 = ["o", "o", "o", "o", "o", "b-LVC"]
    assert decode_bigappy_spans(tags1, tags2) == [(2, 3, 5, 6, "LVC")]


def test_bigappy_contiguous_only_unaffected():
    tags1 = ["O", "B-VID", "I-VID", "O"]
    tags2 = ["o", "o", "o", "o"]
    assert decode_bigappy_spans(tags1, tags2) == [(1, 3, "VID")]


def test_bigappy_orphan_layer2_fragment_ignored():
    # katman 2'de kategori eşleşen bir katman-1 parçası yoksa yok sayılır
    tags1 = ["O", "O", "O"]
    tags2 = ["o", "b-VID", "o"]
    assert decode_bigappy_spans(tags1, tags2) == []


def test_viterbi_empty_input_returns_empty():
    import torch
    TAGS = ["O", "B-VID", "I-VID", "B-LVC", "I-LVC"]
    assert viterbi_decode(torch.empty(0, len(TAGS)), TAGS) == []


def test_viterbi_output_always_bio_valid():
    import torch
    TAGS = ["O", "B-VID", "I-VID", "B-LVC", "I-LVC"]
    torch.manual_seed(0)
    for _ in range(50):
        logits = torch.rand(12, len(TAGS))
        path = viterbi_decode(logits, TAGS)
        for prev, cur in zip(["O"] + path, path):
            if cur.startswith("I-"):
                assert prev in (f"B-{cur[2:]}", f"I-{cur[2:]}"), (prev, cur, path)
