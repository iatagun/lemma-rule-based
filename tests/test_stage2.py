# -*- coding: utf-8 -*-
"""İki-aşamalı boru hattı (Fikir 3) — yayınlanan v3 davranışının regresyon testleri.

Kod incelemesinde bulundu: stage-2 mantığı (span-dict dönüşümü + skorlama) ÜÇ yerde
kopyalanmıştı ve `-LIT` işlemesinde çoktan sapmıştı; tek kaynağa indirildi
(`spans_from_bigappy`, `span_p_literal`). Ayrıca export'taki anahtar remap'inin
(`encoder.*`→`stage2_encoder.*`) hiç testi yoktu.

Çalıştır:  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_stage2.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dizgebert_idiom.modeling_dizgebert_idiom import spans_from_bigappy  # noqa: E402


# ─── spans_from_bigappy — decode_bigappy_spans çıktısı → span sözlükleri ───
WORDS = "Yalanları ortaya çıkınca patron gözden düştü .".split()


def test_contiguous_span():
    got = spans_from_bigappy([(4, 6, "VID")], WORDS)
    assert got == [{"text": "gözden düştü", "start": 4, "end": 6,
                    "category": "VID", "gappy": False}]


def test_gappy_span_5tuple():
    ws = "sahip olduğu geniş yetkilerle olarak".split()
    got = spans_from_bigappy([(0, 1, 4, 5, "VID")], ws)
    assert got[0]["gappy"] is True
    assert got[0]["text"] == "sahip ... olarak"
    assert (got[0]["start"], got[0]["end"], got[0]["start2"], got[0]["end2"]) == (0, 1, 4, 5)


def test_lit_category_dropped_by_default():
    # Fikir 4 kalıntısı: -LIT (deyim-biçimin literal kullanımı) gerçek span değil
    assert spans_from_bigappy([(4, 6, "VID-LIT")], WORDS) == []


def test_lit_category_kept_when_asked():
    got = spans_from_bigappy([(4, 6, "VID-LIT")], WORDS, keep_literal=True)
    assert got == [{"text": "gözden düştü", "start": 4, "end": 6,
                    "category": "VID", "literal": True, "gappy": False}]


def test_mixed_real_and_lit():
    got = spans_from_bigappy([(1, 3, "VID"), (4, 6, "VID-LIT")], WORDS)
    assert [s["category"] for s in got] == ["VID"]


# ─── export anahtar remap: IdiomaticityClf.state_dict → stage2_encoder.* / stage2_head.* ───
def _remap(k: str) -> str:
    """train_idiom_bert.export_hf içindeki remap kuralının kopyası — sözleşmeyi kilitler."""
    return ("stage2_" + k) if k.startswith("head.") else k.replace("encoder.", "stage2_encoder.", 1)


@pytest.mark.parametrize("src,dst", [
    ("head.weight", "stage2_head.weight"),
    ("head.bias", "stage2_head.bias"),
    ("encoder.embeddings.word_embeddings.weight", "stage2_encoder.embeddings.word_embeddings.weight"),
    ("encoder.encoder.layer.11.output.LayerNorm.bias", "stage2_encoder.encoder.layer.11.output.LayerNorm.bias"),
])
def test_stage2_key_remap(src, dst):
    assert _remap(src) == dst


def test_stage2_key_remap_only_first_encoder_replaced():
    # 'encoder.encoder...' → yalnız ilk 'encoder.' değişmeli
    assert _remap("encoder.encoder.layer.0.attention.self.query.weight") == \
        "stage2_encoder.encoder.layer.0.attention.self.query.weight"
