# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom — Türkçe deyim (VID) / eşdizim (LVC.full) span etiketleyici."""
from .configuration_dizgebert_idiom import DizgeBertIdiomConfig
from .modeling_dizgebert_idiom import (
    DizgeBertIdiomForTokenClassification,
    align_words,
    decode_bigappy_spans,
    decode_bio_spans,
    viterbi_decode,
)

__all__ = ["DizgeBertIdiomConfig", "DizgeBertIdiomForTokenClassification",
           "align_words", "decode_bigappy_spans", "decode_bio_spans", "viterbi_decode"]

try:
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("dizgebert-idiom", DizgeBertIdiomConfig)
    AutoModel.register(DizgeBertIdiomConfig, DizgeBertIdiomForTokenClassification)
except Exception:
    pass
