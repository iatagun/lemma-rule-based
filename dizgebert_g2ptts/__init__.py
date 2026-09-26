# -*- coding: utf-8 -*-
"""DizgeBERT-G2PTTS — Türkçe TTS ön ucu için sözcük başına vurgu + sınır etiketleyici (hibrit: kural + ELECTRA)."""
from .configuration_dizgebert_g2ptts import DizgeBertG2ptttsConfig

__all__ = ["DizgeBertG2ptttsConfig"]

try:  # modeling_*.py yalnız export klasöründe stress_rules/normalize/symbols yanında çalışır (kaynak: dizgetts/frontend); kaynak ağaçta yalnız config kayıtlı
    from transformers import AutoConfig

    AutoConfig.register("dizgebert-g2ptts", DizgeBertG2ptttsConfig)
except Exception:
    pass
