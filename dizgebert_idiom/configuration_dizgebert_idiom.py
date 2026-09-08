# -*- coding: utf-8 -*-
"""DizgeBERT-Idiom HF yapılandırması."""
from __future__ import annotations

from transformers import PretrainedConfig


class DizgeBertIdiomConfig(PretrainedConfig):
    model_type = "dizgebert-idiom"

    def __init__(
        self,
        encoder_name: str = "dbmdz/electra-base-turkish-cased-discriminator",
        tags: list[str] | None = None,
        tags2: list[str] | None = None,
        dropout: float = 0.15,
        max_len: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.tags = tags or ["O"]
        # bigappy-unicrossy tarzı 2. katman — YALNIZ gap'li (süreksiz) span'lerin 2. parçası
        # için (bkz. prepare_idiom_data.py docstring). Katman 1 tüm bitişik span'leri taşır.
        self.tags2 = tags2 or ["o"]
        self.dropout = dropout
        self.max_len = max_len
