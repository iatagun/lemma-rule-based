# -*- coding: utf-8 -*-
"""DizgeBERT-Morph HF yapılandırması."""
from __future__ import annotations

from transformers import PretrainedConfig


class DizgeBertMorphConfig(PretrainedConfig):
    model_type = "dizgebert-morph"

    def __init__(
        self,
        encoder_name: str = "dbmdz/electra-base-turkish-cased-discriminator",
        upos_labels: list[str] | None = None,
        xpos_labels: list[str] | None = None,
        feats_label_space: dict[str, list[str]] | None = None,
        treebanks: list[str] | None = None,
        default_scheme: str = "kenet",
        tb_emb_dim: int = 48,
        dropout: float = 0.15,
        max_len: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.upos_labels = upos_labels or []
        self.xpos_labels = xpos_labels or ["_"]
        self.feats_label_space = feats_label_space or {}
        self.treebanks = treebanks or ["kenet", "boun"]
        self.default_scheme = default_scheme
        self.tb_emb_dim = tb_emb_dim
        self.dropout = dropout
        self.max_len = max_len
