# -*- coding: utf-8 -*-
"""DizgeBERT-Joint — tek geçişte UPOS+XPOS+FEATS+HEAD+DEPREL (Türkçe, UD)."""
from .configuration_dizgebert_joint import DizgeBertJointConfig
from .modeling_dizgebert_joint import DizgeBertJointForParsing

__all__ = ["DizgeBertJointConfig", "DizgeBertJointForParsing"]

try:
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("dizgebert-joint", DizgeBertJointConfig)
    AutoModel.register(DizgeBertJointConfig, DizgeBertJointForParsing)
except Exception:
    pass
