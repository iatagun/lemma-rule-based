# -*- coding: utf-8 -*-
"""DizgeBERT-Morph — UD-uyumlu ELECTRA morfolojik belirsizlik gidericisi (Türkçe)."""
from .configuration_dizgebert_morph import DizgeBertMorphConfig
from .modeling_dizgebert_morph import DizgeBertMorphForMorphology

__all__ = ["DizgeBertMorphConfig", "DizgeBertMorphForMorphology"]

try:
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("dizgebert-morph", DizgeBertMorphConfig)
    AutoModel.register(DizgeBertMorphConfig, DizgeBertMorphForMorphology)
except Exception:
    pass
