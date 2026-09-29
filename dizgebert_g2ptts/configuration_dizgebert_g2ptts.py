# -*- coding: utf-8 -*-
"""DizgeBERT-G2PTTS HF yapılandırması."""
from __future__ import annotations

from transformers import PretrainedConfig


class DizgeBertG2ptttsConfig(PretrainedConfig):
    model_type = "dizgebert-g2ptts"

    def __init__(
        self,
        encoder_name: str = "dbmdz/electra-base-turkish-cased-discriminator",
        dropout: float = 0.1,
        n_stress: int = 5,       # 0 = vurgusuz, 1+r = vurgulu seslem sondan r (r>=3 -> 4)
        n_boundary: int = 3,     # 0 = sınır yok (<60 ms), 1 = ip (60-250 ms), 2 = IP (>=250 ms)
        tau: float = 0.25,       # sınır eşiği: P(ip)+P(IP) > tau -> sınır (val'de seçildi)
        max_subwords: int = 254,  # eğitimdeki max_length=256 - [CLS] - [SEP]; uzun metin cümle sınırlarında parçalanır
        lexical_tiers: list[str] | None = None,  # kural yolunda uygulanan sözlüğe kapılı katmanlar (pekiştirme, -CIk sıfat)
        rules_version: str = "",  # dışa aktarmadaki kural+sözlük özeti (resources/ + stress_rules.py)
        train_info: dict | None = None,
        phonemes: bool = True,           # tag() sesbirim de üretsin (dizge==0.1.6 + söyleyiş sözlüğü + uzun ünlü kuralları); v0 paketlerinde alan yok -> açık
        pron_exceptions: bool = True,    # resources/pronunciation_exceptions.tsv (alıntı sözcükler, kağıt/hakim, şapka)
        length_rules: bool = True,       # ğ/y uzun ünlü kuralları
        register: str = "özenli",        # "özenli" | "gündelik" (iddaa, kılinik, kaat)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.dropout = dropout
        self.n_stress = n_stress
        self.n_boundary = n_boundary
        self.tau = tau
        self.max_subwords = max_subwords
        self.lexical_tiers = lexical_tiers or ["pek", "cik"]
        self.rules_version = rules_version
        self.train_info = train_info or {}
        self.phonemes = phonemes
        self.pron_exceptions = pron_exceptions
        self.length_rules = length_rules
        self.register = register
