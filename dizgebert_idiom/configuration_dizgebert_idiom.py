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
        stage2: bool = False,
        stage2_thresh: float = 0.3,
        ensemble: bool = False,
        ensemble_extra2: bool = False,
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
        # Aşama-2 idyomatiklik sınıflandırıcısı (Fikir 3 — iki aşamalı boru hattı):
        # ayrı ELECTRA gövdesi + span ilk⊕son pooling → {idyomatik, literal}. Açıksa
        # predict_spans() bitişik VID adaylarını filtreden geçirir, "güvenli literal"
        # olanı eler (LVC + gap'li span'ler dokunulmaz). Yalnız stage2 ağırlıkları
        # pakete gömülüyse anlamlı. Varsayılan eşik 0.3 (v8 sonrası, 2026-09-20) —
        # Çavuşoğlu eşik-taramasında 0.5'ten ölçülebilir şekilde daha iyi çıktı (doğru-ayırt
        # %65.2→%68.2, yanlış-poz %21.2→%14.6), bkz. MODEL_CARD.md.
        self.stage2 = stage2
        self.stage2_thresh = stage2_thresh
        # Deney O — stage-1 checkpoint ensemble'ı (bkz. .claude/skills/idiom/SKILL.md).
        # Açıksa ikinci bir tam stage-1 gövdesi (`encoder_b`/`tag_head_b`/`tag_head2_b`)
        # kurulur; iki gövdenin aday span'leri birleşim/oy-sayımıyla birleştirilir, sonra
        # (varsa) aynı stage-2 filtresinden geçirilir. Farklı veri dilimleriyle eğitilmiş
        # iki checkpoint (vE+vL) görülmemiş-deyim doğru-ayırtta tek modelden daha iyi
        # ölçüldü; lineage-yakın çiftler (vL+vF) YARARLI DEĞİL — bkz. deney günlüğü.
        self.ensemble = ensemble
        # Deney X — 3. bağımsız stage-1 gövdesi (`encoder_c`/`tag_head_c`/`tag_head2_c`).
        # Yalnız `ensemble=True` ile birlikte anlamlı; union(a,b,c) görülmemiş-deyim
        # doğru-ayırtta union(a,b)'yi de geçti (bkz. deney günlüğü, Deney X).
        self.ensemble_extra2 = ensemble_extra2
