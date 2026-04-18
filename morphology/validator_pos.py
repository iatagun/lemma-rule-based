#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POS Tag Doğrulama ve Güven Skorlama.

Analyzer çıktısını external POS tag'ler (Stanza vb.) ile
cross-check eder. Heceleme-farkında alıntı sözcük tespiti.

SOLID:
  SRP – Yalnızca POS validasyonundan sorumlu.
  OCP – Yeni confidence metriği kolaylıkla eklenir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .phonology import syllabify, is_loanword_candidate


@dataclass
class ValidationResult:
    """POS doğrulama sonucu."""
    
    # Temel bilgi
    form: str
    lemma: str
    
    # Çözümlemeler
    upos_predicted: str  # Analyzer çıktısı
    upos_external: Optional[str] = None  # Stanza vb.
    
    # Heceleme analizi
    syllables: list[str] = field(default_factory=list)
    is_loanword_likely: bool = False
    
    # Güven
    confidence: float = 0.0
    confidence_factors: dict[str, float] = field(default_factory=dict)
    
    # Açıklama
    explanation: str = ""
    
    @property
    def upos_agreement(self) -> bool:
        """Predicted ve external POS uyuşuyor mu?"""
        if self.upos_external is None:
            return True
        return self.upos_predicted == self.upos_external


class POSValidator:
    """
    Analyzer çıktısını doğrula ve güven skoru hesapla.
    
    Heceleme yardımcı olarak alıntı sözcük tespiti yapar.
    """
    
    def __init__(self):
        pass
    
    def validate(
        self,
        form: str,
        lemma: str,
        upos_predicted: str,
        upos_external: Optional[str] = None,
    ) -> ValidationResult:
        """
        POS doğrulama yap.
        
        Args:
            form: Sözcüğün yazı biçimi
            lemma: Lemma
            upos_predicted: Analyzer tarafından tahmin edilen POS tag
            upos_external: Harici POS tag (Stanza vb.)
        
        Returns:
            ValidationResult içerisinde çıktı.
        """
        # Heceleme
        syllables = syllabify(form)
        is_loanword = is_loanword_candidate(form)
        
        result = ValidationResult(
            form=form,
            lemma=lemma,
            upos_predicted=upos_predicted,
            upos_external=upos_external,
            syllables=syllables,
            is_loanword_likely=is_loanword,
        )
        
        # Güven skoru hesapla
        self._calculate_confidence(result)
        
        # Açıklama oluştur
        self._generate_explanation(result)
        
        return result
    
    def _calculate_confidence(self, result: ValidationResult) -> None:
        """Güven skoru hesapla (heceleme-farkında)."""
        
        factors = {}
        
        # 1. POS uyuşması
        if result.upos_external is not None:
            if result.upos_agreement:
                factors["upos_agreement"] = 1.0
            else:
                factors["upos_agreement"] = 0.6
        else:
            factors["upos_agreement"] = 1.0  # Harici veri yok → ignore
        
        # 2. Heceleme-tabanlı loanword tespiti
        if result.is_loanword_likely:
            # Alıntı sözcükler RelaxedHarmony ile çalışır
            # Confidence hafif düşer (belirsizlik var)
            factors["loanword_likelihood"] = 0.85
        else:
            # Yerli sözcükler kesin
            factors["loanword_likelihood"] = 1.0
        
        # 3. POS tipi doğrulama (ek kurallar)
        # Örn: VERB'ler sözlük kırmış mı vs
        factors["upos_consistency"] = 1.0  # Şimdilik neutral
        
        # Ortalama
        result.confidence_factors = factors
        result.confidence = sum(factors.values()) / len(factors)
    
    def _generate_explanation(self, result: ValidationResult) -> None:
        """İnsan-okunur açıklama oluştur."""
        
        parts: list[str] = []
        
        # Heceleme bilgisi
        parts.append(f"Heceleme: {'-'.join(result.syllables)}")
        
        # Loanword durumu
        if result.is_loanword_likely:
            parts.append(f"Sözcük tipi: Alıntı (potansiyel)")
        else:
            parts.append(f"Sözcük tipi: Yerli")
        
        # POS uyuşması
        if result.upos_external is not None:
            if result.upos_agreement:
                parts.append(f"POS tag: {result.upos_predicted} ✓ (Stanza eşleşiyor)")
            else:
                parts.append(
                    f"POS tag: {result.upos_predicted} ⚠ "
                    f"(Stanza: {result.upos_external})"
                )
        else:
            parts.append(f"POS tag: {result.upos_predicted} (no external)")
        
        # Güven
        confidence_pct = result.confidence * 100
        if confidence_pct >= 95:
            conf_mark = "✓ Yüksek"
        elif confidence_pct >= 85:
            conf_mark = "✓ Orta"
        else:
            conf_mark = "⚠ Düşük"
        parts.append(f"Güven: {confidence_pct:.1f}% ({conf_mark})")
        
        result.explanation = " | ".join(parts)
    
    def batch_validate(
        self,
        analyses: list[dict],
    ) -> list[ValidationResult]:
        """
        Çoklu analizi bir seferde doğrula.
        
        Args:
            analyses: Her biri form, lemma, upos_predicted, upos_external 
                     (optional) içeren dict listesi
        
        Returns:
            ValidationResult listesi.
        """
        results = []
        for analysis in analyses:
            result = self.validate(
                form=analysis.get("form", ""),
                lemma=analysis.get("lemma", ""),
                upos_predicted=analysis.get("upos_predicted", "X"),
                upos_external=analysis.get("upos_external"),
            )
            results.append(result)
        return results
    
    def filter_by_confidence(
        self,
        results: list[ValidationResult],
        threshold: float = 0.85,
    ) -> tuple[list[ValidationResult], list[ValidationResult]]:
        """
        Güven skoru ile filtrele.
        
        Returns:
            (confident_analyses, uncertain_analyses)
        """
        confident = [r for r in results if r.confidence >= threshold]
        uncertain = [r for r in results if r.confidence < threshold]
        return confident, uncertain
