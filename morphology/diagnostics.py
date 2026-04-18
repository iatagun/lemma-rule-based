#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostics: Morfoloji çözümlemesini debug etmek.

Heceleme-farkında açıklama ve izleme fonksiyonları.

Fonksiyonlar:
  - explain_analysis() — adım-adım çözümleme izle
  - trace_syllables() — ek eklenme işlemini heceleme ile göster
  - check_loanword_status() — alıntı/yerli tespiti ayrıntısı
  - compare_with_stanza() — analyzer vs Stanza karşılaştırması
"""

from __future__ import annotations

from typing import Optional

from .phonology import syllabify, get_syllable_nuclei, is_loanword_candidate


class MorphologyDiagnostics:
    """Morfoloji çözümlemesi için diagnostics."""
    
    @staticmethod
    def explain_analysis(
        form: str,
        lemma: Optional[str] = None,
        morphemes: Optional[list[str]] = None,
        upos: str = "X",
    ) -> str:
        """
        Sözcüğün çözümlemesini adım-adım açıkla.
        
        Args:
            form: Sözcüğün yazı biçimi
            lemma: Kök/lemma
            morphemes: Çözümlenen morfemler (ek sırası)
            upos: POS tag
        
        Returns:
            İnsan-okunur açıklama.
        """
        lines: list[str] = []
        lines.append(f"Sözcük: {form}")
        lines.append(f"POS: {upos}")
        
        # Heceleme
        syllables = syllabify(form)
        lines.append(f"Heceleme: {'-'.join(syllables)}")
        
        # Lemma
        if lemma:
            lines.append(f"Kök/Lemma: {lemma}")
        
        # Morfemler
        if morphemes:
            morph_str = " + ".join(morphemes)
            lines.append(f"Morfofonemik çözümleme: {morph_str}")
        
        # Alıntı kontrolü
        if is_loanword_candidate(form):
            lines.append("ℹ Potansiyel alıntı sözcük (o/ö > 1. hecede)")
        
        return "\n  ".join(lines)
    
    @staticmethod
    def trace_syllables(
        form: str,
        lemma: str,
        morphemes: Optional[list[str]] = None,
    ) -> str:
        """
        Ek eklenme işlemini heceleme ile izle.
        
        Adım adım: kök → morfofonemik → son konum
        
        Args:
            form: Sonuç sözcüğü
            lemma: Kök
            morphemes: Çözümlenen morfemler
        
        Returns:
            Heceleme izlemesi.
        """
        lines: list[str] = []
        lines.append(f"Heceleme İzlemesi: {form}")
        lines.append("─" * 50)
        
        # Kökün hecelenmesi
        root_syllables = syllabify(lemma)
        lines.append(f"1. Kök '{lemma}' → {'-'.join(root_syllables)}")
        
        # Morfemler varsa
        if morphemes and len(morphemes) > 1:
            current = lemma
            for i, morph in enumerate(morphemes[1:], 1):
                current = current + morph
                curr_syllables = syllabify(current)
                lines.append(
                    f"{i+1}. + '{morph}' → {'-'.join(curr_syllables)}"
                )
        
        # Son konum
        final_syllables = syllabify(form)
        nuclei = get_syllable_nuclei(form)
        lines.append(f"\nSonuç: {'-'.join(final_syllables)}")
        lines.append(f"Ünlü çekirdekleri: {', '.join(nuclei)}")
        
        return "\n  ".join(lines)
    
    @staticmethod
    def check_loanword_status(word: str) -> str:
        """
        Alıntı sözcük tespiti ayrıntısı.
        
        Args:
            word: Sözcük
        
        Returns:
            Alıntı tespiti raporu.
        """
        lines: list[str] = []
        lines.append(f"Alıntı Sözcük Analizi: {word}")
        lines.append("─" * 50)
        
        syllables = syllabify(word)
        nuclei = get_syllable_nuclei(word)
        is_loanword = is_loanword_candidate(word)
        
        lines.append(f"Heceleme: {'-'.join(syllables)}")
        lines.append(f"Ünlü çekirdekleri: {', '.join(nuclei)}")
        
        # Kuralı açıkla
        lines.append("\nKural:")
        lines.append("  Türkçe yerli: Yuvarlak geniş ünlüler (o, ö)")
        lines.append("               yalnızca 1. hecede bulunur")
        lines.append("  Alıntı: o/ö 2. veya sonraki hecelerde olabilir")
        
        # Tespiti
        if is_loanword:
            # Nerede?
            first_nuclei = nuclei[0] if nuclei else None
            later_nuclei = [n for n in nuclei[1:] if n in ("o", "ö")]
            lines.append(f"\n➤ ALIINTI (potansiyel)")
            lines.append(f"  1. hecenin ünlüsü: {first_nuclei}")
            lines.append(f"  Sonraki hecelerde o/ö: {later_nuclei}")
        else:
            lines.append(f"\n➤ YERLİ SÖZCÜK")
            lines.append(f"  Yuvarlak geniş ünlü yalnızca 1. hecede")
        
        return "\n  ".join(lines)
    
    @staticmethod
    def compare_with_stanza(
        form: str,
        upos_predicted: str,
        upos_stanza: Optional[str] = None,
        syllables_predicted: Optional[list[str]] = None,
    ) -> str:
        """
        Analyzer vs Stanza karşılaştırması.
        
        Args:
            form: Sözcük
            upos_predicted: Analyzer tarafından tahmin edilen POS
            upos_stanza: Stanza'dan gelen POS (optional)
            syllables_predicted: Analyzer'dan heceleme (optional)
        
        Returns:
            Karşılaştırma raporu.
        """
        lines: list[str] = []
        lines.append(f"Analyzer vs Stanza: {form}")
        lines.append("─" * 50)
        
        # Stanza
        if upos_stanza:
            stanza_syllables = syllabify(form)
            agreement = upos_predicted == upos_stanza
            agreement_mark = "✓" if agreement else "✗"
            
            lines.append(f"\nAnalyzer      : POS={upos_predicted}")
            if syllables_predicted:
                lines.append(f"                Heceleme={'-'.join(syllables_predicted)}")
            
            lines.append(f"\nStanza        : POS={upos_stanza}")
            lines.append(f"                Heceleme={'-'.join(stanza_syllables)}")
            
            if agreement:
                lines.append(f"\nSonuç: {agreement_mark} Anlaşma ✓")
            else:
                lines.append(f"\nSonuç: {agreement_mark} Uyuşmazlık ⚠")
        else:
            lines.append(f"\nAnalyzer: POS={upos_predicted}")
            lines.append("Stanza: (veri yok)")
        
        return "\n  ".join(lines)


def explain_analysis(
    form: str,
    lemma: Optional[str] = None,
    morphemes: Optional[list[str]] = None,
    upos: str = "X",
) -> str:
    """Kısayol: explain_analysis()"""
    return MorphologyDiagnostics.explain_analysis(form, lemma, morphemes, upos)


def trace_syllables(
    form: str,
    lemma: str,
    morphemes: Optional[list[str]] = None,
) -> str:
    """Kısayol: trace_syllables()"""
    return MorphologyDiagnostics.trace_syllables(form, lemma, morphemes)


def check_loanword_status(word: str) -> str:
    """Kısayol: check_loanword_status()"""
    return MorphologyDiagnostics.check_loanword_status(word)


def compare_with_stanza(
    form: str,
    upos_predicted: str,
    upos_stanza: Optional[str] = None,
    syllables_predicted: Optional[list[str]] = None,
) -> str:
    """Kısayol: compare_with_stanza()"""
    return MorphologyDiagnostics.compare_with_stanza(
        form, upos_predicted, upos_stanza, syllables_predicted
    )
