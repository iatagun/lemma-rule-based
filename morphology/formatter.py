"""
Çözümleme sonuçlarının biçimlendirilmesi.

SOLID:
  SRP – Yalnızca gösterimle ilgilenir; hesaplama yapmaz.
  ISP – Çağıranlar MorphemeAnalysis alır, I/O gerektirmez.
"""

from __future__ import annotations

from .analyzer import MorphemeAnalysis
from .harmony import check_major_harmony, check_minor_harmony
from .phonology import BACK_VOWELS, UNROUNDED, get_vowels


class AnalysisFormatter:
    """Morfolojik çözümleme sonuçlarını metin olarak biçimlendirir."""

    def format_analysis(self, word: str, analysis: MorphemeAnalysis) -> str:
        """Ayrıntılı çözümleme raporu üretir."""
        lines: list[str] = []
        lines.append(f"  Sözcük : {word}")

        # Kök: temel kök biçimi (root > stem)
        kok = analysis.root if analysis.root else analysis.stem
        lines.append(f"  Kök    : {kok}")

        # Gövde: türetilmiş gövde (yalnızca kökten farklıysa)
        if analysis.lemma and analysis.lemma != kok:
            lines.append(f"  Lemma  : {analysis.lemma}")

        if analysis.suffixes:
            sfx_str = " + ".join(
                f"-{s[0]} ({s[1]})" for s in analysis.suffixes
            )
            lines.append(f"  Ekler  : {sfx_str}")
        else:
            lines.append(f"  Ekler  : (ek bulunamadı)")

        lines.append(f"  Ayrım  : {' + '.join(analysis.parts)}")
        lines.append(f"  Uyum   :")
        lines.append(self.vowel_harmony_report(word))
        lines.append("")  # trailing newline
        return "\n".join(lines)

    def format_multi_analysis(
        self, word: str, analyses: list[MorphemeAnalysis],
    ) -> str:
        """Birden fazla çözümlemeyi biçimlendirir.

        Tek çözümleme varsa normal rapor üretir.  Birden fazla varsa
        her birini numaralayarak listeler ve bağlam bilgisi ekler.
        """
        if not analyses:
            return ""
        if len(analyses) == 1:
            return self.format_analysis(word, analyses[0])

        lines: list[str] = [f"  Sözcük : {word}"]
        lines.append(
            f"  ⚠ {len(analyses)} olası çözümleme"
            " (doğru olan bağlama göre belirlenir)"
        )
        lines.append("")

        for i, analysis in enumerate(analyses, 1):
            kok = analysis.root if analysis.root else analysis.stem
            lines.append(f"  ── Çözümleme {i} {'─' * 30}")
            lines.append(f"  Kök    : {kok}")
            if analysis.lemma and analysis.lemma != kok:
                lines.append(f"  Lemma  : {analysis.lemma}")
            if analysis.suffixes:
                sfx_str = " + ".join(
                    f"-{s[0]} ({s[1]})" for s in analysis.suffixes
                )
                lines.append(f"  Ekler  : {sfx_str}")
            else:
                lines.append(f"  Ekler  : (ek bulunamadı)")
            lines.append(f"  Ayrım  : {' + '.join(analysis.parts)}")
            lines.append("")

        lines.append("  Uyum   :")
        lines.append(self.vowel_harmony_report(word))
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def vowel_harmony_report(word: str) -> str:
        """Sözcükteki ardışık ünlü çiftlerinin uyum durumunu raporlar."""
        vowels = get_vowels(word)
        if len(vowels) < 2:
            return "  (tek ünlülü sözcük – uyum kontrolü gerekmez)"

        lines: list[str] = []
        for i in range(1, len(vowels)):
            prev, curr = vowels[i - 1], vowels[i]
            maj = "✓" if check_major_harmony(prev, curr) else "✗"
            mnr = "✓" if check_minor_harmony(prev, curr) else "✗"

            prev_desc = (
                ("kalın" if prev in BACK_VOWELS else "ince")
                + ", "
                + ("düz" if prev in UNROUNDED else "yuvarlak")
            )
            curr_desc = (
                ("kalın" if curr in BACK_VOWELS else "ince")
                + ", "
                + ("düz" if curr in UNROUNDED else "yuvarlak")
            )

            lines.append(
                f"  {prev} ({prev_desc}) → {curr} ({curr_desc})"
                f"  BÜU:{maj}  KÜU:{mnr}"
            )
        return "\n".join(lines)
