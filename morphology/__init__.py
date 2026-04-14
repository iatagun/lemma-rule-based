"""
morphology paketi – Türkçe kural tabanlı morfolojik çözümleme.

Kolay kullanım:
    from morphology import create_default_analyzer, AnalysisFormatter

    analyzer = create_default_analyzer()
    result = analyzer.analyze("evlerinden")

    fmt = AnalysisFormatter()
    print(fmt.format_analysis("evlerinden", result))

Sözlük destekli kullanım:
    analyzer = create_default_analyzer(dictionary_path="turkish_words.txt")
"""

from __future__ import annotations

from pathlib import Path

from .analyzer import (
    DictionaryStemValidator,
    HarmonyStrategy,
    MorphemeAnalysis,
    MorphologicalAnalyzer,
    StemValidator,
)
from .dictionary import TurkishDictionary
from .formatter import AnalysisFormatter
from .sentence import SentenceAnalyzer, SentenceToken
from .dependency import DependencyParser, DepToken
from .harmony import (
    RelaxedHarmonyChecker,
    StrictHarmonyChecker,
    check_word_internal_harmony,
)
from .phonology import get_syllable_nuclei, is_loanword_candidate, syllabify
from .morphotactics import MorphotacticFSM
from .suffix import SuffixRegistry

__all__ = [
    "create_default_analyzer",
    "MorphologicalAnalyzer",
    "MorphemeAnalysis",
    "AnalysisFormatter",
    "SuffixRegistry",
    "HarmonyStrategy",
    "StemValidator",
    "DictionaryStemValidator",
    "TurkishDictionary",
    "StrictHarmonyChecker",
    "RelaxedHarmonyChecker",
    "syllabify",
    "get_syllable_nuclei",
    "is_loanword_candidate",
    "check_word_internal_harmony",
    "SentenceAnalyzer",
    "SentenceToken",
    "DependencyParser",
    "DepToken",
    "MorphotacticFSM",
]


def create_default_analyzer(
    dictionary_path: str | Path | None = None,
) -> MorphologicalAnalyzer:
    """
    Varsayılan yapılandırmayla bir çözümleyici üretir.

    Sözlük dosyası verilirse 4 katmanlı strateji kullanılır:
      1. StrictHarmony + Sözlük doğrulama
      2. RelaxedHarmony + Sözlük doğrulama (alıntı sözcükler)
      3. StrictHarmony + Sezgisel doğrulama (sözlükte olmayan sözcükler)
      4. RelaxedHarmony + Sezgisel doğrulama (son çare)

    Sözlük verilmezse eski davranış (2 katmanlı sezgisel) korunur.
    """
    registry = SuffixRegistry.create_default()

    dictionary: TurkishDictionary | None = None
    dict_validator: DictionaryStemValidator | None = None

    if dictionary_path is not None:
        path = Path(dictionary_path)
        if path.exists():
            dictionary = TurkishDictionary.from_file(path)
            dict_validator = DictionaryStemValidator(dictionary)

    strategies: list[HarmonyStrategy] = []

    # Sözlük destekli stratejiler (varsa, öncelikli)
    if dict_validator is not None:
        strategies.extend([
            HarmonyStrategy(
                checker=StrictHarmonyChecker(),
                stem_validator=dict_validator,
            ),
            HarmonyStrategy(
                checker=RelaxedHarmonyChecker(),
                min_stem_override=3,
                stem_validator=dict_validator,
            ),
        ])

    # Sezgisel stratejiler (sözlük yoksa veya eşleşme bulunamazsa)
    strategies.extend([
        HarmonyStrategy(checker=StrictHarmonyChecker()),
        HarmonyStrategy(
            checker=RelaxedHarmonyChecker(), min_stem_override=3
        ),
    ])

    return MorphologicalAnalyzer(
        registry=registry,
        strategies=strategies,
        dictionary=dictionary,
    )
