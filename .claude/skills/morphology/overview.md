# Morphology Overview

>Detaylı bilgi için: `.skills/00_overview.md`

## Mimari

- **4 katmanlı strateji:** StrictHarmony → Relaxed → Sezgisel → Fallback
- **Benchmark:** BOUN Treebank (10,182 token)
- **Mevcut doğruluk:** ~89.9%

## Dosya Yapısı

```
morphology/
├── analyzer.py      # Ana motor (1700+ satır)
├── suffix.py      # 191 ek tanımı
├── harmony.py    # Ünlü/ünsüz uyumu
├── dictionary.py # TDK sözlük
└── ...
```

## API

```python
from morphology import create_default_analyzer
a = create_default_analyzer()
r = a.analyze("elma", upos="NOUN")
# r.stem, r.suffixes, r.lemma
```

---

Detaylı: `.skills/00_overview.md`