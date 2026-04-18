---
name: morphology
description: Turkish rule-based morphological analyzer - lemma/suffix decomposition
allowed-tools: Bash, Read, Edit
user-invokable: true
---

# Turkish Morphology Analyzer

Morpholoji görevleri için bu skill'i kullan.

## Kullanım

```
word="elma"
upos="NOUN"
result = analyze(word, upos)  # stem=el, suffixes=[ma]
```

## Temel Komutlar

```bash
# Benchmark test
python -X utf8 benchmark/evaluate.py

# Tek sözcük analizi
python -X utf8 -c "from morphology import *; print(analyze('evler', 'NOUN'))"
```

## Dosyalar

- `overview.md` - Proje mimarisi
- `phonology.md` - Ses bilgisi kuralları  
- `benchmarking.md` - Test prosedürleri
- `adding_suffixes.md` - Yeni ek ekleme
- `suffix_reference.md` - 191 ek referansı