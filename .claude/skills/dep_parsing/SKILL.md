---
name: dep_parsing
description: BERT + Biaffine dependency parsing (UD Turkish)
allowed-tools: Bash, Read, Edit
user-invokable: true
---

# Dependency Parsing

Türkçe sözdizim ağacı çıkarma - head/relation prediction.

## Eğitim

```bash
python -X utf8 train_dep_bert.py --epochs 10
```

## Test

```bash
python -X utf8 train_dep_bert.py --eval --checkpoint dep_data/best_dep_parser.pt
```

## Dosyalar

- `bert_parsing.md` - Dozat & Manning (2017) mimarisi
- `training.md` - Eğitim hyperparameter'ları

## Not

⚠️ Eski `best_parser.pt` yeni mimariyle uyumsuz. Sıfırdan eğit.