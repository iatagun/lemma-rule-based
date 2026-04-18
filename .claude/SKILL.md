---
name: lemma-rule-based
description: Turkish morphological analyzer + BERT dependency parsing
allowed-tools: Bash, Read, Edit, Write
user-invokable: true
---

# lemma-rule-based Project Skills

Bu proje iki ana modülden oluşur:

## Skills

### morphology
Türkçe kural-tabanlı morfolojik çözümleme
- Lemma/ek çıkarma
- ~90% doğruluk (BOUN benchmark)

```bash
python -X utf8 benchmark/evaluate.py
```

### dep_parsing  
BERT + Biaffine dependency parsing
- UD Turkish sözdizim ağacı
- Eğitim devam ediyor

```bash
python -X utf8 train_dep_bert.py --epochs 10
```

### karpathy
Karpathy-inspired coding guidelines
- Think before coding
- Simplicity first
- Surgical changes
- Goal-driven execution

## Detaylı Bilgi

Tam dokümantasyon: `.skills/` klasörü

## Proje Durumu

| Modül | Durum |
|-------|-------|
| morphology | ✅ Aktif |
| dep_parsing | 🔄 Eğitim |
| archive | 25 dosya temizlendi |