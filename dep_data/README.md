# BERT Fine-tuning Veri Hazırlığı

## Mevcut Durum

| Veri | Token | Doğruluk |
|------|------|----------|
| Toplam | 169,744 | N/A (gold lemma) |

## Format

BERT için:
- Input: "[CLS] word [SEP] pos [SEP]"
- Output: "[PAD] lemma [PAD]"

## Örnek

```
[CLS] geliyor [SEP] VERB [SEP] --> [PAD] gel [PAD]
[CLS] güzel [SEP] ADJ [SEP] --> [PAD] güzel [PAD]
```

## Sonraki Adımlar

1. HuggingFace formatına çevir
2. Tokenizer ile tokenize et
3. Train/val split yap
4. BERT fine-tune et
