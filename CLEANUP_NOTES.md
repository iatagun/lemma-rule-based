# Proje Temizliği — Durum: ✅ TAMAMLANDI

## Temizlik Özeti

| Kategori | Taşınan | Kalan |
|----------|---------|-------|
| Kök demo/*.py | 6 | 0 |
| Kök find/check/*.py | 9 | 0 |
| dep_data/bert/fix | 8 | 0 |
| Kök test/*.py | 2 | 0 |
| **Toplam** | **25** | — |

---

## ✅ Taşınan Dosyalar (archive/)

### Kök Dizin → archive/
```
demo_text.py
demo_sentence.py  
demo_dep.py
test_real.py
check_checkpoint.py
eval_dep.py
test_parser.py
find_specific.py
find_corrections.py
find_patterns.py
analyze_errors.py
check_boun.py
test_boun.py
test_detector.py
find_lemma.py
check_dep.py
test_tagger.py
extract_lemma_data.py
```

### dep_data/bert/ → dep_data/archive/
```
fix_complete.py
fix_final.py
fix_more.py
fix_all.py
fix_train.py
check_more.py
check_data.py
summary.py
```

## Kalan Dosyalar

### Ana Proje (KORUNACAK)
```
morphology/           # Ana paket
benchmark/           # Test
mcp_server/          # MCP server
turkish_words.txt    # Sözlük
AGENTS.md           # Agent talimatları
skill.md            # Uzman analizi
```

### Yeni Eklenen (KORUNACAK)
```
skills.py           # Skill loader
train_dep_bert.py  # BERT+Biaffine eğitim
```

## ✅ Temiz Yapı

```
lemma-rule-based/
├── morphology/           ✓ Ana paket
├── benchmark/             ✓ Test
├── mcp_server/            ✓ MCP server  
├── skills.py              ✓ Skill loader
├── archive/               ✓ Arşiv (25 dosya)
├── dep_data/              ✓ Eğitim verileri
│   ├── bert/
│   └── best_parser.pt
├── turkish_words.txt      ✓ Sözlük
├── skills.py             ✓ Skill manager
├── train_dep_bert.py    ✓ BERT eğitim
├── AGENTS.md            ✓ Agent talimatları
├── skill.md            ✓ Uzman analizi
└── CLEANUP_NOTES.md    ✓ Bu dosya
```