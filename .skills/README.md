# Skill Dosyaları İndexi

Bu dizin, lemma-rule-based projesi için tüm skill dosyalarını içerir.

## 🎯 Hızlı Başlangıç

```
Görev ne?
├── Morfoloji (lemma/ek)     → 00_overview.md + 01_phonology.md
├── Benchmark/test           → 02_benchmarking.md
├── Yeni ek ekleme          → 03_adding_suffixes.md
├── Sorun giderme           → 04_debugging.md
└── Dependency parsing     → 07_bert_dep_parsing.md
```

## 📁 Proje Yapısı

Proje iki bağımsız modülden oluşur:

| Modül | Amaç | Durum |
|-------|------|-------|
| **morphology/** | Kural-tabanlı morfolojik çözümleme | ✅ Aktif |
| **Root** | BERT+Biaffine dependency parsing | 🔄 Eğitim başladı |

## Dosyalar

### Core Skills (Morfoloji)

| Dosya | İçerik | Öncelik |
|-------|--------|---------|
| `00_overview.md` | Proje özeti, mimari, API | ⭐⭐⭐ |
| `01_phonology.md` | Ses bilgisi, ünlü/ünsüz uyumu | ⭐⭐⭐ |
| `02_benchmarking.md` | Test prosedürleri, regresyon kontrolü | ⭐⭐⭐ |
| `03_adding_suffixes.md` | Yeni ek ekleme rehberi | ⭐⭐ |
| `04_debugging.md` | Hata ayıklama | ⭐⭐ |
| `05_coding_standards.md` | Kod stili | ⭐ |
| `06_suffix_reference.md` | Ek referansı (191 ek) | ⭐⭐⭐ |

### Dependency Parsing Skills

| Dosya | İçerik | Öncelik |
|-------|--------|---------|
| `07_bert_dep_parsing.md` | BERT + Biaffine eğitim, Dozat&Manning (2017) | ⭐⭐⭐ |

## Kullanım Sırası

```
1. Görev tipini belirle:
   - Morfoloji (lemma/ek çözümleme) → 00-06 dosyaları
   - Dependency parsing (sözdizim ağacı) → 07 dosyası

2. İlgili skill dosyasını oku

3. AGENTS.md'deki talimatları kontrol et

4. Benchmark ile doğrula
```

## Önemli Notlar

1. **Aynı modeli kullanma**: Eski `best_parser.pt` (farklı mimari) ile yeni `train_dep_bert.py` uyumsuz. Sıfırdan eğit.

2. **Training devam ediyor**: Eğitim ~1 saat/epoch. Batch size 4, grad accum 4.

3. **Skill güncelleme**: Her yeni özellik eklemeden önce ilgili skill dosyasını güncelle.

## Kaynaklar

- `AGENTS.md` — Ana agent talimatları
- `skill.md` — Uzman panel analizi
- `ARCHITECTURE.md` — Mimari dokümantasyon
- `dep_data/bert/` — Eğitim verileri (13,842 train / 2,553 dev / 2,562 test)