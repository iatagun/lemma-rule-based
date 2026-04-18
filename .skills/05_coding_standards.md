# Kodlama Standartları ve Katkı Rehberi

---

## Genel Kurallar

### PowerShell / Python

```powershell
# Türkçe karakter içeren komutlar
$env:PYTHONIOENCODING = "utf-8"
python -X utf8 script.py

# Windows yolları
C:\Users\...\lemma-rule-based
```

### Dosya Organizasyonu

| Dosya | Sorumluluk | Değiştirme Kuralı |
|-------|-----------|-------------------|
| `phonology.py` | Ses sınıflandırma | Yalnızca fonetik değişiklikler |
| `harmony.py` | Uyum kontrolü | Yalnızca uyum kuralları |
| `suffix.py` | Ek tanımları | Yeni ek eklerken |
| `dictionary.py` | Sözlük | Ses değişim kuralları |
| `analyzer.py` | Ana motor | **Dikkat!** 1700+ satır |
| `morphotactics.py` | FSM | Yalnızca geçiş tablosu |

---

## Kod Stili

### Docstring Formatı

```python
def find_root(self, stem: str) -> str | None:
    """
    Kök adayının sözlük biçimini bulmaya çalışır.

    Sırasıyla dener:
      1. Doğrudan eşleşme
      2. Fiil kökü (kök + mak/mek)
      ...

    Returns:
        Sözlük biçimi veya None.
    """
```

### Fonksiyon Adlandırma

| Tür | Örnek |
|-----|-------|
| Sınıf | `MorphologicalAnalyzer` |
| Metot | `analyze`, `_strip_suffixes_all` |
| Sabit | `VOWELS`, `_FORBIDDEN_SUFFIX_BIGRAMS` |
| Değişken | `stem`, `sfxs`, `upos` |

### Tip İpuçları

```python
def analyze(self, word: str, upos: str | None = None) -> MorphemeAnalysis:
    """..."""
    
def _strip_suffixes_all(
    self,
    word: str,
    max_results: int = 5,
    upos: str | None = None,
) -> list[tuple[str, list[tuple[str, str]]]]:
```

---

## Commit Mesajları

```
feat: yeni ek ekleme (-DIkçA zarf-fiil)
fix: circumflex lemma düzeltmesi (hal→hâl)
refactor: _rank_analyses ayrıştırma
test: yeni test sözcükleri ekleme
docs: AGENTS.md güncelleme
```

---

## Değişiklik Öncesi Kontrol Listesi

- [ ] Değişiklik ne yapıyor? (tek sorumluluk)
- [ ] Başka bir dosyı etkiliyor mu?
- [ ] Benchmark regresyon riski var mı?
- [ ] Yeni ek ise, morfotaktik FSM güncellendi mi?
- [ ] Test edildi mi?

---

## Yasaklar

❌ `analyzer.py`'de büyük yeniden yapılanma
❌ Benchmark sonuçlarını düşüren değişiklikler
❌ POS bazlı > %1 regresyon
❌ Kod içinde yorum ekleme (isterseniz açıklarım)
