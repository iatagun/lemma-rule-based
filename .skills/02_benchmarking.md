# Benchmark ve Test Prosedürleri

> Her değişiklikten sonra benchmark çalıştır!

---

## Benchmark Çalıştırma

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -X utf8 benchmark/evaluate.py
```

**Çıktı:**
```
Toplam token (PUNCT hariç) : 10182
Doğru                      : 9018
Yanlış                     : 1164
Genel doğruluk             : 88.6%
```

---

## Regresyon Kontrolü

Her değişiklikten sonra benchmark çalıştır. Kabul edilebilir sonuçlar:

| Kriter | Minimum | Uyarı |
|---------|---------|-------|
| Genel doğruluk | ≥ %88.6 | %87.5 altı |
| word==lemma | ≥ %95 | %94 altı |
| Herhangi bir POS | ≤ %1 düşüş | > %1 düşüş |

---

## POS Bazlı Mevcut Durum

```
✓ DOKUNMA (≥99%)
  DET:99.8%  PART:100%  SCONJ:100%  CCONJ:99.7%

⚠️ DİKKATLİ OL (96-99%)
  PRON:96.0%  ADP:96.2%  AUX:96.2%

📈 İYİLEŞTİRİLEBİLİR (88-96%)
  ADV:94.8%  ADJ:92.7%  NUM:88.0%

🎯 ANA HEDEF (<88%)
  NOUN:87.3%  PROPN:80.2%  VERB:82.5%
```

---

## Hızlı Test Komutları

### Tek Sözcük Testi
```powershell
python -X utf8 -c "
from morphology import create_default_analyzer
a = create_default_analyzer(dictionary_path='turkish_words.txt')
r = a.analyze('evlerinden', upos='NOUN')
print(f'stem={r.stem}, suffixes={r.suffixes}')
"
```

### Birden Fazla Çözümleme
```python
results = a.analyze_all('gelirin', upos='VERB', max_results=5)
for i, r in enumerate(results):
    print(f'{i+1}. stem={r.stem}, lemma={r.lemma}, suffixes={r.suffixes}')
```

### Sözlük Kontrolü
```python
from morphology import TurkishDictionary
d = TurkishDictionary.from_file('turkish_words.txt')
print(d.contains('ev'))        # True
print(d.find_root('kitab'))    # kitap
```

---

## Test Sözcükleri

### Ünlü Uyumu
```
evler         → ev+ler
okullar       → okul+lar
görüşüyorum   → gör+üş+üyor+um
```

### Ünsüz Benzeşmesi
```
gittim        → git+tim
geldi         → gel+di
kitabı        → kitap+ı
```

### Türetme
```
yazdırıldı    → yaz+dır+ıl+dı
güzel         → güzel
```

### Düzensiz Fiiller
```
dedi          → de+di
yedi          → ye+di
```

---

## Birim Testleri

Testler `tests/` dizininde:

```powershell
python -X utf8 -m pytest tests/ -v
```

---

## En Sık Hatalar (Gözlemlenen)

| Sözcük | Beklenen | Bulunan | POS | Öncelik |
|--------|----------|---------|-----|---------|
| çıkarmaz | çık | çıkar | VERB | Yüksek |
| ister | iste | ister | VERB | Orta |
| hayattan | hayat(ı) | hayat | NOUN | Orta |
| cezaevi | cezaev | cezaevi | NOUN | Düşük |
