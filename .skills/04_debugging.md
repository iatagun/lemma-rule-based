# Hata Ayıklama Rehberi

> Çözümleme sorunlarını nasıl tespit edip düzelteceğiniz.

---

## Yaygın Sorunlar

### 1. Yanlış Kök Seçimi

**Belirti:** `halinde → hali` (hâl olmalı)

**Kontrol Listesi:**
```python
# Sözlükte var mı?
d = TurkishDictionary.from_file('turkish_words.txt')
print(d.contains('hal'))  # True olmalı

# find_root ne döndürüyor?
print(d.find_root('hal'))  # 'hal' veya 'hâl' olmalı

# Analiz sonuçlarını incele
a = create_default_analyzer(dictionary_path='turkish_words.txt')
results = a.analyze_all('halinde', upos='NOUN', max_results=5)
for r in results:
    print(f'stem={r.stem}, lemma={r.lemma}, root={r.root}')
```

### 2. Ek Eşleşmiyor

**Belirti:** `kitabı → kitap` (kitab+ı olmalı)

**Kontrol:**
```python
# Suffix tablosunda var mı?
for sfx in a._registry.suffixes:
    if 'sı' in sfx.form or 'si' in sfx.form:
        print(f'{sfx.form} -> {sfx.label}')

# Ünsüz yumuşaması gerekli mi?
# kitap→kitab için _SOFTENING_REVERSE tablosu kullanılır
```

### 3. Uyum Hatası

**Belirti:** Uyumlu olmayan ekler kabul ediliyor veya reddediliyor

```python
from morphology.harmony import check_vowel_harmony

# Manuel kontrol
result = check_vowel_harmony("ev", "ler")
print(result)  # (True/False, açıklama)

# Hangi ünlüler algılanıyor?
from morphology.phonology import get_vowels, last_vowel
print(get_vowels("ev"))      # ['e']
print(last_vowel("ev"))       # 'e'
```

---

## Debug Modu

### Analiz İzleme

```python
from morphology import create_default_analyzer
from morphology.diagnostics import explain_analysis

analyzer = create_default_analyzer(dictionary_path='turkish_words.txt')
result = analyzer.analyze('evlerinden', upos='NOUN')

# Çözümlemeyi açıkla
explain = explain_analysis('evlerinden', result)
print(explain)
```

### BFS Adımlarını Görme

```python
# _strip_suffixes_all sonuçlarını incele
raw = analyzer._strip_suffixes_all('halinde', max_results=20, upos='NOUN')
print(f"Found {len(raw)} decomposition paths")
for stem, sfxs in raw[:10]:
    print(f"  {stem} + {sfxs}")
```

---

## Sözlük Sorunları

### Sözcük Eksik

```python
# Yeni sözcük ekle
with open('turkish_words.txt', 'a', encoding='utf-8') as f:
    f.write('yenisözcük\n')

# Veya geçici olarak test et
from morphology.dictionary import TurkishDictionary
d = TurkishDictionary.from_file('turkish_words.txt')
d._words |= {'yenisözcük'}
```

### Şapkalı Biçim Sorunu

BOUN standardına göre `hâl`, `âdet`, `kâr` gibi lemmalar korunmalı.

```python
# dictionary.py'de _CIRCUMFLEX_MAPPING kontrol et
from morphology.dictionary import _CIRCUMFLEX_MAPPING
print(_CIRCUMFLEX_MAPPING)

# find_root sonucunu doğrula
d = TurkishDictionary.from_file('turkish_words.txt')
print(d.find_root('halin'))  # 'hâl' olmalı
```

---

## Morfotaktik Sorunları

### Geçersiz Ek Sırası

**Belirti:** Dilbilgisel olarak imkânsız bir ek kombinasyonu kabul ediliyor

```python
# FSM geçişlerini kontrol et
fsm = analyzer._fsm
print(fsm.can_follow("SIFAT_FİİL", "KİŞİ_1Ç"))  # False olmalı
```

### Forbidden Bigram Ekleme

```python
# analyzer.py'de _FORBIDDEN_SUFFIX_BIGRAMS'e ekle
("SIFAT_FİİL", "KİŞİ_1Ç"),  # Örnek
```

---

## Loglama

```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = create_default_analyzer(dictionary_path='turkish_words.txt')
result = analyzer.analyze('testword')
```

---

## Sık Sorulan Sorular

### Q: Neden bu ek eşleşmiyor?
A: 
- `min_stem_length` kontrol et
- Uyum kuralları kontrol et
- `_FORBIDDEN_SUFFIX_BIGRAMS` kontrol et

### Q: Neden yanlış kök seçiliyor?
A: 
- Sözlükte var mı kontrol et
- `find_root()` sonucunu kontrol et
- Sıralama algoritmasını (`_rank_analyses`) incele

### Q: Neden benchmark farklı sonuç veriyor?
A: 
- POS bilgisi kullanılıyor mu kontrol et (`upos` parametresi)
- Sözlük yükleniyor mu kontrol et
- `turkish_lower()` Türkçe karakterleri doğru işliyor mu
