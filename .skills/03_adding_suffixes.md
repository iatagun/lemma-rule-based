# Yeni Ek Ekleme Prosedürü

> Yeni bir ek eklemek için 5 adım.

---

## Adım 1: Ek Tanımı

`morphology/suffix.py` dosyasında `DEFAULT_SUFFIX_DEFINITIONS` listesine ekle:

```python
_D("{D}{I}kç{A}", "ZARF_FİİL_-DIkçA"),  # gördükçe
```

### Şablon Formatı

| Değişken | Açılım | Kural |
|----------|--------|-------|
| `{A}` | a, e | BÜU (2-yollu) |
| `{I}` | ı, i, u, ü | BÜU + KÜU (4-yollu) |
| `{D}` | d, t | Ötümsüz→t, diğer→d |
| `{C}` | c, ç | Ötümsüz→ç, diğer→c |

### Parametreler

```python
SuffixDefinition(
    template="{D}{I}kç{A}",
    label="ZARF_FİİL_-DIkçA",
    harmony_exempt=False,    # Varsayılan: False
    min_stem_length=2,       # Minimum kök uzunluğu
)
```

---

## Adım 2: Etiket Adlandırma

Standart etiket isimlendirmesi:

| Kategori | Önek | Örnek |
|----------|------|-------|
| Çoğul | ÇOĞUL | ÇOĞUL |
| İyelik | İYELİK_ | İYELİK_1T, İYELİK_2Ç |
| Hal | BULUNMA, YÖNELME, BELIRTME, AYRILMA | BULUNMA |
| Fiil zamanı | GEÇMİŞ_ZAMAN, ŞİMDİKİ_ZAMAN | GEÇMİŞ_ZAMAN |
| Sıfat-fiil | SIFAT_FİİL | SIFAT_FİİL |
| Zarf-fiil | ZARF_FİİL | ZARF_FİİL_-Ip |
| Yapım | YAPIM_ | YAPIM_-lI |

---

## Adım 3: Morfotaktik FSM Güncelleme

`morphology/morphotactics.py` dosyasında geçiş tablosuna ekle:

```python
"SIFAT_FİİL": [
    ("İYELİK_1T", SLOT_POSSESSIVE),
    ("İYELİK_2T", SLOT_POSSESSIVE),
    ("İYELİK_3T", SLOT_POSSESSIVE),
    # ... yeni geçişler
],
```

---

## Adım 4: Uyum Muafiyeti

Sabit biçimli ekler (uyum kurallarına tabi değil):

```python
_D("yor", "ŞİMDİKİ_ZAMAN", harmony_exempt=True),  # o ünlüsü değişmez
_D("ken", "ZARF_FİİL_-ken", harmony_exempt=True),
_D("ki", "İLGİ_-ki", harmony_exempt=True),
```

---

## Adım 5: Test ve Regresyon Kontrolü

```powershell
# Hızlı test
python -X utf8 -c "
from morphology import create_default_analyzer
a = create_default_analyzer()
r = a.analyze('gördükçe', upos='VERB')
print(r.stem, r.suffixes)
"

# Full benchmark
python -X utf8 benchmark/evaluate.py
```

---

## Örnek: -DIkçA Ekleme

### 1. Suffix tanımı
```python
# suffix.py
_D("{D}{I}kç{A}", "ZARF_FİİL_-DIkçA"),
```

### 2. Şablon açılımı
```
{D}{I}kç{A} → dıkça, dikçe,dukça, dükçe, tıkça, tikçe, tukça, tükçe
```

### 3. Morfotaktik
```python
# morphotactics.py
"SIFAT_FİİL": [
    ...
    ("ZARF_FİİL_-DIkçA", SLOT_ADVERB),
],
```

### 4. Test
```python
r = a.analyze("gördükçe")
# Beklenen: stem="gör", suffixes=[("dükçe", "ZARF_FİİL_-DIkçA")]
```

---

## Forbidden Bigram Ekleme

Bazı ek kombinasyonları dilbilgisel olarak imkânsız. Bunları engellemek için:

### Prosedür

1. Hata analizinden adayı belirle
2. Doğru çözümlemelerde kaç kez göründüğünü say → **0 olmalı**
3. Hata çözümlemelerinde kaç kez göründüğünü say → **≥2 olmalı**
4. `_FORBIDDEN_SUFFIX_BIGRAMS` tablosuna ekle

### Mevcut Yasaklı Çiftler

```python
_FORBIDDEN_SUFFIX_BIGRAMS = frozenset({
    ("SIFAT_FİİL", "KİŞİ_1Ç"),
    ("SIFAT_FİİL", "KİŞİ_2Ç"),
    ("GENİŞ_ZAMAN", "YÖNELME"),
    ("GEÇMİŞ_ZAMAN", "AYRILMA"),
    ("EDİLGEN", "İYELİK_3Ç"),
    ...
})
```
