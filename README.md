# Türkçe Kural Tabanlı Morfolojik Çözümleyici

Ünlü uyumu kurallarına dayalı, sözlük destekli kök-ek ayırma sistemi.

```
Sözcük : bölüşülmüştür
Kök    : böl
Lemma  : bölüş
Ekler  : -ül (EDİLGEN) + -müş (DUYULAN_GEÇMİŞ) + -tür (BİLDİRME/ETTİRGEN)
Ayrım  : bölüş + ül + müş + tür
Uyum   :
  ö (ince, yuvarlak) → ü (ince, yuvarlak)  BÜU:✓  KÜU:✓
  ü (ince, yuvarlak) → ü (ince, yuvarlak)  BÜU:✓  KÜU:✓
  ü (ince, yuvarlak) → ü (ince, yuvarlak)  BÜU:✓  KÜU:✓
  ü (ince, yuvarlak) → ü (ince, yuvarlak)  BÜU:✓  KÜU:✓
```

## Temel Fikir

Türkçe sondan eklemeli (aglütinatif) bir dildir — sözcükler bir köke eklenen zincirleme eklerle oluşur. Bu ekler **ünlü uyumu** kurallarına tabidir: kökteki son ünlü, eke gelecek ünlüyü belirler. Bu fonetik kısıtlama, morfolojik sınırları tespit etmek için güçlü bir sinyal oluşturur.

Sistem bu sezgiyi algoritmaya çevirir:
1. Sözcüğün sağından başlayarak olası ekleri tanı
2. Her aday eki ünlü uyumu kurallarıyla doğrula
3. Kalan gövdeyi sözlükle karşılaştır
4. Morfofonemik dönüşümleri (ünsüz yumuşaması, ünlü düşmesi) çözümle

## Özellikler

- **Büyük ve Küçük Ünlü Uyumu** kontrolü ile ek doğrulama
- **48.715 sözcüklük TDK sözlüğü** ile kök doğrulama
- **Morfofonemik çözümleme**: ünsüz yumuşaması (kitab→kitap), ünlü düşmesi (burn→burun), ünlü daralması (di→de), kaynaştırma harfi (suy→su)
- **Çok katmanlı türetim**: `yazdırılabileceklerdenmişsiniz` → yaz + dır + ıl + abil + ecek + ler + den + miş + siniz
- **Kök / Lemma ayrımı**: kök=yaz, lemma=yazdır (türetilmiş gövde)
- **Çoklu çözümleme** (belirsizlik): `gelirin` → gelir+in VEYA gel+ir+in
- **Ek hiyerarşisi**: yasaklı ek çiftleri ile dilbilimsel sıra kontrolü
- **56 ek şablonu** → 285 somut ek biçimi ({A}→a/e, {I}→ı/i/u/ü, {D}→d/t, {C}→c/ç)
- **4 katmanlı strateji**: sözlük+katı → sözlük+gevşek → sezgisel+katı → sezgisel+gevşek
- **BOUN Treebank** üzerinde %87.0 lemmatizasyon doğruluğu

## Kurulum

```bash
# Gereksinimler: Python 3.10+
git clone <repo-url>
cd lemma-rule-based

# Benchmark çalıştırmak için (opsiyonel)
pip install conllu
```

Harici kütüphane bağımlılığı yoktur — `morphology` paketi saf Python'dur.

## Kullanım

### Komut Satırı (CLI)

```bash
# Tek sözcük çözümleme
python find_lemma.py kitaplarımızdan

# Birden fazla sözcük
python find_lemma.py evlerinden gelmişler yazdırılabileceklerdenmişsiniz

# Etkileşimli mod
python find_lemma.py -i

# Yerleşik test sözcükleri
python find_lemma.py
```

### Python API

```python
from morphology import create_default_analyzer, AnalysisFormatter

# Çözümleyici oluştur (sözlük destekli)
analyzer = create_default_analyzer(dictionary_path="turkish_words.txt")
formatter = AnalysisFormatter()

# Tek çözümleme
result = analyzer.analyze("evlerinden")
print(result.stem)      # "ev"
print(result.root)      # None (morfofonemik dönüşüm yok)
print(result.suffixes)  # [("leri", "İYELİK_3Ç"), ("nden", "AYRILMA")]
print(result.parts)     # ["ev", "leri", "nden"]

# Biçimlendirilmiş çıktı
print(formatter.format_analysis("evlerinden", result))
```

#### `analyzer.analyze(word, upos=None)`

Sözcüğü çözümleyerek `MorphemeAnalysis` döndürür.

```python
result = analyzer.analyze("kitabından")
result.stem     # "kitab"       — ek soyma sonrası kalan gövde
result.root     # "kitap"       — morfofonemik çözümleme sonrası kök (b→p)
result.lemma    # None          — türetim yok, yalnızca çekim ekleri
result.suffixes # [("ı", "İYELİK_3T/BELIRTME"), ("ndan", "AYRILMA")]
result.parts    # ["kitab", "ı", "ndan"]
```

**`upos` parametresi**: UD POS etiketi verildiğinde bağlama göre özel kurallar devreye girer (AUX için kopula tablosu, PRON için zamir tablosu, DET/ADP için doğrudan eşleme vb.)

#### `analyzer.analyze_all(word, max_results=5)`

Birden fazla olası çözümlemeyi döndürür (belirsizlik desteği):

```python
results = analyzer.analyze_all("gelirin")
# Çözümleme 1: gelir + in  (isim: gelirin)
# Çözümleme 2: gel + ir + in  (fiil: gelmek geniş zaman + tamlayan)
```

Sıralama kalitesine göre yapılır: sözlük eşleşmesi, fiil kökü doğrulaması, gövde uzunluğu.

#### `MorphemeAnalysis` Veri Modeli

| Alan | Tür | Açıklama |
|------|-----|----------|
| `stem` | `str` | Ek soyma sonrası kalan gövde (`kitab`, `gel`, `çalış`) |
| `root` | `str \| None` | Morfofonemik çözümleme sonrası kök (`kitap`, `gel`, `çalış`) |
| `lemma` | `str \| None` | Türetilmiş gövde, yalnızca kökten farklıysa (`yazdır`, `bölüş`) |
| `suffixes` | `list[tuple[str, str]]` | Ek listesi: `[(biçim, etiket), ...]` |
| `parts` | `list[str]` | Gövde + ek parçaları: `["ev", "ler", "in", "den"]` |

# Kök / Lemma ayrımı
- `evlerinden` → kök=`ev`, lemma=`None` (türetim yok)
- `yazdırılmış` → kök=`yaz`, lemma=`yazdır` (ettirgen türetim: yaz→yazdır)
- `bölüşülmüştür` → kök=`böl`, lemma=`bölüş` (işteş türetim: böl→bölüş)

### Fonoloji Araçları

```python
from morphology import syllabify, get_syllable_nuclei, is_loanword_candidate
from morphology import check_word_internal_harmony
from morphology.phonology import turkish_lower, get_vowels, last_vowel

# Heceleme
syllabify("evlerinden")       # ["ev", "le", "rin", "den"]
syllabify("yazdırılabileceklerdenmişsiniz")
# ["yaz", "dı", "rı", "la", "bi", "le", "cek", "ler", "den", "miş", "si", "niz"]

# Hece çekirdekleri
get_syllable_nuclei("öğretmen")  # ["ö", "e", "e"]

# Alıntı sözcük tespiti (o/ö ikinci+ hecede)
is_loanword_candidate("doktor")     # True  (o ikinci hecede)
is_loanword_candidate("kitap")      # False

# Sözcük-içi uyum kontrolü
info = check_word_internal_harmony("televizyon")
info["full_ok"]     # False (alıntı sözcük — uyum bozuk)
info["violations"]  # [{"pos": 2, "v1": "i", "v2": "o", "buu": False, "kuu": False}]

# Türkçe büyük-küçük harf dönüşümü
turkish_lower("İSTANBUL")  # "istanbul" (İ→i, I→ı)
turkish_lower("DİYARBAKIR") # "diyarbakır"

# Ünlü araçları
get_vowels("evlerinden")  # ["e", "e", "i", "e"]
last_vowel("kitap")       # "a"
```

### Uyum Kontrolü

```python
from morphology.harmony import (
    check_major_harmony,
    check_minor_harmony,
    check_vowel_harmony,
    check_consonant_harmony,
)

# Büyük Ünlü Uyumu (kalınlık-incelik)
check_major_harmony("a", "ı")  # True  (kalın→kalın)
check_major_harmony("a", "i")  # False (kalın→ince ✗)

# Küçük Ünlü Uyumu (düzlük-yuvarlaklık)
check_minor_harmony("ö", "ü")  # True  (yuvarlak→dar yuvarlak)
check_minor_harmony("ö", "i")  # False (yuvarlak→düz dar ✗)

# Kök-ek arası tam uyum (BÜU + KÜU)
check_vowel_harmony("ev", "ler")   # True
check_vowel_harmony("ev", "lar")   # False

# Ünsüz benzeşmesi
check_consonant_harmony("kitap", "da")  # False (p sert → d olmamalı)
check_consonant_harmony("kitap", "ta")  # True  (p sert → t olmalı)
```

### Sözlük

```python
from morphology import TurkishDictionary

dictionary = TurkishDictionary.from_file("turkish_words.txt")

# Doğrudan arama
dictionary.contains("kitap")     # True
dictionary.contains("kitab")     # False (yüzey biçimi)

# Morfofonemik kök çözümleme (5 adımlı)
dictionary.find_root("kitab")    # "kitap"  (ünsüz yumuşaması: b→p)
dictionary.find_root("burn")     # "burun"  (ünlü düşmesi: rn→run)
dictionary.find_root("suy")      # "su"     (kaynaştırma: y kaldır)
dictionary.find_root("gid")      # "git"    (ünsüz yumuşaması: d→t)
dictionary.find_root("gel")      # "gel"    (doğrudan + fiil: gelmek)

# Ünlü daralması dahil çözümleme (diyor → de)
dictionary.find_root_with_narrowing("di")  # "de" (diyor → de+yor)
```

### Ek Sistemi

```python
from morphology import SuffixRegistry
from morphology.suffix import SuffixDefinition

# Varsayılan ek kümesi
registry = SuffixRegistry.create_default()
print(len(registry.suffixes))  # 285 somut biçim

# Özel ek ekleme
registry.register(SuffixDefinition(
    template="{D}{A}ki",
    label="BULUNMA_ki",
    harmony_exempt=False,
    min_stem_length=2,
))

# Şablon açılımı: {D}{A}ki → daki, deki, taki, teki
```

**Şablon Değişkenleri:**

| Değişken | Açılım | Dilbilimsel Karşılık |
|----------|--------|----------------------|
| `{A}` | a, e | Geniş ünlü (2 yönlü) |
| `{I}` | ı, i, u, ü | Dar ünlü (4 yönlü) |
| `{D}` | d, t | Ünsüz benzeşmesi |
| `{C}` | c, ç | Ünsüz benzeşmesi |

### Biçimlendirici

```python
from morphology import AnalysisFormatter, create_default_analyzer

analyzer = create_default_analyzer(dictionary_path="turkish_words.txt")
formatter = AnalysisFormatter()

# Tek çözümleme raporu
result = analyzer.analyze("güzelliklerini")
print(formatter.format_analysis("güzelliklerini", result))

# Çoklu çözümleme raporu (belirsizlik uyarısıyla)
results = analyzer.analyze_all("yazar")
print(formatter.format_multi_analysis("yazar", results))
# ⚠ 2 olası çözümleme (doğru olan bağlama göre belirlenir)
# ── Çözümleme 1: yaz + ar  (fiil kökü)
# ── Çözümleme 2: yazar     (isim, sözlükte)

# Ünlü uyumu raporu
print(AnalysisFormatter.vowel_harmony_report("evlerinden"))
```

## Mimari

```
lemma-rule-based/
├── find_lemma.py              # CLI giriş noktası
├── turkish_words.txt          # TDK sözlüğü (48.715 sözcük)
├── morphology/
│   ├── __init__.py            # Fabrika: create_default_analyzer()
│   ├── phonology.py           # Ses bilgisi: ünlü/ünsüz kümeleri, heceleme
│   ├── harmony.py             # Uyum kuralları: BÜU, KÜU, ünsüz benzeşmesi
│   ├── suffix.py              # Ek şablonları ve açılım (56 şablon → 285 biçim)
│   ├── dictionary.py          # Sözlük + morfofonemik kök çözümleme
│   ├── analyzer.py            # Çözümleme motoru (greedy sağdan-sola)
│   └── formatter.py           # Çıktı biçimlendirme
├── benchmark/
│   ├── evaluate.py            # BOUN Treebank değerlendirme betiği
│   ├── test.conllu            # Test kümesi (979 cümle, 10.182 token)
│   └── dev.conllu             # Geliştirme kümesi
├── ARCHITECTURE.md            # Detaylı mimari dokümantasyonu
├── AGENTS.md                  # Copilot talimat dosyası
└── skill.md                   # 4 uzman dilbilimci panel raporu
```

### Modül Sorumlulukları

| Modül | Sorumluluk | SOLID |
|-------|-----------|-------|
| `phonology.py` | Ses sabitleri, heceleme, Türkçe harf dönüşümü | SRP |
| `harmony.py` | Ünlü/ünsüz uyumu kuralları, `HarmonyChecker` protokolü | SRP, OCP, DIP |
| `suffix.py` | Ek tanımları, şablon açılımı, `SuffixRegistry` | SRP, OCP |
| `dictionary.py` | Sözlük yükleme, morfofonemik kök çözümleme | SRP, OCP |
| `analyzer.py` | Çözümleme algoritması, strateji yönetimi | SRP, DIP |
| `formatter.py` | Çıktı biçimlendirme, uyum raporu | SRP, ISP |

### Çözümleme Algoritması

```
     Sözcük: "kitaplarımızdan"
            │
            ▼
    ┌─ Sağdan-sola ek soyma ──────────────────────┐
    │                                              │
    │  "kitaplarımızdan"                           │
    │    └─ "-dan" (AYRILMA) ✓ BÜU ✓ KÜU         │
    │  "kitaplarımız"                              │
    │    └─ "-ımız" (İYELİK_1Ç) ✓ BÜU ✓ KÜU     │
    │  "kitaplar"                                  │
    │    └─ "-lar" (ÇOĞUL) ✓ BÜU ✓ KÜU           │
    │  "kitap"                                     │
    │    └─ Sözlükte ✓ → Dur                       │
    │                                              │
    └──────────────────────────────────────────────┘
            │
            ▼
    Sonuç: kök="kitap", ekler=[lar, ımız, dan]
```

**4 Katmanlı Strateji Sistemi:**

| Katman | Uyum | Doğrulama | Kullanım |
|--------|------|-----------|----------|
| 1 | Katı (BÜU+KÜU) | Sözlük | Öz Türkçe sözcükler |
| 2 | Gevşek (yalnız KÜU) | Sözlük | Alıntı sözcükler (doktor, saat) |
| 3 | Katı (BÜU+KÜU) | Sezgisel | Sözlükte olmayan sözcükler |
| 4 | Gevşek (yalnız KÜU) | Sezgisel | Son çare |

## Benchmark

[UD Turkish BOUN Treebank](https://universaldependencies.org/) test kümesi üzerinde lemmatizasyon doğruluğu:

> **Not:** BOUN Treebank literatürde kabul görmüş bir referans veri setidir, ancak mükemmel değildir — annotator tutarsızlıkları, tartışmalı lemma kararları (ör. "bulun" vs "bul") ve alıntı sözcüklerde belirsizlikler içerir. Aşağıdaki doğruluk oranları bu veri setine göre ölçülmüştür; mutlak bir başarı ölçütü olarak değil, görece bir karşılaştırma aracı olarak değerlendirilmelidir.

| Metrik | Sonuç |
|--------|-------|
| **Genel doğruluk** | **%87.0** (8857/10182) |
| DET | %99.8 |
| PRON | %96.6 |
| ADP | %96.2 |
| AUX | %96.2 |
| ADV | %94.6 |
| ADJ | %91.5 |
| NOUN | %86.2 |
| PROPN | %80.5 |
| VERB | %77.4 |

**Gelişim seyri:**

```
%69.1  İlk sözlük entegrasyonu
  │
  ├── +3.9  Sözlük koruması + fiil gövde koruması
  ├── +1.7  Buffer ünsüzler + türetim kök çıkarma
  ├── +3.5  Sezgisel doğrulama (min_stem, hece) + fiil kontrolü
  ├── +2.7  Zamir tablosu + yönelme eki düzeltmesi
  ├── +2.1  Kopula tablosu + buffer-n
  ├── +1.7  Sıra numaraları + postpozisyon + demek/yemek/etmek
  ├── +0.4  Unicode normalizasyon
  ├── +0.3  Yasaklı ek çiftleri (forbidden bigrams)
  └── +0.1  İşteş eki (İŞTEŞ) + istisna listesi
      │
      ▼
%87.0  Güncel
```

```bash
# Benchmark çalıştırma
pip install conllu
python benchmark/evaluate.py
```

## Dilbilimsel Arka Plan

### Büyük Ünlü Uyumu (Kalınlık-İncelik)

Ekteki ünlüler, kökün son ünlüsüyle aynı kalınlık/incelik grubunda olmalıdır:

| Kök sonu | Ek ünlüsü | Örnek |
|----------|-----------|-------|
| Kalın (a, ı, o, u) | Kalın | **okul** + **dan** ✓ |
| İnce (e, i, ö, ü) | İnce | **ev** + **den** ✓ |
| Kalın | İnce | **okul** + ~~den~~ ✗ |

### Küçük Ünlü Uyumu (Düzlük-Yuvarlaklık)

| Kök sonu | İzin verilen ek ünlüsü |
|----------|----------------------|
| Düz (a, e, ı, i) | Düz (a, e, ı, i) |
| Yuvarlak (o, ö, u, ü) | Dar yuvarlak (u, ü) veya geniş düz (a, e) |

### Ünsüz Benzeşmesi

Sert ünsüzlerden (ç, f, h, k, p, s, ş, t) sonra d→t, c→ç dönüşümü olur:
- **kitap** + **da** → kitap**ta**
- **ağaç** + **dan** → ağaç**tan**

### Morfofonemik Süreçler

| Süreç | Yüzey → Sözlük | Örnek |
|-------|----------------|-------|
| Ünsüz yumuşaması | b→p, c→ç, d→t, g→k, ğ→k | kitab→kitap |
| Ünlü düşmesi | CCC → CVCC | burn→burun |
| Ünlü daralması | a→ı, e→i, o→u, ö→ü | diyor→de+yor |
| Kaynaştırma | y/n kaldırma | suy→su |

## Bilinen Sınırlamalar

- **Leksikalleşmiş türetimler**: `çalışmak` (çal+ış değil, bağımsız sözcük) — istisna listesiyle yönetilir ama kapsamlı değildir
- **Dönüşlü fiiller**: `bulunmak`, `düşünmek` gibi -ın/-in/-un/-ün türetimleri henüz desteklenmez (yüksek yanlış-pozitif riski)
- **Sirkumfleks**: â, î, û harfleri ünlü gruplarında tanımlı değil
- **Bazı eksik ekler**: -CA (eşitlik), -AmA- (yetersizlik), -mAdAn (zarf-fiil), -DIkçA
- **Bağlam bağımlılığı**: Aynı sözcüğün farklı bağlamlarda farklı çözümlemeleri olabilir (çoklu çözümleme ile kısmen desteklenir)

## İlham

- [Dizge](https://github.com/dizge/dizge) — Bu projenin temelini oluşturan Türkçe fonoloji ve sesbilim kütüphanesi. Dizge'deki ünlü/ünsüz sınıflandırma sistemi ve heceleme mantığı, buradaki kural tabanlı çözümleyicinin çekirdeğini oluşturur. Dizge aynı zamanda bu projenin yazarının önceki çalışmasıdır.
- [UD Turkish BOUN Treebank](https://universaldependencies.org/treebanks/tr_boun/) — Altın standart değerlendirme verisi
- Oflazer (1994) — Türkçe morfolojik çözümleyici (FST tabanlı)

## Lisans

MIT
