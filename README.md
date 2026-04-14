# Türkçe Kural Tabanlı Morfolojik Çözümleyici ve Bağımlılık Ayrıştırıcı

Ünlü uyumu kurallarına dayalı morfolojik çözümleyici, cümle analizi ve kural tabanlı sözdizimsel bağımlılık ayrıştırıcı.

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

```
Cümle: Büyük şehirlerde insanlar hızlı yaşar.
└── yaşar [root]
    ├── şehirlerde [obl]
    │   └── Büyük [amod]
    ├── insanlar [nsubj]
    ├── hızlı [advmod]
    └── . [punct]
```

## Temel Fikir

Türkçe sondan eklemeli (aglütinatif) bir dildir — sözcükler bir köke eklenen zincirleme eklerle oluşur. Bu ekler **ünlü uyumu** kurallarına tabidir: kökteki son ünlü, eke gelecek ünlüyü belirler. Bu fonetik kısıtlama, morfolojik sınırları tespit etmek için güçlü bir sinyal oluşturur.

Sistem bu sezgiyi algoritmaya çevirir:
1. Sözcüğün sağından başlayarak olası ekleri tanı
2. Her aday eki ünlü uyumu kurallarıyla doğrula
3. Kalan gövdeyi sözlükle karşılaştır
4. Morfofonemik dönüşümleri (ünsüz yumuşaması, ünlü düşmesi) çözümle
5. Cümle düzeyinde morfolojik belirsizliği çöz (13 yeniden sıralama kuralı)
6. Morfolojik etiketleri kullanarak sözdizimsel bağımlılık ağacı oluştur (21 kural)

## Özellikler

### Morfolojik Çözümleme
- **Büyük ve Küçük Ünlü Uyumu** kontrolü ile ek doğrulama
- **48.715 sözcüklük TDK sözlüğü** ile kök doğrulama
- **Morfofonemik çözümleme**: ünsüz yumuşaması (kitab→kitap), ünlü düşmesi (burn→burun), ünlü daralması (di→de), kaynaştırma harfi (suy→su)
- **Çok katmanlı türetim**: `yazdırılabileceklerdenmişsiniz` → yaz + dır + ıl + abil + ecek + ler + den + miş + siniz
- **Kök / Lemma ayrımı**: kök=yaz, lemma=yazdır (türetilmiş gövde)
- **Çoklu çözümleme** (belirsizlik): `gelirin` → gelir+in VEYA gel+ir+in
- **Ek hiyerarşisi**: yasaklı ek çiftleri ile dilbilimsel sıra kontrolü
- **56 ek şablonu** → 285 somut ek biçimi ({A}→a/e, {I}→ı/i/u/ü, {D}→d/t, {C}→c/ç)
- **4 katmanlı strateji**: sözlük+katı → sözlük+gevşek → sezgisel+katı → sezgisel+gevşek

### Cümle Analizi
- **Tokenizer**: akıllı noktalama ve kısaltma ayrımı
- **Çok sözcüklü ifade (MWT)** tespiti: ondan → o + n + dan
- **13 yeniden sıralama kuralı** ile morfolojik belirsizlik çözümü
- **Bağlam duyarlı** çözümleme: önceki/sonraki sözcüğe göre en uygun ayrıştırma

### Sözdizimsel Bağımlılık Ayrıştırma (Dependency Parsing)
- **21 SOLID kural** ile Universal Dependencies (UD) uyumlu bağımlılık ağacı
- **UPOS çıkarımı**: morfolojik etiketlerden otomatik POS tespiti
- **Hal eki tabanlı görev atama**: BELIRTME→obj, YÖNELME/BULUNMA/AYRILMA→obl
- **İyelik tamlaması**: genitif-posesif zincir çözümlemesi (nmod:poss)
- **Koordinasyon**: bağlaç + virgüllü asindeton + ilişkili bağlaçlar (hem...hem)
- **Yan cümleler**: zarf-fiil→advcl, sıfat-fiil→acl, mastar→csubj
- **Hafif fiil**: compound:lvc (yardım etmek, sabır göstermek)
- **Son-işlem**: obj limiter (yüklem başına max 1 obj) + root-swap (UD ilk-eşgüdüm kuralı)
- **CoNLL-U çıktı**: standart format desteği
- **ASCII ağaç görselleştirme**

## Kurulum

```bash
# Gereksinimler: Python 3.10+
git clone https://github.com/iatagun/lemma-rule-based.git
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

### Cümle Analizi

```python
from morphology import create_default_analyzer
from morphology.sentence import SentenceAnalyzer

analyzer = create_default_analyzer(dictionary_path="data/turkish_words.txt")
sa = SentenceAnalyzer(analyzer)

tokens = sa.analyze("Ali okula gitti.")
for t in tokens:
    print(f"{t.word:15s} kök={t.analysis.stem if t.analysis else '-':10s} "
          f"ekler={t.analysis.suffixes if t.analysis else []}")
```

### Bağımlılık Ayrıştırma (Dependency Parsing)

```python
from morphology import create_default_analyzer
from morphology.sentence import SentenceAnalyzer
from morphology.dependency import DependencyParser

analyzer = create_default_analyzer(dictionary_path="data/turkish_words.txt")
sa = SentenceAnalyzer(analyzer)
dp = DependencyParser()

text = "Büyük şehirlerde insanlar hızlı yaşar."
tokens = sa.analyze(text)
dep_tokens = dp.parse(tokens, text=text)

# CoNLL-U çıktı
print(DependencyParser.to_conllu(dep_tokens, text))

# ASCII ağaç
print(DependencyParser.to_tree(dep_tokens))
# └── yaşar [root]
#     ├── şehirlerde [obl]
#     │   └── Büyük [amod]
#     ├── insanlar [nsubj]
#     ├── hızlı [advmod]
#     └── . [punct]
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
├── find_lemma.py              # CLI giriş noktası (sözcük çözümleme)
├── demo_dep.py                # Dependency parser demo (24 test cümlesi)
├── demo_sentence.py           # Cümle analizi demo
├── data/
│   └── turkish_words.txt      # TDK sözlüğü (48.715 sözcük)
├── morphology/
│   ├── __init__.py            # Fabrika: create_default_analyzer()
│   ├── phonology.py           # Ses bilgisi: ünlü/ünsüz kümeleri, heceleme
│   ├── harmony.py             # Uyum kuralları: BÜU, KÜU, ünsüz benzeşmesi
│   ├── suffix.py              # Ek şablonları ve açılım (56 şablon → 285 biçim)
│   ├── morphotactics.py       # 16-durum FSM (ek sıralama kuralları)
│   ├── dictionary.py          # Sözlük + morfofonemik kök çözümleme
│   ├── analyzer.py            # BFS çözümleme motoru (sağdan-sola)
│   ├── sentence.py            # Cümle analizi + 13 yeniden sıralama kuralı
│   ├── dependency.py          # Dependency parser (21 SOLID kural)
│   └── formatter.py           # Çıktı biçimlendirme
├── benchmark/
│   ├── evaluate.py            # Morfoloji benchmark (BOUN Treebank)
│   ├── eval_dep.py            # Dependency benchmark (UAS/LAS/UPOS)
│   ├── test.conllu            # Test kümesi (979 cümle, 10.182 token)
│   └── dev.conllu             # Geliştirme kümesi
├── tests/                     # Birim testleri
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
| `morphotactics.py` | Morfo-taktik FSM (16 durum, ek geçiş kuralları) | SRP, OCP |
| `dictionary.py` | Sözlük yükleme, morfofonemik kök çözümleme | SRP, OCP |
| `analyzer.py` | BFS çözümleme algoritması, strateji yönetimi | SRP, DIP |
| `sentence.py` | Cümle tokenizer, MWT, 13 yeniden sıralama kuralı | SRP, OCP |
| `dependency.py` | 21 kural tabanlı bağımlılık ayrıştırıcı | SRP, OCP, DIP |
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

[UD Turkish BOUN Treebank](https://universaldependencies.org/) test kümesi üzerinde değerlendirme:

> **Not:** BOUN Treebank literatürde kabul görmüş bir referans veri setidir, ancak mükemmel değildir — annotator tutarsızlıkları, tartışmalı lemma kararları (ör. "bulun" vs "bul") ve alıntı sözcüklerde belirsizlikler içerir. Aşağıdaki doğruluk oranları bu veri setine göre ölçülmüştür; mutlak bir başarı ölçütü olarak değil, görece bir karşılaştırma aracı olarak değerlendirilmelidir.

### Morfoloji (Lemmatizasyon)

| Metrik | Sonuç |
|--------|-------|
| **Genel doğruluk** | **%89.8** (8857/10182) |
| DET | %99.8 |
| PRON | %96.6 |
| ADP | %96.2 |
| AUX | %96.2 |
| ADV | %94.6 |
| ADJ | %91.5 |
| NOUN | %86.2 |
| PROPN | %80.5 |
| VERB | %77.4 |

### Sözdizimsel Bağımlılık (Dependency Parsing) — v15.1

| Metrik | Sonuç |
|--------|-------|
| **UAS** (bağlantı doğruluğu) | **%47.0** |
| **LAS** (bağlantı + etiket) | **%37.1** |
| **Deprel** (etiket doğruluğu) | **%51.2** |
| **UPOS** (sözcük türü) | **%82.2** |

**İlişki bazında doğruluk (LAS):**

| İlişki | Altın | Doğru | Oran |
|--------|-------|-------|------|
| root | 976 | 635 | %65.1 |
| punct | 1192 | 1044 | %87.6 |
| amod | 765 | 292 | %38.2 |
| nmod:poss | 1053 | 340 | %32.3 |
| obl | 755 | 235 | %31.1 |
| conj | 673 | 181 | %26.9 |
| obj | 739 | 189 | %25.6 |
| advmod | 497 | 157 | %31.6 |
| nsubj | 827 | 171 | %20.7 |
| det | 486 | 315 | %64.8 |
| case | 334 | 114 | %34.1 |
| csubj | 72 | 10 | %13.9 |

**Sürüm ilerlemesi:**

| Metrik | v1 | v7 | v10 | v14 | **v15.1** |
|--------|----|----|-----|-----|-----------|
| UAS | %34.7 | %42.3 | %45.3 | %46.5 | **%47.0** |
| LAS | %22.1 | %31.2 | %35.0 | %36.2 | **%37.1** |
| UPOS | %75.1 | %80.9 | %81.6 | %82.2 | **%82.2** |

```bash
# Benchmark çalıştırma
pip install conllu

# Morfoloji benchmark
python benchmark/evaluate.py

# Dependency benchmark
python -X utf8 benchmark/eval_dep.py -m
```

## Bağımlılık Ayrıştırıcı — Kural Tabanlı Mimari

### İşlem Hattı (Pipeline)

```
Ham metin
  │
  ▼
SentenceAnalyzer.analyze()     ← Morfolojik çözümleme + MWT + yeniden sıralama
  │
  ▼
[SentenceToken]                ← stem, suffixes[(form, label)], alternatives
  │
  ▼
DependencyParser.parse()       ← 21 kural zinciri + son-işlem
  │
  ▼
[DepToken]                     ← head, deprel, upos, CoNLL-U uyumlu çıktı
```

### Kural Zinciri (21 Kural — Sıra Önemli)

Her kural `DependencyRule` arayüzünü uygular (Strategy Pattern):

| # | Kural | Çıktı | Açıklama |
|---|-------|-------|----------|
| 1 | PredicateRule | root | Son çekimli fiili ana yüklem ata |
| 2 | NominalPredicateRule | root | Fiilsiz cümlelerde kopula/nominal root |
| 3 | PostpositionRule | case | Edatları (için, gibi, kadar) solundaki isme bağla |
| 4 | PossessiveRule | nmod:poss | TAMLAYAN+İYELİK iyelik tamlaması çözümle |
| 5 | DeterminerRule | det | Belirleyicileri (bu, şu, bir, her) sağdaki isme bağla |
| 6 | NummodRule | nummod | Sayıları sağdaki isme bağla |
| 7 | FlatNameRule | flat | Çok sözcüklü özel isimleri düz yapı olarak bağla |
| 8 | CoordinationRule | cc, conj | Bağlaç + virgüllü/ilişkili koordinasyon |
| 9 | LightVerbRule | compound:lvc | Hafif fiil yapıları (yardım et, sabır göster) |
| 10 | CompoundNounRule | compound | Birleşik isim tespiti |
| 11 | ConverbRule | advcl | Zarf-fiil → yan cümle (koşarak, gelip, gelince) |
| 12 | InfinitiveRule | csubj | Mastar → özne yan cümlesi (yapmak zor) |
| 13 | ParticipleRule | acl | Sıfat-fiil → sıfat yan cümlesi (gelen adam) |
| 14 | TemporalAdvmodRule | obl:tmod | Zamansal zarfları (bugün, dün) bağla |
| 15 | AdvmodEmphRule | advmod:emph | Pekiştirme zarfları (çok, en, pek) |
| 16 | CopulaRule | cop | Kopula eki (-dir, -dır) ve idi/imiş |
| 17 | AdjAdvDisambiguationRule | amod/advmod | ADJ↔ADV belirsizlik çözümü |
| 18 | AdvmodRule | advmod | Zarf bağımlılığı |
| 19 | CaseRoleRule | nsubj/obj/obl | Hal eki tabanlı sözdizimsel görev atama |
| 20 | AdjectiveRule | amod | Sıfat tamlayıcıları bağla |
| 21 | FallbackRule | dep | UPOS-farkında yedek atama |

### Son-İşlem Adımları (Post-processing)

1. **obj Limiter**: Yüklem başına max 1 obj — fazlası nmod:poss veya obl'a dönüştürülür
2. **Root-Swap**: UD koordinasyon kuralı — ilk eşgüdümlü fiil (virgüllü) root olur

### UPOS Çıkarımı

Morfolojik etiketlerden otomatik UPOS (Universal POS) atanır:

| Morfolojik Sinyal | UPOS |
|-------------------|------|
| Çekimli fiil ekleri (zaman/kişi) | VERB |
| Hal ekleri, çoğul, iyelik | NOUN |
| Bilinen belirleyiciler (bu, şu, bir) | DET |
| Bilinen edatlar (için, gibi) | ADP |
| COMMON_ADJECTIVES listesi | ADJ |
| Sayısal ifadeler | NUM |
| Noktalama işaretleri | PUNCT |

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

### Morfoloji
- **Leksikalleşmiş türetimler**: `çalışmak` (çal+ış değil, bağımsız sözcük) — istisna listesiyle yönetilir ama kapsamlı değildir
- **Dönüşlü fiiller**: `bulunmak`, `düşünmek` gibi -ın/-in/-un/-ün türetimleri henüz desteklenmez (yüksek yanlış-pozitif riski)
- **Sirkumfleks**: â, î, û harfleri ünlü gruplarında tanımlı değil
- **Bazı eksik ekler**: -CA (eşitlik), -AmA- (yetersizlik), -mAdAn (zarf-fiil), -DIkçA
- **Bağlam bağımlılığı**: Aynı sözcüğün farklı bağlamlarda farklı çözümlemeleri olabilir (çoklu çözümleme ile kısmen desteklenir)

### Bağımlılık Ayrıştırma
- **İYELİK_3T/BELIRTME belirsizliği**: Türkçede 3. tekil iyelik eki (-ı/-i) ile belirtme hali eki (-ı/-i) biçimsel olarak aynıdır → nsubj↔obj karışması çözülemez
- **Serbest sözcük sırası**: Türkçe SOV, OSV, OVS olabilir → yalın halli en soldaki = özne sezgisi her zaman doğru değildir
- **Pro-drop**: Gizli özne tespiti yapılamaz (yalnızca kişi ekinden çıkarım)
- **İç içe tamlama zinciri**: "Türkiye'nin en büyük şehrinin nüfusu" — zincir uzadıkça doğruluk düşer
- **Compound tespiti**: Birleşik isim ayırma henüz %0 doğrulukta (216 altın token)
- **Morfoloji hata yayılımı**: Yanlış morfolojik çözümleme → yanlış UPOS → yanlış bağımlılık

## İlham

- [Dizge](https://github.com/dizge/dizge) — Bu projenin temelini oluşturan Türkçe fonoloji ve sesbilim kütüphanesi. Dizge'deki ünlü/ünsüz sınıflandırma sistemi ve heceleme mantığı, buradaki kural tabanlı çözümleyicinin çekirdeğini oluşturur. Dizge aynı zamanda bu projenin yazarının önceki çalışmasıdır.
- [UD Turkish BOUN Treebank](https://universaldependencies.org/treebanks/tr_boun/) — Altın standart değerlendirme verisi
- Oflazer (1994) — Türkçe morfolojik çözümleyici (FST tabanlı)

## Lisans

MIT
