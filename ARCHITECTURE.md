# Türkçe Kural Tabanlı Morfolojik Çözümleyici

## Mimari Dokümantasyon & Tasarım Kararları

> **Proje:** lemma-rule-based  
> **Tarih:** 2026-04-13  
> **Referans:** [dizge/dizge](https://github.com/dizge/dizge) – fonoloji tanımları

---

## 1. Problem Tanımı

Türkçe **sondan eklemeli** (agglutinative) bir dildir. Tek bir sözcük, kök üzerine
zincirleme eklenen çok sayıda ek ile oluşturulabilir:

```
yazdırılabileceklerdenmişsiniz
yaz + dır + ıl + abil + ecek + ler + den + miş + siniz
 ↑     ↑    ↑     ↑      ↑     ↑     ↑     ↑      ↑
kök  ettir edil yeter  gelec  çoğ  ayrıl duyul  kişi
```

Bu proje, **sözlük kullanmadan**, yalnızca Türkçe ses uyumu kurallarını
(ünlü uyumu + ünsüz benzeşmesi) uygulayarak sözcükleri kök ve eklerine
ayırmayı amaçlar.

---

## 2. Düşünce Süreci (Kronolojik)

### 2.1 Başlangıç: Dizge Projesinin İncelenmesi

İlk adımda dizge projesinin fonoloji modülleri incelendi:

```
dizge/tools/phonology.py         → Ünlü/ünsüz kümeleri, ses özellikleri
dizge/competence/phonology.py    → Uyum kuralları, ses değişim tabloları
```

**Çıkarımlar:**
- Ünlüler 3 eksen üzerinde sınıflandırılır: kalınlık, düzlük, açıklık
- Ekler şablonlarla temsil edilebilir: `{A}` → a/e, `{I}` → ı/i/u/ü
- Ünsüz uyumu basit kural: sert ünsüzden sonra d→t, c→ç

### 2.2 İlk Prototip (~350 satır, tek dosya)

**Karar:** Hızlı iterasyon için monolitik yapı tercih edildi.

```
find_lemma.py
├── Fonoloji sabitleri (VOWELS, CONSONANTS, ...)
├── Uyum kontrol fonksiyonları
├── Şablon açılım motoru (expand_template)
├── Ek tanımları tablosu
├── Greedy sağ-sol çözümleme algoritması
├── Çıktı biçimlendirme
└── CLI giriş noktası
```

### 2.3 Kritik Hatalar & Düzeltmeler

| Hata | Neden | Çözüm |
|------|-------|-------|
| `{I}m{I}z` → yalnızca `{I}mız` açılıyor | `str.replace()` ilk eşleşmeyi değiştirir | `while changed` döngüsü |
| `öğretm+en` geçerli ayrım sayılıyor | Kök sonu ünsüz kümesi kontrolü yok | `valid_stem()` heuristiği |
| `gel+iyor` → `geli+yor` kalıyor | Bağlayıcı ünlü işlenmiyor | `_handle_yor_connector()` |
| `saat+lerinde` bulunamıyor | Alıntı sözcüklerde BÜU bozuk | Relaxed harmony fallback |

### 2.4 Edge Case Testleri (28 sözcük)

```
Sonuç: 22/28 doğru ayrım (sözlüksüz)

✓ Doğru çözümlenen fenomenler:
  - Tam uyumlu ekler (evlerinden, kitaplarımızda, ...)
  - Ünsüz benzeşmesi (gittiğimiz: git+tiğ+imiz)
  - Yuvarlak ünlü uyumu (görüşünüzden: gör+üş+ünüz+den)
  - İstisna ekler (-ki, -ken, -yor)
  - 8 ek zinciri (yazdırılabileceklerdenmişsiniz)
  - Alıntı sözcükler (saatlerinde, otobüslerden)

✗ Sözlük gerektiren fenomenler (6 sözcük):
  - Ünsüz yumuşaması: kitap→kitab (p→b)
  - Ünlü daralması: de→di, ye→yi
  - Derin türetim: güzelleştirilemez
  - Leksem sınırı: çalışkan (çalışk+an değil)
```

### 2.5 SOLID Refactoring

Monolitik 517 satırlık dosya → 6 modüllü paket mimarisine dönüştürüldü.
**Çıktı birebir korundu** (28 test sözcüğü doğrulandı).

---

## 3. Dilbilimsel Kurallar

### 3.1 Büyük Ünlü Uyumu (BÜU) — Kalınlık-İncelik

Türkçe'de kökteki son ünlünün kalınlık/incelik özelliği, ekteki ünlüyü
belirler:

```
          KALIN          İNCE
        ┌─────────┐   ┌─────────┐
        │ a ı o u │   │ e i ö ü │
        └────┬────┘   └────┬────┘
             │              │
             ▼              ▼
    ek ünlüsü KALIN   ek ünlüsü İNCE

    okul + lar  ✓      ev + ler   ✓
    okul + ler  ✗      ev + lar   ✗
```

**Formül:** `(kök_ünlü ∈ KALIN) == (ek_ünlü ∈ KALIN)`

### 3.2 Küçük Ünlü Uyumu (KÜU) — Düzlük-Yuvarlaklık

```
    Kök son ünlüsü          Ek ünlüsü
    ──────────────          ──────────
    DÜZ (a, e, ı, i)  ───→ DÜZ (a, e, ı, i)

    YUVARLAK (o, ö, u, ü)
         │
         ├─ dar ek ünlüsü ───→ DAR YUVARLAK (u, ü)
         │
         └─ geniş ek ünlüsü ──→ GENİŞ DÜZ (a, e)
```

**Karar ağacı:**

```
    kök son ünlü DÜZ mü?
    ├── EVET → ek ünlüsü DÜZ olmalı
    └── HAYIR (yuvarlak)
        ├── ek ünlüsü DAR mı? → YUVARLAK olmalı
        └── ek ünlüsü GENİŞ mi? → DÜZ olmalı
```

### 3.3 Ünsüz Benzeşmesi

```
    Kök son ünsüz SERT (ç,f,h,k,p,s,ş,t)?
    ├── EVET → ek başı: d→t, c→ç
    │          git + dir → git + tir  ✓
    │          git + dir             ✗
    └── HAYIR → ek başı değişmez
               gel + dir             ✓
               gel + tir             ✗
```

### 3.4 Şablon Sistemi

Ekler, ses uyumu değişkenlerini içeren şablonlarla tanımlanır:

```
    Değişken    Açılım               Kural
    ────────    ──────               ─────
    {A}         a, e                 2-yönlü BÜU
    {I}         ı, i, u, ü          4-yönlü BÜU+KÜU
    {D}         d, t                 Ünsüz uyumu
    {C}         c, ç                 Ünsüz uyumu

    Örnek: "{D}{I}r" → dır, dir, dur, dür, tır, tir, tur, tür (8 biçim)
```

**Açılım algoritması** — birden fazla aynı değişken için iteratif:

```
    Girdi: "{I}m{I}z"

    Tur 1: {I}m{I}z → ım{I}z, im{I}z, um{I}z, üm{I}z
    Tur 2: ım{I}z  → ımız, imiz, umuz, ümüz  (×4 her biri)
           im{I}z  → imız, imiz, imuz, imüz
           um{I}z  → umız, umiz, umuz, umüz
           üm{I}z  → ümız, ümiz, ümuz, ümüz

    Toplam: 16 somut biçim
    (Bunların çoğu uyum kontrolünde elenir)
```

---

## 4. Çözümleme Algoritması

### 4.1 Ana Akış — Greedy Sağ-Sol Soyma

```
    ┌──────────────────────────────────────────────┐
    │             SÖZCÜK GİRİŞİ                    │
    │         "evdekilerden"                       │
    └───────────────────┬──────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────────┐
    │  1. lowercase + strip                        │
    └───────────────────┬──────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────────┐
    │  2. DÖNGÜ (maks. 10 iterasyon)               │
    │                                              │
    │   ┌─────────────────────────────────────┐    │
    │   │ 2a. Strict strateji ile en uzun     │    │
    │   │     eşleşen eki ara                  │    │
    │   │     (BÜU + KÜU + Ünsüz)            │    │
    │   └──────────────┬──────────────────────┘    │
    │                  │                           │
    │            Bulundu mu?                        │
    │           ╱          ╲                       │
    │         EVET         HAYIR                    │
    │          │              │                     │
    │          │    ┌─────────┴───────────────┐    │
    │          │    │ 2b. Relaxed strateji    │    │
    │          │    │     (KÜU + Ünsüz)      │    │
    │          │    │     min_stem ≥ 3        │    │
    │          │    └─────────┬───────────────┘    │
    │          │              │                     │
    │          │        Bulundu mu?                 │
    │          │       ╱          ╲                │
    │          │     EVET         HAYIR → ÇIKIŞ    │
    │          │      │                            │
    │          ▼      ▼                            │
    │   ┌─────────────────────────────────────┐    │
    │   │ Eki kaydet, kökü güncelle            │    │
    │   │ current = stem_candidate             │    │
    │   │ suffixes.insert(0, ek)               │    │
    │   └──────────────┬──────────────────────┘    │
    │                  │                           │
    │                  └──── (döngü başına dön)     │
    └───────────────────┬──────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────────┐
    │  3. -yor bağlayıcı ünlü post-processing     │
    │     geli+yor → gel + iyor                    │
    └───────────────────┬──────────────────────────┘
                        │
                        ▼
    ┌──────────────────────────────────────────────┐
    │  4. MorphemeAnalysis(stem, suffixes) döndür   │
    └──────────────────────────────────────────────┘
```

### 4.2 Ek Eşleşme Detayı — `_try_strategy()`

Her ek adayı için 5 aşamalı filtre:

```
    SUFFIX_TABLE (uzunluğa göre sıralı, uzun ekler önce)
    ┌────────────────┐
    │ sınız (5 harf) │──┐
    │ ımız  (4 harf) │  │
    │ ler   (3 harf) │  │  Her ek için sırayla:
    │ de    (2 harf) │  │
    │ ...             │  │
    └────────────────┘  │
                        ▼
              ┌─────────────────┐     HAYIR
              │ 1. Uzunluk:     │──────────→ sonraki ek
              │ len(ek) < len(  │
              │   current)?     │
              └────────┬────────┘
                       │ EVET
                       ▼
              ┌─────────────────┐     HAYIR
              │ 2. Eşleşme:     │──────────→ sonraki ek
              │ current.endswith │
              │   (ek)?         │
              └────────┬────────┘
                       │ EVET
                       ▼
              ┌─────────────────┐     HAYIR
              │ 3. Kök uzunluğu │──────────→ sonraki ek
              │ len(kök) ≥      │
              │   min_stem?     │
              └────────┬────────┘
                       │ EVET
                       ▼
              ┌─────────────────┐     HAYIR
              │ 4. Kök geçerli? │──────────→ sonraki ek
              │ valid_stem()    │
              └────────┬────────┘
                       │ EVET
                       ▼
              ┌─────────────────┐     HAYIR
              │ 5a. Uyum muaf?  │──────┐
              │  (yor/ken/ki)   │      │
              └────────┬────────┘      │
                       │ HAYIR         │
                       ▼               │
              ┌─────────────────┐      │    HAYIR
              │ 5b. Ünlü uyumu │──────┼──→ sonraki ek
              │  checker.check_ │      │
              │  vowel_harmony()│      │
              └────────┬────────┘      │
                       │ EVET          │
                       ▼               ▼
              ┌─────────────────┐     HAYIR
              │ 6. Ünsüz uyumu │──────────→ sonraki ek
              │  checker.check_ │
              │  consonant_     │
              │  harmony()      │
              └────────┬────────┘
                       │ EVET
                       ▼
                  ✅ EŞLEŞTİ!
              (stem, form, label)
```

### 4.3 Kök Doğrulama Heuristiği — `StemValidator`

```
    Kök adayı: "öğretm"
                        ┌─────────────────────┐
                        │ En az 1 ünlü var mı? │
                        └──────────┬──────────┘
                                   │ EVET
                                   ▼
                        ┌─────────────────────┐
                        │ Sondaki ünsüz sayısı │
                        └──────────┬──────────┘
                                   │
                        ┌──────────┼──────────┐
                        │          │          │
                     0 veya 1      2        3+
                        │          │          │
                        ▼          ▼          ▼
                      GEÇERLİ   İlk ünsüz   GEÇERSİZ
                               akıcı mı?
                              (l,m,n,r,y,
                               ş,s,z,ğ)?
                              ╱         ╲
                           EVET         HAYIR
                            │              │
                            ▼              ▼
                        GEÇERLİ       GEÇERSİZ

    Örnekler:
      "göz"    → 1 ünsüz       → ✓
      "Türk"   → rk (r akıcı)  → ✓
      "oğl"    → ğl (ğ akıcı)  → ✓
      "öğretm" → tm (t akıcı ✗) → ✗ (öğretmen kökünü korur)
```

### 4.4 İki Geçişli Strateji Sistemi

```
    ┌─────────────────────────────────────────────────────────┐
    │                  STRATEJİ AKIŞI                         │
    │                                                         │
    │  1. GEÇİŞ: StrictHarmonyChecker                        │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ ✓ Büyük Ünlü Uyumu (BÜU)                        │  │
    │  │ ✓ Küçük Ünlü Uyumu (KÜU)                        │  │
    │  │ ✓ Ünsüz Benzeşmesi                               │  │
    │  │ ○ min_stem = ek tanımındaki değer (varsayılan 2) │  │
    │  └───────────────────────────────────────────────────┘  │
    │                         │                               │
    │                   Eşleşme yok?                          │
    │                         │                               │
    │                         ▼                               │
    │  2. GEÇİŞ: RelaxedHarmonyChecker                       │
    │  ┌───────────────────────────────────────────────────┐  │
    │  │ ✗ Büyük Ünlü Uyumu ATLANIR                       │  │
    │  │ ✓ Küçük Ünlü Uyumu (KÜU)                        │  │
    │  │ ✓ Ünsüz Benzeşmesi                               │  │
    │  │ ● min_stem ≥ 3 (aşırı bölünmeyi önler)          │  │
    │  └───────────────────────────────────────────────────┘  │
    │                                                         │
    │  Neden gerekli?                                         │
    │  "saat" → Arapçadan alıntı, BÜU bozuk (a→e)           │
    │  "otobüs" → Fransızcadan, BÜU bozuk (o→ü)            │
    │  min_stem≥3 olmadan "sa+at" gibi yanlış ayrımlar olur  │
    └─────────────────────────────────────────────────────────┘
```

### 4.5 -yor Bağlayıcı Ünlü İşleme

```
    Giriş: stem="geli", suffixes=[("yor", "ŞİMDİKİ_ZAMAN")]

    ┌──────────────────────────────────────────────────┐
    │ İlk ek "yor" ve ŞİMDİKİ_ZAMAN mı?              │
    │                                                  │
    │ Kök dar ünlüyle bitiyor mu?                      │
    │   geli → son harf "i" ∈ {ı,i,u,ü} ✓             │
    │                                                  │
    │ Ünlüyü çıkar:                                   │
    │   remaining = "gel"                              │
    │   connector = "i"                                │
    │                                                  │
    │ Kalan kök geçerli mi?                            │
    │   "gel" → ünsüzle bitiyor ✓, ünlü var ✓         │
    │                                                  │
    │ Bağlayıcı ünlü uyumlu mu?                       │
    │   check_vowel_harmony("gel", "i") → ✓            │
    │                                                  │
    │ Sonuç: stem="gel", suffix="iyor"                 │
    └──────────────────────────────────────────────────┘

    gel + i + yor  →  gel + iyor
     ↑        ↑        ↑     ↑
    kök   bağlayıcı   kök   birleşik ek
```

---

## 5. Yazılım Mimarisi

### 5.1 Paket Yapısı

```
    lemma-rule-based/
    ├── find_lemma.py              ← İnce CLI sarıcı (geriye dönük uyumlu)
    ├── turkish_words.txt          ← Sözcük listesi
    └── morphology/                ← Ana paket
        ├── __init__.py            ← Fabrika + dışa aktarım
        ├── phonology.py           ← Fonetik sabitler & yardımcılar
        ├── harmony.py             ← Uyum kontrol sistemi
        ├── suffix.py              ← Ek tanımları & şablon motoru
        ├── analyzer.py            ← Çözümleme algoritması
        └── formatter.py           ← Çıktı biçimlendirme
```

### 5.2 Modül Bağımlılık Grafiği

```
    ┌─────────────────┐
    │  find_lemma.py   │  CLI giriş noktası
    │  (ince sarıcı)   │
    └────────┬─────────┘
             │ import
             ▼
    ┌─────────────────┐
    │  __init__.py     │  Fabrika + re-export
    │  create_default_ │
    │  analyzer()      │
    └──┬────┬────┬─────┘
       │    │    │
       │    │    └──────────────────────────────┐
       │    │                                   │
       │    ▼                                   ▼
       │  ┌─────────────────┐        ┌─────────────────┐
       │  │  analyzer.py     │        │  formatter.py    │
       │  │                  │        │                  │
       │  │ MorphologicalAn- │        │ AnalysisFormat-  │
       │  │   alyzer         │        │   ter            │
       │  │ StemValidator    │        │ vowel_harmony_   │
       │  │ HarmonyStrategy  │        │   report()       │
       │  │ MorphemeAnalysis │        │                  │
       │  └──┬─────────┬────┘        └──┬──────────────┘
       │     │         │                │
       │     │         │                │  import
       │     ▼         ▼                ▼
       │  ┌─────────────────┐  ┌─────────────────┐
       │  │  harmony.py      │  │  (harmony.py)    │
       │  │                  │  │  check_major_    │
       │  │ «protocol»       │  │  harmony()       │
       │  │ HarmonyChecker   │  │  check_minor_    │
       │  │                  │  │  harmony()       │
       │  │ StrictHarmony-   │  └────────┬─────────┘
       │  │   Checker        │           │
       │  │ RelaxedHarmony-  │           │
       │  │   Checker        │           │
       │  └──────────┬───────┘           │
       │             │                   │
       │             ▼                   ▼
       │  ┌──────────────────────────────────┐
       │  │         phonology.py              │
       │  │                                   │
       │  │  VOWELS, CONSONANTS, BACK_VOWELS  │
       │  │  FRONT_VOWELS, UNROUNDED, ROUNDED │
       │  │  OPEN_VOWELS, CLOSE_VOWELS        │
       │  │  VOICELESS, SONORANT_SIBILANT     │
       │  │  get_vowels(), last_vowel()       │
       │  └───────────────────────────────────┘
       │
       ▼
    ┌─────────────────┐
    │  suffix.py       │
    │                  │
    │ SuffixDefinition │
    │ SuffixForm       │
    │ SuffixRegistry   │
    │ DEFAULT_SUFFIX_  │
    │   DEFINITIONS    │
    └──────────────────┘
```

### 5.3 Sınıf Diyagramı

```
    ┌─────────────────────────────────────┐
    │        «protocol»                    │
    │       HarmonyChecker                 │
    ├─────────────────────────────────────┤
    │ + check_vowel_harmony(stem, suffix)  │
    │ + check_consonant_harmony(stem, sfx) │
    └──────────────┬──────────────────────┘
                   │ implements
          ┌────────┴────────┐
          ▼                 ▼
    ┌───────────────┐ ┌────────────────┐
    │ StrictHarmony │ │ RelaxedHarmony │
    │   Checker     │ │   Checker      │
    ├───────────────┤ ├────────────────┤
    │ BÜU: ✓        │ │ BÜU: ✗ (atlar) │
    │ KÜU: ✓        │ │ KÜU: ✓         │
    │ Ünsüz: ✓      │ │ Ünsüz: ✓       │
    └───────────────┘ └────────────────┘

    ┌─────────────────────────────────────┐
    │   «frozen dataclass»                 │
    │   HarmonyStrategy                    │
    ├─────────────────────────────────────┤
    │ + checker: HarmonyChecker            │
    │ + min_stem_override: int | None      │
    └─────────────────────────────────────┘

    ┌─────────────────────────────────────┐
    │   MorphologicalAnalyzer              │
    ├─────────────────────────────────────┤
    │ - _registry: SuffixRegistry          │
    │ - _strategies: list[HarmonyStrategy] │
    │ - _validator: StemValidator           │
    │ + MAX_ITERATIONS = 10                │
    ├─────────────────────────────────────┤
    │ + analyze(word) → MorphemeAnalysis   │
    │ - _find_suffix_match(current)        │
    │ - _try_strategy(current, strategy)   │
    │ - _handle_yor_connector(stem, sfx)   │
    └─────────────────────────────────────┘

    ┌─────────────────────────────────────┐
    │   SuffixRegistry                     │
    ├─────────────────────────────────────┤
    │ - _suffixes: list[SuffixForm]        │
    │ - _sorted: bool                      │
    │ + TEMPLATE_VARS: dict                │
    ├─────────────────────────────────────┤
    │ + register(definition)               │
    │ + register_many(definitions)         │
    │ + suffixes → list[SuffixForm] (lazy) │
    │ + create_default() → SuffixRegistry  │
    │ - _expand(template) → list[str]      │
    └─────────────────────────────────────┘

    ┌───────────────────┐  ┌──────────────────┐
    │«frozen dataclass» │  │«frozen dataclass»│
    │ SuffixDefinition  │  │ SuffixForm       │
    ├───────────────────┤  ├──────────────────┤
    │ template: str     │  │ form: str        │
    │ label: str        │  │ label: str       │
    │ harmony_exempt    │  │ harmony_exempt   │
    │ min_stem_length   │  │ min_stem_length  │
    └───────────────────┘  └──────────────────┘
          girdi formatı          açılmış biçim

    ┌─────────────────────────────────────┐
    │   «frozen dataclass»                 │
    │   MorphemeAnalysis                   │
    ├─────────────────────────────────────┤
    │ + stem: str                          │
    │ + suffixes: list[tuple[str, str]]    │
    │ + parts → list[str]  (property)      │
    └─────────────────────────────────────┘

    ┌─────────────────────────────────────┐
    │   StemValidator                      │
    ├─────────────────────────────────────┤
    │ + is_valid(stem) → bool              │
    └─────────────────────────────────────┘

    ┌─────────────────────────────────────┐
    │   AnalysisFormatter                  │
    ├─────────────────────────────────────┤
    │ + format_analysis(word, analysis)    │
    │ + vowel_harmony_report(word) static  │
    └─────────────────────────────────────┘
```

### 5.4 Nesne Oluşturma Akışı — `create_default_analyzer()`

```
    create_default_analyzer()
    │
    ├─→ SuffixRegistry.create_default()
    │       │
    │       ├─→ SuffixRegistry()
    │       │
    │       └─→ register_many(DEFAULT_SUFFIX_DEFINITIONS)
    │               │
    │               └─→ Her tanım için:
    │                   _expand(template) → somut biçimler
    │                   SuffixForm(form, label, exempt, min)
    │
    ├─→ StrictHarmonyChecker()
    │
    ├─→ RelaxedHarmonyChecker()
    │
    ├─→ HarmonyStrategy(strict, min_stem_override=None)
    │
    ├─→ HarmonyStrategy(relaxed, min_stem_override=3)
    │
    └─→ MorphologicalAnalyzer(registry, [strict_strat, relaxed_strat])
```

---

## 6. SOLID Prensipleri Analizi

### 6.1 Single Responsibility Principle (SRP)

Her modül tam olarak bir sorumluluğa sahiptir:

```
    ┌─────────────┬──────────────────────────────────────┐
    │ Modül       │ Tek Sorumluluk                       │
    ├─────────────┼──────────────────────────────────────┤
    │ phonology   │ Fonetik sabitler & ses sınıflandırma │
    │ harmony     │ Ses uyumu kuralları                  │
    │ suffix      │ Ek verisi & şablon açılımı           │
    │ analyzer    │ Çözümleme algoritması                │
    │ formatter   │ Sonuç gösterimi                      │
    │ __init__    │ Birleşim & fabrika                   │
    └─────────────┴──────────────────────────────────────┘
```

**Önceki ihlal:** Tek dosya hem fonoloji, hem uyum, hem ek tanımı, hem
algoritma, hem biçimlendirme, hem CLI yapıyordu.

### 6.2 Open/Closed Principle (OCP)

Sisteme yeni davranış eklemek mevcut kodu değiştirmeden mümkün:

```
    Yeni ek eklemek:
    ┌────────────────────────────────────────────────┐
    │ registry = SuffixRegistry.create_default()     │
    │ registry.register(SuffixDefinition(            │
    │     template="{I}yor{D}{I}",                   │
    │     label="YENİ_EK",                           │
    │ ))                                             │
    └────────────────────────────────────────────────┘
    → Mevcut hiçbir sınıf değiştirilmedi ✓

    Yeni uyum stratejisi eklemek:
    ┌────────────────────────────────────────────────┐
    │ class SuperRelaxedChecker:                     │
    │     def check_vowel_harmony(self, s, x):       │
    │         return True  # Her şeyi kabul et       │
    │     def check_consonant_harmony(self, s, x):   │
    │         return True                            │
    │                                                │
    │ strategies.append(                             │
    │     HarmonyStrategy(SuperRelaxedChecker())     │
    │ )                                              │
    └────────────────────────────────────────────────┘
    → HarmonyChecker Protocol'ünü karşılayan her sınıf çalışır ✓
```

### 6.3 Liskov Substitution Principle (LSP)

```
    StrictHarmonyChecker ve RelaxedHarmonyChecker,
    HarmonyChecker Protocol'ünü tam olarak karşılar.
    Biri yerine diğeri konabilir:

    analyzer = MorphologicalAnalyzer(
        registry=reg,
        strategies=[
            HarmonyStrategy(checker=StrictHarmonyChecker()),   ← Bu
            HarmonyStrategy(checker=RelaxedHarmonyChecker()),  ← veya bu
            HarmonyStrategy(checker=HerhangiBirChecker()),     ← veya bu
        ]
    )

    Analyzer hiçbir zaman isinstance() kontrolü yapmaz.
    Yalnızca Protocol'deki metotları çağırır. ✓
```

### 6.4 Interface Segregation Principle (ISP)

```
    ┌──────────────────────────────────────────────────────┐
    │ HarmonyChecker Protocol: yalnızca 2 metot            │
    │   check_vowel_harmony()                              │
    │   check_consonant_harmony()                          │
    │                                                      │
    │ İstemcilerin bilmek zorunda olmadığı detaylar         │
    │ Protocol'e dahil DEĞİL:                              │
    │   × check_major_harmony()     (dahili detay)         │
    │   × check_minor_harmony()     (dahili detay)         │
    │   × BACK_VOWELS, ROUNDED ...  (veri)                 │
    └──────────────────────────────────────────────────────┘

    MorphemeAnalysis: istemciler sadece veri alır
    ┌──────────────────────────────────────────────────────┐
    │ result = analyzer.analyze("evlerinden")              │
    │ result.stem      → "ev"            (salt okunur)     │
    │ result.suffixes  → [("leri", ..)]  (salt okunur)     │
    │ result.parts     → ["ev", "leri"]  (hesaplanır)      │
    │                                                      │
    │ Biçimlendirme AYRI sınıfta:                          │
    │ formatter.format_analysis(word, result)               │
    │                                                      │
    │ → İstemci veriyi almak için I/O'ya bağımlı değil ✓   │
    └──────────────────────────────────────────────────────┘
```

### 6.5 Dependency Inversion Principle (DIP)

```
    ÖNCE (monolitik):
    ┌──────────────────────┐
    │ find_morphemes()     │
    │                      │
    │ doğrudan çağırır:    │
    │ → check_vowel_harm() │  ← somut fonksiyona bağımlı
    │ → check_minor_harm() │  ← somut fonksiyona bağımlı
    │ → valid_stem()       │  ← somut fonksiyona bağımlı
    └──────────────────────┘

    SONRA (SOLID):
    ┌────────────────────────────┐
    │ MorphologicalAnalyzer      │
    │                            │
    │ bağımlılıkları:            │
    │ → HarmonyChecker Protocol  │  ← soyutlamaya bağımlı ✓
    │ → StemValidator (enjekte)  │  ← enjekte edilebilir ✓
    │ → SuffixRegistry (enjekte) │  ← enjekte edilebilir ✓
    └────────────────────────────┘

    Üst düzey modül (analyzer) alt düzey modüllere
    (phonology, harmony) doğrudan bağımlı değil;
    Protocol soyutlaması aracılığıyla iletişim kurar.
```

---

## 7. Veri Akışı: Uçtan Uca Örnek

### Girdi: `"koşuyormuş"`

```
ADIM 1: lowercase + strip
        "koşuyormuş"

ADIM 2: İterasyon 1 — Sağdan en uzun ek ara
        Strict strateji:
          current = "koşuyormuş"
          En uzun eşleşme: "muş" → DUYULAN_GEÇMİŞ
          Kök: "koşuyor"
          Uyum: last_vowel("koşuyor")="o", get_vowels("muş")[0]="u"
            BÜU: o∈KALIN, u∈KALIN → ✓
            KÜU: o∈YUVARLAK, u∈DAR+YUVARLAK → ✓
            Ünsüz: "r"→"m" (ikisi de ötümsüz değil → sorun yok) → ✓
          valid_stem("koşuyor"): ünlü var, sonda "r" (1 ünsüz) → ✓
          ✅ Eşleşti!

        suffixes = [("muş", "DUYULAN_GEÇMİŞ")]
        current = "koşuyor"

ADIM 3: İterasyon 2
        current = "koşuyor"
        En uzun eşleşme: "yor" → ŞİMDİKİ_ZAMAN (harmony_exempt=True)
        Kök: "koşu"
        Uyum kontrolü: ATLA (uyum muaf)
        Ünsüz: "u"→"y" → ✓
        valid_stem("koşu"): ünlü var → ✓
        ✅ Eşleşti!

        suffixes = [("yor", "ŞİMDİKİ_ZAMAN"), ("muş", "DUYULAN_GEÇMİŞ")]
        current = "koşu"

ADIM 4: İterasyon 3
        current = "koşu"
        Hiçbir ek eşleşmedi → DÖNGÜ BİTTİ

ADIM 5: -yor bağlayıcı ünlü işleme
        İlk ek: "yor" + "ŞİMDİKİ_ZAMAN" → koşul sağlandı
        Kök: "koşu" → son harf "u" ∈ CLOSE_VOWELS ✓
        remaining = "koş", connector = "u"
        "koş" ünsüzle bitiyor ✓, ünlü var ✓
        check_vowel_harmony("koş", "u"):
          last_vowel("koş")="o", "u"
          BÜU: o∈KALIN, u∈KALIN → ✓
          KÜU: o∈YUVARLAK, u∈DAR+YUVARLAK → ✓
        ✅ Bağlayıcı ünlü birleştirildi!

        stem = "koş"
        suffixes = [("uyor", "ŞİMDİKİ_ZAMAN"), ("muş", "DUYULAN_GEÇMİŞ")]

ÇIKTI:
        MorphemeAnalysis(
            stem="koş",
            suffixes=[("uyor", "ŞİMDİKİ_ZAMAN"),
                      ("muş", "DUYULAN_GEÇMİŞ")]
        )
        → koş + uyor + muş
```

---

## 8. Ek Kataloğu

Sistemde tanımlı 51 ek şablonu, toplam ~300+ somut biçime açılır:

```
    Kategori              Şablon          Biçim sayısı   Örnek
    ─────────────────────────────────────────────────────────────
    ÇOĞUL                 l{A}r           2              lar, ler
    İYELİK_1T             {I}m            4              ım, im, um, üm
    İYELİK_2T             {I}n            4              ın, in, un, ün
    İYELİK_3T             s{I}            4              sı, si, su, sü
    İYELİK_1Ç             {I}m{I}z        16             ımız, imiz, ...
    İYELİK_2Ç             {I}n{I}z        16             ınız, iniz, ...
    İYELİK_3Ç             l{A}r{I}        8              ları, leri, ...
    BULUNMA               {D}{A}          4              da, de, ta, te
    AYRILMA               {D}{A}n         4              dan, den, tan, ten
    AYRILMA               n{D}{A}n        4              ndan, nden, ...
    YÖNELME               y{A} / n{A}     2+2            ya, ye, na, ne
    BELIRTME              y{I} / n{I}     4+4            yı, yi, ..., nı
    TAMLAYAN              n{I}n           4              nın, nin, nun, nün
    GEÇMİŞ_ZAMAN         {D}{I}          8              dı, di, ..., tü
    DUYULAN_GEÇMİŞ       m{I}ş           4              mış, miş, muş, müş
    ŞİMDİKİ_ZAMAN        {I}yor          4              ıyor, iyor, ...
    ŞİMDİKİ_ZAMAN        yor             1              yor (uyum muaf)
    GENİŞ_ZAMAN          {I}r / {A}r     4+2            ır, ir, ..., ar, er
    GENİŞ_ZAMAN_OLMSZ    m{A}z           2              maz, mez
    GELECEK_ZAMAN         {A}c{A}k        4              acak, ecek, ...
    BİLDİRME/ETTİRGEN    {D}{I}r         8              dır, dir, ..., tür
    EDİLGEN              {I}l            4              ıl, il, ul, ül
    ETTİRGEN              l{A}t           2              lat, let
    İŞTEŞ                {I}ş            4              ış, iş, uş, üş
    YETERLİLİK            {A}bil          2              abil, ebil
    OLUMSUZ               m{A}            2              ma, me
    MASTAR                m{A}k           2              mak, mek
    SIFAT_FİİL            {D}{I}k         8              dık, dik, ..., tük
    SIFAT_FİİL            {D}{I}ğ         8              dığ, diğ, ..., tüğ
    SIFAT_FİİL            {A}n            2              an, en
    ZARF_FİİL             {I}p / y{I}p    4+4            ıp, ip, ..., yüp
    ZARF_FİİL             ken             1              ken (uyum muaf)
    ZARF_FİİL             {A}r{A}k        4              arak, erek, ...
    ZARF_FİİL             {I}nc{A}        8              ınca, ince, ...
    DİLEK_ŞART            s{A}            2              sa, se
    KİŞİ_1Ç              {I}z            4              ız, iz, uz, üz
    KİŞİ_2Ç              s{I}n{I}z       16             sınız, siniz, ...
    KİŞİ_2T              s{I}n           4              sın, sin, sun, sün
    EMİR_3Ç              s{I}nl{A}r      8              sınlar, sinler, ...
    VASITA                l{A}            2              la, le
    İLGİ                  ki              1              ki (uyum muaf)
    YAPIM_-lI             l{I}            4              lı, li, lu, lü
    YAPIM_-lIk            l{I}k           4              lık, lik, luk, lük
    YAPIM_-sIz            s{I}z           4              sız, siz, suz, süz
    YAPIM_-CI             {C}{I}          8              cı, ci, ..., çü
    YAPIM_-CIlIk          {C}{I}l{I}k     16             cılık, cilik, ...
    YAPIM_-lAş            l{A}ş           2              laş, leş
    YAPIM_-lAn            l{A}n           2              lan, len
```

### Uyum Muaf Ekler

```
    ┌──────────┬───────────────────────────────────────────┐
    │ Ek       │ Neden muaf?                               │
    ├──────────┼───────────────────────────────────────────┤
    │ -yor     │ Değişmeyen yuvarlak ünlü (o)              │
    │ -ken     │ Değişmeyen ince düz ünlüler (e)           │
    │ -ki      │ Değişmeyen ince düz ünlü (i)              │
    └──────────┴───────────────────────────────────────────┘
```

---

## 9. Bilinen Sınırlamalar & Gelecek İyileştirmeler

### 9.1 Sözlük Gerektiren Fenomenler

```
    Fenomen                  Örnek              Beklenen        Bulunan
    ──────────────────────  ─────────────────  ──────────────  ──────────────
    Ünsüz yumuşaması        kitabından         kitap+ından     kitabı+ndan
      (p→b, k→g, t→d, ç→c)                    (p→b gerekir)

    Ünlü daralması           diyor             de+iyor         di+yor
      (e→i, a→ı)            yiyorlar           ye+iyor+lar     yi+yor+lar

    Derin türetim zinciri    güzelleştirilemez  güzel+leş+tir   güzelleştiri
                                                +il+e+mez      +le+mez

    Leksem sınırı            çalışkanlıklar-   çalışkan+lık    çalışk+an+lık
                              ından             +ları+ndan      +ları+ndan
```

### 9.2 Olası İyileştirme Yolları

```
    ┌─────────────────────────────────────────────────────────────┐
    │  1. SÖZLÜK ENTEGRASYONU                                    │
    │     Kök sözlüğü ile stem_candidate kontrolü                │
    │     → Ünsüz yumuşaması ve ünlü daralmasını çözer          │
    │                                                             │
    │  2. GERİ İZLEME (BACKTRACKING)                              │
    │     Greedy yerine geri izlemeli arama                       │
    │     → güzelleştirilemez gibi derin zincirleri çözer         │
    │                                                             │
    │  3. MORFOFONEMIK KURALLAR                                   │
    │     p→b, k→g, t→d, ç→c dönüşüm kuralları                  │
    │     → Sözlükle birleşince ünsüz yumuşamasını çözer         │
    │                                                             │
    │  4. İSTATİSTİKSEL SIRALAMA                                  │
    │     Birden fazla olası çözümlemeyi skorlama                 │
    │     → En olası ayrımı seçme                                │
    │                                                             │
    │  5. TEST ALTYAPISI                                          │
    │     pytest ile birim testler                                │
    │     → Regresyon kontrolü                                   │
    └─────────────────────────────────────────────────────────────┘
```

---

## 10. Kullanım Örnekleri

### 10.1 CLI

```bash
# Yerleşik test sözcükleri
python find_lemma.py

# Tek sözcük
python find_lemma.py evlerinden

# Etkileşimli mod
python find_lemma.py -i
```

### 10.2 Python API

```python
from morphology import create_default_analyzer, AnalysisFormatter

# Çözümleyici oluştur
analyzer = create_default_analyzer()

# Çözümle
result = analyzer.analyze("evlerinden")
print(result.stem)       # "ev"
print(result.suffixes)   # [("leri", "İYELİK_3Ç"), ("nden", "AYRILMA")]
print(result.parts)      # ["ev", "leri", "nden"]

# Biçimlendir
fmt = AnalysisFormatter()
print(fmt.format_analysis("evlerinden", result))
```

### 10.3 Genişletme — Yeni Ek Ekleme

```python
from morphology import create_default_analyzer, SuffixRegistry
from morphology.suffix import SuffixDefinition

analyzer = create_default_analyzer()
analyzer._registry.register(SuffixDefinition(
    template="{D}{I}kç{A}",
    label="ZARF_FİİL_-DIkçA",
))
# Artık "gördükçe" gibi sözcükler de çözümlenir
```

### 10.4 Genişletme — Yeni Uyum Stratejisi

```python
from morphology import MorphologicalAnalyzer, HarmonyStrategy, SuffixRegistry

class NoHarmonyChecker:
    """Uyum kontrolü yapmayan strateji (deney amaçlı)."""
    def check_vowel_harmony(self, stem, suffix):
        return True
    def check_consonant_harmony(self, stem, suffix):
        return True

analyzer = MorphologicalAnalyzer(
    registry=SuffixRegistry.create_default(),
    strategies=[HarmonyStrategy(checker=NoHarmonyChecker())],
)
```

---

## 11. Sonuç

Bu proje, Türkçe morfolojik çözümlemenin **sözlüksüz, salt kural tabanlı**
sınırlarını test eder. Ses uyumu kuralları tek başına 28 test sözcüğünün
22'sini (%79) doğru ayırabilmektedir.

Kalan 6 sözcük, morfofonemik dönüşümler (ünsüz yumuşaması, ünlü daralması)
nedeniyle sözlük bilgisi gerektirir — bu, kural tabanlı yaklaşımın doğal
sınırıdır.

SOLID prensipleriyle yapılandırılmış mimari, sözlük entegrasyonu veya
istatistiksel modeller gibi gelecek iyileştirmelerin mevcut kodu bozmadan
eklenmesini mümkün kılar.

```
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   "Bir dil ne kadar düzenli kurallar içeriyorsa,     │
    │    kural tabanlı çözümleme o kadar etkili olur.      │
    │    Türkçe'nin ünlü uyumu bu düzenin en güzel         │
    │    örneklerinden biridir."                            │
    │                                                      │
    └──────────────────────────────────────────────────────┘
```
