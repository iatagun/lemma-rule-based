# 🧠 Uzman Panel Değerlendirmesi — Türkçe Kural-Tabanlı Morfolojik Çözümleyici

> **Tarih:** 2026-04-13  
> **Proje:** `lemma-rule-based`  
> **Mevcut Doğruluk:** %86.9 (BOUN Treebank, 8848/10182)  
> **Panel:** 4 uzman agent paralel inceleme + çapraz tartışma sentezi

---

## 📋 Panel Üyeleri

| # | Uzman | Odak Alanı | Temel Sorular |
|---|-------|-----------|---------------|
| 🔤 | **Morfoloji & Sözdizim** | Ek hiyerarşisi, çatı ekleri, nominalizasyon, etiketleme | Ek envanteri yeterli mi? Slot modeli doğru mu? |
| 🔊 | **Sesbilim (Fonoloji)** | Ünlü/ünsüz uyumu, ses olayları, hece yapısı | Ses kuralları doğru kodlanmış mı? |
| 💻 | **Hesaplamalı Dilbilim** | Algoritma, mimari, SOLID, benchmark, FST karşılaştırma | Doğru yaklaşım mı? Ölçeklenebilir mi? |
| 📖 | **Sözlükbilim & Derlem** | TDK sözlüğü, lemmatizasyon, BOUN uyumu, lookup tabloları | Sözlük yeterli mi? Gold standard uyuşuyor mu? |

---

## 1. 🏗️ MİMARİ — UZMANLAR NE DİYOR?

### 1.1 Sağdan Sola Ek Sıyırma (Suffix Stripping)

**🔤 Morfoloji:** "Sondan eklemeli (agglutinative) yapı için meşru bir temel strateji. En uzun eşleşme (longest match) önceliği, `evlerinden` gibi sözcüklerde `-nden` yerine doğru biçimde `-den` eşleşmesini sağlıyor."

**💻 Hesaplamalı:** "Dilbilimsel olarak doğal ve uygun. O(n × m) karmaşıklıkla sabit üst sınır — performans açısından sorunsuz. Ancak FST'lerin deterministik morfotaktik modeline kıyasla belirsizlik çözümlemesi zayıf."

**UZLAŞI:** ✅ Doğru temel strateji. Ancak **backtracking eksikliği** en büyük sınırlama:
```
sinirden → sin+ir(GENİŞ_ZAMAN)+den(AYRILMA) ✗  [greedy hata]
sinirden → sinir+den(AYRILMA) ✓                  [doğru, backtracking gerekir]
```

### 1.2 Dört Katmanlı Strateji Sistemi

```
Katman 1: StrictHarmony + Sözlük  → Tam uyumlu Türkçe sözcükler
Katman 2: RelaxedHarmony + Sözlük → Alıntı sözcükler (saat, otobüs)
Katman 3: StrictHarmony + Sezgisel → Sözlükte olmayan sözcükler
Katman 4: RelaxedHarmony + Sezgisel → Son çare (fallback)
```

**🔊 Sesbilim:** "Alıntı sözcük stratejisi akıllıca. `is_loanword_candidate()` fonksiyonundaki 'ilk hece dışında o/ö' sezgisi fonotaktik olarak sağlam — Türkçe kökenli sözcüklerde yuvarlak geniş ünlüler yalnızca ilk hecede bulunur."

**💻 Hesaplamalı:** "Çok iyi düşünülmüş. Dilbilimsel olarak doğru hiyerarşi. Precision → Recall gradient'i doğru yönde."

**🔤 Morfoloji:** "Çok isabetli bir tasarım. `saat+ler+inde` ile `ev+ler+inde` arasındaki farkı doğru biçimde yakalıyor."

**UZLAŞI:** ✅ **Tüm uzmanlar hemfikir — en güçlü mimari karar.**

> **💻 Eleştiri:** "Katmanlar arası geçiş sözcük düzeyinde yapılıyor, ama ideal olarak ek düzeyinde olmalı. `televizyonlarından` sözcüğünde kök `televizyon` relaxed'e düşerken, ekler `lar+ın+dan` strict uyuma tabidir."

### 1.3 SOLID Prensipleri

**💻 Hesaplamalı (detaylı değerlendirme):**

| Prensip | Durum | Açıklama |
|---------|-------|----------|
| **SRP** | ⚠️ Kısmi | `phonology.py`, `harmony.py`, `suffix.py` mükemmel. Ama `analyzer.py` (850+ satır) hem algoritmayı hem 7 veri tablosunu barındırıyor |
| **OCP** | ✅ İyi | `SuffixRegistry.register()` + Protocol-tabanlı strateji — yeni ek/kural eklemek mevcut kodu değiştirmeden yapılıyor |
| **DIP** | ✅ İyi | `HarmonyChecker` Protocol'ü — gerçek DIP, `isinstance()` yok |
| **LSP** | ✅ İyi | `StrictHarmonyChecker` ve `RelaxedHarmonyChecker` yer değiştirebilir |
| **ISP** | ✅ İyi | Küçük, odaklı arayüzler |

**SRP İhlali — `analyzer.py`:** 7 veri tablosu (~150 satır saf veri) + çözümleme algoritması tek dosyada. Ayrı `lexicon.py` veya `exceptions.py` modülüne taşınmalı.

---

## 2. 🔊 SESBİLİM — ÜNLÜ/ÜNSÜZ UYUMU

### 2.1 Büyük Ünlü Uyumu (BÜU / Palatal Harmony)

**🔊 Sesbilim:** "✅ Doğru. Kalınlık-incelik uyumu ikili karşıtlığa dayalı: `BACK_VOWELS = {a, ı, o, u}` vs `FRONT_VOWELS = {e, i, ö, ü}`. Kod, gövdenin son ünlüsüyle ekin ilk ünlüsünün aynı kümeye ait olup olmadığını doğru denetler."

```
ev + ler → e(ince) → e(ince) ✅
kitap + lar → a(kalın) → a(kalın) ✅
*kitap + ler → a(kalın) → e(ince) ❌ → doğru reddedilir
```

### 2.2 Küçük Ünlü Uyumu (KÜU / Labial Harmony)

**🔊 Sesbilim:** "✅ Doğru. Düzlük-yuvarlaklık uyumu iki boyutlu matris ile kodlanmış:"

| Gövde ünlüsü | Ek ünlüsü: dar | Ek ünlüsü: geniş |
|---|---|---|
| **Düz** (a, e, ı, i) | düz-dar: ı, i | düz-geniş: a, e |
| **Yuvarlak** (o, ö, u, ü) | yuvarlak-dar: u, ü | düz-geniş: a, e |

"BÜU ve KÜU bağımsız boyutlar olarak ayrı kontrol edilip `and` ile birleştirilmesi, Türkçe ünlü uyumunun iki bağımsız özellik (feature) üzerine kurulu olduğunu doğru yansıtır: [±kalın] ve [±yuvarlak]."

### 2.3 Şablon Değişkenleri

**🔊 Sesbilim:** "✅ Sesbilimsel olarak tam ve doğru."

| Değişken | Karşılıklar | Kural |
|----------|-------------|-------|
| `{A}` | a, e | Yalnızca BÜU'ya bağlı (2-yollu) |
| `{I}` | ı, i, u, ü | BÜU + KÜU (4-yollu) |
| `{D}` | d, t | Ünsüz ötümlülük benzeşmesi |
| `{C}` | c, ç | Ünsüz ötümlülük benzeşmesi |

### 2.4 🐛 Ünsüz Uyumunda {C} Hatası

**🔊 Sesbilim:** "Ünlüden sonra `ç` yanlışlıkla kabul ediliyor!"

| Gövde | Ek | Beklenen | Kod Sonucu | Durum |
|---|---|---|---|---|
| `araba` + `cı` | arabacı | True | True | ✅ |
| `araba` + `çı` | *arabaçı | **False** | **True** | 🐛 HATA |
| `balık` + `çı` | balıkçı | True | True | ✅ |

**Düzeltme:**
```python
# Mevcut (hatalı):
if first_ch == "ç" and last_ch in CONSONANTS and last_ch not in VOICELESS:
# Düzeltilmiş:
if first_ch == "ç" and last_ch not in VOICELESS:
```

### 2.5 Şapkalı Ünlüler (â, î, û)

**🔊 Sesbilim:** "🔴 Eksik — `VOWELS` kümesinde yok."

```python
VOWELS = frozenset("aeıioöuü")  # â, î, û YOK!
get_vowels("rüzgâr")  # → ['ü'] — 'â' atlanıyor!
```

**📖 Sözlükbilim:** "65 BOUN token şapkalı lemma bekliyor ama TDK sözlüğünde 0 şapkalı sözcük var. Sistem asla şapkalı lemma üretemez. `hâl` kelimesi tek başına 7 hata üretiyor."

**UZLAŞI:** Hem fonolojik hem sözlüksel düzeltme gerekiyor:
1. `VOWELS` kümesine `â, î, û` ekle
2. `_CIRCUMFLEX_MAP = {"hal": "hâl", "adet": "âdet", "kar": "kâr", ...}` tablosu ekle

### 2.6 Ünlü Daralması — Çok Heceli Fiiller

**🔊 Sesbilim:** "🔴 Kritik eksiklik."

| Sözcük | Beklenen | Sistemin Çıktısı | Durum |
|---|---|---|---|
| `başlıyor` | başla + yor | stem=başl, root=None | ❌ |
| `bekliyor` | bekle + yor | stem=bekl, root=None | ❌ |
| `söylüyor` | söyle + yor | stem=söy, root=None | ❌ |
| `ağlıyor` | ağla + yor | stem=ağ, root=None | ❌ |
| `diyor` | de + yor | stem=di, root=de | ✅ |

**Kök neden:** `_handle_yor_connector()` ünsüz kümesi yeniden düzenlemesi, ünlü-sonlu fiillerde yanlış tetikleniyor. `başl+ıyor` → `baş+lıyor` yapılıyor, ama doğrusu `başla+yor` (daralma: a→ı).

**Çözüm:** Ünsüz kümesi kontrolünden ÖNCE daralma tersine çevirme denemesi:
```python
if stem[-1] in CONSONANTS and connector in _NARROWING_REVERSE:
    widened_stem = stem + _NARROWING_REVERSE[connector]
    if dictionary.find_root(widened_stem) is not None:
        return widened_stem, [("yor", "ŞİMDİKİ_ZAMAN")]  # daralma kabul
```

---

## 3. 🔤 MORFOLOJİ — EK ENVANTERİ VE HİYERARŞİ

### 3.1 Ek Envanteri: 53 Ek Yeterli mi?

**🔤 Morfoloji:** "Temel çerçeveyi kaplıyor, ancak **önemli boşluklar** var."

#### Eksik Çekim Ekleri (kritik)

| Ek | Şablon | Örnek | Üretkenlik |
|---|---|---|---|
| **-CA** (eşitlik) | `{C}{A}` | *Türkçe, bence, güzelce* | ⭐⭐⭐ Çok yüksek |
| **-AmA-** (yeterlilik olumsuz) | `-{A}m{A}` | *gelemez, yapamaz* | ⭐⭐⭐ |
| **-mAdAn** (olumsuz zarf-fiil) | `m{A}d{A}n` | *gelmeden, bakmadan* | ⭐⭐⭐ |
| **-DIkçA** (süreklilik) | `{D}{I}kç{A}` | *geldikçe, baktıkça* | ⭐⭐ |
| **-mAktA** (sürerlik) | `m{A}kt{A}` | *gelmekte, yazmakta* | ⭐⭐ |

#### Eksik Yapım Ekleri

| Ek | Şablon | Örnek |
|---|---|---|
| **-IcI** (sıfat yapım) | `{I}c{I}` | *yapıcı, öğretici, kurucu* |
| **-DAş** (eşlik) | `{D}{A}ş` | *vatandaş, meslektaş* |
| **-sAl** (niteleme) | `s{A}l` | *tarihsel, bilimsel* |

### 3.2 Etiket Belirsizlikleri

**🔤 Morfoloji:** "Belirsiz etiketler çözümleme kalitesini düşürüyor."

| Etiket | İşlev 1 | İşlev 2 | Sorun |
|---|---|---|---|
| `BİLDİRME/ETTİRGEN` | *güzel**dir*** (bildirme) | *yaz**dır*** (ettirgen) | Çatı ≠ Bildirme |
| `OLUMSUZ/İSİM_FİİL` | *gel**me*** (olumsuz) | *oku**ma*** (isim-fiil) | Sözdizimi tamamen farklı |
| `İYELİK_3T/BELIRTME` | *ev**i*** (onun evi) | *ev**i*** (evi gördüm) | Syncretism, ama ayrım önemli |

### 3.3 Çatı Ekleri (Voice)

**🔤 Morfoloji:** "Çatı eki dizilimi kısmen doğru, ama eksik."

```
Türkçe standart: KÖK → [Ettirgen] → [Edilgen] → [İşteş] → [Dönüşlü]
Sistemde tanımlı: -Il (EDİLGEN), -Iş (İŞTEŞ), -lAt (ETTİRGEN)
Eksik: -t (kısa ettirgen), -Ir (ettirgen), -In (dönüşlü)
```

**Somut etki:** `yürüt-`, `oku-t-`, `bitir-`, `düşür-`, `giyinmek` gibi çok sık kullanılan fiillerin çatı analizi yapılamıyor.

### 3.4 Forbidden Bigram vs Tam Slot Modeli

**TARTIŞMA NOKTASI — Uzmanlar bölünmüş:**

**🔤 Morfoloji:** "Yasaklı ikili (forbidden bigram) yaklaşımı yetersiz. 20 yasak çift, Türkçe'deki onlarca imkânsız geçişin küçük bir alt kümesi. Çift hal (`*evdeden`), çift çoğul (`*evlerler`), kişi sonrası hal engellenmiyor. **Tam slot-tabanlı FSM gerekli.**"

**💻 Hesaplamalı:** "Veri güdümlü kısıtlama yaklaşımı etkili — empirik doğruluk sağlıyor. Ancak 'hangi çiftler geçerli?' sorusuna cevap vermiyor. Tam FSA ~30-40 durum ve ~100 geçiş kuralıyla modellenebilir."

**📖 Sözlükbilim:** "Forbidden bigram +31 token kazandırdı, 0 kayıp. Pragmatik yaklaşım çalışıyor."

**UZLAŞI:** Mevcut forbidden bigram sistemi iyi bir **ilk adım**, ama uzun vadede tam slot modeline evrilmeli. Kısa vadede yasaklı çift listesini genişletmek (çift hal, çift çoğul vb.) düşük maliyetli kazanç sağlar.

---

## 4. 📖 SÖZLÜK & LEMMATİZASYON

### 4.1 Sözlük Koruması (Dictionary Protection)

**📖 Sözlükbilim:** "Üç seviyeli koruma mekanizması dilbilimsel olarak sağlam tasarlanmış."

```
Seviye 1: Fiil gövdesi → stem + mak/mek sözlükte → atomik
           (başla+mak=başlamak → "başla" korunur)
Seviye 2: Sözlük sözcüğü → yalnızca fiil kökü + fiil eki izni
           (neden → ne+den ✗, olur → ol+ur ✓)
Seviye 3: Morfofonemik çözümleme → find_root() ile doğrulama
```

**Aşırı koruma sorunu:**
- `ister` sözlükte bağımsız girdi → `ist+er` ayrıştırması engelleniyor → BOUN gold: `iste` (6 hata)
- `değer` sözlükte → `değ+er+ler+in` çözümlenemiyor → `değ` üretiliyor (6 hata)

### 4.2 Morfofonemik Çözümleme (5 Adım)

**🔊 Sesbilim + 📖 Sözlükbilim ortak değerlendirme:**

| Adım | Mekanizma | Örnek | Durum |
|---|---|---|---|
| 1 | Doğrudan eşleşme | `ev → ev` | ✅ |
| 2 | Fiil kökü (+mak/mek) | `gel → gelmek ✓` | ✅ |
| 3 | Ünsüz yumuşaması tersine | `kitab → kitap` | ✅ |
| 4 | Ünlü düşmesi tersine | `burn → burun` | ✅ |
| 5 | Kaynaştırma harfi kaldırma | `suy → su` | ✅ |

**Eksik morfofonemik kurallar:**
- **Ünsüz ikizleşmesi (gemination):** `hak → hakk-ı`, `ret → redd-i`, `his → hiss-i`
- **`-e/-a` sonlu fiil kökleri:** `iste`, `söyle`, `bekle` sözlükte bağımsız girdi değil → çekimli biçimlerinde sorun

### 4.3 BOUN Treebank Uyumu

**📖 Sözlükbilim:** "Kritik uyumsuzluk noktaları var."

| Konu | BOUN Standardı | Sistemin Davranışı | Etki |
|---|---|---|---|
| Fiil lemması | Yalın gövde: `iste`, `gel` | `root` veya `stem` | Türetilmiş fiillerde fazla geri gidiyor |
| Şapka | `hâl`, `âdet`, `kâr` | Şapkasız: `hal`, `adet`, `kar` | 65 token otomatik hata |
| Birleşik sözcük | `cezaev` (iyelik düşürülür) | `cezaevi` (sözlükteki biçim) | Birleşik sözcük ayrıştırma yok |

### 4.4 Lookup Tabloları

**📖 Sözlükbilim (detaylı denetim):**

| Tablo | Kapsam | Eksikler |
|---|---|---|
| Zamirler | ~65 biçim | `sen` paradigması (sana, seni), `nere` (nerede), `şura` formları, `hiçbiri` |
| Kopula | ~80 biçim | `ydük`, `ymışsınız`, `değillerdir` gibi nadir biçimler |
| Sıra sayıları | 22 biçim | ✅ Yeterli |
| Son-çekim edatları | ~55 biçim | `dış`, `taraf`, `karşı`, `neden`, `göre`, `rağmen` eksik |

---

## 5. 💻 BENCHMARK & LİTERATÜR KARŞILAŞTIRMASI

### 5.1 Doğruluk Karşılaştırması

| Sistem | Yaklaşım | Lemma Doğruluğu | Not |
|--------|----------|----------------|-----|
| **Bu proje** | **Kural-tabanlı + Sözlük** | **%86.9** | Sözlük destekli |
| Oflazer (1994) | FST | ~%95-97 | Tam morfolojik analiz |
| Zemberek | FSA + Sözlük | ~%93-95 | Java, production-ready |
| TRmorph (Çöltekin) | FST (HFST) | ~%92-94 | Açık kaynak |
| UDPipe (neural) | BiLSTM-CRF | ~%95+ | Bağlam-duyarlı |
| Stanza (Stanford) | Neural | ~%96+ | Transformer-tabanlı |

**💻 Hesaplamalı:** "%86.9, saf kural-tabanlı bir sistem için iyi bir başarı. Sözlük entegrasyonu büyük fark yaratmış (%79 → %86.9). Ancak production-grade için yetersiz — her 7-8 sözcükten biri hatalı."

### 5.2 Bu Projenin FST'ye Göre Avantajları

| Kriter | Bu Proje | FST (Oflazer/TRmorph) |
|--------|---------|----------------------|
| Prototipleme hızı | Dakikalar | Saatler |
| Okunabilirlik | Python kodu | XFST/lexc formalizmi |
| Debug edilebilirlik | Her adım izlenebilir | Çok zor |
| Genişletilebilirlik | `register()` ile trivial | Transducer yeniden derleme |
| Morfotaksi | Kısmi (forbidden bigram) | Tam (durum geçiş grafiği) |
| Belirsizlik çözümleme | İlk bulunan (greedy) | Tüm olası çözümler |

### 5.3 POS Bazlı Hata Analizi

```
DET: 99.8%  ██████████████████████████████  (mükemmel)
PART:100.0% ██████████████████████████████  (mükemmel)
SCONJ:100%  ██████████████████████████████  (mükemmel)
CCONJ:99.7% ██████████████████████████████  (mükemmel)
PRON: 96.6% █████████████████████████████   (çok iyi)
ADP:  96.2% █████████████████████████████   (çok iyi)
AUX:  96.2% █████████████████████████████   (çok iyi)
ADV:  94.6% ████████████████████████████    (iyi)
ADJ:  91.5% ███████████████████████████     (iyi)
NUM:  88.0% ██████████████████████████      (orta)
INTJ: 86.4% █████████████████████████       (orta)
NOUN: 86.2% █████████████████████████       (orta - iyileştirme potansiyeli)
PROPN:80.5% ████████████████████████        (zayıf)
VERB: 77.1% ███████████████████████         (zayıf - en büyük sorun)
```

---

## 6. 🤝 UZMAN UZLAŞISI — GÜÇLÜ YANLAR

### ✅ G1: Dört Katmanlı Strateji Sistemi (4/4 uzman hemfikir)

Strict→Relaxed, Dict→Heuristic fallback zinciri, Türkçe'nin en büyük istisna kaynağı olan alıntı sözcükleri zarif biçimde ele alıyor. Protocol-tabanlı soyutlama, yeni strateji eklemeyi trivial kılıyor.

### ✅ G2: Morfofonemik Dönüşüm Derinliği (4/4)

5 adımlı `find_root()` + ünlü daralması + `-yor` bağlayıcı ünlü + düzensiz fiil tabloları — Türkçe'nin temel ses değişim kurallarının büyük çoğunluğu kapsanmış.

### ✅ G3: Sözlük Entegrasyonu ve Koruma Mekanizması (4/4)

48,715 TDK girişli sözlük + üç seviyeli koruma (fiil gövdesi → sözlük sözcüğü → morfofonemik). `neden→ne+den`, `için→iç+in` gibi klasik tuzaklar başarıyla engelleniyor.

---

## 7. 🤝 UZMAN UZLAŞISI — ZAYIF YANLAR

### ❌ Z1: Çok Heceli Fiillerde Ünlü Daralması (🔊 + 📖 kritik)

`başlıyor`, `bekliyor`, `söylüyor`, `ağlıyor` — Türkçe'nin en üretken fiil sınıflarından biri çözümlenemiyor. Benchmark'ta tahminen **30+ token** etkiliyor.

### ❌ Z2: Şapkalı Ünlü + Circumflex Eşleme Eksikliği (🔊 + 📖 kritik)

`VOWELS` kümesinde `â, î, û` yok + TDK sözlüğünde 0 şapkalı sözcük. BOUN'un 65 tokeni otomatik hata. `hâl` tek başına 7 hata.

### ❌ Z3: Ek Envanteri Boşlukları (🔤 + 💻)

`-CA` (eşitlik), `-AmA-` (yeterlilik olumsuz), `-mAdAn`, `-DIkçA`, `-t` (kısa ettirgen), `-In` (dönüşlü çatı) gibi üretken ekler eksik.

### ❌ Z4: Backtracking Eksikliği (🔤 + 💻)

Greedy algoritma yanlış uzun ek eşleştirdiğinde düzeltemez. VERB doğruluğunun %77.1'de kalmasının temel nedeni.

### ❌ Z5: `-e/-a` Sonlu Fiil Kökleri (📖 + 🔊)

`iste-`, `söyle-`, `bekle-`, `düzenle-`, `belirle-` sözlükte bağımsız girdi değil → çekimli biçimlerde sorun. En az 30+ token etkiliyor.

---

## 8. 🎯 ORTAK ÖNERİLER — ÖNCELİK SIRALI

### 🔴 Öncelik 1: Şapkalı Ünlü Düzeltmesi (tahmini etki: +65 token, ~%0.6)

**Maliyet:** Düşük (eşleme tablosu + 2 satır kod)  
**Risk:** Sıfır  
**Tüm uzmanlar hemfikir.**

```python
# phonology.py
VOWELS = frozenset("aeıioöuüâîû")

# analyzer.py
_CIRCUMFLEX_MAP = {
    "hal": "hâl", "adet": "âdet", "kar": "kâr",
    "ruzgar": "rüzgâr", "imkan": "imkân", "hikaye": "hikâye",
    "dukkan": "dükkân", "kase": "kâse", ...
}
```

### 🔴 Öncelik 2: Ünlü Daralması Düzeltmesi (tahmini etki: +30-50 token, ~%0.3-0.5)

**Maliyet:** Orta (`_handle_yor_connector` yeniden tasarım)  
**Risk:** Düşük (yalnızca `-yor` bağlamı)

### 🔴 Öncelik 3: Eksik Eklerin Eklenmesi (tahmini etki: +50-80 token, ~%0.5-0.8)

**Öncelik sırası:** `-CA` > `-AmA-` > `-mAdAn` > `-DIkçA` > `-t` (ettirgen) > `-In` (dönüşlü)

### 🟡 Öncelik 4: Türetim Zinciri Genişletme (tahmini etki: +100-150 token, ~%1.0-1.5)

`_DERIVATIONAL_VERB_SUFFIXES` listesine `-t`, `-n`, `-lAş`, `-lA`, `-At` ekleri.

### 🟡 Öncelik 5: Forbidden Bigram Genişletme (tahmini etki: +20-40 token)

Çift hal, çift çoğul, kişi sonrası hal gibi temel imkânsızlıklar eklenmeli.

### 🔵 Öncelik 6 (Uzun Vade): Slot-Tabanlı FSM

Forbidden bigram → tam pozitif kısıtlı sonlu durum makinesi. ~30-40 durum, ~100 geçiş kuralı.

### 🔵 Öncelik 7 (Uzun Vade): Sınırlı Backtracking / Beam Search

1-2 seviye geri dönüş veya en iyi 3-5 aday üretip skorlama.

---

## 9. 🐛 BULUNAN HATALAR

| # | Konum | Hata | Önem |
|---|-------|------|------|
| 1 | `harmony.py:82` | `{C}` uyumunda ünlü sonrası `ç` kabul ediliyor | Orta |
| 2 | `phonology.py:13` | `VOWELS`'da `â, î, û` eksik | Yüksek |
| 3 | `analyzer.py:683` | `-yor` daralma + ünsüz kümesi çakışması | Yüksek |

---

## 10. 📊 PROJEKSİYON

```
Mevcut:     %86.9  (8848/10182)
                    │
Circumflex:  +0.6%  │  %87.5  (düşük maliyet, sıfır risk)
Daralma:     +0.4%  │  %87.9  (orta maliyet, düşük risk)
Eksik ekler: +0.7%  │  %88.6  (orta maliyet, orta risk)
Türetim:     +1.2%  │  %89.8  (yüksek maliyet, orta risk)
Bigram genişletme: +0.3% │ %90.1
                    │
Hedef:      ~%90    ╧  (kural-tabanlı tavan)
```

**💻 Hesaplamalı:** "Kural-tabanlı yaklaşımın pratik tavanı %90-92 civarıdır. Bu noktadan sonra makine öğrenmesi hibrit yaklaşımları (neural reranking, bağlam-duyarlı belirsizlik çözümleme) gerekecektir."

---

## 11. SONUÇ

Bu proje, **sözlük-destekli kural-tabanlı bir Türkçe morfolojik çözümleyici** olarak dilbilimsel açıdan sağlam temeller üzerine inşa edilmiştir. Dört katmanlı strateji sistemi, morfofonemik dönüşüm desteği ve forbidden bigram yaklaşımı özgün ve etkili çözümlerdir.

Dört uzman paneli şu konularda **tam uzlaşıya** varmıştır:
1. Mimari tasarım SOLID prensipleriyle büyük ölçüde uyumlu
2. Ses uyumu kuralları dilbilimsel olarak doğru kodlanmış
3. En acil iyileştirme: circumflex eşleme (düşük maliyet, yüksek getiri)
4. En büyük yapısal sorun: backtracking eksikliği ve eksik ek envanteri
5. Uzun vadeli hedef: slot-tabanlı morfotaktik FSM

> *"Kural-tabanlı bir sistem için %86.9 rekabetçi bir sonuçtur. Önerilen düzeltmelerle %90 ulaşılabilir bir hedeftir."* — Hesaplamalı Dilbilim Uzmanı
