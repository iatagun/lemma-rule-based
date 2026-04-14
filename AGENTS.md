# Türkçe Kural-Tabanlı Morfolojik Çözümleyici — Agent Talimatları

> Bu dosya, Copilot agent'ın bu proje üzerinde çalışırken bilmesi gereken
> dilbilimsel kuralları, mimari kararları ve kısıtlamaları içerir.
> Detaylı uzman analizi için bkz. `skill.md`

---

## Proje Kimliği

- **Amaç:** Türkçe sözcükleri kök + eklerine ayıran kural-tabanlı morfolojik çözümleyici
- **Yaklaşım:** Sağdan sola ek sıyırma (right-to-left suffix stripping) + ünlü/ünsüz uyumu + TDK sözlük desteği
- **Benchmark:** BOUN Treebank (UD_Turkish-BOUN) test seti — 10,182 token (PUNCT hariç)
- **Mevcut doğruluk:** %88.6 (9018/10182)
- **Dil:** Python 3.10+, UTF-8, Windows ortamı

---

## Mimari Kılavuz

### Dosya Sorumlulukları (SOLID — Tek Sorumluluk)

| Dosya | Sorumluluk | Değiştirme Kuralı |
|-------|-----------|-------------------|
| `morphology/phonology.py` | Ses sınıflandırma (ünlü/ünsüz kümeleri, turkish_lower) | Yalnızca fonetik değişiklikler |
| `morphology/harmony.py` | Ünlü/ünsüz uyumu denetimi (BÜU, KÜU, ünsüz benzeşmesi) | Yalnızca uyum kuralları |
| `morphology/suffix.py` | Ek tanımları, şablon genişletme ({A},{I},{D},{C}) | Yeni ek eklerken buraya |
| `morphology/dictionary.py` | TDK sözlük, morfofonemik çözümleme (5 adım) | Ses değişim kuralları |
| `morphology/validator.py` | Gövde/kök geçerlilik doğrulama | Fonotaktik kurallar |
| `morphology/analyzer.py` | Ana analiz motoru + BFS + lookup tabloları | Dikkatli — 1500+ satır |
| `morphology/morphotactics.py` | Morfotaktik FSM (16 durum, ek sıra doğrulama) | Yalnızca geçiş tablosu |
| `morphology/sentence.py` | Cümle düzeyinde bağlamsal yeniden sıralama (16 kural) | Kural ekleme |
| `morphology/formatter.py` | Çözümleme çıktı biçimlendirme (tekli + çoklu) | Yalnızca gösterim |
| `morphology/__init__.py` | Fabrika metodu, 4 katmanlı strateji oluşturma | Strateji ekleme/değiştirme |
| `benchmark/evaluate.py` | BOUN Treebank değerlendirme | Metrik/raporlama |

### 4 Katmanlı Strateji Sistemi

```
Katman 1: StrictHarmony + Sözlük    → Tam uyumlu Türkçe sözcükler
Katman 2: RelaxedHarmony + Sözlük   → Alıntı sözcükler (saat, otobüs)
Katman 3: StrictHarmony + Sezgisel  → Sözlükte olmayanlar
Katman 4: RelaxedHarmony + Sezgisel → Son çare (fallback)
```

Bu sıralama kasıtlıdır — precision'dan recall'a doğru kayan gradient. Sırayı değiştirme.

### Sözlük Koruması (Dictionary Protection) — 3 Seviye

1. **Fiil gövdesi atomik koruma:** `stem + mak/mek ∈ sözlük → stem parçalanmaz`
2. **Sözlük sözcüğü koruma:** Sözlükteki sözcük → yalnızca fiil kökü + fiil eki izni
3. **Morfofonemik koruma:** `find_root()` çözümleyebiliyorsa → korunur

Bu mekanizmayı bypass eden değişiklik yaparken dikkatli ol. `neden→ne+den` ve `için→iç+in` gibi tuzaklar bu sayede engelleniyor.

### Çoklu Çözümleme (Belirsizlik Desteği)

Türkçe morfolojisi doğası gereği belirsizdir. Aynı sözcük birden fazla geçerli çözümlemeye sahip olabilir:

- **Kök-ek sınırı belirsizliği:** `gelirin` → gelir+in (İYELİK) / gel+ir+in (GENİŞ_ZAMAN+İYELİK)
- **Sözcüksel belirsizlik:** `yazar` → yaz+ar (fiil) / yazar (isim — yazar)
- **Ek yorumu belirsizliği:** `kuzuları` → kuzu+ları (İYELİK_3Ç) / kuzu+lar+ı (ÇOĞUL+BELİRTME)

Doğru çözümleme bağlamdan (cümle, POS etiketi, sözdizimsel konum) belirlenir.

**API:**
- `analyze(word, upos)` → tek `MorphemeAnalysis` (en güvenilir, geriye uyumlu)
- `analyze_all(word, upos, max_results)` → `list[MorphemeAnalysis]` (tüm adaylar, sıralı)

**BFS Algoritması** (`_strip_suffixes_all`): Tüm geçerli ek sıyırma yollarını keşfeder.
Morfotaktik FSM (16 durum) ile geçersiz ek sıraları budanır.
Kalite filtresi sözlük destekli kökleri tercih eder; gürültüyü eler.

**Biçimlendirme** (`format_multi_analysis`): Birden fazla çözümleme varsa numaralı liste + "bağlama göre belirlenir" uyarısı gösterir.

---

## Türkçe Dilbilgisi Kuralları

### Ünlü Uyumu

- **Büyük Ünlü Uyumu (BÜU):** Kalın(a,ı,o,u) ↔ İnce(e,i,ö,ü) — ekin ilk ünlüsü kökün son ünlüsüyle aynı kümede olmalı
- **Küçük Ünlü Uyumu (KÜU):** Düzlük-yuvarlaklık uyumu — yuvarlak gövde + dar ek → yuvarlak; diğer durumlar → düz
- İki uyum **bağımsız boyutlardır**, `and` ile birleştirilir
- Alıntı sözcüklerde BÜU bozulabilir ama KÜU genellikle korunur

### Şablon Değişkenleri

| Değişken | Karşılıklar | Kural |
|----------|-------------|-------|
| `{A}` | a, e | BÜU (2-yollu) |
| `{I}` | ı, i, u, ü | BÜU + KÜU (4-yollu) |
| `{D}` | d, t | Ötümsüz ünsüzden sonra t, diğer durumlarda d |
| `{C}` | c, ç | Ötümsüz ünsüzden sonra ç, diğer durumlarda c |

### Ek Hiyerarşisi (Slot Modeli)

Türkçe'de ekler kesin bir sıra izler. Bu sıra ASLA ihlal edilmez:

```
İSİM: KÖK → [Yapım] → [Çoğul] → [İyelik] → [Hal] → [-ki]
FİİL: KÖK → [Çatı] → [Olumsuz] → [Yeterlilik] → [Zaman/Kip] → [Kişi] → [Bildirme]
```

**Nominalizasyon sıfırlama:** Sıfat-fiil (-An, -DIk) veya isim-fiil (-mA, -Iş) ekinden sonra isim slotları yeniden başlar:
```
yaşadığını = yaşa + dığ(SIFAT_FİİL) + ı(İYELİK) + nı(HAL)
```

### Morfofonemik Ses Olayları

| Ses Olayı | Kural | Örnek |
|-----------|-------|-------|
| Ünsüz yumuşaması | p→b, ç→c, t→d, k→g/ğ + ünlü eki | kitap → kitab+ı |
| Ünlü düşmesi | 2. hece dar ünlüsü düşer + ünlü eki | burun → burn+u |
| Ünlü daralması | a→ı, e→i / __{-yor} | başla+yor → başlıyor |
| Kaynaştırma y | V-kök + V-ek arası | su+y+u |
| Kaynaştırma n | İşaret zamirleri | o+n+un, bu+n+a |
| Kaynaştırma s | İyelik 3T V-kök | araba+s+ı |

### Düzensiz Fiiller

Türkçe'de yalnızca 2 gerçek düzensiz fiil var:
- **demek:** de-, di-, diy-, ded-, den-, deni-, denil-
- **yemek:** ye-, yi-, yiy-, yed-, yen-

`etmek` ve `gitmek` düzensiz değil ama ünsüz yumuşaması gösterir (et→ed, git→gid).

---

## Bilinen Sorunlar ve Kısıtlamalar

### 🐛 Aktif Hatalar

1. ~~**{C} ünsüz uyumu hatası**~~ → ✅ Düzeltildi (v24). `last_ch not in VOICELESS` koşulu uygulandı.

2. **Şapkalı ünlüler kısmen destekleniyor** (`phonology.py`): `â, î, û` artık VOWELS ve alt kümelere eklendi ✅. Ancak `hal→hâl`, `adet→âdet` eşleme tablosu henüz yok → 65+ BOUN token hâlâ hatalı.

3. ~~**Çok heceli fiillerde ünlü daralması başarısız**~~ → ✅ Düzeltildi (v20). `başlıyor`, `bekliyor`, `söylüyor` çözümleniyor.

### ⚠️ Yapısal Eksiklikler

- ~~**Backtracking yok**~~ → ✅ BFS (genişlik-öncelikli arama) uygulandı, tüm geçerli yollar keşfedilir
- ~~**Slot-tabanlı FSM**~~ → ✅ 16-durumlu morfotaktik FSM entegre (morphotactics.py)
- **Eksik ekler:** -CA (eşitlik), -AmA- (yeterlilik olumsuz), -mAdAn, -DIkçA, -t (kısa ettirgen), -In (dönüşlü çatı)
- **Etiket belirsizliği:** BİLDİRME/ETTİRGEN (-DIr), OLUMSUZ/İSİM_FİİL (-mA) ayrışmamış
- **Circumflex eşleme tablosu yok:** `hal→hâl`, `adet→âdet` dönüşümü eksik (65 BOUN token etkili)
- **Birleşik sözcük ayrıştırma yok:** `cezaevi→cezaev` dönüşümü yapılamıyor

---

## Benchmark Kuralları

### Çalıştırma

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -X utf8 benchmark/evaluate.py
```

### Regresyon Kontrolü

Her değişiklikten sonra benchmark çalıştır. Kabul edilebilir sonuçlar:
- Genel doğruluk **düşmemeli** (≥ %88.6)
- Hiçbir POS kategorisinde **1%'den fazla** gerileme olmamalı
- "Birebir" (word==lemma) doğruluğu ≥ %95 kalmalı

### POS Bazlı Mevcut Durum

```
DET:99.8%  PART:100%  SCONJ:100%  CCONJ:99.7%  — DOKUNMA
PRON:96.0%  ADP:96.2%  AUX:96.2%               — DİKKATLİ OL
ADV:94.8%  ADJ:92.7%  NUM:88.0%                 — İYİLEŞTİRİLEBİLİR
NOUN:87.3%  PROPN:80.2%  VERB:82.5%             — ANA HEDEF
```

### BOUN Lemma Standardı

- **Fiiller:** Yalın gövde — `gel`, `yaz`, `iste` (mastar eki yok!)
- **İsimler:** Yalın kök — `ev`, `kitap`
- **Birleşik isimler:** Tamlama eki düşürülür — `cezaevi → cezaev`
- **Şapkalı biçimler:** Korunur — `hâl`, `âdet`, `kâr`

---

## Kodlama Kuralları

### PowerShell / Python

- Türkçe karakter içeren komutlar: `$env:PYTHONIOENCODING = "utf-8"` + `python -X utf8`
- Heredoc: `$code = @'...'@ | python -X utf8`
- Yollar Windows-style: `C:\Users\...`

### Ek Ekleme Prosedürü

1. `morphology/suffix.py` → `SuffixDefinition` olarak tanımla
2. `harmony_exempt` gerekiyorsa (sabit biçimli ekler: -yor, -ken, -ki) belirt
3. `min_stem_length` ayarla (varsayılan: 2)
4. Benchmark çalıştır, regresyon kontrolü yap
5. Gerekiyorsa `_FORBIDDEN_SUFFIX_BIGRAMS` tablosuna yeni yasaklı çiftler ekle

### Lookup Tablosu Ekleme Prosedürü

1. `analyzer.py` içindeki ilgili tabloya ekle
2. Benchmark çalıştır — eklemenin doğruluğu artırdığını doğrula
3. Hiçbir POS'ta 1%'den fazla gerileme olmadığını kontrol et

### Forbidden Bigram Ekleme Prosedürü

1. Adayı benchmark hata analizinden belirle
2. Doğru çözümlemelerde kaç kez göründüğünü say → **0 olmalı**
3. Hata çözümlemelerinde kaç kez göründüğünü say → **≥2 olmalı**
4. `_FORBIDDEN_SUFFIX_BIGRAMS` frozenset'ine `(iç_ek, dış_ek)` tuple ekle
5. Benchmark çalıştır, net kazancı doğrula

---

## İyileştirme Yol Haritası (Öncelik Sıralı)

| # | İş | Tahmini Etki | Maliyet | Risk |
|---|---|---|---|---|
| 1 | Circumflex eşleme tablosu (hal→hâl) | +%0.6 | Düşük | Sıfır |
| 2 | Eksik ekler (-CA, -AmA-, -mAdAn, -t) | +%0.7 | Orta | Orta |
| 3 | Türetim zinciri genişletme (-t, -n, -lAş) | +%1.2 | Yüksek | Orta |
| 4 | Morfofonemik-farkında BFS (ünsüz yumuşaması ters) | +%0.5 | Orta | Orta |
| 5 | Forbidden bigram genişletme | +%0.3 | Düşük | Düşük |

**Kural-tabanlı yaklaşımın pratik tavanı: ~%90-92**

---

## Referans Belgeler

- `skill.md` — 4 uzman paneli detaylı analiz raporu
- `ARCHITECTURE.md` — Proje mimarisi ve tasarım kararları
- `benchmark/test.conllu` — BOUN Treebank test seti (CoNLL-U formatı)
- `turkish_words.txt` — 48,715 TDK madde başı listesi
