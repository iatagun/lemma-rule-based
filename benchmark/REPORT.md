# BOUN Treebank Benchmark Raporu

## Özet

**Sistem**: Kural-tabanlı Türkçe morfolojik çözümleyici (ünlü uyumu + sözlük)  
**Gold Standard**: UD Turkish BOUN Treebank (test seti)  
**Tarih**: 2025

| Metrik | Değer |
|--------|-------|
| Toplam token (PUNCT hariç) | 10,182 |
| Doğru tahmin | 7,036 |
| **Genel doğruluk** | **%69.1** |
| Birebir eşleşen (word=lemma) | %82.2 (3,834/4,662) |
| Değişen (word≠lemma) | %58.0 (3,202/5,520) |

## POS Bazlı Sonuçlar

```
POS        Toplam    Doğru     Oran
─────────────────────────────────────
PART          162      162   100.0%   ← Edatlar
SCONJ          26       26   100.0%   ← Bağlama edatları
DET           546      537    98.4%   ← Belirleyiciler
CCONJ         337      318    94.4%   ← Bağlaçlar
ADP           264      248    93.9%   ← İlgeçler
NUM           276      229    83.0%   ← Sayılar
ADV           479      362    75.6%   ← Zarflar
PROPN         677      497    73.4%   ← Özel isimler
NOUN        3,952    2,725    69.0%   ← İsimler
ADJ           681      453    66.5%   ← Sıfatlar
PRON          321      212    66.0%   ← Zamirler
INTJ           22       14    63.6%   ← Ünlemler
VERB        2,199    1,238    56.3%   ← Fiiller
AUX           240       15     6.2%   ← Yardımcı fiiller (ek-fiil)
```

## Hata Analizi

### Hata Dağılımı (3,131 toplam hata)

| Kategori | Sayı | Oran | Açıklama |
|----------|------|------|----------|
| **Fazla parçalama** | 1,774 | %56.7 | `neden→ne`, `zaman→zam`, `dünya→dün` |
| **Az parçalama** | 1,066 | %34.0 | `olan→olan` (olması gereken: `ol`) |
| **Ek-fiil** | 225 | %7.2 | `dir→i`, `ise→i`, `tı→y` |
| **Diğer** | 66 | %2.1 | Uzunluk aynı ama farklı kök |

### Temel Sorunlar

#### 1. Fazla Parçalama (Over-decomposition) — %56.7

En büyük sorun. Sözlükte bağımsız sözcük olarak bulunan kelimeler
yanlışlıkla parçalanıyor:

```
neden    → ne + den(AYRILMA)      — "neden" bağımsız sözcük
zaman    → zam + an(SIFAT_FİİL)   — "zaman" bağımsız sözcük
hayır    → hay + ır(GENİŞ_ZAMAN)  — "hayır" bağımsız sözcük
deniz    → den + iz(KİŞİ_1Ç)      — "deniz" bağımsız sözcük
kimse    → kim + se(DİLEK_ŞART)   — "kimse" bağımsız sözcük
```

**Kök neden**: TDK sözlüğü hem uzun sözcüğü (neden) hem de kısa kökü
(ne) içeriyor. Greedy algoritma "ne+den" eşleşmesini buluyor ve
kabul ediyor — neden'in bütünlüğünü koruyamıyor.

**Çözüm yolu**: Sözlükteki sözcüklere "bütünlük skoru" vererek,
parçalama yerine tam sözcüğü tercih eden bir mekanizma gerekiyor.

#### 2. Az Parçalama (Under-decomposition) — %34.0

Sistem bazı ekleri tanıyamıyor veya soyamıyor:

```
olan     → olan    (olması gereken: ol + an)
buna     → buna    (olması gereken: bu + na)
ister    → ister   (olması gereken: iste + r)
olur     → olur    (olması gereken: ol + ur)
```

**Kök neden**: Bazı ek kalıpları eksik veya eşleşme koşulları
(ünlü uyumu, minimum kök uzunluğu) çok katı.

#### 3. Ek-fiil (Copula) — %7.2

Türkçe ek-fiil sistemi (i-mek) tamamen eksik:

```
dir/dır/tir/tır  → lemma: "i"  (ek-fiil şimdiki zaman)
ise              → lemma: "i"  (ek-fiil koşul)
ti/tı/di/dı      → lemma: "y"  (ek-fiil geçmiş zaman)
değil            → lemma: "değil" (olumsuz ek-fiil)
```

**Çözüm yolu**: Ek-fiil paradigmasının ayrı bir modül olarak eklenmesi.

## Uygulanan Düzeltmeler

Bu benchmark sırasında yapılan iyileştirmeler:

| Düzeltme | Etki | Detay |
|----------|------|-------|
| Türkçe İ/I case-folding | +118 token | `İ→i`, `I→ı` (Python varsayılanı yanlış) |
| Apostrof işleme | +74 token | `Türkiye'ye → türkiye` (özel isim koruma) |
| -yor connector genişletme | +8 token | `görm+üyor → gör+müyor` |
| **Toplam iyileştirme** | **+200 token** | **%67.1 → %69.1** |

## Literatür Karşılaştırması

| Sistem | Tür | Yaklaşık Doğruluk |
|--------|-----|-------------------|
| Oflazer (1994) | FST, kapsamlı sözlük | ~%95+ |
| Zemberek-NLP | Kural+sözlük, morfotatik | ~%90+ |
| TRmorph (Çöltekin) | Açık kaynak FST | ~%85-90 |
| **Bizim sistem** | **Kural-tabanlı, greedy** | **%69.1** |

**Not**: Diğer sistemlerin doğruluk oranları farklı test setleri ve
metrikler üzerinde raporlanmıştır. Doğrudan karşılaştırma yapılamaz.

## İyileştirme Yol Haritası

### Yüksek Etki (tahmini +5-10%)
1. **Bütünlük koruması**: Sözlükteki sözcüklerin gereksiz parçalanmasını
   önleyen skor-tabanlı tercih mekanizması
2. **Eksik ek kalıpları**: `-An` (sıfat-fiil), `-ArAk` (zarf-fiil) gibi
   verimli ek kalıplarının eklenmesi/düzeltilmesi

### Orta Etki (tahmini +3-5%)
3. **Ek-fiil modülü**: `dir/dır/ise/ti/tı` gibi ek-fiillerin tanınması
4. **Zamir çekimleri**: `bana→ben`, `sana→sen`, `onun→o` gibi düzensiz
   zamir biçimlerinin eklenmesi

### Düşük Etki (tahmini +1-2%)
5. **Morfotatik kısıtlar**: Hangi ekin hangi ekten sonra gelebileceğini
   belirleyen kurallar (şu an yok)
6. **Frekans tabanlı belirsizlik çözümü**: Birden fazla analiz mümkünse
   en olası olanı seçme

## Teknik Notlar

- **Test seti**: UD_Turkish-BOUN test.conllu (979 cümle, 10,182 token)
- **Sözlük**: turkish_words.txt (48,715 TDK lemma/madde başı)
- **Değerlendirme**: Token bazlı lemma eşleştirme (büyük/küçük harf duyarsız)
- **Hariç tutulan**: PUNCT, SYM, X POS etiketleri
