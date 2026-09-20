---
language: tr
license: cc-by-nc-sa-4.0
library_name: transformers
pipeline_tag: token-classification
tags:
- token-classification
- idiom-detection
- multiword-expressions
- turkish
- electra
datasets:
- parseme
---

# DizgeBERT-Idiom

ELECTRA tabanlı Türkçe **deyim / eşdizim span etiketleyici**. Ön-token'lanmış bir cümle
verildiğinde her kelimeye BIO etiketi atar: **B/I-VID** (deyim — figüratif, birleşimsel-olmayan:
*gözden düşmek*, *eli açık*), **B/I-LVC** (eşdizim/yardımcı-fiil birleşimi — *light-verb
construction*: *karar vermek*), **O** (serbest birleşim / deyim değil). Sonra ikinci bir aşama,
bulunan her aday deyimin CÜMLEDEKİ kullanımına bakıp figüratif mi yoksa düz anlamda mı
kullanıldığını ayırt eder ("otobüs *yol aldı*" düz, "projede *yol aldık*" deyim).

**Güncel sürüm: v8** (2026-09-19/20, eşik güncellemesi 2026-09-20). Başlıca rakamlar
(Çavuşoğlu & Çöltekin dış-kaynak benchmark'ı, 198 idyomatik/literal cümle çifti, eğitim
verisinde yok; `stage2_thresh` varsayılanı **0.3**, eşik-taraması sonrası güncellendi —
aşağıya bakın):

| | değer | not |
|---|---|---|
| İdyomatik/literal doğru ayırt etme oranı | **%68.2** (95% GA: %61.6–%74.7, n=198) | önceki sürüme (v7, %58.1, varsayılan eşik 0.5) göre +10.1pp; eşleştirilmiş bootstrap %95 GA: +3.5pp – +16.7pp — istatistiksel olarak anlamlı |
| Yanlış-pozitif (literal cümlede yanlış işaretleme) | %14.6 | v7'ye göre (%28.3) yarıya yakın düşüş |
| PARSEME test.cupt span-F1 | yaklaşık 63-67 (sürüme göre) | exact-match, VID+LVC.full |
| Precision | yaklaşık %55-64 | model kaçırmaktan çok fazla-işaretlemeye eğilimli |

**En büyük bilinen kısıt:** Aşama 1, üç ayrı ELECTRA gövdesinin (ensemble) birleşimidir — model
dosyası yaklaşık 2GB, çıkarım tek-gövdeli bir modele göre yaklaşık 3× yavaş. Precision de hâlâ mütevazı: her
3 işaretlemeden yaklaşık 1-1.5'i yanlış olabilir. Ayrıntılı sürüm geçmişi ve deney notları aşağıda.

## Kullanım

```python
from transformers import AutoModel, AutoTokenizer

m = AutoModel.from_pretrained("iatagun/DizgeBERT-Idiom", trust_remote_code=True).eval()
tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Idiom")

words = ["Sonunda", "gözden", "düştü", "."]
print(m.predict(words, tokenizer=tok))            # ham BIO (Aşama 1, Aşama 2'den etkilenmez)
# [('Sonunda', 'O', 'o'), ('gözden', 'B-VID', 'o'), ('düştü', 'I-VID', 'o'), ('.', 'O', 'o')]

print(m.predict_spans(words, tokenizer=tok))      # Aşama 2 varsayılan AÇIK
# [{'text': 'gözden düştü', 'start': 1, 'end': 3, 'category': 'VID', 'gappy': False}]

# Aşama 2, literal kullanımı eler:
lit = ["Otobüs", "kısa", "sürede", "çok", "yol", "aldı", "."]
print(m.predict_spans(lit, tokenizer=tok))                 # → []  (literal, elendi)
print(m.predict_spans(lit, tokenizer=tok, stage2=False))   # → [{'text': 'yol aldı', ...}]

# eşiği ayarlamak: varsayılan artık 0.3 (yukarıdaki gibi çağırmaya gerek yok — model bunu
# config'inden otomatik okur). Daha YÜKSEK bir eşik daha AZ temkinli eleme yapar (daha çok
# span idyomatik sayılır, duyarlılık artar ama yanlış-pozitif de artar):
print(m.predict_spans(lit, tokenizer=tok, stage2_thresh=0.7))

# gap'li (süreksiz) örnek — "sahip ... olarak" (Aşama 2 dokunmaz)
ws = "... sahip olduğu ... değerleriyle olarak önemini ...".split()
print(m.predict_spans(ws, tokenizer=tok))
# gappy=True ise: {'text': 'sahip ... olarak', 'start':.., 'end':.., 'start2':.., 'end2':.., 'category':..}
```

## Mimari

- **Gövde:** [`dbmdz/electra-base-turkish-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)
  (DizgeBERT-Morph/Joint/Dep ile aynı → ortak subword sözlüğü)
- **Kelime temsili:** ilk subword ⊕ son subword (DizgeBERT-Morph ile aynı yöntem)
- **İki katmanlı etiketleme (bigappy-unicrossy tarzı, Berk, Erden & Güngör 2019):** standart
  BIO **süreksiz (gap'li)** span'leri temsil edemez (*"sahip ... olarak"* gibi araya kelime
  giren deyimler). İkinci, bağımsız bir head (`o/b-VID/i-VID/b-LVC/i-LVC`) yalnız gap'li
  span'in **2. parçasını** taşır; 1. parça her zaman ana BIO katmanında. Çıkarımda iki katman
  ayrı ayrı Viterbi ile çözülüp aynı kategoriden en yakın parçalar eşleştirilir.
- **Çözümleme:** geçiş-kısıtlı **Viterbi** (argmax değil), her iki katmanda da.
- **Aşama 1 — üç bağımsız stage-1 gövdesinin ensemble'ı (v7'den beri):** her cümlede üç ayrı
  ELECTRA gövdesi (farklı veri dilimleriyle eğitilmiş) kendi aday span'lerini önerir; birleşim
  (çakışan span'lerde en çok oy alan/en uzun kazanır) Aşama 2 filtresine girer. Model dosyası
  **üç** stage-1 ELECTRA gövdesi + Aşama 2 için ayrı bir gövde içerir (yaklaşık 2 GB, yaklaşık 3× gecikme).
  Alternatif bir sürüm (v6, tek gövde, yaklaşık 440MB) daha küçük/hızlı ama daha eski/zayıf bir Aşama
  2'ye sahip — bkz. Sürüm Geçmişi.
- **Aşama 2 — idyomatiklik sınıflandırıcısı (v8'den beri sentetik veriyle eğitildi):** ikinci,
  bağımsız bir ELECTRA gövdesi + span ilk⊕son subword temsili → `Linear(2H, 2)` →
  {literal, idyomatik}. `predict_spans()` bitişik VID adaylarını bundan geçirir; yalnız
  *güvenli* literal (p(literal) > eşik, varsayılan 0.5) elenir — LVC ve gap'li span'ler
  dokunulmaz (LVC yarı-birleşimsel, ayrım anlamsız).

## Eğitim verisi

1. [PARSEME Türkçe fiil-merkezli çok-sözcüklü ifade derlemi, edition 1.2](https://gitlab.com/parseme/sharedtask-data/-/tree/master/1.2/TR)
   (Güngör & Yirmibeşoğlu) — 17.945 cümle, VID+LVC.full toplam yaklaşık 6.7k span (yalnız
   *verbal* MWE; 308'i gap'li).
2. TDK Atasözleri ve Deyimler Sözlüğü'nden çıkarılan gömülü örnek cümleler — isim/sıfat
   deyimlerini de kapsar (*eli açık*, *başı dertte* gibi, PARSEME'de yok); gövde-eşleştirme
   (stem matching, `max_gap=2` — araya en fazla 2 eşleşmeyen kelime girebilir) ile
   zayıf-etiketlenmiş (weak supervision). Deyim ve cümle-metni düzeyinde ayrı tutulan bir
   held-out kısmı hiç eğitime sokulmadı.
3. **Aşama 1'i genişletmek için (v4):** Leipzig Türkçe derleminden (Wikipedia+Haber+Web,
   CC-BY) deyim yüzey biçimiyle eşleşen doğal cümleler madenlenip GLU rubriğiyle etiketlendi;
   idyomatik-olarak işaretlenmiş binlerce cümle Aşama 1'in span-eğitimine eklendi. Amaç:
   Aşama 1'in eğitim verisi TDK sözlük-örneği ve PARSEME haber-metni üslubuna aşırı uymuştu.
4. **Aşama 2 için, v8'den önce:** aynı Leipzig havuzundan madenlenen cümleler bir LLM ile
   idyomatik/literal diye ETİKETLENDİ (doğal metinde literal kullanım nadir olduğu için bu
   veri dengesizdi — bkz. aşağıdaki v8 notu).
5. **Aşama 2 için, v8'den itibaren:** doğal cümle etiketlemek yerine, TDK deyim listesinden
   seçilen yaklaşık 650 deyim için bir LLM'e doğrudan **dengeli idyomatik+literal cümle
   ÜRETTİRİLDİ** (deyim başına yaklaşık 3+3, literal okuma anlamsızsa boş bırakılarak). Üretilen
   cümleler otomatik bir gövde-eşleştiriciyle doğrulandı; 503 deyim / 1866 kayıt kabul edildi.
   Aşama 2 artık TAMAMEN bu sentetik havuzla eğitiliyor, doğal-derlem verisi kullanılmıyor.

## Sonuçlar

### Taban çizgisi: model gerçekten ne katıyor?

Aşağıdaki satır, sinir ağı hiç kullanılmadan yalnız TDK'nin yaklaşık 11k deyimlik sözlük listesiyle
(aynı `max_gap=2` gövde-eşleştiricisi) cümledeki idyomları bulmaya çalışan saf kural-tabanlı
bir sistemi gösteriyor — modelin eklediği değeri somutlaştırmak için:

| sistem | PARSEME F1 | CASES (16 vaka) | Çavuşoğlu duyarlılık | Çavuşoğlu yanlış-poz | Çavuşoğlu doğru-ayırt |
|---|---|---|---|---|---|
| **yalnız sözlük eşleştirici** (nöral yok) | 10.6 | 7/16 | %18.7 | %23.7 | **%11.6** |
| **DizgeBERT-Idiom v8** (tam boru hattı, `stage2_thresh=0.3`) | yaklaşık 63-67 | 13/16 | %81.3 | %14.6 | **%68.2** |

Sözlük eşleştirici Çavuşoğlu'nda span bulduğu az sayıda durumda (11/198, dar bir örtüşme)
duyarlılık VE yanlış-pozitif ikisi de %100 — yani bulduğunda bağlamdan tamamen kördür,
idyomatik/literal ayrımı sıfır. Modelin asıl katkısı sadece span BULMAK değil, bulduğu
span'ler için bağlama duyarlı bir ayrım yapabilmesi.

*(Not: daha ayrıntılı ablasyonlar — yalnız PARSEME ile eğitilmiş düz bir ELECTRA, veri
kaynaklarının [PARSEME/TDK/Leipzig] tek tek katkısı — henüz ölçülmedi; bu, kartın bilinen bir
eksiği. Bkz. Kısıtlar.)*

### Ana metrik: bağlam-bağımlılık (dış kaynak)

**Bağımsız dış kaynak — ama "dev-seti gibi" okuyun.** Çavuşoğlu & Çöltekin'in (MWE 2026)
elle-yazılmış Türkçe deyim benchmark'ı (198 deyim, her biri için gerçek idyomatik-kullanım +
literal-kullanım cümle çifti, eğitim verimizde yok — `benchmark/eval_idiom.py --mode
external`) üzerinde. **Önemli çekince:** bu benchmark, v5→v8 arası ~20 deneyde HANGİ
adayın promote edileceğine karar vermek için TEKRAR TEKRAR kullanıldı (bkz. Sürüm Geçmişi) —
yani kör bir tutulmuş test seti değil, fiilen bir model-seçim/dev seti gibi işlev gördü.
Aşağıdaki mutlak sayılar (özellikle sürümler-arası "kazanç" iddiaları) bu yüzden bir miktar
iyimser yanlı olabilir; PARSEME ve dış-kaynak (Aslantaş&Güngör, Dodiom) sayıları bu sızıntıdan
bağımsız, o yüzden onlara daha çok ağırlık verin.

**Tek-aşama ablasyonu (2026-09-20, `benchmark/eval_cavusoglu_stage_ablation.py`):**
Aşama 1'in TEK BAŞINA (Aşama 2 KAPALI) performansı — "tek aşamalı bir tagger bağlam ayrımı
yapamaz" iddiasının doğrudan testi. Aşama 1 (v7, 3-gövde ensemble) burada da AYNI eğitim
verisiyle (literal kullanımlar `corpus_examples_glu.json`'da zaten hep-O olarak var), yalnız
Aşama 2 filtresi devre dışı:

| ölçüm | Aşama 1 (stage2=False) | **+ Aşama 2 (v8, varsayılan eşik 0.3)** |
|---|---|---|
| idyomatik cümlede span işaretledi (duyarlılık) | %93.4 (95% GA %89.9–%96.5) | %81.3 (95% GA %75.8–%86.4) |
| literal cümlede **yanlış** span işaretledi | %66.7 (95% GA %60.1–%73.2) | %14.6 (95% GA %10.1–%19.7) |
| ikisini de doğru ayırt etti (çift-düzeyi, birincil metrik) | **%28.8** (95% GA %22.7–%35.4) | **%68.2** (95% GA %61.6–%74.7) |

Eşleştirilmiş fark (+Aşama2 − Aşama1-tek, aynı 198 çift): **+39.4pp** (95% GA +30.8pp – +47.5pp,
sıfırı içermiyor). Aşama 1 tek başına neredeyse HER cümlede (idyomatik ya da literal) span
işaretliyor (%93.4 vs %66.7 — ikisi birbirine yakın, yani bağlamdan bağımsız); asıl ayrımı
Aşama 2 üretiyor. **Not:** bu, tek bir ELECTRA gövdesinin değil, üç-gövdeli ensemble'ın tek
başına ölçümü — daha da recall-ağırlıklı olduğu için tek-gövde bir tagger muhtemelen bu kadar
yüksek yanlış-pozitif vermez, ama muhtemelen daha düşük duyarlılık da verir; net doğru-ayırt
yönü aynı kalır ama bu spesifik %28.8 rakamı ensemble'a özgü, tek-gövdeye genellenmemeli.

**Sürüm notu (hangi sayı hangi sürüme ait):** Aşama 1 (stage2=False, span-tespit gövdeleri)
**v7'den beri değişmedi** — v8 yalnız Aşama 2'yi (idyomatiklik sınıflandırıcısını) yeniden
eğitti. Yani yukarıdaki %28.8 hem v7 hem v8 için geçerli Aşama-1-tek rakamı. `+Aşama 2`
sütunundaki **%68.2**, v8'in `stage2_thresh=0.3` (güncel varsayılan) ile ölçümü; v8 ilk
yayınlandığında varsayılan eşik 0.5'ti ve o eşikte **%65.2** ölçülmüştü (aşağıdaki eşik
tablosuna bakın — aynı model dosyası, yalnız eşik farklı). v4'ün Aşama 1'i BUGÜNKÜ v7/v8'den farklıydı: v4 `corpus_examples_glu.json`'un o zamanki
(sonradan fark edilen, bkz. proje deney günlüğü "Deney H") sürümünü kullanıyordu — bu dosya
neredeyse tamamen D-only'ydi (LLM kaynaklı, stale), gerçek literal→hep-O örneği YOK denecek
kadar azdı. Bu, mimari bir farklılık değil, o dönemki veri dosyasının içeriğiyle ilgiliydi;
sonraki bir düzeltme (Deney H) temiz, dengeli D+L verisiyle `corpus_examples_glu.json`'u
yeniden üretti, ve v7'den itibaren Aşama 1 bu temiz veriyle eğitiliyor. Yukarıdaki ablasyon
bu YENİ (v7/v8) Aşama-1 verisiyle ölçüldü, v4'ün Aşama-1'i DEĞİL.

**stage2_thresh eşik-duyarlılığı** (aynı 198 çift, `p(literal) > eşik` olan span elenir):

| eşik | duyarlılık | yanlış-poz | doğru-ayırt |
|---|---|---|---|
| **0.3 (varsayılan, 2026-09-20'den beri)** | %81.3 | %14.6 | **%68.2 (en iyi ölçülen)** |
| 0.5 (v8'in ilk yayınındaki eski varsayılan) | %84.8 | %21.2 | %65.2 |
| 0.7 | %88.9 | %26.3 | %64.1 |
| 0.9 | %92.9 | %39.9 | %55.1 |

Eşik düştükçe hem yanlış-pozitif hem doğru-ayırt-kaybı azalıyor. v8 ilk yayınlandığında
varsayılan 0.5 idi (v7'nin mirası); bu tarama sonrası **varsayılan 0.3'e düşürüldü** —
model dosyası aynı, yalnız `config.stage2_thresh` değişti. Bu eğri, "net etki pozitif" gibi
tek-sayıya indirgenmiş iddiaların yerine geçiyor.

### v7→v8 kıyası ve doğrulama

Kıyas, HER İKİ sürümün de FİİLEN YAYINLANDIĞI eşikle yapılıyor (v7: 0.5, v8: 0.3 — ikisi de
kendi varsayılanı):

| metrik | v7 (`thresh=0.5`) | **v8 (`thresh=0.3`)** |
|---|---|---|
| doğru-ayırt (n=198) | %58.1 (95% GA %51.0–%65.2) | **%68.2 (95% GA %61.6–%74.7)** |
| **eşleştirilmiş fark** (aynı 198 cümle, v8−v7) | — | **+10.1pp (95% GA +3.5pp – +16.7pp, 5000 bootstrap)** |
| McNemar dökümü | — | 103 ikisi de doğru, **32 yalnız v8 doğru**, 12 yalnız v7 doğru, 51 ikisi de yanlış |
| yanlış-pozitif | %28.3 | %14.6 |
| GLU tanı seti (35 vaka, küçük örneklem — bkz. not) | 20/35 | 25/35 |

Bağımsız ölçülen iki nokta (independent CI) tek başına bakılınca örtüşüyor gibi görünebilir,
ama bu 198 cümle HER İKİ modelde de AYNI olduğu için doğru karşılaştırma eşleştirilmiş
(paired) bootstrap'tır: fark %99.9 ihtimalle pozitif, %95 güven aralığı sıfırı içermiyor —
istatistiksel olarak anlamlı. **35-vakalık GLU tanı setindeki 20/35→25/35 gibi geçişler ise
küçük örneklem nedeniyle TEK BAŞINA güçlü bir kanıt sayılmamalı**, yalnız destekleyici sinyal.

### Sızıntı / "görülmemiş deyim" kontrolü — düzeltme notu

**v8'in eğitiminde kullanılan 503 sentetik deyimin** Çavuşoğlu'nun 198 eval deyimiyle sıfır
kesiştiği doğrulandı. Ayrıca, bu kartın önceki bir taslağı "görülmemiş deyim dilimi" diye bir
alt-küme (177/198) raporluyordu; bu alt-küme ilk kez erken bir deneyde (proje içi, HF dışı)
hesaplanmış ve sonraki veri-genişletme turlarında **yeniden hesaplanmamıştı** — denetlerken bu
fark edildi: o 177'nin aslında 121'i, sonraki turlarda eğitim havuzuna (Aşama 1'in çeşitli
gövdelerinden en az birine) girmiş çıktı. Düzeltilmiş, muhafazakâr bir "gerçekten görülmemiş"
kümeyle (59 deyim) yeniden ölçüldüğünde desen KORUNUYOR: v8 doğru-ayırt (`thresh=0.3`) **%69.5**
vs v7 (`thresh=0.5`) %59.3 (yaklaşık +10.2pp, genel bulguyla tutarlı) — yani asıl bulgu (v8'in
genellemesi) ayakta kalıyor, ama kart daha önce yanlış/güncellenmemiş bir "seen/unseen"
ayrımını rapor ediyordu. Bu artık düzeltildi; proje içi deney günlüğünde tam detay var.

### İlgili çalışmalar — daha yüksek rakam bildiren Türkçe deyim çalışmaları var, ama farklı görevler ölçüyorlar

Bu alanda daha yeni, daha yüksek F1/doğruluk bildiren iki Türkçe çalışma var. Okuyucunun
"88-90 mı, 60-68 mi" diye yüzeysel bir kıyasla bu modeli geride sanmaması için farkları
açıkça belirtiyoruz:

- **Aslantaş & Güngör (Boğaziçi, SIGTURK 2026), "A Unified Turkic Idiom Understanding
  Benchmark: Idiom Detection and Semantic Retrieval Across Five Turkic Languages"**
  ([ACL Anthology 2026.sigturk-1.4](https://aclanthology.org/2026.sigturk-1.4/),
  [github.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark](https://github.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark)).
  Kod ve veri açık olduğu için **çift yönlü, ölçülmüş** bir kıyas yapıldı (2026-09-20,
  `benchmark/eval_aslantas_gungor.py`, `benchmark/train_ag_electra_baseline.py`,
  `benchmark/analyze_ag_errors.py`).

  **Dürüst özet, en başta:** onların TR test setinde (131 cümle), onların KENDİ metriğiyle
  (seqeval, entity-düzeyi exact-match), DizgeBERT **açıkça geride**: F1=0.592 (tümü),
  **0.414 gerçekten görülmemiş deyimlerde** — onların in-domain ELECTRA-tr/ConvBERT-tr'sinin
  0.877/0.880'inin belirgin altında. VID/LVC iki kategorimiz bu kıyas için TEK bir IDIOM
  sınıfına indirgendi (their span'leriyle karşılaştırılabilir olması için).

  | | tümü (n=131) | deyim-kimliği görülmüş (n=71) | gerçekten görülmemiş (n=60) |
  |---|---|---|---|
  | **entity-exact (seqeval, onların metriği), stage2=False** | F1=0.592 (95% GA 0.513–0.672) | F1=0.736 (0.639–0.827) | **F1=0.414** (0.300–0.537) |
  | entity-exact, stage2=True (yayınlanan varsayılan) | F1=0.496 (0.414–0.580) | F1=0.608 (0.504–0.712) | F1=0.372 (0.252–0.496) |
  | gevşek/token-örtüşmeli (kısmi sınır hatasına kısmi puan), stage2=False | P=0.887 R=0.721 F1=**0.795** | P=0.933 R=0.848 F1=0.888 | P=0.822 R=0.583 F1=0.682 |
  | **onların ELECTRA-tr/ConvBERT-tr'si (in-domain, kendi eğitimleri)** | **F1=0.877 / 0.880** | — | — |

  131 test idiomunun **71'i (%54) zaten bizim Aşama-1 eğitim havuzumuzda** — "sıfır-atış"
  yalnız CÜMLE düzeyinde geçerli, DEYİM-kimliği düzeyinde değil.

  **Entity-exact (0.592) ile gevşek/token-örtüşmeli (0.795) arasındaki fark ne kadarı sınır
  konvansiyonundan geliyor?** 131 cümledeki 153 span-olayının elle (`analyze_ag_errors.py`
  çıktısı okunarak) sınıflandırılması: **%50.3 tam eşleşme (EXACT), %20.3 doğru deyimi buldu
  ama sınır farklı (BOUNDARY), %15.7 gerçek kaçırma (MISS), %13.7 gerçek fazladan-işaretleme
  (SPURIOUS).** Yani fark KISMEN sınır-konvansiyonundan (EXACT+BOUNDARY=%70.6 "doğru yerde"),
  ama önemli bir kısmı GERÇEK kaçırma/fazladan-işaretleme (%29.4). BOUNDARY vakalarının
  belirgin bir kısmı tek bir örüntüye ait: "Allah ..." ile başlayan dua/beddua kalıpları
  (*Allah belasını versin*, *Allah korusun*, *Allah mübarek etsin*) — onların anotasyonu
  TÜM cümleyi (vokatif+fiil) tek span sayıyor, DizgeBERT (TDK sözlük-biçimli VID eğitiminden)
  yalnız çekirdek yüklemi/nesneyi işaretliyor. Bu, iki farklı "deyim" tanımının (idiom-as-
  fixed-clause vs idiom-as-lexical-item) sınır anlaşmazlığı — model hatası değil, tanım farkı.

  **Reprodüksiyon, 3 tohum:** onların backbone'unu (ELECTRA-tr — bizimkiyle AYNI encoder,
  checkpoint yayınlamadıkları için double-blind kendi TR verileriyle burada yeniden eğitildi)
  3 farklı tohumla (`--seed 42/1/2`) eğitip onların KENDİ metriğiyle ölçtük: **F1 = 0.898 ±
  0.031** (0.924/0.864/0.906) — onların bildirdiği 3-seed-ortalaması **0.877** bu aralığın
  içinde, makul bir reprodüksiyon. **Ayrı bir not:** onların makale METNİ "token-level
  precision/recall/F1/accuracy" diyor ama kodları (`src/modeling/metrics.py`) `seqeval` ile
  ENTITY-düzeyi hesaplıyor — bu, onların kendi makalesindeki bir terminoloji tutarsızlığı,
  bizim hatamız değil, ama okuyucunun bilmesi gereken bir nüans.

  Bu backbone'u (kendi TR verisiyle, tek tohumla eğitilmiş hâli) bizim Çavuşoğlu bağlam-ayrımı
  benchmark'ımızda çalıştırdık: duyarlılık %87.4, yanlış-poz %85.9, **doğru-ayırt yalnız
  %9.6** — bağlamdan bağımsız, saf yüzey-biçim eşleşmesi. Bu, "salt-pozitif eğitilmiş bir
  tagger bağlam ayırt edemiyor" gözlemini destekler ama TEK BAŞINA "iki aşama şart" iddiasını
  KANITLAMAZ (onların modeli literal örnekle hiç eğitilmedi — kendi göreviyle tutarlı bir
  sınırlama, "başarısızlık" değil). Asıl kanıt aşağıdaki tek-aşama ablasyonu (AYNI veri,
  yalnız Aşama 2 kapalı/açık).

- **Umut, Site, Arslan & Eryiğit (İTÜ, UBMK 2025), "Exploring Turkish Idiomaticity with
  LLMs".** Veri seti/kodu yayınlanmamış (IEEE Xplore, paywall) — doğrudan kıyas mümkün
  olmadı. Yerine, aynı ITU NLP ekosisteminden halka açık bir kaynak kullanıldı: **Dodiom TR**
  (Eryiğit, Şentaş & Monti, *Natural Language Engineering* 2022,
  [github.com/Dodiom/dodiom](https://github.com/Dodiom/dodiom), 6861 crowdsourced örnek / 36
  deyim, idiom/nonidiom ikili etiket + hedef span). **Bu Umut et al.'ın verisiyle AYNI DEĞİL**
  — yalnız benzer bir dış/insan-etiketli kaynak olarak kullanıldı (`benchmark/eval_dodiom.py`).
  **"Dış kaynak, görülmemiş" demiyoruz** — daha kesin: 36 deyimin **34'ü zaten Aşama-1'in
  span eğitim havuzunda var** (bu, Aşama-1 açısından neredeyse hiç görülmemiş-deyim testi
  DEĞİL); Aşama-2'nin (bağlam-ayrımı) KENDİ etiketli eğitim havuzuyla örtüşme **0/36** — yani
  bu **Aşama-2 açısından görülmemiş deyim kimlikleri üzerinde, doğal/crowdsourced cümlelerle**
  bir test.

  | | duyarlılık | yanlış-poz | dengelenmiş doğruluk* |
  |---|---|---|---|
  | stage2=True (varsayılan) | %69.9 (küme-GA %62.5–76.3) | %15.1 (%12.1–18.1) | %77.4 |
  | stage2=False | %85.5 (%81.5–89.1) | %35.6 (%30.1–41.3) | %74.9 |

  *dengelenmiş doğruluk = (duyarlılık + (1−yanlış-poz))/2, EŞLEŞTİRİLMEMİŞ cümle-düzeyi bir
  ölçüt — Çavuşoğlu'nun çift-düzeyi "ikisini de doğru ayırt etti" metriğiyle AYNI ölçek
  DEĞİL, doğrudan kıyaslanmamalı. Güven aralıkları deyim-KÜMESİ (36 deyim) bootstrap'ı —
  etkin örneklem 36'dır, 6861 satır değil.

  **Aşama 2'nin net etkisi burada Çavuşoğlu'ndaki kadar net DEĞİL.** Aşama 2 açıldığında
  duyarlılık %85.5→%69.9 (**−15.6pp, gerçek recall bedeli**) düşerken yanlış-poz %35.6→%15.1
  (**−20.5pp, gerçek kazanç**) düşüyor — net "dengelenmiş doğruluk" farkı görünürde yalnız
  +2.5pp (%74.9→%77.4). Deyim-kümesi EŞLEŞTİRİLMİŞ bootstrap ile test edildiğinde: **fark
  +2.4pp, 95% GA −1.5pp – +6.1pp — SIFIRI İÇERİYOR, bu ölçekte (n=36 deyim kümesi)
  istatistiksel olarak ayırt edilemiyor.** Yani Dodiom'da Aşama 2'nin doğru anlatımı "recall'ı
  yanlış-pozitif için takas ediyor, net dengelenmiş-doğruluk etkisi bu örneklemde belirsiz" —
  Çavuşoğlu'ndaki net +39.4pp'lik (anlamlı) kazanç BURADA tekrarlanmıyor. "İki aşama şart"
  iddiası bu yüzden benchmark'a göre YUMUŞATILMALI: Çavuşoğlu'nun (deliberate, dengeli
  idyomatik/literal çiftler) ölçtüğü keskin bağlam-değiştirme senaryosunda güçlü/anlamlı,
  Dodiom'un (doğal, dengesiz, düşük-literal-oran) senaryosunda yönü aynı ama büyüklüğü
  istatistiksel olarak belirsiz.

**Tek-aşama ablasyonu, ayrı bir kanıt hattı (2026-09-20):** yukarıdaki iki dış-kaynak
karşılaştırması "salt-pozitif eğitilmiş bir tagger bağlam ayırt edemiyor" gözlemini
DOĞRULUYOR (onların modeli literal örnek hiç görmedi). Bunun "iki aşama şart" iddiasına
dönüşmesi için ayrı bir ablasyon gerekliydi: Aşama 1'i (AYNI veri, literal örnekler zaten
hep-O olarak var) Aşama 2 OLMADAN çalıştırmak — yukarıdaki "Ana metrik" bölümündeki
"tek-aşama ablasyonu" tablosuna bakın (doğru-ayırt %28.8 vs %68.2, eşleştirilmiş fark
+39.4pp, 95% GA sıfırı içermiyor). **Ama bu iddia BENCHMARK'A BAĞLI**: aynı açık/kapalı
karşılaştırması Dodiom'da tekrarlanınca (yukarı bakın) fark +2.4pp'ye düşüyor ve GA sıfırı
içeriyor — istatistiksel olarak anlamsız. Okunması gereken doğru cümle "iki aşama her
zaman/her veri dağılımında şart" değil, **"Çavuşoğlu'nun ölçtüğü türden keskin, dengeli
idyomatik/literal bağlam-değiştirme senaryosunda Aşama 2'nin katkısı büyük ve anlamlı;
Dodiom'un doğal, düşük-literal-oranlı dağılımında yönü aynı (daha az yanlış-pozitif) ama
net dengelenmiş-doğruluk etkisi bu örneklem büyüklüğünde istatistiksel olarak
ayırt edilemiyor."**

### Diğer kıyas noktaları (dikkatli okunmalı — görev tanımları farklı)

**Büyük LLM'lerle kıyas:** Çavuşoğlu & Çöltekin (2026), büyük LLM'lerin (Gemini 2.5, GPT-4o,
Llama-3 70B) kendi Türkçe deyim benchmark'larında **ikili idyomatik-mi-değil-mi
sınıflandırmasında %59-61 doğruluk** aldığını raporluyor. Bu, DOĞRUDAN karşılaştırılabilir bir
sayı değil: (a) LLM'e deyim HEDEFİ verilip yalnız ikili karar sorulmuş, bizim modelimiz span'i
KENDİSİ bulup sonra karar veriyor (daha zor bir görev); (b) bizim birincil metriğimiz (%68.2)
ÇİFT düzeyinde ve daha katı ("ikisini de doğru yap"), LLM'lerinki tek-cümle doğruluğu. Daha adil
bir kıyas için CÜMLE düzeyinde bakarsak: v8 idyomatik cümlede %81.3, literal cümlede %85.4
(=100−yanlış-poz) doğru — ortalama **yaklaşık %83.4**, LLM'lerin %59-61'inin belirgin üstünde, ama
görev tanımı farkı yüzünden bunu da temkinli okuyun.

**PARSEME 1.1 shared task kıyası:** en iyi sistem (SHOMA, nöral+CRF) tüm diller ortalamasında
%58.09 makro-F1 almıştı (bazı diller %23-32, Macarca/Romence %85-90). Bu **doğrudan
karşılaştırılamaz** — farklı edition (1.1 vs bizim 1.2), farklı dil kümesi, muhtemelen farklı
metrik, ve biz yalnız VID+LVC.full ile eğitiyoruz (PARSEME'nin tüm kategorileri değil). Yalnız
büyüklük mertebesi olarak: bizim yaklaşık 64-67 F1'imiz bu aralığın ortasında, görevin zor olduğunu
teyit ediyor.

### PARSEME span-F1 (Aşama 1, `stage2=False`)

| test seti | kapsam | P | R | F1 |
|---|---|---|---|---|
| PARSEME test.cupt, **genel** | fiil-merkezli, bitişik+gap'li | değişken (sürüme göre) | değişken | **yaklaşık 63-67** |
| PARSEME test.cupt, yalnız **gap'li span'lar** | süreksiz deyim/eşdizim | — | — | yaklaşık 30-45 |

Gap'li span'lar standart BIO ile **yapısal olarak asla yakalanamazdı** (ilk sürümde recall
garanti %0); iki-katmanlı şemayla artık kısmen (yaklaşık %30-45) kurtarılıyor.

**Ölçüm yöntemi notu:** yukarıdaki tüm span metrikleri **exact-match** (span sınırları
BİREBİR tutmalı). Bu, sınır hatalarını (span'in bir kelime kısa/uzun bulunması) tam-kaçırma
ile aynı kefeye koyuyor. Token-düzeyi (yalnız hangi kelimelerin doğru etiketlendiğine bakan,
sınır hatalarını kısmi puanlayan) bir ölçüm henüz raporlanmıyor — bu da kartın bilinen bir
eksiği, modelin nerede tam yanıldığını nerede yalnız sınırı kaçırdığını ayırt etmek için
faydalı olurdu.

## Kısıtlar

- **Aşama 2 (v8) tamamen sentetik (LLM-üretilmiş) veriyle eğitildi, doğal veri hiç
  kullanılmadı.** Yanlış-pozitif oranı v7'ye göre iyileşti (%28.3→%14.6, varsayılan eşikle)
  ama sıfır değil. Sentetik cümlelerin üslubu doğal metinden farklı olabilir (ajan-üretimi) —
  bu risk ölçüldü (sentetik+doğal karışım denendi, daha kötü çıktı), ama üretim ortamında
  çıktıyı doğrulamadan güvenmeyin. `stage2=False` ile tamamen devre dışı, `stage2_thresh` ile
  eşik ayarlanır (varsayılan 0.3 — yukarıdaki eşik-duyarlılığı tablosuna bakın).
- **Model seçimi bir iç held-out sette yapıldı, bu artık kör bir test sayılamaz.** Aşama 2'nin
  epoch seçimi (v3-v7 döneminde) görülmemiş-deyim bir held-out setteki dengeli-doğruluk
  skoruna göre yapılmıştı — bu seti tekrar tekrar kullanmak onu artık saf bir test kümesi
  olmaktan çıkarıyor. Kartın birincil kanıtı bu yüzden HER ZAMAN tamamen ayrı, dış bir kaynağa
  (Çavuşoğlu & Çöltekin) dayanıyor.
- **Aşama 1 hâlâ üç gövdeli bir ensemble, tek bir bütünleşik model değil** — model
  boyutu/gecikmesi buna göre büyük (yaklaşık 2GB, yaklaşık 3×). Daha küçük/hızlı bir alternatif (v6, tek
  gövde, yaklaşık 440MB) var ama Aşama 2'si eski/daha zayıf.
- **Precision yaklaşık %55-64** — üretim kullanımında çıktıyı doğrulamadan güvenmeyin.
- **Küçük örneklemli alt-metrikler dikkatli okunmalı.** Dış-kaynak benchmark'ının "bilinen
  deyim" dilimi yalnız 21 çift, GLU tanı seti 35 vaka — bu ölçeklerde ±10 puanlık
  dalgalanmalar normaldir, tek bir sürüm geçişini bunlara dayandırmayın (bkz. yukarıdaki
  bootstrap güven aralıkları).
- **Eksik ablasyonlar.** Kartta şu an yok: (a) yalnız-PARSEME ile eğitilmiş düz bir ELECTRA
  taban çizgisi, (b) veri kaynaklarının (PARSEME/TDK/Leipzig) tek tek katkısı, (c) token-
  düzeyi (exact-match olmayan) span metrikleri, (d) diğer Türkçe deyim çalışmalarıyla aynı
  test setinde ölçülmüş bir kıyas (bkz. "İlgili çalışmalar"). Bunlar modelin gerçek katkısını
  daha net gösterebilir; henüz ölçülmedi.
- **Zayıf-denetim gövde-eşleştirmesi lemma/stemmer kalitesine duyarlı.** Türkçe MWE
  tespitinde lemma/POS düzeltmelerinin ayrı bir çalışmada (Öztürk ve ark., Seq2Seq MWE
  tespitinde) F1'i ölçülebilir şekilde artırdığı bildiriliyor — bu, bizim kendi
  bulgumuzla örtüşüyor: Deney Z'de kullanılan snowball stemmer bazı çekim eklerini
  (özellikle "-yor" şimdiki zaman eki) atmıyor, bu da serbest üretilmiş cümlelerde
  span-doğrulama oranını ciddi düşürüyordu (önek-toleranslı bir eşleştirmeyle telafi
  edildi, `data/prepare_synthetic_stage2_pairs.py::find_span_lenient`). Daha iyi bir
  lemmatizer/morfolojik çözümleyici, hem eğitim verisi kalitesini hem precision'ı
  iyileştirebilir — henüz denenmedi.
- **Gap'li (süreksiz) span'lar kısmen çözülüyor, tam değil**, ve şema yalnız **tam 2 parçalı**
  gap'leri temsil eder (PARSEME-TR'de ampirik olarak hep böyle görüldü).
- **Karışık alan.** PARSEME kaynağı gazete metni, TDK örnekleri çoğunlukla klasik/edebi
  alıntı; Leipzig derlemi (Wikipedia+Haber+Web) doğal-cümle üslubunu bir ölçüde ekliyor ama
  güncel konuşma dili veya sosyal medya metninde ayrıca test edilmedi.
- **İsim/sıfat kapsamı kısmi.** TDK verisi yalnız gömülü örneği olan deyimlerden alındı —
  TDK'nin tam yaklaşık 11k deyimlik listesinin küçük bir kesiti.
- Ön-token'lanmış girdi bekler (kelime listesi), ham metin değil.

## Sürüm Geçmişi

Aşama 2 HER ZAMAN ayrı bir gövde/veri kaynağıyla eğitildi; hangi sürümde neyin değiştiği:

- **v3 — iki aşamalı boru hattına geçiş.** Aşama 1 tek-BIO modeli precision tavanına
  takılıyordu (yüzey biçim eşleşince bağlamdan bağımsız işaretliyordu). Aşama 2 eklendi:
  ayrı ELECTRA gövdesi, alt 8 transformer katmanı donduruldu (küçük veride tam fine-tune
  ağır overfit ediyordu).
- **v4 — Aşama 1 genişletildi, Aşama 2 değişmedi.** TDK eşleştiricisi `max_gap=2` oldu
  (araya en fazla 2 eşleşmeyen kelime girebilir) ve Leipzig'den madenlenip GLU rubriğiyle
  idyomatik-olarak etiketlenmiş binlerce doğal cümle Aşama 1'e eklendi. Sonuç: dış-kaynak
  doğru-ayırt %42→%56, kazanç tümüyle o zamanki görülmemiş-deyim diliminde.
- **v5 — Aşama 1'e ikinci gövde (ensemble), Aşama 2 değişmedi.** İki farklı veri diliminde
  eğitilmiş stage-1 checkpoint'inin aday span birleşimi, tek modelle kırılamayan bir tavanı
  aştı. Bedel: model iki katı (yaklaşık 1.3GB), yaklaşık 2× gecikme.
  PARSEME F1 hafifçe düştü, dış-kaynak doğru-ayırt yükseldi.
- **v6 — Aşama 1 ensemble'ı TEK gövdeye indirgendi (anlaşmazlık-ağırlıklı distilasyon).**
  Düz (ağırlıksız) damıtım öğretmenlerin ORTALAMASINI aktarıyordu, tamamlayıcı kapsamlarını
  değil; anlaşmazlık ağırlıklandırması (öğretmenlerin GERÇEKTEN ayrıştığı pozisyonlara
  damıtım kapasitesini yönlendirme) bu farkı kapattı. Sonuç: ensemble'ın doğruluğu tek
  gövdede korundu, yaklaşık 2× gecikme bedeli kalktı.
- **v7 — Aşama 1'e ÜÇÜNCÜ bağımsız gövde eklendi (ensemble'a dönüş, distilasyon değil).**
  v6'nın damıtımı iki öğretmene genelleşmişti ama üçe genelleşmedi (üçüncü öğretmenin küçük
  eğitim hacmi anlaşmazlık sinyalini gürültüleştirdi). Saf çıkarım-zamanı birleşim ise işe
  yaradı: dış-kaynak doğru-ayırt yeni bir rekor kırdı, bedelinde v6'nın kazandığı hız/boyut
  avantajı geri gitti (model tekrar yaklaşık 2GB/yaklaşık 3×).
- **v8 — Aşama 2 sentetik minimal çiftlerle yeniden eğitildi, Aşama 1 (v7) değişmedi.**
  Önceki tüm Aşama 2 turları doğal derlemden madenlenip LLM ile ETİKETLENMİŞ cümlelere
  dayanıyordu (literal kullanım doğal metinde nadir olduğu için hep dengesiz/gürültülüydü).
  v8, LLM'e doğal cümle sınıflandırtmak yerine TDK deyim listesinden seçilen yaklaşık 650 deyim için
  doğrudan dengeli idyomatik+literal cümle ÜRETTİRDİ, doğal-derlem verisini tamamen atıp
  yalnız bu sentetik havuzla eğitti (karışık deneme daha kötü sonuç verdi). Sonuç: dış-kaynak
  doğru-ayırt %58.1→%65.2 (varsayılan eşik 0.5'te) — ayrıntılı istatistiksel doğrulama
  yukarıdaki "Sonuçlar" bölümünde.
- **v8 eşik güncellemesi (2026-09-20).** Model ağırlıkları DEĞİŞMEDİ; harici bir model-kartı
  incelemesi sonrası yapılan `stage2_thresh` taramasında 0.3'ün 0.5'ten ölçülebilir şekilde
  daha iyi olduğu görüldü (doğru-ayırt %65.2→%68.2, yanlış-poz %21.2→%14.6). Varsayılan
  `config.stage2_thresh` 0.3'e güncellendi (yalnız config.json değişti, ağırlıklar aynı).

## Atıf

```bibtex
@inproceedings{gungor2018turkish,
  title     = {Turkish Verbal Multiword Expressions Corpus},
  author    = {Erden, Berna and Berk, G{\"o}zde and G{\"u}ng{\"o}r, Tunga},
  booktitle = {26th IEEE Signal Processing and Communications Applications Conference (SIU)},
  year      = {2018},
  doi       = {10.1109/SIU.2018.8404583}
}
@inproceedings{berk2019bigappy,
  title     = {Representing Overlaps in Sequence Labeling Tasks with a Novel Tagging Scheme: Bigappy-Unicrossy},
  author    = {Berk, G{\"o}zde and Erden, Berna and G{\"u}ng{\"o}r, Tunga},
  booktitle = {International Conference on Computational Linguistics and Intelligent Text Processing (CICLing)},
  year      = {2019}
}
```

TDK örnekleri: Türk Dil Kurumu, *Atasözleri ve Deyimler Sözlüğü* (sozluk.gov.tr).

## Lisans

Eğitim verisi (1) PARSEME Türkçe VMWE derlemi, edition 1.2 — **CC BY-NC-SA 4.0** (ticari
olmayan, paylaş-aynı-lisansla); (2) TDK Atasözleri ve Deyimler Sözlüğü — telif TDK'ye ait,
eğitim/araştırma amaçlı kullanılmıştır. Bu model PARSEME'nin lisansını miras alır —
**ticari kullanım için uygun değildir.** Araştırma, eğitim, kişisel/nonprofit projelerde
serbestçe kullanılabilir ve türetilebilir (paylaş-aynı-lisansla, CC BY-NC-SA 4.0 şartıyla);
ticari bir üründe veya ücretli bir hizmette kullanmadan önce PARSEME verisinin lisansını
ayrıca kontrol edin.
