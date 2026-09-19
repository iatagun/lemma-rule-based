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
construction*: *karar vermek*), **O** (serbest birleşim / deyim değil).

**İki aşamalı boru hattı (v3):** Aşama 1 aday span'leri bulur (yukarıdaki BIO modeli); **Aşama 2**
ayrı bir idyomatiklik sınıflandırıcısıdır — her bitişik **VID** adayını *(cümle, span)* olarak alıp
{idyomatik / literal} kararı verir ve **güvenli literal kullanımları eler** ("otobüs yol aldı" gibi).
Aşama 2 varsayılan olarak `predict_spans()` içinde açıktır; `stage2=False` ile kapatılır. Tek-BIO
modeli yüzey biçim eşleşince bağlamdan bağımsız işaretliyordu — Aşama 2 bunu düzeltmeyi hedefler.

**v4 — Aşama 1 (span tespiti) genişletildi, Aşama 2 değişmedi.** Aşama 1 artık (a) TDK
zayıf-denetim eşleştiricisinde sınırlı-boşluklu (`max_gap=2`) eşleşmeye izin veriyor (araya
en fazla 2 eşleşmeyen kelime girebilir, sıra korunur — permütasyon değil) ve (b) Leipzig
derleminden madenlenip GLU rubriğiyle elle idyomatik-olarak etiketlenmiş 4.622 doğal cümle
ile eğitildi. Sonuç: Aşama 1'in **görülmemiş deyimlerde hiç aday önermeme oranı** doğal
(sözlük-dışı) cümlelerde %71→%40'a düştü, ve bu — DEĞİŞMEMİŞ Aşama 2 filtresiyle birleşince —
dış-kaynak bağlam-ayrımını (Çavuşoğlu & Çöltekin) %42'den **%56**'ya çıkardı; kazanç **tümüyle
görülmemiş-deyim diliminde** (seen deyimlerde ayrım değişmedi) — ezber değil genelleme.
Aşama 2 küçük elle-etiketli veriyle eğitildi; literal kullanımların bir kısmı hâlâ geçiyor
(Aşama 2 gövdesinin alt katmanları donduruldu → overfit azaldı, `stage2_thresh` eşiği anlamlı
çalışıyor). precision (~%58-64) ve sınırlamalar bölümüne bakın.

**v5 — Aşama 1'e ikinci gövde (ensemble), Aşama 2 değişmedi.** İki ayrı stage-1 checkpoint'i
(v4'teki gövde + farklı bir corpus-glu diliminde eğitilmiş bağımsız bir varyant) her cümlede
aday span önerir; iki listenin **birleşimi** (çakışan span'lerde en çok oy alan/en uzun
kazanır) DEĞİŞMEMİŞ Aşama 2 filtresine girer. İki checkpoint farklı veri dilimleriyle
eğitildiği için görülmemiş-deyim kör noktaları örtüşmüyor — birleşim, ne kadar hacim eklense
de tek bir modelle kırılamayan bir tavanı aşıyor. Sonuç: dış-kaynak doğru-ayırt %55.6→**%57.1**
(genel), görülmemiş-deyim diliminde %54.2→**%55.9** — proje tarihinin en iyi ölçümü. Bedel:
**modelin iki katına çıkması** (üç ELECTRA gövdesi: iki stage-1 + bir stage-2, ~1.3 GB) ve
buna bağlı ~2× çıkarım gecikmesi (stage-1 iki kez çalışıyor). PARSEME F1 hafifçe düştü
(66.16→65.57) — iki gövdenin birleşimi biraz daha fazla yanlış-pozitif de ekliyor
(yanlış-poz %26.3→%27.8), net etki yine pozitif.

**v6 — Aşama 1 ensemble'ı TEK gövdeye indirgendi (anlaşmazlık-ağırlıklı distilasyon),
sonuç KORUNDU/GELİŞTİ, model tekrar tek boyut/hız.** v5'in iki-gövde ensemble'ı (vE+vL)
çalıştırma maliyetini kaldırmak için, iki gövdenin bilgisini tek bir öğrenci gövdeye
damıtmak denendi — ama düz (tüm token'lara eşit ağırlıklı) damıtım öğretmenlerin yalnız
ORTALAMASINI aktarıyordu, tamamlayıcı kapsamlarını değil (dış-kaynak doğru-ayırt ensemble'ın
altında kaldı). Çözüm: damıtım kaybı, iki öğretmenin per-token GERÇEKTEN anlaştığı mı yoksa
ayrıştığı mı olduğuna göre ağırlıklandırıldı — kapasite, öğretmenlerin zaten hemfikir olduğu
(gereksiz) değil, ayrıştığı (tamamlayıcı bilgi taşıyan) pozisyonlara yönlendirildi. Sonuç: **tek
gövdede** dış-kaynak doğru-ayırt %57.1'e (ensemble'la birebir eşit) ulaştı, PARSEME F1
65.57→**66.07** (ensemble'ı bile geçti), yanlış-pozitif %27.8→**%23.7** (düştü). Model
tekrar **tek ELECTRA gövdesi** (~440MB stage-1 + stage-2, v4'ten büyük değil), **~2×
gecikme bedeli tamamen ortadan kalktı**.

**v7 — Aşama 1'e ÜÇÜNCÜ bir bağımsız gövde eklendi (ensemble'a geri dönüş, distilasyon
değil).** v6'nın tek-gövde damıtımı, iki öğretmene (vE+vL) genelleştirildiğinde işe yaramıştı
(bkz. yukarı) ama **üç** öğretmenle (vE+vL+ üçüncü, dar bir veri diliminde eğitilmiş bağımsız
bir gövde) denendiğinde damıtım BAŞARISIZ oldu — üçüncü öğretmenin görece küçük eğitim hacmi
yüzünden token-düzeyi anlaşmazlıkları çoğunlukla gerçek tamamlayıcı sinyal değil gürültüydü;
anlaşmazlık-ağırlıklı kayıp bu gürültüye ağırlık verip PARSEME F1'i yükseltirken dış-kaynak
doğru-ayırtı v5/v6'nın ALTINA düşürdü. Buna karşılık **saf çıkarım-zamanı birleşim** (üç
gövdenin aday span'lerinin birleşimi, damıtım yok) yeni bir rekor verdi: dış-kaynak
doğru-ayırt %57.1→**%58.1** (genel), görülmemiş-deyim diliminde %55.4→**%57.1**. Bedel v6'nın
avantajını geri verdi: model artık **üç** stage-1 ELECTRA gövdesi taşıyor (+ stage-2, ~2 GB)
ve stage-1 üç kez çalışıyor (~3× gecikme) — v5'in kabul ettiği takasın bir adım ilerisi.

- **Gövde:** [`dbmdz/electra-base-turkish-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)
  (DizgeBERT-Morph/Joint/Dep ile aynı → ortak subword sözlüğü)
- **Kelime temsili:** ilk subword ⊕ son subword (DizgeBERT-Morph ile aynı yöntem)
- **İki katmanlı etiketleme (bigappy-unicrossy tarzı, Berk, Erden & Güngör 2019):** standart
  BIO **süreksiz (gap'li)** span'leri temsil edemez (*"sahip ... olarak"* gibi araya kelime
  giren deyimler). İkinci, bağımsız bir head (`o/b-VID/i-VID/b-LVC/i-LVC`) yalnız gap'li
  span'in **2. parçasını** taşır; 1. parça her zaman ana BIO katmanında. Çıkarımda iki katman
  ayrı ayrı Viterbi ile çözülüp aynı kategoriden en yakın parçalar eşleştirilir.
- **Çözümleme:** geçiş-kısıtlı **Viterbi** (argmax değil), her iki katmanda da.
- **Aşama 2 — idyomatiklik sınıflandırıcısı:** ikinci, bağımsız bir ELECTRA gövdesi + span
  ilk⊕son subword temsili → `Linear(2H, 2)` → {literal, idyomatik}. `predict_spans()` bitişik
  VID adaylarını bundan geçirir; yalnız *güvenli* literal (p(literal) > eşik, varsayılan 0.5)
  elenir — LVC ve gap'li span'ler dokunulmaz (LVC yarı-birleşimsel, ayrım anlamsız).
- **Aşama 1 — üç bağımsız stage-1 gövdesinin ensemble'ı (v7):** her cümlede üç ayrı ELECTRA
  gövdesi (farklı veri dilimleriyle eğitilmiş) kendi aday span'lerini önerir; birleşim
  (çakışan span'lerde en çok oy alan/en uzun kazanır) DEĞİŞMEMİŞ Aşama 2 filtresine girer.
  Model dosyası **üç** stage-1 ELECTRA gövdesi + Aşama 2 için ayrı bir gövde içerir (~2 GB,
  ~3× gecikme — aşağıdaki "Eğitim" bölümüne bakın). Önceki v6 sürümü bunun yerine iki
  gövdeyi tek gövdeye damıtıyordu (1× bedel); üçüncü gövdeyle aynı damıtım denendi ama
  başarısız oldu (bkz. yukarıdaki v7 notu) — bu yüzden v7 ham ensemble olarak kaldı.
- **Eğitim verisi:**
  1. [PARSEME Türkçe fiil-merkezli çok-sözcüklü ifade derlemi, edition 1.2](https://gitlab.com/parseme/sharedtask-data/-/tree/master/1.2/TR)
     (Güngör & Yirmibeşoğlu) — 17.945 cümle, VID+LVC.full toplam ~6.7k span (yalnız *verbal* MWE;
     bunun 308'i gap'li — artık atılmıyor, 2. katmana taşınıyor).
  2. TDK Atasözleri ve Deyimler Sözlüğü'nden çıkarılan 2.629 gömülü örnek cümle (2.501 benzersiz
     deyim) — isim/sıfat deyimlerini de kapsar (*eli açık*, *başı dertte* gibi, PARSEME'de yok);
     gövde-eşleştirme (stem matching) ile zayıf-etiketlenmiş (weak supervision), deyim
     sözlük-biçimi ile örnek cümledeki çekimli yüzey biçimi aynı gövde dizisine indirgenerek.
     313 deyim (323 cümle, ayrıca 313 başka deyim 328
     cümlelik bir dev parçasında) tamamen ayrı tutulup hiç eğitime sokulmadı; bölme **hem deyim
     hem cümle metni düzeyinde** yapıldı (aynı alıntı cümle birden fazla deyime örnek
     verilebiliyor — sızıntı riski görülüp düzeltildi) — held-out genelleme testi için.
  3. **Aşama 2 için:** Leipzig Türkçe derleminden (Wikipedia + Haber + Web, CC-BY) deyim yüzey
     biçimiyle eşleşen cümleler madenlenip **elle etiketlendi** (GLU deyim etiketleme kılavuzu
     rubriğiyle) — 1661 kullanım (910 idyomatik + 751 literal). 975'i eğitim (622 idyomatik +
     353 literal, sınıf ağırlığıyla dengelendi), 686'sı **118 hiç görülmemiş deyimden** held-out.
  4. **Aşama 1 için (v4):** aynı GLU-etiketli Leipzig havuzundaki **idyomatik-olarak işaretli
     4.622 doğal cümle** de Aşama 1'in span-eğitimine eklendi (yalnız D-etiketli; L-etiketli
     kullanımlar zaten Aşama 2'nin negatifleri). Amaç: Aşama 1'in eğitim verisi TDK sözlük-örnek
     ve PARSEME haber-metni üslubuna aşırı uymuştu — sözlük/haber dışı doğal cümlelerde aday
     önermeyi öğrenmiyordu (görülmemiş deyimde %71 hiç aday yok). Bu ekleme yalnız üslup/kullanım
     çeşitliliği katıyor, deyim-kimliği kapsamını genişletmiyor (aynı TDK sözlük-deyim havuzundan
     madenlendi) — kazancın tamamen görülmemiş-deyim diliminde çıkması (aşağıya bakın) bunun
     ezber değil genelleme olduğunu doğruluyor. Ayrıca TDK zayıf-denetim eşleştiricisi artık
     `max_gap=2` (araya en fazla 2 eşleşmeyen kelime — sıra korunur, permütasyon değil).

## Sonuçlar

Aşağıdaki tablo **Aşama 1** (tek-BIO, `stage2=False`) performansıdır — span-düzeyi, exact-match,
Viterbi çözümlemeyle:

| test seti | kapsam | P | R | F1 |
|---|---|---|---|---|
| PARSEME test.cupt (held-out), **genel** | fiil-merkezli, bitişik+gap'li | 56.44 | 78.22 | **65.57** |
| PARSEME test.cupt, yalnız **gap'li span'ler** | süreksiz deyim/eşdizim | 41.67 | 31.91 | 36.14 |
| TDK held-out (görülmemiş deyimler, v4 tek-gövde sayısı*) | isim/sıfat dahil karışık | 61.54 | 60.38 | **60.38** |

(v5/ensemble sayıları — v4'e göre F1 hafifçe düştü (67.69→65.57), asıl kazanım aşağıdaki
dış-kaynak bağlam-ayrımı tablosunda. *TDK satırı ensemble ile yeniden ölçülmedi. **v6
(anlaşmazlık-damıtımı) PARSEME ALL F1 66.07** — v5'i de geçti, aşağıya bakın.)

Gap'li satır önemli: bu span'ler standart BIO ile **yapısal olarak asla yakalanamaz**dı (v1'de
recall garanti %0). İki-katmanlı şemayla artık ~%38-47 (test/dev) kurtarılıyor — kusursuz değil
ama sıfırdan gerçek bir kazanım. Kontiguous (bitişik) span'lerdeki performans korunmuş (tek-katman
öncesi sürümle aynı büyüklük mertebesinde — 2. head eklenmesi ana görevi bozmadı).

### Aşama 2 (varsayılan, `stage2=True`) — bağlam ayrımı

**Bağımsız dış kaynak — bağlam-bağımlılık testi.** Çavuşoğlu & Çöltekin'in (MWE 2026)
elle-yazılmış Türkçe deyim benchmark'ı (198 deyim, her biri için gerçek idyomatik-kullanım
+ literal-kullanım cümle çifti, eğitim verimizde yok — `benchmark/eval_idiom.py --mode
external`) üzerinde:

| ölçüm | Aşama 1 (stage2=False) | **+ Aşama 2 (varsayılan)** |
|---|---|---|
| idyomatik cümlede span işaretledi (duyarlılık) | %77.8 | %77.8 |
| literal cümlede **yanlış** span işaretledi | — | %23.7 |
| ikisini de doğru ayırt etti | — | **%57.1** |

**v3→v4→v5→v6→v7 kıyası (Aşama 2 HİÇ değişmedi; her sürümde tek değişen Aşama 1):**

| ölçüm | v3 | v4 | v5 (2-gövde ensemble) | v6 (2-öğretmen distilasyon) | **v7 (3-gövde ensemble)** |
|---|---|---|---|---|---|
| doğru-ayırt — **görülmemiş deyim** dilimi (n=177) | %39.0 | %54.2 | %55.9 | %55.4 | **%57.1 (en iyi)** |
| doğru-ayırt — bilinen deyim dilimi (n=21) | %66.7 | %66.7 | %66.7 | %71.4 (en iyi) | %66.7 |
| doğru-ayırt — genel (n=198) | %41.9 | %55.6 | %57.1 | %57.1 | **%58.1 (en iyi)** |
| yanlış-pozitif | %16.2 | %26.3 | %27.8 | %23.7 (en düşük) | %28.3 |
| duyarlılık | %55.6 | %78.8 | %81.3 | %77.8 | %83.3 |
| GLU tanı seti (35 vaka) | 21/35 | 20/35 | 20/35 | 20/35 | 20/35 (aynı) |
| **model boyutu / gecikme** | 1× | 1× | 2×, ~1.3GB | **1× (bedel yok)** | **3×, ~2GB (en pahalı)** |

v7'yi TEK gövdeye damıtmak (v6'nın yaptığı gibi) denendi ama başarısız oldu — üçüncü
öğretmenin küçük eğitim hacmi anlaşmazlık sinyalini gürültüleştirdi, sonuç v3 tabanından
bile kötü çıktı. v7 bu yüzden **ham ensemble** olarak yayınlandı, v6'nın tek-gövde
avantajını taşımıyor.

v3→v4 kazancı **tümüyle görülmemiş-deyim diliminde** çıkmıştı — bilinen deyimlerde ayrım
birebir aynı kalmıştı (Aşama 1'in daha önce hiç aday önermediği görülmemiş deyimlerde artık
aday önerebilmesi sayesinde, ezber değil genelleme). v4→v5 aynı deseni tekrarlıyor: ikinci
stage-1 gövdesi farklı bir veri diliminde eğitildiği için v4'ün hâlâ kaçırdığı bazı
görülmemiş-deyim adaylarını yakalıyor. **v5→v6, v5'in bedelini (2×/1.3GB) v5'in doğruluğunu
kaybetmeden kaldırıyor** — anlaşmazlık-ağırlıklı damıtım, ensemble'ın iki gövdesinin
tamamlayıcı kapsamını (bilinen deyimde bile artık daha iyi, görülmemiş deyimde ensemble'a
neredeyse eşit) tek gövdeye aktarabildi; ilk denenen DÜZ (ağırlıksız) damıtım bunu
başaramamıştı (doğru-ayırt ensemble'ın belirgin altında kalmıştı). PARSEME test'te Aşama 2
F1'i düşürme eğilimi v6'da tersine döndü (**67.69 → 66.16 → 65.57 → 66.07**) — v6 hem
PARSEME'de hem dış-kaynakta v5'i geçti, saf takas değil.

Model idyomatik/literal ayrımını **kısmen** çözüyor — dürüst, bilinen bir sınırlama. Aşama 2
küçük veriyle eğitildiğinden literal kullanımların önemli bir kısmı hâlâ geçiyor.

**Çözümleme: Viterbi, argmax değil.** Ham token-düzeyi argmax yapısal olarak geçersiz diziler
üretebilir (`O` sonrası yetim `I-VID`, ya da `B-VID` sonrası kategori-karışık `I-LVC`). Çıkışa
geçiş-kısıtlı Viterbi kod çözme uygulanır (yeniden eğitim gerektirmez, yalnız çıkarım-zamanı) —
argmax'a göre ölçülebilir kazanım, hem precision hem recall'da (saf P/R takası değil — bozuk
sınırları düzelterek kaçırılan doğru span'ları da kurtarıyor).

**Precision hakkında dürüst not:** ~%58-64 (v4'te bir miktar düştü — Aşama 1'e eklenen doğal
Leipzig cümleleri recall'ı yükseltirken precision'dan biraz feragat etti). Class-weight
kaldırma ve çıkarım-zamanı güven eşiği taraması bunu anlamlı ölçüde değiştirmedi. Recall
yüksek (~%76-84) — model kaçırmaktan çok fazla-işaretlemeye eğilimli.

**Dış kıyas noktası:** PARSEME 1.1 shared task'ta en iyi sistem (SHOMA, nöral+CRF) tüm diller
ortalamasında %58.09 makro-F1 almıştı (bazı diller %23-32 gibi çok düşük, Macarca/Romence
%85-90). Bu bağlamda PARSEME-TR'de aldığımız ~70 makul/rekabetçi — görev doğası gereği zor.
Aynı zamanda büyük LLM'ler (Gemini 2.5, GPT-4o, Llama-3 70B) bağımsız bir Türkçe deyim
benchmark'ında (Çavuşoğlu & Çöltekin 2026) ikili idyomatik-mi-değil-mi sınıflandırmasında
yalnızca **%59-61 doğruluk** (rastgele tahminin biraz üstü) alıyor — deyim tespiti büyük
modeller için de hâlâ zor bir problem.

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

# eşiği gevşetmek (daha az eleme, recall koru): stage2_thresh yüksek
print(m.predict_spans(lit, tokenizer=tok, stage2_thresh=0.9))

# gap'li (süreksiz) örnek — "sahip ... olarak" (Aşama 2 dokunmaz)
ws = "... sahip olduğu ... değerleriyle olarak önemini ...".split()
print(m.predict_spans(ws, tokenizer=tok))
# gappy=True ise: {'text': 'sahip ... olarak', 'start':.., 'end':.., 'start2':.., 'end2':.., 'category':..}
```

## Kısıtlar

- **Aşama 2 deneysel, küçük veriyle eğitildi** (975 örnek, elle etiketli), v3'ten beri
  değişmedi. Literal kullanımların önemli bir kısmını hâlâ yakalayamıyor (yanlış-pozitif
  ~%24-28). `stage2=False` ile tamamen devre dışı, `stage2_thresh` ile eşik ayarlanır.
- **v7 üç gövdeli bir ensemble, tek bir bütünleşik model değil** — üçüncü gövde görece dar
  bir veri diliminde eğitildi ve yalnız çıkarım-zamanı birleşim olarak katkı sağlıyor (kendi
  başına tek-gövde damıtımı denendiğinde başarısız oldu, bkz. yukarı). Model boyutu/gecikmesi
  buna göre büyük (~2GB, ~3×) — hız/bellek kısıtlı ortamlarda v6 (tek gövde, ~440MB) daha
  uygun bir seçim olabilir.
- **Precision ~%58-64** (yukarıya bakın) — üretim kullanımında çıktıyı doğrulamadan güvenmeyin.
- **Gap'li (süreksiz) span'ler kısmen çözülüyor, tam değil.** İki-katmanlı şema ~%38-47'sini
  kurtarıyor (yukarıya bakın); geri kalanı hâlâ kaçıyor. Ayrıca şema yalnız **tam 2 parçalı**
  gap'leri temsil eder (PARSEME-TR'de ampirik olarak hep böyle — 3+ parçalı hiç görülmedi).
- **Karışık alan.** PARSEME kaynağı gazete metni, TDK örnekleri çoğunlukla klasik/edebi alıntı
  (yazar isimli); v4'te Leipzig derlemi (Wikipedia+Haber+Web) eklenerek doğal-cümle üslubu
  genelleme sorunu büyük ölçüde iyileştirildi (dış-kaynak görülmemiş-deyim doğru-ayırt
  %39→%54) — ama güncel konuşma dili veya sosyal medya metninde hâlâ ayrıca test edilmedi.
- **İsim/sıfat kapsamı kısmi.** TDK verisi yalnız gömülü örneği olan (~%43) deyimlerden ve
  bunların da ~%67'si (stem-eşleştirme başarılı) kullanıldı — TDK'nin tam ~11k deyimlik
  listesinin küçük bir kesiti.
- Ön-token'lanmış girdi bekler (kelime listesi), ham metin değil.

## Eğitim

- `dbmdz/electra-base-turkish-cased-discriminator` + kelime-düzeyi first⊕last pooling
- **İki bağımsız BIO head'i**: katman 1 (5 sınıf: O, B/I-VID, B/I-LVC — tüm span'lerin 1. parçası)
  + katman 2 (5 sınıf: o, b/i-VID, b/i-LVC — yalnız gap'li span'lerin 2. parçası)
- AdamW, linear warmup; katman 1 ters-frekans class-weighting opsiyonel (`--class-weights`,
  ölçülebilir fark yaratmadı), katman 2'de HER ZAMAN açık (dengesizlik çok daha aşırı —
  ~326k tokenden ~300'ü non-'o')
- PARSEME train + TDK train (`max_gap=2` zayıf-denetim eşleştirici) + 4.622 Leipzig-madenli
  GLU-etiketli doğal cümle (v4) karışık; PARSEME dev ile model seçimi (epoch 5/10, en iyi
  dev span-F1)
- Çıkarım: token-düzeyi argmax değil, geçiş-kısıtlı **Viterbi** (her iki katmanda ayrı ayrı)

**Aşama 1 — anlaşmazlık-ağırlıklı ensemble distilasyonu (v6):** v5'in iki bağımsız stage-1
gövdesi (vE, vL — farklı corpus-glu dilimleriyle eğitilmiş, bkz. v5 notu) "öğretmen" olarak
kullanıldı. Weak-supervision eğitim kayıtlarının (TDK + corpus-glu; PARSEME altın verisi HİÇ
dokunulmadı) her token'ı için (a) iki öğretmenin ortalama softmax'ı ("yumuşak hedef") VE (b)
iki öğretmenin per-token toplam-varyasyon uzaklığı ("anlaşmazlık ağırlığı", [0,1]) önceden
hesaplandı (`scripts/distill_ensemble_labels.py`). Öğrenci (v4'ün BİREBİR aynı tek-gövde
mimarisi) normal sabit-etiket kaybına ek olarak bu yumuşak hedefle bir KL kaybı görüyor —
ama KL'nin katkısı ağırlıksız ortalama DEĞİL, anlaşmazlık ağırlıklı ortalama: öğretmenlerin
zaten hemfikir olduğu pozisyonlarda KL sinyali neredeyse sıfıra iniyor, ayrıştıkları
(tamamlayıcı bilgi taşıyan) pozisyonlarda tam ağırlıkta kalıyor. İlk denenen ağırlıksız
(düz ortalama) damıtım aynı mimariyle denenmiş ama Çavuşoğlu doğru-ayırtında ensemble'ın
belirgin altında kalmıştı — anlaşmazlık ağırlıklandırması bu farkı kapattı.

**Aşama 2 (idyomatiklik sınıflandırıcısı):** ayrı ELECTRA gövdesi + span ilk⊕son pooling →
`Linear(2H, 2)`. 975 elle-etiketli örnek (Leipzig derleminden madenlenip GLU rubriğiyle
sınıflandırıldı), sınıf ağırlığı literal'e; **alt 8 transformer katmanı donduruldu** (975
örnekte tam fine-tune ağır overfit ediyordu → softmax doygun, eşik ayarı ölü). 118 görülmemiş
deyimlik held-out ile dengeli-doğruluk seçimi (best: acc ~%86, idyom-recall %90, literal-eleme
%82). Aşama 1'den tamamen bağımsız eğitildi.

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
**ticari kullanım için uygun değildir**.
