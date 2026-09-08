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

**v3 — Aşama 2 deneyseldir.** Bağlam ayrımını ölçülebilir biçimde iyileştirir (dış kaynakta
yanlış-pozitif %25→%16, doğru-ayırt %37→%42) ama küçük elle-etiketli veriyle eğitildi;
literal kullanımların ~%18'i hâlâ geçiyor. precision (~%60-71) ve sınırlamalar bölümüne bakın.
(v3: Aşama 2 gövdesinin alt katmanları donduruldu → overfit azaldı, `stage2_thresh` eşiği
artık anlamlı çalışıyor.)

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
  elenir — LVC ve gap'li span'ler dokunulmaz (LVC yarı-birleşimsel, ayrım anlamsız). Model
  dosyası bu yüzden iki ELECTRA gövdesi içerir (~880 MB).
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

## Sonuçlar

Aşağıdaki tablo **Aşama 1** (tek-BIO, `stage2=False`) performansıdır — span-düzeyi, exact-match,
Viterbi çözümlemeyle:

| test seti | kapsam | P | R | F1 |
|---|---|---|---|---|
| PARSEME test.cupt (held-out), **genel** | fiil-merkezli, bitişik+gap'li | 64.35 | 75.78 | 69.60 |
| PARSEME test.cupt, yalnız **gap'li span'ler** | süreksiz deyim/eşdizim | 39.13 | 38.30 | **38.71** |
| TDK held-out (313 deyim, **eğitimde/hiçbir split'te hiç görülmedi**) | isim/sıfat dahil karışık | 72.26 | 73.37 | 72.81 |

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
| idyomatik cümlede span işaretledi (duyarlılık) | %59.1 | %55.6 |
| literal cümlede **yanlış** span işaretledi | %25.3 | **%16.2** |
| ikisini de doğru ayırt etti | %37.4 | **%41.9** |

Aşama 2, literal cümledeki yanlış-pozitifleri ~1/3 azaltır ve doğru-ayırt oranını yükseltir —
bedeli birkaç puan duyarlılık (bazı gerçek idyomatik kullanımlar da elenir). Aynı yönde:
GLU tanı seti 16/35 → 21/35. PARSEME test'te Aşama 2 F1'i **69.60 → 67.60** düşürür — ama bu
yapaydır: o benchmark'ta **tüm** span'ler idyomatik kullanımdır (literal yok), dolayısıyla her
eleme bir false-negative'dir. Gerçek metinde (idyomatik + literal karışık) kazanç nettir.

Model idyomatik/literal ayrımını **kısmen** çözüyor — dürüst, bilinen bir sınırlama. Aşama 2
küçük veriyle eğitildiğinden literal kullanımların ~%18'i hâlâ geçiyor.

**Çözümleme: Viterbi, argmax değil.** Ham token-düzeyi argmax yapısal olarak geçersiz diziler
üretebilir (`O` sonrası yetim `I-VID`, ya da `B-VID` sonrası kategori-karışık `I-LVC`). Çıkışa
geçiş-kısıtlı Viterbi kod çözme uygulanır (yeniden eğitim gerektirmez, yalnız çıkarım-zamanı) —
argmax'a göre ölçülebilir kazanım, hem precision hem recall'da (saf P/R takası değil — bozuk
sınırları düzelterek kaçırılan doğru span'ları da kurtarıyor).

**Precision hakkında dürüst not:** ~%64-71 — yani işaretlenen her 3 span'den yaklaşık 1'i
yanlış pozitif olabilir. Class-weight kaldırma ve çıkarım-zamanı güven eşiği taraması bunu
anlamlı ölçüde değiştirmedi; bu veri ölçeğinde pratik bir tavan gibi görünüyor. Recall yüksek
(~%76-83) — model kaçırmaktan çok fazla-işaretlemeye eğilimli.

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

- **Aşama 2 deneysel, küçük veriyle eğitildi** (975 örnek, elle etiketli). Literal kullanımların
  ~%18'ini yakalayamıyor; bazı gerçek idyomatik kullanımları da yanlışlıkla eliyor (dış kaynak
  duyarlılık %59→%56). `stage2=False` ile tamamen devre dışı, `stage2_thresh` ile eşik ayarlanır.
- **Precision ~%60-71** (yukarıya bakın) — üretim kullanımında çıktıyı doğrulamadan güvenmeyin.
- **Gap'li (süreksiz) span'ler kısmen çözülüyor, tam değil.** İki-katmanlı şema ~%38-47'sini
  kurtarıyor (yukarıya bakın); geri kalanı hâlâ kaçıyor. Ayrıca şema yalnız **tam 2 parçalı**
  gap'leri temsil eder (PARSEME-TR'de ampirik olarak hep böyle — 3+ parçalı hiç görülmedi).
- **Karışık alan.** PARSEME kaynağı gazete metni, TDK örnekleri çoğunlukla klasik/edebi alıntı
  (yazar isimli) — güncel konuşma dili veya sosyal medya metninde genelleme test edilmedi.
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
- PARSEME train + TDK train (2.629 örnek) karışık; PARSEME dev ile model seçimi (epoch 10/10)
- Çıkarım: token-düzeyi argmax değil, geçiş-kısıtlı **Viterbi** (her iki katmanda ayrı ayrı)

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
