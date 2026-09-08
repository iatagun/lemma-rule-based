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

**v1 erken sürüm** — sınırlamalar bölümüne bakın, özellikle precision (~%64-71) hakkında.

- **Gövde:** [`dbmdz/electra-base-turkish-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)
  (DizgeBERT-Morph/Joint/Dep ile aynı → ortak subword sözlüğü)
- **Kelime temsili:** ilk subword ⊕ son subword (DizgeBERT-Morph ile aynı yöntem)
- **İki katmanlı etiketleme (bigappy-unicrossy tarzı, Berk, Erden & Güngör 2019):** standart
  BIO **süreksiz (gap'li)** span'leri temsil edemez (*"sahip ... olarak"* gibi araya kelime
  giren deyimler). İkinci, bağımsız bir head (`o/b-VID/i-VID/b-LVC/i-LVC`) yalnız gap'li
  span'in **2. parçasını** taşır; 1. parça her zaman ana BIO katmanında. Çıkarımda iki katman
  ayrı ayrı Viterbi ile çözülüp aynı kategoriden en yakın parçalar eşleştirilir.
- **Çözümleme:** geçiş-kısıtlı **Viterbi** (argmax değil), her iki katmanda da.
- **Eğitim verisi:**
  1. [PARSEME Türkçe fiil-merkezli çok-sözcüklü ifade derlemi, edition 1.2](https://gitlab.com/parseme/sharedtask-data/-/tree/master/1.2/TR)
     (Güngör & Yirmibeşoğlu) — 17.945 cümle, VID+LVC.full toplam ~6.7k span (yalnız *verbal* MWE;
     bunun 308'i gap'li — artık atılmıyor, 2. katmana taşınıyor).
  2. TDK Atasözleri ve Deyimler Sözlüğü'nden çıkarılan 2.629 gömülü örnek cümle (2.501 benzersiz
     deyim) — isim/sıfat deyimlerini de kapsar (*eli açık*, *başı dertte* gibi, PARSEME'de yok);
     repo'nun kural-tabanlı morfoloji çözümleyicisiyle gövde-eşleştirme (stem matching) ile
     zayıf-etiketlenmiş (weak supervision). 313 deyim (323 cümle, ayrıca 313 başka deyim 328
     cümlelik bir dev parçasında) tamamen ayrı tutulup hiç eğitime sokulmadı; bölme **hem deyim
     hem cümle metni düzeyinde** yapıldı (aynı alıntı cümle birden fazla deyime örnek
     verilebiliyor — sızıntı riski görülüp düzeltildi) — held-out genelleme testi için.

## Sonuçlar (span-düzeyi, exact-match, Viterbi çözümlemeyle)

| test seti | kapsam | P | R | F1 |
|---|---|---|---|---|
| PARSEME test.cupt (held-out), **genel** | fiil-merkezli, bitişik+gap'li | 64.35 | 75.78 | 69.60 |
| PARSEME test.cupt, yalnız **gap'li span'ler** | süreksiz deyim/eşdizim | 39.13 | 38.30 | **38.71** |
| TDK held-out (313 deyim, **eğitimde/hiçbir split'te hiç görülmedi**) | isim/sıfat dahil karışık | 72.26 | 73.37 | 72.81 |

Gap'li satır önemli: bu span'ler standart BIO ile **yapısal olarak asla yakalanamaz**dı (v1'de
recall garanti %0). İki-katmanlı şemayla artık ~%38-47 (test/dev) kurtarılıyor — kusursuz değil
ama sıfırdan gerçek bir kazanım. Kontiguous (bitişik) span'lerdeki performans korunmuş (tek-katman
öncesi sürümle aynı büyüklük mertebesinde — 2. head eklenmesi ana görevi bozmadı).

**Bağımsız dış kaynak — bağlam-bağımlılık testi.** Çavuşoğlu & Çöltekin'in (MWE 2026)
elle-yazılmış Türkçe deyim benchmark'ı (198 deyim, her biri için gerçek idyomatik-kullanım
+ literal-kullanım cümle çifti, eğitim verimizde yok — `benchmark/eval_idiom.py --mode
external`) üzerinde:

| ölçüm | sonuç |
|---|---|
| idyomatik cümlede span işaretledi (duyarlılık) | %59.1 (117/198) |
| literal cümlede **yanlış** span işaretledi | %25.3 (50/198) |
| ikisini de doğru ayırt etti | %37.4 (74/198) |

Bu, kullanıcının en baştaki endişesini (bağlam-bağımlılık) doğrudan ölçüyor: model
idyomatik/literal ayrımını **kısmen** çözüyor ama üçte birinin biraz üstünde tam doğru — bu
küme çoğunlukla isim/sıfat deyimlerinden oluştuğu için (PARSEME'nin zayıf olduğu kategori),
sayı beklenenden düşük çıkıyor. Dürüst, bilinen bir sınırlama.

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
print(m.predict(words, tokenizer=tok))
# [('Sonunda', 'O', 'o'), ('gözden', 'B-VID', 'o'), ('düştü', 'I-VID', 'o'), ('.', 'O', 'o')]
print(m.predict_spans(words, tokenizer=tok))
# [{'text': 'gözden düştü', 'start': 1, 'end': 3, 'category': 'VID', 'gappy': False}]

# gap'li (süreksiz) örnek — "sahip ... olarak"
ws = "... sahip olduğu ... değerleriyle olarak önemini ...".split()
print(m.predict_spans(ws, tokenizer=tok))
# gappy=True ise: {'text': 'sahip ... olarak', 'start':.., 'end':.., 'start2':.., 'end2':.., 'category':..}
```

## Kısıtlar

- **Precision ~%64-71** (yukarıya bakın) — üretim kullanımında çıktıyı doğrulamadan güvenmeyin.
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
