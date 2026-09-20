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
| İdyomatik/literal doğru ayırt etme oranı (kendi benchmark'ımız) | **%68.2** (95% GA: %61.6–%74.7, n=198) | önceki sürüme (v7, %58.1, varsayılan eşik 0.5) göre +10.1pp; **her iki sürüm de AYNI 198 çiftte kendi en iyi eşiğine ayarlandı** — bkz. "dev-seti gibi" çekincesi altta |
| Yanlış-pozitif (literal cümlede yanlış işaretleme) | %14.6 | v7'ye göre (%28.3) yarıya yakın düşüş |
| PARSEME test.cupt span-F1 | 63.48 | exact-match, VID+LVC.full |
| Precision | yaklaşık %55-64 | model kaçırmaktan çok fazla-işaretlemeye eğilimli |

**Üç bağımsız DIŞ değerlendirme (bizim seçtiğimiz eşik/sürüm yok, ayrıntı aşağıda "İlgili
çalışmalar"):**

| dış kaynak | görev | sonuç | not |
|---|---|---|---|
| Aslantaş & Güngör TR test (131 cümle) | span exact-match (seqeval) | **F1=0.592** (gerçekten görülmemiş 60 deyimde 0.414) | onların in-domain modelleri 0.877-0.880 — DizgeBERT geride |
| Dodiom TR (6861 örnek/36 deyim) | idiomaticity (duyarlılık/yanlış-poz) | dengelenmiş-doğruluk %77.4, ama Aşama-2 açık/kapalı farkı **istatistiksel olarak anlamsız** (95% GA sıfırı içeriyor) | farklı ölçek, Çavuşoğlu'yla doğrudan kıyaslanamaz |
| Umut et al. (2025) | — | ölçülmedi | veri seti yayınlanmamış, Dodiom yerine kullanıldı |

**En büyük bilinen kısıt:** Aşama 1, üç ayrı ELECTRA gövdesinin (ensemble) birleşimidir — model
dosyası yaklaşık 2GB, çıkarım tek-gövdeli bir modele göre yaklaşık 3× yavaş. Precision de hâlâ mütevazı: her
3 işaretlemeden yaklaşık 1-1.5'i yanlış olabilir. Ayrıntılı sürüm geçmişi ve deney notları aşağıda.

## Kullanım

**Model dosyası ~2GB ve `trust_remote_code=True` uzak Python kodu çalıştırır** (`modeling_
dizgebert_idiom.py`) — üretimde bir `revision=` (commit hash) pinlemeniz önerilir, aksi
halde repo'ya yeni bir push geldiğinde farkında olmadan farklı kod/ağırlık çekebilirsiniz.
Güncel commit: `984e48b` (bu kartın kendisi). **v6 (tek gövde, ~440MB, daha küçük/hızlı ama
daha eski/zayıf bir Aşama 2 — bkz. Sürüm Geçmişi)** hâlâ `revision="fd002bdd32"` ile erişilebilir
(HF repo'nun commit geçmişinde, ayrı bir model/branch DEĞİL — sonraki commit'ler üstüne yazdı).

```python
from transformers import AutoModel, AutoTokenizer

REV = "984e48b757236b43641cd1b77c55aa49b8e2b99d"  # pinlemek için; v6 için "fd002bdd32..." kullanın
m = AutoModel.from_pretrained("iatagun/DizgeBERT-Idiom", trust_remote_code=True,
                               revision=REV).eval()
tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Idiom", revision=REV)

words = ["Sonunda", "gözden", "düştü", "."]
print(m.predict(words, tokenizer=tok))            # ham BIO (Aşama 1, Aşama 2'den etkilenmez)
# [('Sonunda', 'O', 'o'), ('gözden', 'B-VID', 'o'), ('düştü', 'I-VID', 'o'), ('.', 'O', 'o')]

print(m.predict_spans(words, tokenizer=tok))      # Aşama 2 varsayılan AÇIK
# [{'text': 'gözden düştü', 'start': 1, 'end': 3, 'category': 'VID', 'gappy': False}]

# Aşama 2, literal kullanımı eler — bu ÇIKTILAR gerçekten çalıştırılıp doğrulandı (2026-09-20,
# güncel commit'e karşı; "otobüs ... yol aldı" projenin v5-v7 arası KALICI yanlış-pozitif
# örneğiydi, v8+thresh=0.3'te ilk kez doğru elendiği burada teyit edildi):
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
   CC-BY) deyim yüzey biçimiyle eşleşen doğal cümleler madenlenip **GLU rubriğiyle**
   (Gülsün Leylâ Uzun'un öğretmen-etiketleme kılavuzundan damıtılmış, bir yapının bağlamda
   deyim mi literal mi olduğuna karar verme çerçevesi — proje-içi kısaltma) etiketlendi;
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
   **Kullanılan model:** Claude (Anthropic), Claude Code alt-ajanları aracılığıyla, TDK
   tanımını girdi alan sabit bir üretim istemi ile (script: `data/prepare_synthetic_stage2_pairs.py`,
   istem metni scriptin içinde/alt-ajan çağrılarında, ayrı bir prompt dosyası yayınlanmadı).
   **Not:** Anthropic'in çıktılarını model eğitiminde kullanma şartları bu kart hazırlanırken
   bağımsız olarak teyit edilmedi — ticari kullanım düşünen kullanıcıların sağlayıcının güncel
   kullanım politikasını kendileri kontrol etmesi önerilir.

## Sonuçlar

**Bu bölümdeki script yolları** (`benchmark/*.py`, `data/*.py`) **ve "Deney X" notları** proje
kod tabanına işaret eder:
[github.com/iatagun/lemma-rule-based](https://github.com/iatagun/lemma-rule-based) (kamuya
açık). Tam deney günlüğü (her deneyin ne, neden, hangi sonuçla reddedildiği/kabul edildiği)
[`.claude/skills/idiom/SKILL.md`](https://github.com/iatagun/lemma-rule-based/blob/main/.claude/skills/idiom/SKILL.md)
dosyasında.

### Taban çizgisi: model gerçekten ne katıyor?

Aşağıdaki satır, sinir ağı hiç kullanılmadan yalnız TDK'nin yaklaşık 11k deyimlik sözlük listesiyle
(aynı `max_gap=2` gövde-eşleştiricisi) cümledeki idyomları bulmaya çalışan saf kural-tabanlı
bir sistemi gösteriyor — modelin eklediği değeri somutlaştırmak için:

**CASES**, 16 elle-seçilmiş "nokta-atışı" idyomatik/literal cümle çifti — repodaki
`benchmark/eval_idiom.py::CASES` sabiti, bağımsız bir dış kaynak DEĞİL, projenin kendi küçük
tanı seti (bkz. `run_cases()`).

| sistem | PARSEME F1 | CASES (16 vaka) | Çavuşoğlu duyarlılık (gevşek) | Çavuşoğlu yanlış-poz (gevşek) | Çavuşoğlu doğru-ayırt (gevşek) |
|---|---|---|---|---|---|
| **yalnız sözlük eşleştirici** (nöral yok) | 10.6 | 7/16 | %18.7 | %23.7 | **%11.6** |
| **DizgeBERT-Idiom v8** (tam boru hattı, `stage2_thresh=0.3`) | 63.48 (bkz. PARSEME bölümü) | 13/16 | %81.3 | %14.6 | **%68.2** |

**"Gevşek" ve "sıkı" iki ayrı ölçüm, karıştırmayın:** yukarıdaki tablo "gevşek" (cümlede
HERHANGİ bir span var mı, hedefte olması şart değil) tanımını kullanıyor — sözlük
eşleştirici için %18.7/%23.7/%11.6. AYRICA, "sıkı" (span hedef deyimle konumca çakışıyor
mu) tanımıyla ölçüldüğünde: sözlük eşleştirici 198 çiftin yalnız **11'inde** hedefi
konumlandırabiliyor (dar örtüşme — TDK'nin sözlük-biçimiyle cümledeki çekimli biçim arasındaki
gövde-eşleşmesi kısıtlı); bu 11 içinde duyarlılık VE yanlış-pozitif ikisi de **%100** — yani
konumlandırdığında bağlamdan tamamen kördür, idyomatik/literal ayrımı sıfır. Modelin asıl
katkısı sadece span BULMAK değil, bulduğu span'ler için bağlama duyarlı bir ayrım yapabilmesi.

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

**Tek-aşama karşılaştırması (2026-09-20, `benchmark/eval_cavusoglu_stage_ablation.py`) —
BU BİR "İKİ AŞAMA ŞART" KANITI DEĞİL, aşağıda açıklanıyor neden.** Aşama 1'i (v7, 3-gövde
ensemble) Aşama 2 KAPALIYKEN çalıştırıp tam boru hattıyla kıyasladık:

| ölçüm | Aşama 1 (stage2=False) | **+ Aşama 2 (v8, varsayılan eşik 0.3)** |
|---|---|---|
| idyomatik cümlede span işaretledi (duyarlılık) | %93.4 (95% GA %89.9–%96.5) | %81.3 (95% GA %75.8–%86.4) |
| literal cümlede **yanlış** span işaretledi | %66.7 (95% GA %60.1–%73.2) | %14.6 (95% GA %10.1–%19.7) |
| ikisini de doğru ayırt etti (çift-düzeyi, birincil metrik) | **%28.8** (95% GA %22.7–%35.4) | **%68.2** (95% GA %61.6–%74.7) |

Eşleştirilmiş fark: **+39.4pp** (95% GA +30.8pp – +47.5pp, sıfırı içermiyor).

**Kritik metodolojik sorun (2026-09-20, 3. dış inceleme ile fark edildi) — bu "aynı veri"
DEĞİL.** Aşama 1'in gördüğü literal→hep-O örnekleri `corpus_examples_glu.json`'dan geliyor
(Leipzig'ten madenlenen doğal cümleler). v8'in Aşama 2'si ise BAMBAŞKA, AYRI bir veri
kaynağıyla eğitildi: TDK'den seçilen 650 deyim için LLM'e YAZDIRILAN 1866 sentetik
dengeli-minimal-çift (`_synthetic_stage2_records.jsonl`, Deney Z). Aşama 1 bu sentetik
çiftleri HİÇ görmedi. Yani yukarıdaki %28.8→%68.2 farkı şunu gösteriyor: **"sentetik dengeli
minimal çiftlerle eğitilmiş AYRI bir Aşama-2 sınıflandırıcısı katkı sağlıyor"** — "iki
aşamalı mimarinin kendisi şart, tek aşama prensipte yetersiz" iddiasını KANITLAMIYOR, çünkü
Aşama 1'e HİÇ bu kaliteli/hedefli veri verilmedi ki adil kıyaslansın.

**Gerçek ablasyon SONUÇLANDI (2026-09-20).** Aynı 1866 sentetik kaydı (literal olanlar
hep-O'ya çevrilerek) Aşama-1'in standart eğitim verisine (TDK+PARSEME+corpus-glu, `--class-
weights --tdk-examples --corpus-glu`) EKLEYİP TEK-GÖVDE bir ELECTRA'yı sıfırdan yeniden
eğittik (`data/build_synthetic_stage1_ablation.py` + `training/train_idiom_bert.py`, 10
epoch, en iyi epoch 10, dev span-F1 66.32). Bu, hem ELECTRA-tek-gövde (Aslantaş&Güngör'ün
ilk eleştirisi) HEM sentetik-veriyi-gören (bu bölümün eleştirisi) bir Aşama-1 — gerçekten
adil bir "tek aşama vs iki aşama" testi:

| model | doğru-ayırt (Çavuşoğlu, n=198) |
|---|---|
| Aşama 1 tek başına, sentetik veri GÖRMEDİ (yukarıdaki ilk satır) | %28.8 (95% GA %22.7–35.4) |
| **Aşama 1 tek-gövde, sentetik veriyi GÖRDÜ (bu ablasyon, stage2=False)** | **%56.6 (95% GA %49.5–63.1)** |
| Tam iki-aşamalı boru hattı (v8, Aşama 1 + ayrı Aşama-2 sınıflandırıcısı) | %68.2 (95% GA %61.6–74.7) |

**Yorum:** sentetik veriyi Aşama-1'e eklemek tek başına büyük bir kazanç veriyor (%28.8→%56.6,
+27.8pp) — yani önceki %39.4pp'lik farkın ÇOĞU (yaklaşık %70'i) gerçekten bir VERİ etkisiydi,
mimari değil, tıpkı bu eleştirinin öngördüğü gibi. AMA ayrı bir Aşama-2 sınıflandırıcısı YİNE
DE ek bir kazanç sağlıyor (%56.6→%68.2, +11.6pp) — aynı veriyi gören tek-gövde bir modelin
ulaşamadığı bir yer. Bu ikisi eşleştirilmiş/aynı-198-cümle bir testte DOĞRUDAN kıyaslanmadı
(ayrı eval koşuları, yalnız nokta tahminleri var) — bu yüzden +11.6pp'nin kendisi için resmi
bir güven aralığı YOK, yalnız iki bağımsız GA'nın (49.5–63.1 vs 61.6–74.7) kısmen örtüştüğü
görülebilir. **Savunulabilir sonuç: bu mimaride ve bu veri rejiminde, ayrı bir Aşama-2
sınıflandırıcısı, aynı veriyi gören tek-gövde bir alternatiften ÖLÇÜLEBİLİR şekilde daha iyi
— ama önceki "neredeyse tüm kazanç mimariden geliyor" izlenimi YANLIŞTI, kazancın büyük
kısmı veriden geliyor.** Checkpoint arşivde (`best_idiom_tagger_vAblationSynth.pt`), kanonik
`best_idiom_tagger.pt` (vL) ve `corpus_examples_glu.json` dokunulmadan geri yüklendi.

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

### v7→v8 kıyası, sızıntı kontrolü, v7'nin kendi eşik-taraması

Özet: v8 (thresh=0.3) doğru-ayırt %68.2 vs v7 (thresh=0.5) %58.1, eşleştirilmiş fark +10.1pp
(95% GA +3.5pp – +16.7pp, anlamlı). v7'nin KENDİ eşiği de ayrıca tarandı (yanlılık kontrolü)
ve en iyisinin gerçekten 0.5 olduğu doğrulandı. Sızıntı kontrolü: v8'in 503 eğitim-deyimi
Çavuşoğlu'nun 198 eval-deyimiyle sıfır kesişiyor; düzeltilmiş "gerçekten görülmemiş" 59-deyim
dilimde de aynı yönde bulgu (+10.2pp) korunuyor. **Tam istatistiksel detay (McNemar dökümü,
GLU tanı seti geçişleri, v7 eşik-taraması tam tablosu, seen/unseen düzeltme notu):**
[CHANGELOG.md](CHANGELOG.md).

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
  `benchmark/analyze_ag_errors.py`). **Dürüst özet:** onların TR test setinde, onların KENDİ
  metriğiyle (seqeval entity-exact), DizgeBERT **açıkça geride** (F1=0.592 tümü, 0.414
  gerçekten görülmemiş 60 deyimde — bkz. üstteki özet tablosu) — onların in-domain
  modelleri 0.877-0.880. Gevşek/token-örtüşmeli metrikle fark küçülüyor (F1=0.795) ama
  span-düzeyi hata dökümü (153 olay: %50.3 tam eşleşme, %20.3 sınır-farkı — bunların 8/31'i
  "Allah ..." dua kalıpları, %15.7 gerçek kaçırma, %13.7 gerçek fazla-işaretleme) gösteriyor ki fark
  KISMEN sınır-konvansiyonu, kısmen gerçek. Onların backbone'unu (aynı encoder, kendi
  verileriyle 3 tohumla burada yeniden eğitildi, F1=0.898±0.031 — makul reprodüksiyon) bizim
  Çavuşoğlu benchmark'ımızda çalıştırınca doğru-ayırt %9.6 (literal örnek hiç görmedi — bu
  tek başına "iki aşama şart" kanıtlamaz). **Tam sayılar, Allah-kalıbı yüzdesi, reprodüksiyon
  detayı:** [CHANGELOG.md](CHANGELOG.md#aslantaş--güngör-kıyası--tam-detay).

- **Umut, Site, Arslan & Eryiğit (İTÜ, UBMK 2025), "Exploring Turkish Idiomaticity with
  LLMs".** Veri seti/kodu yayınlanmamış (IEEE Xplore, paywall) — doğrudan kıyas mümkün
  olmadı. Yerine, aynı ITU NLP ekosisteminden halka açık bir kaynak kullanıldı: **Dodiom TR**
  (Eryiğit, Şentaş & Monti, *Natural Language Engineering* 2022,
  [github.com/Dodiom/dodiom](https://github.com/Dodiom/dodiom), 6861 crowdsourced örnek / 36
  deyim). **Bu Umut et al.'ın verisiyle AYNI DEĞİL.** "Dış kaynak, görülmemiş" demiyoruz —
  daha kesin: 36 deyimin **34'ü zaten Aşama-1'in span eğitim havuzunda var**; Aşama-2'nin
  KENDİ etiketli havuzuyla örtüşme **1/36** — yani bu **Aşama-2 açısından görülmemiş deyim
  kimlikleri, doğal/crowdsourced cümlelerle** bir test. Sonuç (bkz. üstteki özet tablosu):
  dengelenmiş-doğruluk %77.4 (stage2 açık), ama Aşama-2 açık/kapalı farkı deyim-kümesi
  eşleştirilmiş bootstrap'ta **istatistiksel olarak anlamsız** (95% GA −1.5pp – +6.1pp,
  sıfırı içeriyor) — Çavuşoğlu'ndaki net +39.4pp'lik kazanç burada tekrarlanmıyor. **Tam
  sayılar, trade-off dökümü:** [CHANGELOG.md](CHANGELOG.md#dodiom-kıyası--tam-detay).

**Sonuç olarak "iki aşama şart" iddiası BENCHMARK'A BAĞLI VE KISMİ:** Çavuşoğlu'nun ölçtüğü
keskin, dengeli bağlam-değiştirme senaryosunda güçlü/anlamlı ama (düzeltilmiş ablasyona göre,
yukarıdaki "Ana metrik" bölümü) kazancın yalnız ~%30'u gerçekten mimariden geliyor (+11.6pp),
~%70'i veri kalitesinden (+27.8pp, tek-gövde bir modele aynı sentetik veriyi eklemek bile
işe yarıyor); Dodiom'un doğal, düşük-literal-oranlı dağılımında ise yönü aynı ama büyüklüğü
bu örneklemde istatistiksel olarak ayırt edilemiyor.

### Diğer kıyas noktaları (dikkatli okunmalı — görev tanımları farklı)

**Büyük LLM'lerle kıyas — temkinli okuyun, iki taraf da benchmark'a uyarlanmış.** Çavuşoğlu &
Çöltekin (2026, Table 2, "Detection" sütunu) büyük dil modellerinin kendi 200-idiom Türkçe
benchmark'larında ikili idyomatik-mi-değil-mi sınıflandırmasında şu doğrulukları bildiriyor:
Llama-3 70B-Instruct 0.614, Gemini 2.5-flash 0.609, GPT-4o 0.594, Llama-3 8B-Instruct 0.544,
Llama-3 3B-Instruct 0.521, Llama-3 1B-Instruct 0.496 (yaklaşık %50-61 aralığı, küçük modeller
şansa yakın). Bu, DOĞRUDAN karşılaştırılabilir bir sayı değil: (a) LLM'e deyim HEDEFİ verilip
yalnız ikili karar sorulmuş, bizim modelimiz span'i KENDİSİ bulup sonra karar veriyor (daha
zor bir görev); (b) bizim birincil metriğimiz (%68.2) ÇİFT düzeyinde ve daha katı ("ikisini de
doğru yap"), LLM'lerinki tek-cümle doğruluğu; (c) **bizim %68.2'miz AYNI 198-çift
benchmark'ında eşik-taranarak seçildi** (yukarıdaki "dev-seti gibi" çekincesine bakın) — LLM'ler
ise sıfır-atış (fine-tuning/eşik-ayarı yok), yani karşılaştırma bizim lehimize ekstra yanlı.
Daha adil bir kıyas için CÜMLE düzeyinde bakarsak: v8 idyomatik cümlede %81.3, literal cümlede
%85.4 (=100−yanlış-poz) doğru — ortalama **yaklaşık %83.4**, LLM'lerin ~%50-61 aralığının
üstünde, ama yukarıdaki üç çekinceyle (özellikle eşik-ayarı yanlılığıyla) birlikte okuyun.

**PARSEME 1.1 shared task kıyası:** en iyi sistem (SHOMA, nöral+CRF) tüm diller ortalamasında
%58.09 makro-F1 almıştı (bazı diller %23-32, Macarca/Romence %85-90). Bu **doğrudan
karşılaştırılamaz** — farklı edition (1.1 vs bizim 1.2), farklı dil kümesi, muhtemelen farklı
metrik, ve biz yalnız VID+LVC.full ile eğitiyoruz (PARSEME'nin tüm kategorileri değil). Yalnız
büyüklük mertebesi olarak: bizim v8 F1'imiz (63.48) bu aralığın ortasında, görevin zor olduğunu
teyit ediyor.

### PARSEME span-F1 (Aşama 1, `stage2=False`)

| test seti | kapsam | P | R | F1 |
|---|---|---|---|---|
| PARSEME test.cupt, **genel** (v8) | fiil-merkezli, bitişik+gap'li | değişken (sürüme göre) | değişken | **63.48** |
| PARSEME test.cupt, yalnız **gap'li span'lar** | süreksiz deyim/eşdizim | — | — | yaklaşık 30-45 |

Gap'li span'lar standart BIO ile **yapısal olarak asla yakalanamazdı** (ilk sürümde recall
garanti %0); iki-katmanlı şemayla artık kısmen (yaklaşık %30-45) kurtarılıyor.

**Ölçüm yöntemi notu:** yukarıdaki PARSEME metrikleri **exact-match** (span sınırları BİREBİR
tutmalı). Bu, sınır hatalarını (span'in bir kelime kısa/uzun bulunması) tam-kaçırma ile aynı
kefeye koyuyor; PARSEME için ayrı bir token-düzeyi/sınır-hatası dökümü henüz yapılmadı. Ancak
YÖNTEM artık var ve başka bir benchmark'ta (Aslantaş & Güngör) uygulandı —
`benchmark/analyze_ag_errors.py`, EXACT/BOUNDARY/MISS/SPURIOUS dökümü (yukarıdaki "İlgili
çalışmalar" bölümüne bakın: sınır-hatalarının ne kadarı gerçek kaçırma değil, konvansiyon
farkı olduğunu gösteriyor). Aynı script PARSEME'ye de uygulanabilir, henüz yapılmadı.

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
- **Hâlâ eksik ablasyonlar:** (a) yalnız-PARSEME ile eğitilmiş düz bir ELECTRA taban çizgisi,
  (b) veri kaynaklarının (PARSEME/TDK/Leipzig) tek tek katkısı. **Artık YAPILDI (2026-09-20)**:
  token-düzeyi/sınır-hatası dökümü (Aslantaş & Güngör testinde,
  `benchmark/analyze_ag_errors.py`) ve diğer Türkçe deyim çalışmalarıyla ölçülmüş, iki yönlü
  bir kıyas (bkz. "İlgili çalışmalar") — bu ikisi artık kartın eksiği değil.
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

v3'ten (iki-aşamalı boru hattına geçiş) v8'e (Aşama 2 sentetik minimal çiftlerle yeniden
eğitildi) kadar sürüm-sürüm ne değişti, tam liste: **[CHANGELOG.md](CHANGELOG.md#sürüm-geçmişi)**.
Özet: v5 Aşama-1'e ensemble ekledi, v6 bunu tek gövdeye damıttı, v7 üçüncü bağımsız gövdeyle
ensemble'a döndü (damıtım 3 öğretmene genelleşmedi), v8 yalnız Aşama-2'yi (sentetik veriyle)
yeniden eğitti — Aşama-1 v7'den beri değişmedi.

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

Leipzig Corpora Collection (Goldhahn, Eckart & Quasthoff, LREC 2012) — **CC BY 4.0**,
Türkçe cümle madenciliği için kullanıldı.

Bu kartta karşılaştırma için kullanılan dış kaynaklar:
- Çavuşoğlu, E. & Çöltekin, Ç. (2026). *An Idiom Benchmark for Turkish.* MWE 2026, s. 103–109.
  [ACL Anthology 2026.mwe-1.12](https://aclanthology.org/2026.mwe-1.12/) ·
  [github.com/coltekin/turkish-idioms](https://github.com/coltekin/turkish-idioms)
- Aslantaş, G. & Güngör, T. (2026). *A Unified Turkic Idiom Understanding Benchmark.*
  SIGTURK 2026, s. 38–51.
  [ACL Anthology 2026.sigturk-1.4](https://aclanthology.org/2026.sigturk-1.4/) ·
  [github.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark](https://github.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark)
- Eryiğit, G., Şentaş, A. & Monti, J. (2022). *Gamified Crowdsourcing for Idiom Corpora
  Construction.* Natural Language Engineering, 29(4), 909–941.
  [github.com/Dodiom/dodiom](https://github.com/Dodiom/dodiom)

## Lisans

Eğitim verisi (1) PARSEME Türkçe VMWE derlemi, edition 1.2 — **CC BY-NC-SA 4.0** (ticari
olmayan, paylaş-aynı-lisansla); (2) TDK Atasözleri ve Deyimler Sözlüğü — telif TDK'ye ait,
eğitim/araştırma amaçlı kullanılmıştır. Bu model PARSEME'nin lisansını miras alır —
**ticari kullanım için uygun değildir.** Araştırma, eğitim, kişisel/nonprofit projelerde
serbestçe kullanılabilir ve türetilebilir (paylaş-aynı-lisansla, CC BY-NC-SA 4.0 şartıyla);
ticari bir üründe veya ücretli bir hizmette kullanmadan önce PARSEME verisinin lisansını
ayrıca kontrol edin.
