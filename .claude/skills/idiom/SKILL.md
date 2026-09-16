---
name: idiom
description: Türkçe deyim (VID) / eşdizim (LVC) span tespiti — DizgeBERT-Idiom. GLU çok-ölçütlü etiketleme karar çerçevesi (bir yapı bu bağlamda deyim mi, eşdizim mi, terim mi, literal mi?) + deney günlüğü (v6-v14 tek-model kaldıraçları TÜKENDİ; iki-aşamalı detect→filter ÇALIŞIYOR; Aşama-1 checkpoint ENSEMBLE'ı (Deney O) çalıştı, YAYINLANDI v5) + stage-2 idyomatiklik sınıflandırıcısı iş akışı. Trigger — DizgeBERT-Idiom üzerinde çalışırken, deyim/MWE eğitim verisi hazırlarken/etiketlerken, idyomatik-literal ayrımı, stage-2 sınıflandırıcı, `find_span`/`prepare_*idiom*`/`filter_corpus_idiomaticity`, `/deyim` ya da `/idiom`.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
user-invokable: true
---

# DizgeBERT-Idiom — Deyim / Eşdizim Span Tespiti

`lemma-rule-based` reposunda ELECTRA tabanlı Türkçe deyim (VID) / eşdizim (LVC.full)
BIO span etiketleyici. **Yayınlandı** (`huggingface.co/iatagun/DizgeBERT-Idiom`).

- **YAYINLANDI v5 (2026-09-16): Aşama-1 ENSEMBLE (Deney O) — vE+vL birleşimi.** `idiom_data/
  best_idiom_tagger.pt` yerelde **vL** (Aşama 3 kanonik). HF paketi artık `best_idiom_tagger.pt`
  tek dosyasından ÜRETİLMİYOR — iki ayrı stage-1 checkpoint'i (`best_idiom_tagger_vE_leipzigglu.pt`
  + `best_idiom_tagger_vL_idiomcoverage.pt`) `--ensemble-ckpt` ile TEK pakete gömülüyor
  (`config.ensemble=True`, `encoder_b`/`tag_head_b`/`tag_head2_b`). HF/Space şimdi bu ensemble'ı
  çalıştırıyor. Detay: aşağıdaki "Deney O" bölümü.
- **YAYINLANDI v4 (2026-09-14): vE (gap=2 + corpus_examples_glu.json) — Çavuşoğlu
  doğru-ayırt %41.9→%55.6, TÜMÜYLE unseen-deyim diliminde (+15.2pp), seen diliminde birebir
  aynı — 9 stage-2 negatifinin sonunda ilk gerçek kaldıraç, ama stage-1 tarafında. Kullanıcı
  onayladı, HF'ye push edildi (`iatagun/DizgeBERT-Idiom`), Space (`iatagun/dizge-demo`)
  restart edilip smoke-test doğrulandı. **NOT: vE'nin veri kaynağı iddiası (aşağıda "zaten
  vetted") sonradan YANLIŞ çıktı — bkz. "Deney H" bölümü, düzeltme + takip deneyi.** Detay:
  aşağıdaki "2026-09-14 turu — TDK-stil vs Leipzig-stil" bölümü.**
- **Stage-1 checkpoint envanteri**: `best_idiom_tagger_v5_bigappy.pt` (kanonik-öncesi, arşivde
  güvende), `best_idiom_tagger_vE_leipzigglu.pt` (YAYINDA, HF), `best_idiom_tagger_vF_cleanglu.pt`
  (YEREL en güncel, temiz-only, henüz push edilmedi — şu an `best_idiom_tagger.pt` bu).
  ELECTRA (`dbmdz/electra-base-turkish-cased-discriminator`),
  bigappy-unicrossy 2-katman + Viterbi. `idiom_data/best_idiom_tagger_vB_gap2.pt` = Deney B
  (bounded-gap matcher) sonucu — PARSEME/TDK-test'te net kazanç AMA tam boru hattında
  (değişmemiş stage-2 v3'le) Çavuşoğlu doğru-ayırt %41.9→%39.9 GERİLEDİ → **PROMOTE EDİLMEDİ**,
  kullanıcı kararıyla v5 yayında kaldı. Detay: aşağıdaki "2026-09-14 turu".
- **Yayınlanan HF sürümü = v3 (iki aşamalı):** v5 span'leri + gömülü stage-2 idyomatiklik
  sınıflandırıcısı (`idiom_data/best_idiomaticity_clf_v3.pt`, ~880MB bundle, `config.stage2=True`).
  `predict_spans()` bitişik VID adaylarını süzer, güvenli literal'i eler; `stage2=False` ile kapatılır.

Tam deney günlüğü: `memory/dizgebert-idiom-project.md`. Deney disiplini: [[finetune-iteration]].
Yayınlama: [[hf-model-publish]].

## Bu skill ne için

1. **`glu_karar_cercevesi.md`** — bir yapının bu bağlamda deyim mi olduğuna KARAR VERME
   çerçevesi (Gülsün Leylâ Uzun, Ankara Üniv. öğretmen etiketleme kılavuzundan damıtıldı).
   Eğitim verisi hazırlarken, örnekleri gözden geçirirken, eval seti kurarken **bunu esas al**.
2. Aşağıdaki **deney dersleri** — nelerin denendiği ve neden işe yaramadığı. Tekrar deneme.

## Çekirdek ders: precision tavanı bir VERİ/ENCODER sorunu DEĞİL

v5 span-F1: PARSEME test ~69.6, TDK held-out ~72.8, **precision ~%64-71**. Dış bağlam-
bağımlılık testinde (Çavuşoğlu & Çöltekin) idyomatik/literal doğru-ayırt yalnız ~%37.

**Denendi, HEPSİ reddedildi (v6-v14, 10 deney):**

| deney | ne | sonuç |
|---|---|---|
| **v6** | ConvBERTurk-mC4 encoder | recall'a kaydı; dış doğru-ayırt −10 |
| **v7** | +TDK crawl verisi (frozen-split) | TDK held-out −3 |
| **v8** | +Leipzig 29.7k madenlenen örnek | precision çöküşü 63.8→53.5 |
| **v9-v13** | GLU-filtreli L→hep-O örnekleri, çeşitli D:L oranları | v12 (1:1 küçük) en dengeli ama bağımsız ayrımda kazanç yok; v13 (focus-L) PARSEME-P −3 |
| **Fikir 4 (v14)** | ayrı `B/I-VID-LIT` etiket sınıfı | görülmemiş deyimde VID-LIT recall %15 — genellemedi; Çavuşoğlu yanlış-poz %25→%39 |
| **recall-boost** | `--span-weight-mult 1.8/1.35` | span modeli olarak iyi (TDK F1 +2.4) ama iki-aşamada stage-2'yi boğuyor |

**Neden (tek-model):** her weak-supervision yöntemi tek ölçüte bakıyor — YÜZEY BİÇİM. Model
"bu kelimeler bu sırada = span" öğreniyor; idyomatik/literal ayrımını öğrenmiyor. 7-yönlü
softmax'ta güçlü B-VID yüzey prior'u yeni sınıf/sinyali bastırıyor. Tükenmiş kaldıraçların
tam listesi: [[finetune-iteration]].

## ÇALIŞAN yaklaşım: İKİ AŞAMA (Fikir 3, YAYINLANDI)

Stage-1 = yüksek-recall BIO (v5, değişmedi). **Stage-2** = ayrı ELECTRA gövdesi + span ilk⊕son
pooling → {literal, idyomatik}; bitişik VID adaylarını süzer (LVC + gap'li dokunulmaz).

- Eğitim verisi: `data/filter_corpus_idiomaticity.py` ile Leipzig derleminden madenlenen deyim
  cümleleri **elle** GLU rubric'iyle etiketlendi (`_corpus_sample_labels.tsv`, D/L/E; 910 D / 751 L).
  `--apply --balance` → `corpus_examples_glu.json` (train) + `corpus_minpair_test.json` (118
  görülmemiş deyim held-out) + `_holdout_idioms.json`.
- **v3 sınıflandırıcı** = alt 8 katman donduruldu + seçim metriği `(idyom_R+literal_eleme)/2`
  (overfit oyun kitabı: [[finetune-iteration]]). held-out literal-eleme %78→%82.
- Boru hattı vs v5-tek: **Çavuşoğlu yanlış-poz %25.3→%16.2, doğru-ayırt %37.4→%41.9, GLU 16→21/35.**
  PARSEME −2 (yapay — o sette literal kullanım yok).
- Pakete gömme: `train_idiom_bert.py --stage2-ckpt` → ~880MB bundle. Bkz. [[hf-model-publish]].

### Stage-2 tavanı da kırılamadı (2026-09-09) — TEKRAR DENEME

Çavuşoğlu doğru-ayırt ~%42 stage-2 tavanı. Denenip REDDEDİLEN:

| deney | ne | sonuç |
|---|---|---|
| **v4 / v4b** | LLM-ölçekli GLU etiketi (Sonnet-5 ikili D/N, 8k cümle / 3980 yeni deyim, `--new-idioms-only`), freeze 8/10 | PARSEME ALL −2.2/−2.9 (VID aşırı filtre), Çavuşoğlu doğru-ayırt düz/−1.5, GLU −3. Model temiz held-out'ta epoch ilerledikçe kötüleşiyor → LLM etiketleri ~%20 sınır gürültüsü (κ 0.57). **Kapsam per-etiket gürültüsünü yenmedi.** |
| **v5ctx** | stage-2 mimarisi: span ilk⊕son → `[CLS] ⊕ span-ortalama ⊕ ilk ⊕ son`, `Linear(4H,2)`; 8k LLM veri | Çavuşoğlu **birebir v3** — mimari ayrım ekseninde sıfır fark, darboğaz etiket kalitesi |
| **v5ctx_clean** | aynı bağlam mimarisi + orijinal ~975 temiz etiket | GLU minimal-çift ayırt 33→44 (ama 9-çift = gürültü), Çavuşoğlu yanlış-poz 16.2→13.6 AMA **doğru-ayırt her eşikte ~41'de sabit**, duyarlılık −3. Wash. |
| **gold f8/f10** (2026-09-10) | 601-kayıt elle GLU altın set (D/V/L/X yargı, `data/verdicts_to_gold.py`→`ingest_gold.py`, stage-2 1081D/968L) | Çavuşoğlu doğru-ayırt 41.9→40.9/40.4, yanlış-poz 16.2→18.2/17.2, PARSEME/CASES/GLU düz. **5. bağımsız doğrulama: elle titiz etiket de tavanı kırmıyor.** |
| **ctx f8/f10** (2026-09-10) | v5ctx tekrarı: head `+[CLS]+span-dışı ortalama` → `Linear(4H,2)`, altın veri | izole metrikte **birebir düz** (aşağı bak). v5ctx negatifini doğruladı, geri alındı. |

**Ders:** stage-2 tavanı (a) daha çok veri, (b) ikili çerçeve, (c) cümle-bağlamı mimarisi,
(d) elle titiz GLU etiketi ile kırılmıyor — tek-model tavanının (v6-v14) stage-2 karşılığı.

**Tavanın anatomisi (2026-09-10, `--mode stage2-iso` ile):** stage-2'yi pipeline'sız,
Çavuşoğlu 179 altın-span çiftinde skorla → **çift-içi sıralama %93** (v3/gold/ctx hepsi):
temsil, deyimsel kullanıma literalden yüksek idyomatiklik veriyor. Ama tek global eşikle
doğru-ayırt sadece ~%59 (@0.5 58.7). Pipeline ~%42 = üç kayıp: stage-1 span-recall (~17p,
görülmemiş deyimde bulamıyor) + eşik kalibrasyonu (%93 sıralama → %59 eşikli) + gerçek
stage-2 hatası (~7p). **Darboğaz "head bağlamı görmüyor" DEĞİL** (ctx eklemek düz) —
göreli sinyal güvenilir, mutlak eşik deyim-kimliğine göre kayıyor. Kaldıraç sırası:
(1) per-deyim/uyarlanır eşik (göreli %93'ü kullan, retrain yok), (2) stage-1 recall (ayrı).

`data/filter_corpus_idiomaticity.py` yeni bayraklar (`stage2-llm-labels` dalı, main'e merge
kararı açık): `--gate` (LLM'i frozen sette kıyasla), `--new-idioms-only` (kapsam örneklemesi),
`--ingest-llm` (append-only). GLU prompt'u İKİLİ (D/N) + `sözlük biçimi: {idiom}` satırda.
anthropic.com uçları: `temperature` at + `thinking:{type:disabled}` (yoksa boş içerik).

### Hâlâ açık (düşük beklenen değer)
- Stage-2 ~1k elle-etiketli örnekte overfit; literal kullanımların ~%16-18'i geçiyor.
- **per-deyim uyarlanır eşik — DENENDİ, NEGATİF (2026-09-10).** Sentetik per-deyim referans
  (span tek başına / "cümlede X var" şablonu) ile eşik. Bare-span prior'ı per-deyim
  midpoint'le sadece r≈0.30 korele → per-deyim eşik %60.9, global-en-iyi eşik %60.3'ün
  üstünde değil. Mutlak kayma çıkarım-zamanı sentetik referanstan kestirilemiyor.
- **adversarial deyim-kimliği silme (GRL) — DENENDİ, ZARARLI (2026-09-10).** GRL head span
  rep'inden deyim ID tahmin ediyor, ters-gradyan. λ 0.5/1.0, freeze 8. Sonuç: **çift-içi
  sıralama 93→91/87 DÜŞTÜ**, literal-cümle p(idyo) 0.28→0.37/0.44 YÜKSELDİ (model literali
  daha az tanıyor), doğru-ayırt @0.5 58.7→53.6/45.8. Deyim kimliğini silmek meşru sinyali
  de siliyor ("kafayı yemek"in mecazi kullanımını tanımak için deyim olduğunu bilmek şart).
  Ayrıca adversary seyrek (91 sınıf, ~2 örnek) → GRL gürültü enjeksiyonu gibi. 7. negatif.
  Geri alındı.
- **STAGE-2 KALDIRAÇLARI TÜKENDİ (7 negatif + 2 pilot, toplam 9).** Kalan tek gerçek kazanç
  pipeline'ın stage-1 tarafında (~17p span-recall, stage-2'den bağımsız). Stage-2 ~%93 sıralama /
  ~%59 eşikli = tek-örnek çıkarımın yapısal sınırı. v3 kanonik, yayında kalır.
- **2026-09-14 pilot turu — "neden başarılı, nasıl ölçeklenir" (gerçek API ile test edildi):**
  (a) **Kendinden-tutarlılık oylaması** (`filter_corpus_idiomaticity.py --votes N`): κ 0.581→0.606
  (n=150, gerçek ama mütevazı) — %90 oybirliği, hata çoğunlukla SİSTEMATİK (rastgele değil),
  oylama bunu çözemiyor. (b) **Hedefli-derin etiketleme** (`data/deep_label_idioms.py`, 25 deyim
  × 10 cümle, votes=3): kalibre edilebilir deyim 87→106, dış-benchmarkla örtüşen **3→22 (7 kat)**
  — veri-açlığı sorunu ÇÖZÜLDÜ. AMA per-deyim kalibrasyon hipotezi artık İSTATİSTİKSEL GÜÇLE
  REDDEDİLDİ: r(kalibrasyon,altın)=0.026/−0.241, `stage2-iso` doğru-ayırt global-eşik %81.8 >
  per-deyim %72.7 (n=22). **Deney F artık "yetersiz veri" değil "test edildi, çürütüldü"
  statüsünde.** Ayrıntı: proje hafızası.
- **`--mode stage2-iso`** (`benchmark/eval_idiom.py`) = güvenilir izole stage-2 tabanı
  (179 çift). 11-çift "sıkı" metriği ve 9-çift GLU minimal-çift ±gürültülüydü; bunu kullan.
- **Minimal çiftler / hard negative** (GLU `glu_karar_cercevesi.md`) hâlâ keskin eval sinyali.

## 2026-09-14 turu — stage-1 recall'a odaklan, sonuç: İLK gerçek kazanç (Deney B)

Plan: `~/.claude/plans/en-son-deyim-ve-clever-crab.md`. Önceki turun "stage-1 recall ~17p,
görülmemiş deyim" iddiası hiç doğrudan ÖLÇÜLMEMİŞTİ — bunu ölçüp hedefli yeni kaldıraçlar
denendi (sentetik-değil, gerçek-veri kalibrasyonu; katı-eşleştirme yerine sınırlı gevşetme).

| deney | ne | sonuç |
|---|---|---|
| **A** | seen/unseen deyim-kimliği ayrımı (`run_external --seen-idioms-file`) | **DOĞRULANDI, büyüdü**: doğru-ayırt seen %66.7 vs unseen %39.0 (~28p, tahmin edilen 17p'den büyük) |
| **C** | TDK sözlük-aday-fallback (`inference/lexicon_candidates.py`, `--lexicon`) | **REDDEDİLDİ — v8 deseni tekrarı.** Muhafazakâr varyantta bile PARSEME F1 −2.4, fp/tp oranı kötü. 9. tek-model tipi negatif. |
| **F** | gerçek-etiketli per-deyim eşik kalibrasyonu (`benchmark/calibrate_stage2.py`) | **İlk turda "veri-açlığı" (n=3, anlamsız) → hedefli-derin etiketleme pilotuyla (aşağı) n=22'ye çıkarıldı → hipotez GÜÇLE REDDEDİLDİ (r≈0, global eşik per-deyimden iyi).** 8. stage-2 negatifi, kesin. |
| **B** | sınırlı-boşluklu `find_span` (`max_gap` parametresi, permütasyon DEĞİL — sıralı + ara-söz toleransı) | **KARMAŞIK — PARSEME/TDK-test'te net kazanç ama tam boru hattında Çavuşoğlu %41.9→%39.9 geriledi → PROMOTE EDİLMEDİ.** Aşağıda detay. |
| **D** | katman-2 (bigappy gap-parçası) ağırlık ayarı (`--span-weight-mult2`) | **REDDEDİLDİ**: GAPLI F1 32.10→22.78 (−9.3) — ağırlık artışı seyrek sınıfı istikrarsızlaştırdı, recall bile DÜŞTÜ |

**Deney B detayı:** `find_span(seq, sent_stems, max_gap=0)` — `max_gap=0` ORİJİNAL katı-bitişik
davranışla bire bir aynı (geriye uyumlu varsayılan). `max_gap>0`: deyim gövdeleri SIRALI
kalır ama aralarına toplam `max_gap` kadar eşleşmeyen sözcük girebilir — **permütasyon değil**,
yalnız ara-söz toleransı (v7/v8'in "veri ekle" veya "fuzzy eşiği gevşet" desenlerinden farklı:
tek, dar, prensipli bir eşleştirme-kuralı değişikliği). Ham verim gap=0 %26.9→gap=2 %28.8
(gap=3'te doygun). **Tek-değişken kıyas (aynı güncel CSV/kod, yalnız max_gap farklı, aynı
dondurulmuş test dosyası):** TDK-test F1 52.94→**59.81 (+6.87)**, precision de YUKARI
(55.10→59.26) — v6/v7/v8/v10'un "recall için precision feda" deseni YOK. Mevcut stage-2 v3'le
(yeniden eğitilmeden) PARSEME ALL F1 67.60→**68.82 (+1.22)**, precision düz, recall +3.05 —
**dağılım-kayması riski gerçekleşmedi**. `find_span` hem TDK eğitim-verisi üretiminde hem
`run_external` eval'inde AYNI fonksiyon olduğundan bu tek değişiklik iki yeri birden düzeltiyor.
**Önceki "eşleştirme kuralı ASLA gevşetilmez, v7/v8 dersi" kuralı YANLIŞ genellenmişti** — o
ders "kontrolsüz veri/eşik gevşetmesi" içindi, PRENSİPLİ sınırlı-pencere gevşetmesi (tek
değişken, aynı frozen test'te ölçülen) PARSEME/TDK-test'te net kazanç verdi.

**AMA — kritik düzeltme: tam boru hattında PROMOTE EDİLMEDİ.** İlk değerlendirme (yalnız
PARSEME+TDK-test+vA0-tabanlı dış-benchmark kıyası) yanıltıcıydı — gerçek karar ölçütü
YAYINDAKİ v5 ile aday vB'nin ikisinin de DEĞİŞMEMİŞ stage-2 v3 ile FULL 198-çift Çavuşoğlu
sonucudur:

| metrik | v5+s2 (yayında) | vB+s2 (aday) |
|---|---|---|
| PARSEME ALL F1 | 67.60 | 68.82 (+1.22) |
| **Çavuşoğlu doğru-ayırt** | **41.9%** | **39.9% (−2.0, GERİLEME)** |
| Çavuşoğlu duyarlılık | 55.6% | 52.5% (−3.1) |

Stage-1-tek (stage2'siz) Çavuşoğlu'nda vB gerçekten iyiydi (aynı-CSV tabana göre +3.5) ama bu
DEĞİŞMEMİŞ stage-2 filtresinden geçince tersine döndü — **dağılım-kayması gerçekleşti**, ilk
turda "geçti" denen risk yanlış metrikte (PARSEME) test edilmişti. **Kullanıcı kararı
(2026-09-14): PUSH ETME.** `best_idiom_tagger.pt` v5'e geri yüklendi (hash doğrulandı),
`dizgebert_idiom_hf/` (push edilmemiş yerel export) silindi. Checkpoint'ler arşivde:
`best_idiom_tagger_vB_gap2.pt` (aday, promote edilmedi), `best_idiom_tagger_vA0_gap0.pt`
(temiz gap=0 taban). **Denendi (2026-09-14) — stage-2'yi vB'nin GERÇEK adaylarıyla yeniden eğitmek: DAHA KÖTÜ,
kapandı.** `train_idiomaticity_clf.py --align-stage1 <ckpt>` eklendi (altın span yerine o
checkpoint'in önerdiği en-çok-örtüşen adayı kullanır, önermezse örneği atar). vB, bu gold-
etiketli (Leipzig-madenli) havuzda örneklerin %66-76'sında HİÇ aday önermedi — TDK-test'teki
iyi recall bu cümle türüne genellemiyor. Kalan çok küçük/çarpık veriyle (363/184) yeniden
eğitilen stage-2: Çavuşoğlu doğru-ayırt 39.9%→**34.8%**, yanlış-poz **15.7%→25.3% (ikiye
katlandı)** — PARSEME F1 68.82→70.20 YÜKSELDİ ama YANILTICI (PARSEME'de literal karşı-örnek
yok, gevşek/az-filtreleyen stage-2 orada iyi görünür). Checkpoint `best_idiomaticity_clf_vB.pt`
arşivde, reddedildi. **Kanonik: v5 + stage-2 v3, DEĞİŞMEDİ, yayında.**

## 2026-09-14 turu (devam) — TDK-stil vs Leipzig-stil dengesizliği: POZİTİF, 9 stage-2
## negatifinden sonra İLK gerçek kaldıraç (stage-1 tarafında)

**Hipotez:** Deney B'nin takip denemesinde ortaya çıkan bulgu — vB (gap=2) stage-1, gold
Leipzig-madenli havuzda örneklerin %65-76'sında HİÇ aday önermiyor — stage-1'in TDK'nin
kendine özgü (sözlük/edebi) cümle üslubuna aşırı uyum sağlamış olabileceğini düşündürdü.

**Veri denetimi (önce):** Pozitif span sayısında TDK aslında AZINLIK (1394/6319 = %22),
PARSEME baskın (%78) — "TDK ağırlıklı" varsayımı ham sayıda yanlış. Ama bu, PARSEME'nin
Leipzig-tarzı doğal metne ne kadar benzediğini göstermiyor; asıl kanıt aşağıdaki ölçüm.

**Yeni ölçüm (ilk kez yapıldı):** `align_to_stage1()` ile v5'in KENDİSİNİN de aynı gold
havuzda aday-önerme oranı ölçüldü (önceden yalnız vB için biliniyordu): v5 train %59.4,
**test (görülmemiş deyim) %71.2** — vB'ninkine (%76.3) çok yakın. Yani sorun `--gap`
parametresinden bağımsız, v5'te de var, gerçek ve büyük.

**Tek-değişken deney (vE):** vB tabanına (gap=2 TDK verisi + PARSEME, `--class-weights
--tdk-examples`) tek şey eklendi: `--corpus-glu` (`idiom_data/corpus_examples_glu.json`,
4622 kayıt, YALNIZ idyomatik-D span'ler; `apply_manual()` held-out deyimleri bu dosyadan
HARİÇ TUTUYOR, o yüzden aşağıdaki held-out ölçümlerde sızıntı yok). Ön-kayıtlı eşik:
aday-yok oranı held-out'ta ≥15pp düşerse BAŞARI, PARSEME F1 >3p düşerse RED.

**DÜZELTME (2026-09-15, kritik):** bu 4622 kaydın "zaten elle-vetted GLU verisi" olduğu
iddiası YANLIŞTI. Dosyanın mtime'ı, STAGE-2 İÇİN REDDEDİLMİŞ 8k LLM-etiketleme çıktısıyla
birebir aynı — `corpus_examples_glu.json` o turda üretilmiş ve stage-2 v4/v4b reddedilip
"frozen veri geri yüklendi"ğinde YENİDEN ÜRETİLMEMİŞ, stale kalmış. 30-kayıtlık örneklemde
29/30'u güncel etiket havuzunda hiç yok. **Gerçek: kullanılan verinin ezici çoğunluğu
stage-2 için gürültülü sayılıp reddedilen LLM etiketiydi.** Bu vE'nin ÖLÇÜLEN sonuçlarını
geçersiz kılmıyor (bağımsız dondurulmuş setlerde ölçüldü) ama mekanizma açıklamasını
değiştiriyor — bkz. aşağıdaki Deney H.

**Sonuç — held-out (deyim düzeyinde tutulan, sızıntısız) gold havuz, vB→vE:**
aday-yok oranı **%76.3→%40.4 (−35.9pp)** — eşiğin çok üstünde.

**Tam boru hattı (vE + DEĞİŞMEMİŞ stage-2 v3), Çavuşoğlu 198 çift — Deney B'nin düştüğü
tuzağa (stage-1-tek iyi görünüp tam boru hattında geri dönme) burada düşülmedi:**

| metrik | v5+s2 (yayında) | vB+s2 (Deney B, red) | **vE+s2 (yeni)** |
|---|---|---|---|
| PARSEME ALL F1 | 67.60 | 68.82 | 66.16 |
| Çavuşoğlu duyarlılık | 55.6% | 52.5% | **78.8%** |
| Çavuşoğlu yanlış-poz | 16.2% | 15.7% | 26.3% |
| **Çavuşoğlu doğru-ayırt** | **41.9%** | **39.9%** | **55.6% (+13.7pp)** |
| CASES (16 vaka) | 11/16 | — | **14/16** |
| GLU vaka (35) | 21/35 | — | 20/35 (düz) |

**Kesin kanıt — seen/unseen kırılımı (Deney A'nın dondurulmuş `_bench_seen/unseen.json`
ile), asıl kazancın nereden geldiğini gösteriyor:**

| dilim | v5+s2 doğru-ayırt | vE+s2 doğru-ayırt |
|---|---|---|
| seen (21 çift) | 66.7% | **66.7% (birebir aynı)** |
| unseen (177 çift) | 39.0% | **54.2% (+15.2pp)** |

Kazanç **TÜMÜYLE unseen dilimde** — seen'de hiç değişim yok. Bu, ezber değil gerçek
genelleme: corpus-glu deyim-KİMLİĞİ kapsamını genişletmiyor (aynı TDK sözlük-deyim havuzundan
madenlendi), yalnız zaten bilinen deyimlere DOĞAL cümle-üslubu ÇEŞİTLİLİĞİ ekliyor — ve bu,
model hiç görmediği deyimlerde bile stage-1'in aday önerme davranışını genelleştiriyor.
~15pp'lik kazanç, 2026-09-10 tavan-anatomisi analizinin öngördüğü "~17p stage-1 recall" payına
neredeyse birebir denk düşüyor.

**PARSEME −1.44 (67.60→66.16) kabul edilebilir görülüyor**: PARSEME literal karşı-örnek
içermiyor (tüm span'ler idyomatik sayılıyor), o yüzden bu eksende ufak bir kayıp — CASES ve
Çavuşoğlu'ndaki büyük, bağımsız kazançla dengede. Precision tarafında gerçek bir maliyet var
(stage-1 raw PARSEME P 63.80→58.22, v6-v13'ün "recall-skew" desenine benziyor) ama F1/CASES/
Çavuşoğlu üçü de net pozitif.

**Durum: 9 bağımsız stage-2 negatifinden sonra ilk gerçek kaldıraç — ama stage-2'de DEĞİL,
stage-1'de. YAYINLANDI v4 (2026-09-14).** `idiom_data/best_idiom_tagger.pt` = vE (kanonik,
`best_idiom_tagger_vE_leipzigglu.pt` olarak da yedekli); v5 arşivde
(`best_idiom_tagger_v5_bigappy.pt`). MODEL_CARD.md v3→v4 güncellendi (yeni tablo, dürüst
precision/yanlış-poz notu), `dizgebert_idiom_hf/` yeniden export edilip round-trip doğrulandı
(yerel `.pt` sayılarıyla birebir), `push_idiom_hf.py` ile `iatagun/DizgeBERT-Idiom`'a push
edildi (440MB delta upload, content-addressed). Space (`iatagun/dizge-demo`) restart edildi
(`HfApi().restart_space`, `RUNNING_APP_STARTING`→`RUNNING`), `gradio_client` ile smoke-test:
"Çocuğu kaldırmak için el verdi" hâlâ doğru filtreleniyor (stage-2 sağlam), "Otobüs ... yol
aldı" artık YANLIŞ-POZİTİF olarak geçiyor — bu v4'ün ölçülmüş/açıklanmış yanlış-poz artışının
(%16→%26) canlıda doğrulanması, bir deploy hatası değil.

## Deney H (2026-09-15) — temiz-only ablasyon: mekanizma doğrulandı, büyüklük HACME bağımlı

vE'nin veri kaynağı hatası fark edilince (yukarı), ücretsiz bir ablasyon yapıldı: GÜNCEL
temiz etiket havuzundan (`_corpus_sample_labels.tsv`: 1237 D / 1062 L, LLM'siz)
`filter_corpus_idiomaticity.py --apply` ile `corpus_examples_glu.json` YENİDEN üretildi
(stale hâli `_corpus_examples_glu_STALE_llm8k.json` olarak yedeklendi) → **1337 kayıt
(826 D-span + 511 L→hepO)**. vF = vB tabanı + bu temiz veri, aynı reçete.

| metrik | v5+s2 | vB+s2 | vE+s2 (LLM-ağırlıklı, yayında) | **vF+s2 (temiz-only)** |
|---|---|---|---|---|
| PARSEME ALL F1 | 67.60 | 68.82 | 66.16 | 65.46 |
| Çavuşoğlu duyarlılık | 55.6% | 52.5% | 78.8% | 61.1% |
| Çavuşoğlu yanlış-poz | 16.2% | 15.7% | 26.3% | **18.7%** |
| **Çavuşoğlu doğru-ayırt** | **41.9%** | **39.9%** | **55.6%** | **46.0%** |
| CASES (16) | 11/16 | — | 14/16 | 13/16 |
| GLU vaka (35) | 21/35 | — | 20/35 | 22/35 |

**Mekanizma bağımsız doğrulandı:** temiz-only 1337 kayıt bile gerçek kazanç veriyor
(+4.1p doğru-ayırt) ve v6-v13'ün precision-feda desenini tekrarlamıyor (yanlış-poz yalnız
+2.5p — vF'te L-negatifleri de var, vE'nin D-only stale dosyasında yoktu). **Ama vE'nin
+13.7p kazancının yalnız ~%30'unu yakalıyor.** Sonuç: **LLM-ölçekli etiketleme stage-2 için
işe yaramasa da (κ~0.57-0.61 sınır gürültüsü idyomatiklik kararını bozuyor) stage-1 için
gerçek bir kaldıraç** — çünkü stage-1'in ihtiyacı yalnız "öbek nerede", LLM'in D/N kararı
yanlış olsa da öbek SINIRI genelde doğru kalıyor. "LLM etiketleme işe yaramıyor" genellemesi
stage-2'ye özeldi, stage-1'e yanlış taşınmıştı.

**Durum (2026-09-15):** `best_idiom_tagger.pt` şu an vF (`best_idiom_tagger_vF_cleanglu.pt`
olarak da yedekli, dürüst/temiz kaynaklı, daha dengeli FP). HF/Space ŞİMDİLİK vE'de kalıyor
(dokunulmadı) — kullanıcı kararı, vF yalnız yerel güvence. Sıradaki adım (ayrı plan): kasıtlı,
`--votes` ile kendinden-tutarlılık oylamalı, daha büyük ölçekli bir LLM etiketleme turu ile
vE'nin hacmini vF'in D+L dengesiyle birleştirmek.

## Deney I (2026-09-15) — DizgeBERT-Morph UPOS özellik enjeksiyonu (precision) — REDDEDİLDİ

Proje boyunca (v5→vF) hiç kıpırdamayan precision (~%58-71) için, erken literatür taramasının
önerdiği ama hiç denenmemiş "POS-özellik enjeksiyonu" fikri, ailenin yayında olan
`iatagun/DizgeBERT-Morph`'u kullanarak test edildi. **Reddedilen "arc-classification"dan
FARKLI**: Dep/Joint'e karar DEVREDİLMİYOR, yalnız Morph'un UPOS'u ELECTRA temsiline EK
özellik (`nn.Embedding(18,32)`, first⊕last'a concat) olarak ekleniyor — karar yine idiom
modelinin kendi temsilinden.

Tutarlılık ilkesi: UPOS her zaman Morph ÇIKARIMINDAN (altın değil, çünkü Space'te de yalnız
tahmin var) — `data/tag_idiom_upos.py` + paylaşılan `load_morph_upos_fn()` (hem toplu
etiketleme hem `eval_idiom.py::make_predictor`'ın çıkarım-zamanı desteği aynı fonksiyonu
kullanır). `--pos-features` bayrağı kapalıyken kod bit-birebir eski davranış (doğrulandı).

vH = vF + `--pos-features`, tek değişken:

| metrik | vF | vH | Δ |
|---|---|---|---|
| PARSEME F1 (raw) | 67.59 | 68.89 | +1.30 |
| PARSEME precision | 62.05 | 63.82 | +1.77 (eşik ≥+3) |
| TDK-test F1 | 65.38 | 59.62 | **−5.76** |
| Çavuşoğlu doğru-ayırt (tam, +s2 v3) | 46.0% | 46.0% | **0** |
| Çavuşoğlu doğru-ayırt (unseen, n=177) | 43.5% | 42.4% | −1.1 (gürültü) |

**RED.** PARSEME'de küçük gerçek kazanç (F1 de yükseldi, saf takas değil) ama asıl karar
metriği (Çavuşoğlu tam boru hattı) sıfır değişti, TDK-test gerçek gerileme verdi. UPOS
düzeyi muhtemelen deyim/düz ayrımı için çok kaba (aynı isim+fiil ikilisi idyomatik ve literal
kullanımda AYNI POS dizisine sahip). Literatür notu artık kapandı. `best_idiom_tagger.pt`
vF'e geri yüklendi; vH arşivde (`best_idiom_tagger_vH_upos.pt`), kullanılmıyor.

**Ortam notu:** bu makinede aynı süreçte 2-3 ELECTRA gövdesi (Idiom+Morph+stage-2) birden
yüklemek harness'ın sessiz "bellek düşük" kill'ini birkaç kez tetikledi. Çözüm: Morph
`dtype=float16`+CUDA ile yükleniyor; ağır `--mode all` yerine `--mode external`/`neural`
gibi tekli modları ayrı ayrı çalıştırmak daha güvenli.

## Aşama 3 (2026-09-16) — deyim-kimliği kapsamı: gerçek, bağımsız kazanç — vL, YENİ YEREL KANONİK

Aşama 2'nin negatif sonucu (yukarı) net bir teşhis verdi: `corpus_examples.json`'daki 5795
benzersiz deyimden **5272'si zaten frozen**, yalnız **526'sı hiç dokunulmamış** (1553 ham
cümle, `sample()`'ın per-idiom≤3 kapağından sonra 757 cümle). Bu, önceki turların hiç
denemediği AYRI bir eksen: bilinen deyimin doğal üslupta tanınması değil, hiç görülmemiş
deyim KİMLİKLERİNİN kapsanması.

`--new-idioms-only` ile bu 526 deyime ait 757 cümle, aynı 3-oylu Claude Code alt-ajan yöntemiyle
(3×260/237'lik parça × 3 oy = 9 ajan çağrısı, ekstra API ücreti yok) etiketlendi: D:447/N:310,
**tam-oybirliği %87.5** (Aşama 2'nin %81'inden yüksek — muhtemelen bu deyimler daha az
belirsiz/homonym-çakışmalı örneklerden oluşuyor). `--ingest-llm`: **757/757 kabul edildi,
526/526 yeni deyim** — Aşama 2'nin aksine SIFIR çakışma (hedef zaten bu 526'ydı).
`--apply`: `corpus_examples_glu.json` 7313→**7938 kayıt (+%8.5)**.

vL (vK tabanı + bu veri) eğitildi (epoch 5'te en iyi, F1 66.25; epoch 8'de bellek-düşük kill
tetiklendi ama en iyi checkpoint zaten diskteydi, sorun yok), stage-2 v3 DEĞİŞMEDEN ölçüldü:

| metrik | vF (taban) | vJ | vK | **vL** | vE (referans, yayında) |
|---|---|---|---|---|---|
| PARSEME ALL F1 | 65.46 | 65.50 | 64.60 | 64.34 (−1.12) | 66.16 |
| Çavuşoğlu tam (198) | 46.0% | 54.0% | 52.5% | **54.0%** | 55.6% |
| **Çavuşoğlu unseen (177)** | 46.0% | 51.4% | 49.7% | **52.0% (en iyi vF-soyu)** | 54.2% |
| CASES (16) | 13/16 | 11/16 | — | 12/16 | 14/16 |
| GLU vaka (35) | 22/35 | 22/35 | 20/35 | 21/35 | 20/35 |

**Ön-kayıtlı karar** (unseen ≥%54.2 VE PARSEME −≤3pp → PROMOTE): unseen %52.0, eşiğin
2.2pp altında → **PARTIAL yine**, ama vJ/vK/vF'in hepsini geçti — gerçek, bağımsız bir kazanç
(vK'nin "iyileşme yok" bulgusunun ardından ekseni doğruluyor: hacim değil kimlik-kapsamı
darboğazı çözülünce ölçülebilir ilerleme geldi). `best_idiom_tagger.pt` **vL'ye yükseltildi**
(vF değil artık) — dört varyant içinde en iyi ölçülen, en temiz kaynaklı (hiç stale/çakışan
veri yok) sonuç. HF/Space dokunulmadı, hâlâ vE.

**Durum (2026-09-16, güncellendi):** 526 deyimlik dokunulmamış havuz o an **tükenmişti**.
Aşağıdaki Leipzig genişletme turu tam bunu denedi — sonuç NEGATİF, bkz. "Deney N".

## Deney N (2026-09-16) — Leipzig derlemini 3M→8M cümleye genişletme: REGRESYON, vL kanonik kaldı

TDK'nin 11172 eşlenebilir deyiminden mevcut 3M-cümlelik Leipzig örneklemede yalnız 5795'i
bulunabilmişti (~%48'i hiç eşleşmemiş) — bu, "daha büyük ham derlem = daha çok yeni deyim
kimliği" hipotezini test etmek için doğal bir fırsattı. 6 ek Leipzig derlemi indirildi
(`tur_wikipedia_2016_1M`, `tur_news_2019/2020/2022/2023_1M`, toplam 3M→8M cümle; `tur_mixed_2012_1M`
mevcut değildi/404). `prepare_tdk_corpus_examples.py --cap 20` yeniden çalıştırıldı (stem-map
cache silinip yeniden kuruldu): kapsanan deyim 5795→6084 gibi görünse de bu ölçüm frozen-set
kaymasıyla kirli (bkz. not); gerçek ölçüt olan "hiç dokunulmamış deyim" sayısı **526→1133'e
çıktı** (6745 ham cümle, sample-cap sonrası 2402).

Bu 2402 cümle 30 Claude Code alt-ajanıyla (10×260/62 parça × 3 oy, ekstra API ücreti yok,
Aşama 3'teki aynı yöntem) etiketlendi: D:1183/N:1219, tam-oybirliği %84.8 — kalite göstergesi
iyi. `--ingest-llm`: 2383 kayıt / 1130 yeni deyim kabul edildi (**not**: 7614 kayıt "join-yok"
ile atlandı — ajanların tek-satır JSON dosyasını parça parça okurken bazı cümle metinlerini
harfiyen yeniden üretememesi; bu, aşağıdaki regresyonun olası bir nedeni). `--apply`:
`corpus_examples_glu.json` 7938→**9837 kayıt (+%24)**.

vM (vL tabanı + bu genişletilmiş veri, epoch 3'te en iyi F1 65.95) stage-2 v3 DEĞİŞMEDEN
ölçüldü:

| metrik | vL (taban, kanonik) | **vM** | Δ |
|---|---|---|---|
| PARSEME ALL F1 | 64.34 | 64.87 | +0.53 (düz/hafif iyi) |
| Çavuşoğlu tam (198) | 54.0% | 51.5% | **−2.5** |
| **Çavuşoğlu unseen (177)** | **52.0%** | **49.2%** | **−2.8 (REGRESYON)** |
| CASES (16) | 12/16 | — | — |
| GLU vaka (35) | 21/35 | 22/35 | +1 (düz) |

**Sonuç: RED.** PARSEME'de değişim yok/hafif pozitif ama asıl karar metriği (Çavuşoğlu
unseen) geriledi — vJ/vK ailesindeki "hacim tek başına yetmiyor" dersini doğruluyor, ama bu
kez daha çarpıcı: vL'nin kazandığı net ilerlemeyi kısmen SİLDİ. Olası nedenler (doğrulanmadı,
gelecek turlar için not): (1) yeni eklenen 5 derlem (özellikle çok-yıllı haber metni) daha
gürültülü/farklı üsluplu olabilir ve stage-1'in TDK-sözlük-üslubuna göre kalibrasyonunu
bozmuş olabilir; (2) %24'lük hacim artışının çoğu (1130 yeni deyim, göreli az örnekli) sınıf
dengesini seyrekleştirmiş olabilir; (3) 7614 "join-yok" kaybı, ajan-tabanlı etiketlemenin tek-
satır JSON okuma güvenilirliğinin bu ölçekte (2402 cümle, 10 batch) düştüğünü gösteriyor —
gelecekte cümle-metnini join anahtarı yapmak yerine idx-tabanlı eşleştirme daha güvenli olur.

`best_idiom_tagger.pt` vL'ye geri yüklendi (kanonik, DEĞİŞMEDİ). vM arşivde
(`best_idiom_tagger_vM_leipzig8m.pt`), kullanılmıyor. **Ders: kimlik-kapsamı ekseni Aşama
3'te (526 deyim, TDK-sözlük-üslubuna yakın orijinal 3-derlem havuzundan) işe yaradı ama Deney
N'de (1133 deyim, 8M'lik daha geniş/gürültülü havuzdan) işe yaramadı — kaldıraç deyim SAYISI
değil, kaynağın üslup/kalite TUTARLILIĞI. Ham derlem büyütme yönü şimdilik kapandı.**

**Kod kalıyor** (varsayılan kapalı, kanonik vF/vE'yi etkilemiyor): `data/tag_idiom_upos.py`,
`idiom_data/upos_labels.json`, tüm idiom JSON'larına eklenmiş `"upos"` alanı,
`IdiomLabelSpace`/`IdiomTagger`'daki `--pos-features` altyapısı.

## Deney O (2026-09-16) — Aşama-1 checkpoint ENSEMBLE'ı: YAYINLANDI v5, en iyi ölçüm

Deney N'nin ardından tek-checkpoint/veri-hacmi eksenleri tükenmiş görünüyordu. Hiç denenmemiş
bir eksen: diskte zaten duran dört stage-1 checkpoint'i (v5, vE, vF, vL) farklı corpus-glu
dilimleriyle eğitildi — görülmemiş-deyim kör noktaları örtüşmeyebilir. Yeni eğitim YOK; yalnız
çıkarım-zamanı birleştirme.

`benchmark/eval_idiom.py --ensemble ck1,ck2,... --ensemble-min-votes N`: her checkpoint kendi
aday span'lerini üretir, `(start,end,category)` anahtarıyla oylanır, çakışan span'ler arasından
en çok oy alan (eşitlikte en uzun) greedy seçilir. `min_votes=1` = birleşim (recall-odaklı),
`min_votes=N` (tüm checkpoint sayısı) = tam-oybirliği (precision-odaklı). Değişmemiş stage-2
v3 birleşik listeye uygulanıyor.

**Sonuçlar (unseen 177-çift Çavuşoğlu, gap=2, stage-2 v3 değişmeden):**

| kombinasyon | doğru-ayırt (unseen) |
|---|---|
| vL tek | 52.0% |
| vE tek | 54.2% |
| union(vL,vF) | 51.4% (vL tekten kötü) |
| agree(vL,vF) 2/2 | 35.0% (ÇÖKTÜ — recall imha) |
| majority(vE,vF,vL) 2/3 | 45.8% (kötü) |
| **union(vE,vL)** | **55.9% (yeni rekor)** |

vL+vF ikilisi İŞE YARAMADI (aynı soy, vF→vJ→vK→vL zinciri — korele hatalar; union gürültü
ekliyor, agree recall'ı katlediyor). **vE+vL işe yaradı** çünkü ikisi gerçekten bağımsız veri
dilimleriyle eğitildi (vE = LLM-hacim ağırlıklı stale veri, vL = temiz kimlik-kapsamı verisi) —
kör noktaları örtüşmüyor. **Ders: ensemble çeşitliliği checkpoint SAYISINDAN değil, eğitim
verisinin GERÇEKTEN BAĞIMSIZ olmasından geliyor** — aynı soyun ardışık sürümlerini
ensemble'lamak zarar veriyor.

**Tam boru hattı doğrulaması (union(vE,vL) + DEĞİŞMEMİŞ stage-2 v3):**

| metrik | vE (tek, önceki yayın) | vL (tek, yerel kanonik) | **union(vE,vL)** |
|---|---|---|---|
| PARSEME ALL F1 | 66.16 | 64.34 | 65.57 |
| Çavuşoğlu doğru-ayırt, tam (198) | 55.6% | 54.0% | **57.1% (yeni rekor)** |
| Çavuşoğlu doğru-ayırt, unseen (177) | 54.2% | 52.0% | **55.9% (yeni rekor)** |
| Çavuşoğlu doğru-ayırt, seen (21) | 66.7% | — | 66.7% (aynı) |
| CASES (16) | 14/16 | 12/16 | 14/16 |
| GLU vaka (35) | 20/35 | 21/35 | 20/35 |

PARSEME F1 hafifçe düştü (66.16→65.57, iki gövdenin birleşimi biraz fazla yanlış-pozitif de
katıyor: %26.3→%27.8) ama karar ölçütü (Çavuşoğlu doğru-ayırt) hem tam hem unseen dilimde
proje tarihinin en iyisi — 9 stage-2 negatifi + birkaç stage-1 PARTIAL turundan sonra ilk kez
hem tam-boru-hattı hem unseen'de net, sorgusuz bir rekor.

**Bedel — bunu bir önceki tüm stage-1 turlarından ayıran şey:** bu bir veri/mimari kazancı
DEĞİL, bir **çıkarım-zamanı maliyet takası**. Model paketi iki stage-1 gövdesi + bir stage-2
gövdesi taşıyor (~1.3GB, önceki ~880MB'den), ve `predict_spans()` Aşama 1'i artık İKİ KEZ
çalıştırıyor (~2× gecikme). Kullanıcı bu takası kabul edip yayınlamaya karar verdi.

**Pakete gömme (`modeling_dizgebert_idiom.py`):** `config.ensemble=True` ise ikinci tam stage-1
gövdesi (`encoder_b`/`tag_head_b`/`tag_head2_b`) kurulur; `predict_spans()` her iki gövdenin
span'lerini `merge_ensemble_spans()` ile birleştirir (eval_idiom.py'deki mantığın TEK kopyası —
orada ayrı iki checkpoint yükleyip ölçülmüştü, burada tek pakette gömülü aynı davranışı verir).
`train_idiom_bert.py --export-hf` artık `--ensemble-ckpt <ikinci-checkpoint>` alıyor (IdiomTagger
state_dict'ini `encoder_b.`/`tag_head_b.`/`tag_head2_b.` önekiyle aynı safetensors'a katar).
Flag kapalıyken (`config.ensemble=False`, varsayılan) davranış bit-birebir eski — v4 paketleri
etkilenmez.

**YAYINLANDI v5 (2026-09-16):** `iatagun/DizgeBERT-Idiom`'a push edildi (vE tabanlı export +
vL ensemble gövdesi + stage-2 v3, ~464MB delta upload). Round-trip doğrulandı (`--hf-repo`
yerel klasörle PARSEME/CASES/external/GLU sayıları birebir eşleşti). Space (`iatagun/dizge-demo`)
kodu da güncellendi (`idiom_tab.py` + `app.py`: v4→v5 rozetleri, ensemble açıklaması, güncel
sayılar) — bu bir kod değişikliği olduğu için restart değil, commit+push ile tam rebuild
tetiklendi (`RUNNING_BUILDING`→`RUNNING_APP_STARTING`→`RUNNING`). `gradio_client` smoke-test:
idyomatik "yol aldık" doğru işaretlendi, bilinen literal yanlış-pozitif ("Otobüs ... yol aldı")
hâlâ geçiyor — bu ölçülmüş/belgelenmiş %27.8 yanlış-pozitifin canlıda doğrulanması, deploy
hatası değil. `idiom_data/best_idiom_tagger.pt` (yerel tek-gövde kanonik) hâlâ **vL** —
ensemble yalnız HF paketinde, yerel tek-checkpoint iş akışını değiştirmedi.

## Deney J (2026-09-15, DEVAM EDİYOR) — vE'nin hacmini vF'in D+L dengesiyle birleştir

Fikir: vE'nin +13.7pp kazancının ~%70'i "hacim" kaynaklıydı (Deney H), ama vE'nin verisi
D-only'ydi (literal negatif yoktu, stale 8k LLM turundan). Zaten diskte, hiç kullanılmamış
`idiom_data/_corpus_idiomaticity_labels.jsonl` (8000 kayıt: 4622 D + 3378 N) içindeki **N**
kısmı `filter_corpus_idiomaticity.py`'nin ana etiketleme döngüsünde sessizce atılıyordu (yalnız
D tutulurdu). `--ingest-llm` ile bu havuz frozen sete katıldı — **ücretsiz** (yeni LLM çağrısı
yok): **7358 kayıt / 3742 yeni deyim eklendi, {D: 4289, L: 3069}**, 640 zaten-frozen-deyim
atlandı. `_corpus_sample_labels.tsv` artık idx 10169'a kadar; **idx <2173 elle, ≥2173 bu
LLM-turundan** (ingest'ten önceki insan-etiketli hâl `_corpus_sample_labels_HUMAN2811.tsv`
olarak yedekli — dosya adındaki "2811" değil, gerçek kesim noktası **idx 2173**'tür).

`--apply` çalıştırıldı (`--balance` KULLANILMADI): `corpus_examples_glu.json` **7222 kayda**
büyüdü (4369 D-span + 2853 L→hepO; beklenen 8000-8600'ün biraz altı — script'in kendi
örnekleme sınırı "deyim başına ≤3, toplam ≤6000" ham havuzdan seçim yapıyor, 10169 etiketli
kaydın tümünü değil). `corpus_minpair_test.json`/`_holdout_idioms.json` yedekleri:
`_corpus_minpair_test_vF.json` / `_holdout_idioms_vF.json` (vF hâli `_corpus_examples_glu_vF1337.json`).

**vJ eğitildi ve ölçüldü — SONUÇ: PARTIAL.**
`train_idiom_bert.py --class-weights --tdk-examples --corpus-glu --epochs 10` (best epoch 5,
span F1 ALL 67.18) → `best_idiom_tagger_vJ_llmbalanced.pt`. Stage-2 v3 DEĞİŞMEDEN, dondurulmuş
takımda:

| metrik | vF (taban) | **vJ** | vE (referans, yayında) |
|---|---|---|---|
| PARSEME ALL F1 | 65.46 | **65.50 (düz)** | 66.16 |
| Çavuşoğlu doğru-ayırt (tam, 198 çift) | 46.0% | **54.0%** | 55.6% |
| Çavuşoğlu doğru-ayırt (**unseen, 177 çift**) | 46.0%* | **51.4%** | 54.2% |
| Çavuşoğlu doğru-ayırt (seen, 21 çift) | — | **76.2%** (n=21, gürültülü) | 66.7% |
| CASES (16) | 13/16 | 11/16 | 14/16 |
| GLU vaka (35) | 22/35 | 22/35 (düz) | 20/35 |
| stage2-iso (sızıntı kontrolü) | — | **93.3% sıralama, %58.7/59.2 eşikli — v3 ile birebir, değişmedi** | aynı |

*vF'in ayrı seen/unseen kırılımı önceki turda ölçülmemişti; 46.0% tam-boru-hattı rakamı.

**Ön-kayıtlı karar** (unseen doğru-ayırt ≥%54.2 VE PARSEME F1 düşüşü ≤3pp → PROMOTE):
unseen **%51.4**, eşiğin **2.8pp altında** → **PROMOTE değil, PARTIAL.** PARSEME F1 pratikte
düz (+0.04), o kısım geçti; tek-değişkenli karar unseen eşiğinde takıldı.

**Önemli nüans — tam-198 rakamı yanıltıcı olabilirdi:** tam-boru-hattı (198 çift) doğru-ayırt
%54.0 — vE'nin %55.6'sına neredeyse eşit, yüzeysel bakışta PROMOTE gibi görünür. Ama bu,
seen dilimdeki büyük sıçramanın (66.7%→76.2%, n=21'de gürültülü olabilir) unseen dilimdeki
daha mütevazı kazancı (46.0%→51.4%) maskelemesinden kaynaklanıyor — tam olarak seen/unseen
ayrımının var olma nedeni. Ön-kayıtlı kural bu yüzden tam-198 değil unseen dilimi esas aldı.

**Aşama 2 ucuz kaçış GEÇTİ** (`--gate --votes 3 --gate-limit 200`): κ 0.606 pilotundan
**0.666**'ya çıktı, 4/4 kapı bandı GEÇ, oy-birliği %92 — kalite artışı gerçek.

**Aşama 2 tam turu — API BÜTÇESİ OLMADAN, Claude Code ajanlarıyla çalıştırıldı, SONUÇ:
İYİLEŞME YOK.** Kullanıcı ek API bütçesi ayıramadığı için `api.anthropic.com`'a gitmek yerine
15 Claude Code alt-ajanı (`general-purpose`, 5×300 cümlelik parça × 3 oy) aynı GLU D/N
rubric'iyle 1500 yeni cümleyi etiketledi (oturum bütçesi içinde, ekstra ücret yok).
Çoğunluk-oyu birleştirme: 1500 kayıt, D:708/N:792, tam-oybirliği oranı %81 (2 partide yalnız
2 oy kullanılabildi — 2 ajan denemesi ara dosyaya yazıp asıl etiket listesini teslim etmedi,
üçüncü denemede biri yine aynı hatayı yaptı, kabul edilip 2-oy ile devam edildi).

**Kritik olumsuz bulgu:** `--ingest-llm` bu 1500 kayıttan yalnız **145'ini** (84 D + 61 L,
141 yeni deyim) frozen sete katabildi — geri kalan 1355'i **zaten** ilk ingest turunda (8000
kayıt, Deney J) frozen hâle gelmiş 3742 deyimin biriyle çakıştı. `--apply` sonrası
`corpus_examples_glu.json` 7222→**7313 kayıt (+%1.3)** — istatistiksel olarak anlamlı bir
hacim artışı DEĞİL. vK (vJ tabanı + bu marjinal veri) eğitildi, sonuç vJ'nin GÜRÜLTÜ bandında,
gerçekte biraz daha kötü:

| metrik | vJ | **vK** |
|---|---|---|
| PARSEME ALL F1 | 65.50 | 64.60 (−0.90) |
| Çavuşoğlu tam (198) | 54.0% | 52.5% |
| Çavuşoğlu unseen (177) | 51.4% | **49.7% (−1.7, iyileşme yok)** |
| CASES (16) | 11/16 | 10/16 (yaklaşık, "20/35" GLU vakası referans) |

**Sonuç:** Aşama 2'nin kalite kaldıracı (κ 0.606→0.666) gerçekti, ama pratikte işe yaramadı —
çünkü ölçek sınırlayıcısı kaliteden ÖNCE hacimdi ve bu turda hacim neredeyse hiç büyümedi
(idiom-kimliği örtüşmesi yüzünden). **Ders: `--new-idioms-only` KULLANMADAN örneklenen yeni
cümleler, ilk ingest turu zaten deyim-kimliği havuzunu 1389→5131'e genişlettiği için, hızla
"zaten frozen" idiom'lara çarpıyor — asıl darboğaz artık idiom-KİMLİĞİ kapsamı, cümle-üslubu
çeşitliliği değil.** Bu, plandaki Aşama 3'ün (deyim-kimliği kapsamı, ayrı eksen) tam olarak
öngördüğü durum; Aşama 2 gibi "aynı havuzdan daha fazla cümle" yollarının artık getirisi
tükenmiş görünüyor. `best_idiom_tagger.pt` vF'e geri yüklendi (ne vJ ne vK promote edildi,
`best_idiom_tagger_vJ_llmbalanced.pt` / `best_idiom_tagger_vK_agentvotes.pt` arşivde).

## Kritik teknik notlar

- **`prepare_tdk_idiom_examples.py` split'i shuffle-slice** (per-key hash DEĞİL). Deyim sayısı
  değişince tüm train/dev/test yeniden karışır → versiyonlar-arası kıyas kirlenir. **Frozen-split
  modu eklendi**: mevcut `tdk_examples_{dev,test}.json` varsa dev/test cümle-metni düzeyinde
  sabitlenir, yeni deyimler yalnız train'e. v5 split yedeği: `idiom_data/_v5_tdk_split_backup/`.
- **Kıyas her zaman aynı eval dosyasında**: v5/v6/v7/v8 hepsi `idiom_data/tdk_examples_test.json`
  (frozen, 312 deyim) üzerinde ölçüldü. Yeni versiyon = aynı dosyada ölç.
- Eğitim `idiom_data/best_idiom_tagger.pt`'yi ÜZERİNE YAZAR. v5 yedekli; yeni run bitince
  hemen `cp best_idiom_tagger.pt best_idiom_tagger_vN_*.pt`, gerekirse v5'i geri yükle.
- `train_idiom_bert.py` bayrakları: `--class-weights --tdk-examples` (v5 reçetesi),
  `--corpus-examples` (Leipzig madenciliği), `--encoder <hf-id>` (encoder A/B override).
- Eşleştirme kuralı her yerde AYNI olmalı: `find_span` (`data/prepare_tdk_idiom_examples.py`;
  `stem()` snowballstemmer) hem TDK eğitim-verisi üretiminde hem `run_external` eval'inde
  kullanılıyor — biri değişirse diğeri de değişmeli. **Güncelleme (2026-09-14, Deney B):**
  "asla gevşetme" kuralı yanlış genellenmişti. `find_span(..., max_gap=N)` — SIRALI kalan ama
  aralarına sınırlı ara-söz toleransı olan (permütasyon DEĞİL) bounded-gap eşleştirme — TDK-test
  F1 +6.87 verdi, precision-korumalı. v7/v8 dersi hâlâ geçerli AMA dar kapsamı var: "kontrolsüz
  veri hacmi ekleme" veya "sınırsız fuzzy/threshold gevşetme" için — TEK, dar, prensipli bir
  eşleştirme-kuralı parametresini (max_gap, varsayılan 0 = eski davranış) tek-değişken olarak
  denemek FARKLI ve işe yaradı. `max_gap=2` şu an önerilen değer (gap=3'te getiri doygunlaşıyor).
- **Stage-2:** `modeling_dizgebert_idiom.py` içinde `config.stage2` ise `stage2_encoder` +
  `stage2_head` kurulur; `predict_spans(stage2=, stage2_thresh=, keep_literal=)`. Yalnız
  bitişik VID süzülür. `_LIT` kategorisi (Fikir 4 kalıntısı, inert) gerçek span sayılmaz.
- **Stage-2 eğitim held-out'u DEYİM düzeyinde**: `_holdout_idioms.json` (118 görülmemiş deyim);
  `train_idiomaticity_clf.py load_pairs` bunu okur (cümle-metni değil — focus-l sonradan
  aynı deyimden cümle ekleyince sızardı).
- **Kanonik span modeli `best_idiom_tagger.pt` = v5, DOKUNULMADI.** İki-aşama = v5 + stage-2;
  recall-boost turları span modeli olarak arşivde ama boru hattına konmadı.

## Komutlar

Scriptler repo kökünden çalıştırılır (`training/`, `data/`, `inference/` alt dizinleri).

```bash
# eğitim (v5 reçetesi)
python training/train_idiom_bert.py --class-weights --tdk-examples --epochs 10

# değerlendirme — 4 eksen
python benchmark/eval_idiom.py --local --checkpoint idiom_data/best_idiom_tagger.pt --mode all
python training/train_idiom_bert.py --eval --checkpoint <ckpt> --eval-file idiom_data/tdk_examples_test.json

# TDK verisi yeniden üret (frozen-split otomatik; snowballstemmer ile gövde-eşleştirme)
python data/prepare_tdk_idiom_examples.py

# Leipzig derlem madenciliği
python data/fetch_leipzig_tr.py
python -u data/prepare_tdk_corpus_examples.py --cap 8

# stage-2 idyomatiklik sınıflandırıcısı (Fikir 3) — --balance stage-2'yi ETKİLEMEZ (yalnız --apply gerekli)
python data/filter_corpus_idiomaticity.py --apply
python training/train_idiomaticity_clf.py --freeze 8 --dropout 0.3 --weight-decay 0.05 --epochs 14

# HF export (stage-2 gömülü) + push
python training/train_idiom_bert.py --checkpoint idiom_data/best_idiom_tagger.pt \
    --stage2-ckpt idiom_data/best_idiomaticity_clf_v3.pt --export-hf dizgebert_idiom_hf
python inference/push_idiom_hf.py
```

## Dosya envanteri

`data/fetch_parseme_tr.py`, `data/prepare_idiom_data.py`, `data/fetch_tdk_deyim.mjs`,
`data/prepare_tdk_idiom_examples.py` (+frozen-split, snowball stemmer, `--max-gap`, `tdk_idioms_by_split.json` dump),
`data/fetch_leipzig_tr.py`, `data/prepare_tdk_corpus_examples.py` (stem-map cache + sıkı eşleşme),
`data/filter_corpus_idiomaticity.py`, `data/prepare_glu_examples.py`,
`training/train_idiom_bert.py` (+`--span-weight-mult2`, +`--ensemble-ckpt` — Deney O), `training/train_idiomaticity_clf.py`
(+`--align-stage1`, `align_to_stage1()`),
`dizgebert_idiom/` (config+modeling+MODEL_CARD, kökte; `config.ensemble` + `encoder_b`/`tag_head_b`/`tag_head2_b` + `merge_ensemble_spans()` — Deney O),
`benchmark/eval_idiom.py` (+`--seen-idioms-file`, `--lexicon`, `--gap`, `--per-idiom-thresh-file`, +`--ensemble`/`--ensemble-min-votes` — Deney O),
`benchmark/calibrate_stage2.py` (Deney F, gerçek-veri per-deyim kalibrasyon + korelasyon raporu),
`data/deep_label_idioms.py` (hedefli-derin deyim etiketleme pilotu, `--votes` ile),
`inference/predict_idiom.py`, `inference/lexicon_candidates.py` (Deney C, TDK sözlük-fallback —
önerilmiyor ama altyapı kalıyor), `inference/push_idiom_hf.py`,
`tests/test_idiom_labelspace.py`. **Arşiv/kullanılmıyor**: `data/prepare_idiom_arc_data.py`,
`training/train_idiom_arc_bert.py` (reddedilen arc-classification).

Kaynak kılavuz: `GLU_Deyim_Etiketleme_Ogretmen_Kilavuzu.pdf` (repo kökü).
