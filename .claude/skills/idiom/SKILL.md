---
name: idiom
description: Türkçe deyim (VID) / eşdizim (LVC) span tespiti — DizgeBERT-Idiom. GLU çok-ölçütlü etiketleme karar çerçevesi (bir yapı bu bağlamda deyim mi, eşdizim mi, terim mi, literal mi?) + deney günlüğü (v6-v14 tek-model kaldıraçları TÜKENDİ; iki-aşamalı detect→filter ÇALIŞIYOR ve YAYINLANDI v3) + stage-2 idyomatiklik sınıflandırıcısı iş akışı. Trigger — DizgeBERT-Idiom üzerinde çalışırken, deyim/MWE eğitim verisi hazırlarken/etiketlerken, idyomatik-literal ayrımı, stage-2 sınıflandırıcı, `find_span`/`prepare_*idiom*`/`filter_corpus_idiomaticity`, `/deyim` ya da `/idiom`.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
user-invokable: true
---

# DizgeBERT-Idiom — Deyim / Eşdizim Span Tespiti

`lemma-rule-based` reposunda ELECTRA tabanlı Türkçe deyim (VID) / eşdizim (LVC.full)
BIO span etiketleyici. **Yayınlandı** (`huggingface.co/iatagun/DizgeBERT-Idiom`).

- **Stage-1 (span modeli) = v5, DEĞİŞMEDİ**: `idiom_data/best_idiom_tagger_v5_bigappy.pt` =
  `best_idiom_tagger.pt` ile aynı, ELECTRA (`dbmdz/electra-base-turkish-cased-discriminator`),
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
`training/train_idiom_bert.py` (+`--span-weight-mult2`), `training/train_idiomaticity_clf.py`
(+`--align-stage1`, `align_to_stage1()`),
`dizgebert_idiom/` (config+modeling+MODEL_CARD, kökte),
`benchmark/eval_idiom.py` (+`--seen-idioms-file`, `--lexicon`, `--gap`, `--per-idiom-thresh-file`),
`benchmark/calibrate_stage2.py` (Deney F, gerçek-veri per-deyim kalibrasyon + korelasyon raporu),
`data/deep_label_idioms.py` (hedefli-derin deyim etiketleme pilotu, `--votes` ile),
`inference/predict_idiom.py`, `inference/lexicon_candidates.py` (Deney C, TDK sözlük-fallback —
önerilmiyor ama altyapı kalıyor), `inference/push_idiom_hf.py`,
`tests/test_idiom_labelspace.py`. **Arşiv/kullanılmıyor**: `data/prepare_idiom_arc_data.py`,
`training/train_idiom_arc_bert.py` (reddedilen arc-classification).

Kaynak kılavuz: `GLU_Deyim_Etiketleme_Ogretmen_Kilavuzu.pdf` (repo kökü).
