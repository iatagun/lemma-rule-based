---
name: idiom
description: Türkçe deyim (VID) / eşdizim (LVC) span tespiti — DizgeBERT-Idiom. YAYINLANDI v8 (2026-09-19, Deney Z — Aşama 2 LLM-üretilmiş sentetik minimal çiftlerle yeniden eğitildi, Çavuşoğlu doğru-ayırt rekoru %65.2/%65.0, v7'den +7pp, 10 ardışık stage-2 reddinden sonraki İLK kazanç). ÖNEMLİ (2026-09-20): `_bench_seen/unseen.json` bayatlamıştı (Deney A'dan beri hiç güncellenmemiş), düzeltildi — asıl bulgu ayakta kaldı ama detay için "KRİTİK DÜZELTME" bölümüne bak. GLU çok-ölçütlü etiketleme karar çerçevesi + deney günlüğü + stage-2 idyomatiklik sınıflandırıcısı iş akışı. Trigger — DizgeBERT-Idiom üzerinde çalışırken, deyim/MWE eğitim verisi hazırlarken/etiketlerken, idyomatik-literal ayrımı, stage-2 sınıflandırıcı, `find_span`/`prepare_*idiom*`/`filter_corpus_idiomaticity`, `/deyim` ya da `/idiom`.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
user-invokable: true
---

# DizgeBERT-Idiom — Deyim / Eşdizim Span Tespiti

`lemma-rule-based` reposunda ELECTRA tabanlı Türkçe deyim (VID) / eşdizim (LVC.full)
BIO span etiketleyici. **Yayınlandı** (`huggingface.co/iatagun/DizgeBERT-Idiom`).

- **Dış-literatür çapraz karşılaştırma (2026-09-20) — DÜZELTİLMİŞ sürüm.** İlk turun
  "%79.5 F1" / "doğru-ayırt %77.4" gibi rakamları YANLIŞ METRİKLE hesaplanmıştı (bizim
  kısmi-örtüşmeye puan veren token-düzeyi ölçütümüz — onların `seqeval` entity-exact-match
  metriğiyle DOĞRUDAN kıyaslanamaz) ve deyim-kimliği örtüşmesi hiç ölçülmemişti. Dış bir
  incelemeden (kullanıcı payı) sonra düzeltildi:
  - **(1) DizgeBERT-Idiom'u Aslantaş & Güngör'ün (SIGTURK 2026,
    `github.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark`) TR test setinde (131
    cümle) çalıştırmak — ONLARIN metriğiyle (seqeval entity-F1):** tümü F1=0.592 (stage2
    kapalı) / 0.496 (açık), onların in-domain ELECTRA-tr/ConvBERT-tr'sinin (0.877/0.880)
    belirgin altında. **131 idiomun 71'i (%54) zaten bizim Aşama-1 havuzumuzda** — gerçekten
    görülmemiş 60 deyimlik dilimde F1 yalnız 0.414 (stage2 kapalı). "Sıfır-atış" yalnız
    CÜMLE düzeyinde geçerliydi, deyim-kimliği düzeyinde değil.
  - **(2) Onların backbone'unu (ELECTRA-tr, kendi verileriyle burada yeniden eğitildi —
    kendi test setlerinde seqeval F1=0.924, makul reprodüksiyon) bizim Çavuşoğlu'muzda
    çalıştırmak:** doğru-ayırt %9.6 (duyarlılık %87.4≈yanlış-poz %85.9). Bu, "salt-pozitif
    eğitilmiş bir tagger bağlam ayırt edemiyor" gözlemini destekliyor — ama TEK BAŞINA "iki
    aşama şart" genellemesini KANITLAMIYOR (onlar hiç literal örnekle eğitilmedi).
  - **Bağımsız, asıl ablasyon (`eval_cavusoglu_stage_ablation.py`):** Aşama 1'i (v7, AYNI
    veri — literal örnekler `corpus_examples_glu.json`'da zaten hep-O) Aşama 2 OLMADAN
    çalıştırınca Çavuşoğlu'nda doğru-ayırt **%28.8** (95% GA %22.7–35.4) vs tam boru hattı
    **%68.2**; eşleştirilmiş fark +39.4pp (95% GA +30.8–47.5, anlamlı). Bu, "bu mimari/veriyle
    tek aşama yetmiyor" iddiasının GERÇEK kanıtı (ensemble'a özgü, tek-ELECTRA-gövdeye
    otomatik genellenmez — not edildi).
  - **(3) Umut et al. (2025, UBMK, IEEE-arkalı) veri seti yayınlanmamış** — Dodiom TR
    (Eryiğit ekibi, 6861 örnek/36 deyim) yerine kullanıldı, AYNI VERİ DEĞİL. Deyim-KÜME
    bootstrap (etkin n=36): duyarlılık %69.9 (küme-GA %62.5–76.3), yanlış-poz %15.1
    (%12.1–18.1), **dengelenmiş doğruluk %77.4** (EŞLEŞTİRİLMEMİŞ, Çavuşoğlu'nun çift-düzeyi
    "doğru-ayırt"ıyla AYNI ÖLÇEK DEĞİL). 36 deyimin **34'ü Aşama-1 span havuzunda zaten var**
    (span-bulma açısından neredeyse hiç görülmemiş-deyim testi değil); Aşama-2'nin KENDİ
    etiketli havuzuyla örtüşme 0/36 (asıl test edilen bağlam-ayrımı sinyali için dış veri).
  - **İkinci düzeltme turu (2026-09-20, dış inceleme #2):** ilk turun bulguları büyük ölçüde
    ayakta ama beş nokta daha netleştirildi: (a) Çavuşoğlu benchmark'ı v5-v8 arası ~20 deneyde
    model-SEÇİMİ için kullanıldı, "dev-seti gibi" okunmalı, kör test değil. (b) Aşama 2'nin
    etkisi BENCHMARK'A BAĞLI — Çavuşoğlu'nda +39.4pp (anlamlı), Dodiom'da deyim-kümesi
    eşleştirilmiş bootstrap'la yalnız +2.4pp (95% GA −1.5–+6.1, SIFIRI İÇERİYOR, anlamsız);
    Dodiom'da Aşama 2 recall'ı (−15.6pp) yanlış-pozitif için (−20.5pp) takas ediyor, net
    etki belirsiz. (c) Dodiom "Aşama-2 açısından görülmemiş" olarak yeniden adlandırıldı
    (34/36 zaten Aşama-1 havuzunda). (d) AG test setinde entity-exact (0.592) vs gevşek/
    token-örtüşmeli (0.795) farkının kaynağı `analyze_ag_errors.py` ile incelendi: 153
    span-olayının %50.3'ü tam eşleşme, %20.3'ü sınır-farkı (çoğu "Allah ..." dua/beddua
    kalıpları — onlar tüm cümleyi span sayıyor, biz çekirdek yüklemi), %15.7 gerçek kaçırma,
    %13.7 gerçek fazladan-işaretleme. (e) AG-ELECTRA-tr reprodüksiyonu 3 tohumla tekrarlandı:
    F1=0.898±0.031 (0.924/0.864/0.906), onların 0.877'si bu aralıkta — makul reprodüksiyon.
    Tüm detay: `dizgebert_idiom/MODEL_CARD.md`. Yeni scriptler: `benchmark/analyze_ag_errors.py`,
    `benchmark/stats_utils.py::cluster_paired_balanced_acc_diff_ci`.
  - **Üçüncü düzeltme turu (2026-09-20, dış inceleme #3) — asıl eksik giderildi.** Önceki
    tek-aşama ablasyonu "aynı veri" DEĞİLDİ (Aşama 1 `corpus_examples_glu.json`'daki doğal-
    derlem L→hep-O'yu gördü, Aşama 2 ise v8'de TAMAMEN AYRI sentetik 1866-kayıtlık minimal-
    çift havuzuyla eğitildi — Aşama 1 bu sentetik veriyi hiç görmedi). Gerçek ablasyon
    (`data/build_synthetic_stage1_ablation.py`): aynı sentetik kayıtlar (literal→hep-O
    çevrilerek) standart Aşama-1 verisine eklenip TEK-GÖVDE bir ELECTRA sıfırdan eğitildi
    (10 epoch, en iyi epoch 10, dev F1 66.32). Sonuç: doğru-ayırt %28.8 (sentetik veri yok)
    → **%56.6** (95% GA %49.5–63.1, sentetik veri VAR, tek gövde) → %68.2 (tam iki-aşamalı
    boru hattı). **Yorum: önceki +39.4pp farkın ~%70'i (+27.8pp) VERİ etkisiydi, ~%30'u
    (+11.6pp) gerçekten mimariden geliyor** — "iki aşama şart" iddiası doğru ama daha
    mütevazı; eleştirinin öngördüğü gibi büyük ölçüde bir veri-confound'du. Ayrıca: (a) Dodiom
    Aşama-2 örtüşme kontrolü düzeltildi (yanlış dosya + boş sorgu bug'ı, "0/36"→doğru "1/36"),
    (b) v7'nin kendi eşiği tarandı (en iyisi gerçekten 0.5 çıktı, v8-v7 kıyası zaten yanlı
    değildi), (c) AG "Allah ..." kalıbı kesin sayıldı (8/31 BOUNDARY vakası), (d) 3-seed
    reprodüksiyon (F1=0.898±0.031), (e) CHANGELOG.md ayrıldı (kart 565→463 satıra indi),
    (f) revision= pin + v6 erişim notu, GLU/CASES tanımları, atıflar (Leipzig/Çavuşoğlu/
    Aslantaş/Dodiom) eklendi. Checkpoint arşivde: `best_idiom_tagger_vAblationSynth.pt`.
    Kanonik `best_idiom_tagger.pt` (vL) ve `corpus_examples_glu.json` DOKUNULMADAN geri
    yüklendi (hash doğrulandı).
  - **Dördüncü düzeltme turu (2026-09-20, dış inceleme #4) — dürüstlük/çerçeveleme
    düzeltmeleri.** (a) LLM-kıyası "(a)" maddesi YANLIŞTI: makalenin gerçek prompt'u
    ("Does the following Turkish sentence contain an idiom...") LLM'e hedef deyim VERMİYOR,
    LLM de span'i kendisi buluyor — düzeltildi, ayrıca negatif-sınıf kompozisyonunun makalede
    belirtilmediği not edildi (repo yalnız veri, kod yok, doğrulanamadı). (b) Ablasyondaki
    "~%30 mimari / ~%70 veri" ayrımı GEÇERSİZDİ — 28.8→56.6 adımında hem sentetik veri
    eklendi hem ensemble 3→1 gövdeye düştü, 56.6→68.2 adımında hem Aşama-2 eklendi hem
    ensemble 1→3'e çıktı; iki değişken aynı anda değişti, yüzdeler kaldırıldı, "farkın kaynağı
    ayrıştırılmadı" olarak yeniden yazıldı, temiz izolasyon için 2 öneri eklendi (henüz
    yapılmadı). (c) Üst özet tablosu dengelendi: Aslantaş stage2=True (0.496) da eklendi
    (yalnız stage2=False/0.592 vardı), Dodiom/Aslantaş satırlarına sürüm/eşik eklendi, "hiç
    ayar yok" ifadesi "bu setlerde ayar yapılmadı"ya düzeltildi (0.3 eşiği Çavuşoğlu'nda
    seçilmişti). (d) Çavuşoğlu'nun başlık metriği "gevşek" olduğu artık AÇIKÇA yazıyor +
    "sıkı" sayı da var (n=11, stem-eşleştirici kısıtı nedeniyle küçük örneklem). (e) İç
    çelişki giderildi: kart hem "Çavuşoğlu birincil KÖR kanıt" hem "dev-seti gibi kullanıldı"
    diyordu — ikinci ifade doğru, birincisi düzeltildi. (f) Eryiğit et al. yılı 2022→2023
    (Çavuşoğlu makalesinin kendi kaynakçasından doğrulandı). (g) Anthropic Usage Policy
    doğrudan okundu: "outputs to train an AI model... without prior authorization" açıkça
    yasaklanıyor — Aşama-2'nin Claude-üretimi sentetik verisi bu kapsama girebilir, kartta
    somut/nötr biçimde belirtildi (hukuki görüş değil). (h) `revision` kendi-kendine-referans
    hatası (bir commit kendi hash'ini içeremez) HF etiketleriyle çözüldü: `revision="v6"` ve
    `revision="v8"` artık kalıcı git tag'leri (doğrulandı, ikisi de doğru state'e çözülüyor).
  - Tam detay + tablolar: `dizgebert_idiom/MODEL_CARD.md` (kısa) + `dizgebert_idiom/CHANGELOG.md`
    (sürüm geçmişi + derin istatistiksel doğrulama), ikisi de HF'ye push edildi
    (commit 8e9a29a, HF etiketleri: v6, v8). **Düzeltme notu:** Anthropic kullanım-politikası
    yorumu ilk turda fazla geniş çıkarım yapmıştı ("train an AI model... without prior
    authorization" genel cümlesinden yasak sonucu) — Anthropic'in resmi Yardım Merkezi SSS'i
    dar-kapsamlı sınıflandırıcı/bilgi-çıkarma araçlarına AÇIK İZİN verdiğini gösteriyor
    (yasak olan genel-amaçlı sohbet botları); DizgeBERT-Idiom bu izin verilen kategoriye
    yakın, düzeltildi.
    Scriptler: `data/fetch_aslantas_gungor_tr.py`, `data/fetch_dodiom_tr.py`,
    `benchmark/eval_aslantas_gungor.py`, `benchmark/train_ag_electra_baseline.py`,
    `benchmark/reeval_ag_baseline_seqeval.py`, `benchmark/eval_ag_baseline_on_cavusoglu.py`,
    `benchmark/eval_dodiom.py`, `benchmark/eval_cavusoglu_stage_ablation.py`,
    `benchmark/stats_utils.py`. **Ölçülmeyen/eksik kalan:** PARSEME'nin kendi VMWE
    lemma-kimlikleri deyim-örtüşme kontrolüne dahil edilmedi (format farkı).
- **EVAL DÜZELTME TURU (2026-09-22) — GLU kılavuzu yeniden incelendi, 2 puanlama hatası +
  1 rubrik çelişkisi bulundu; v9 kararı YENİDEN ÖLÇÜLDÜ ve DEĞİŞMEDİ.** GLU vaka skoru
  eşdizim zor-negatiflerinde LVC'yi hata sayıyordu (kendi yorumuyla çelişiyordu) → düzeltilince
  HERKES +4: v8 25/35→**29/35**, 900+terim adayı 24/35→**28/35**. Fark (1 puan) ve CASES
  (13/16 vs 12/16) AYNEN korundu — yani v8'de kalma kararı hatalı ölçüme dayanmıyordu.
  Ayrıca `söz vermek` rubrik çelişkisi PARSEME altınıyla çözüldü (→LVC). Detay: aşağıdaki
  "Eval düzeltme turu" bölümü.
- **ARAŞTIRMA TURU KAPANDI, v9 YAYINLANMADI (2026-09-20) — Deney AA: sentetik havuz 650→900
  deyime ölçeklendi, GLU terim-regresyonu kök-nedeni bulunup kısmen düzeltildi, KULLANICI
  KARARIYLA v8'de kalındı.** En iyi aday (900+terim, thresh=0.6): Çavuşoğlu doğru-ayırt %66.2
  (v8: %65.2), yanlış-poz %17.2 (v8: %21.2) — AMA CASES 12/16 ve GLU 24/35 (v8: 13/16, 25/35),
  2 gerçek deyim (kafa tuttu, söz aldım) kaçırılıyor. Kullanıcı bu bedeli kabul etmedi.
  Checkpoint'ler arşivde, HF/Space DOKUNULMADI. Detay: aşağıdaki "Deney AA" bölümü.
- **YAYINLANDI v8 (2026-09-19): Aşama-2 SENTETİK MİNİMAL ÇİFTLERLE yeniden eğitildi
  (Deney Z) — Çavuşoğlu doğru-ayırt rekoru %65.2 (tam) / %65.0 (unseen), v7'den +7pp.**
  10 ardışık stage-2 reddinden sonraki İLK gerçek kazanç: LLM'e doğal cümle etiketletmek
  yerine 650 TDK deyimi için dengeli idyomatik+literal cümle YAZDIRILDI, doğal derlem verisi
  tamamen atılıp yalnız bu sentetik havuzla eğitildi. Aşama 1 (v7, 3-gövde ensemble) hiç
  değişmedi. Detay: aşağıdaki "Deney Z" bölümü.
- **YAYINLANDI v7 (2026-09-19): Aşama-1 3-GÖVDE ENSEMBLE (Deney X) — vE+vL+vX3
  birleşimi, Çavuşoğlu doğru-ayırt rekoru %58.1 (tam) / %57.1 (unseen).** v6'nın tek-gövde
  distilasyonu üç öğretmene genelleştirilmeye çalışıldı ama BAŞARISIZ oldu (vL tabanından da
  kötü); kullanıcı kararıyla ham 3-gövde ensemble yayınlandı (~1.76GB, ~3× gecikme — v5'in
  kabul ettiği takasın bir adım ilerisi). Detay: aşağıdaki "Deney X" bölümü.
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

## Deney Y (2026-09-19) — morfosentaktik kanonik-biçim sapması özelliği (Fazly/Cook/Stevenson
## 2009): FP oranını gerçekten düşürdü ama doğru-ayırtı hareket ettirmedi — REDDEDİLDİ

Literatür taraması sonrası (`~/.claude/plans/swift-finding-token.md`), Deney I'in ("UPOS
enjeksiyonu çok kaba") teşhisini bir üst seviyeye taşıyan fikir test edildi: kaba POS dizisi
yerine, span'in NESNE bileşeninin hâl-eki/çoğulluk/belirlilik ve FİİL bileşeninin çatısının
idiomun KANONİK biçiminden sapıp sapmadığı — Fazly, Cook & Stevenson (2009)'un "verb-noun
idiomatic combinations come in fixed syntactic forms" bulgusu. DizgeBERT-Morph zaten bu
FEATS'i (Case/Number/Definite/Voice) tahmin ediyor; per-deyim öğrenilmiş istatistik yerine
(çoğu deyimin örneği az, gürültülü olurdu) dilbilimsel öncül kural kullanıldı: kanonik biçim
hâl-eksiz/belirsiz nesne + etken çatı.

**Yeni modül** `data/tag_idiom_morph_feats.py::load_morph_feats_fn`/`morph_deviation_vec` —
`tag_idiom_upos.py`'nin FEATS ikizi, ama UPOS'u da (nesne/fiil ayrımı için) taşıyor. Smoke-test
mekanizmayı doğruladı: "yol aldık" (idiyomatik) → vec [0,0,0,0], "yolu aldı" (literal, hâl
ekli) → vec [1,0,0,0] — tam projenin kendi kanonik yanlış-pozitif vakasında beklenen ayrım.

**Mimari** (Deney V'nin `compat_gap` deseniyle birebir aynı yol): `IdiomaticityClf(morph_feat=
bool)` → head girdisi `base_dim+4`; `morph_feat=False` iken eskisiyle bit-birebir aynı.
`span_p_literal_morph` (modeling_dizgebert_idiom.py, `span_p_literal_gap`'in ikizi) çıkarım-
zamanı kullanılıyor; `wrap_stage2` yalnız gerekince (`need_morph`) Morph modelini yüklüyor.

vL span modeli + DEĞİŞMEMİŞ stage-2 eğitim verisi + `--freeze 8 --dropout 0.3 --weight-decay
0.05 --epochs 14 --morph-feat` (v3'ün BİREBİR aynı reçetesi, tek değişken). Best epoch 3,
dev macro 73.7.

**Tam boru hattı (v7 stage-1 ensemble DEĞİŞMEDEN + vMorph stage-2):**

| metrik | v7 (v3 stage-2) | **Deney Y (vMorph)** |
|---|---|---|
| PARSEME ALL F1 | 66.02 | 64.26 (−1.76) |
| CASES (16) | 13/16 | 12/16 |
| **Çavuşoğlu doğru-ayırt (tam)** | **58.1%** | **58.1% (birebir aynı)** |
| **Çavuşoğlu doğru-ayırt (unseen)** | **57.1%** | **57.6% (+0.5, gürültü bandında)** |
| Çavuşoğlu yanlış-poz (tam) | 28.3% | **24.7% (−3.6, gerçek iyileşme)** |
| GLU vaka (35) | 20/35 | 21/35 |

**Ön-kayıtlı eşik (doğru-ayırt ≥+2pp) geçilmedi → REDDEDİLDİ.** Ama Deney I'den farklı bir
sonuç: yanlış-pozitif oranı gerçekten düştü (mekanizma bir şey yakalıyor — smoke-test'teki
tam ayrımı doğruluyor), yalnız bu, tüm 198 çiftte doğru-ayırtı hareket ettirecek kadar GÜÇLÜ
değil. Olası neden: 4-boyutlu kural-tabanlı sinyal yalnız "hâl eki VAR/YOK" gibi kaba bir
ikili ayrım veriyor — birçok idiomun literal kullanımı da hâl-eksiz kalabiliyor (Türkçe SOV
+ düşük-hâl-işaretleme bağlamlarında), yani sinyal DOĞRU ama SEYREK devreye giriyor.
Checkpoint arşivde (`best_idiomaticity_clf_vMorph.pt`), kanonik `best_idiomaticity_clf_v3.pt`
DOKUNULMADI, stage-1 hiç değişmedi. Kod kalıcı (`data/tag_idiom_morph_feats.py`,
`IdiomaticityClf(morph_feat=)`, `span_p_literal_morph`, `--morph-feat`) — ileride per-deyim
öğrenilmiş kanonik-biçim istatistiğiyle (Deney Z'nin sentetik verisi yeterince büyürse)
tekrar denenebilir, düşük öncelik.

## Deney Z (2026-09-19) — LLM-ÜRETİLMİŞ dengeli minimal çiftler: 10 stage-2 reddinden
## sonraki İLK GERÇEK KAZANÇ, projenin en büyük tek-sürüm sıçraması, YAYINLANDI v8

Deney Y'nin ardından, plan dosyasındaki (`~/.claude/plans/swift-finding-token.md`) ikinci
öneri denendi: 9 stage-2 negatifinin (P/Q/R/S/T/U/V/focal/LLM-ölçekli-etiketleme) hepsi AYNI
kök soruna dayanıyordu — LLM doğal-derlem cümlelerini D/L diye SINIFLANDIRIYORDU (κ~0.57-0.66
sınır gürültüsü), ve literal kullanım doğal metinde nadir olduğu için bu veri hep dengesizdi.
Literatür (EDM2025 "Bridging the Data Gap") farklı bir rejim öneriyordu: LLM'e doğal cümle
etiketletmek yerine, her deyim için dengeli D+L cümle YAZDIRMAK.

**Uygulama:** `data/prepare_synthetic_stage2_pairs.py` (yeni) — `--select` Çavuşoğlu'nun 198
GERÇEK eval çifti + frozen + held-out ile örtüşmeyen bir TDK deyim havuzu seçer, Claude Code
alt-ajanlarına (Aşama 2/3'teki aynı ücretsiz yöntem) dispatch için parçalar; her ajan bir
deyim listesi + TDK tanımı alıp 3 idyomatik + ≤3 literal cümle üretir (literal okuma anlamsızsa
boş bırakır — zorlama yok), STRICT JSON döner. `--build` üretilen cümlelerde deyimin
kelimelerinin gerçekten geçtiğini doğrular.

**Kritik keşif — span doğrulama gevşetilmeli:** katı stem-eşit `find_span` ile yalnız
%17 (101/605) kayıt kabul edildi — TDK örnek cümlelerinden farklı olarak SERBEST üretilen
cümleler snowball stemmer'ın atlamadığı çekim ekleri (özellikle "-yor" şimdiki zaman) yüzünden
stem eşleşmiyordu. `find_span_lenient` (önek-toleranslı eşleşme, kısa token'larda güvenlik
sınırı) eklenince kabul oranı %87'ye çıktı — **yalnız bu scriptte**, paylaşılan `find_span`
dokunulmadı.

**150-deyimlik pilot (6 alt-ajan, 406 kayıt) — tam boru hattı (v7 ensemble DEĞİŞMEDEN):**

| metrik | v7 (taban) | **Deney Z pilot (sentetik-yalnız)** |
|---|---|---|
| Çavuşoğlu doğru-ayırt (tam) | 58.1% | **62.1% (+4.0pp)** |
| Çavuşoğlu doğru-ayırt (unseen) | 57.1% | **61.6% (+4.5pp)** |
| Çavuşoğlu doğru-ayırt (seen) | 66.7% | 66.7% (birebir aynı — ezber değil genelleme) |
| yanlış-poz | 28.3% | 25.8% |
| GLU vaka (35) | 20/35 | 24/35 |

**Kritik ablasyon — "sentetik+doğal karışım" DAHA KÖTÜ:** aynı 406 kaydı 9694 doğal kayıtla
karıştırıp eğitmek doğru-ayırtı **56.6%/55.9%'a düşürdü** (tabanın bile altı) — küçük temiz
sentetik sinyal, büyük gürültülü doğal havuzda boğuluyor. **Sentetik-YALNIZ eğitim** (doğal
veri TAMAMEN atılıp yalnız sentetik havuzla) işe yaradı — dev macro'da EN KÖTÜ sonucu verdi
(67.5, doğal-veri varyantlarının 73'üne karşı) ama Çavuşoğlu'nda EN İYİ sonucu verdi. **Ders:**
Çavuşoğlu'nun kendisi de deliberate-inşa edilmiş dengeli minimal-çift bir test seti — sentetik
verinin dağılımı buna doğal-derlem-madenli dağılımdan daha yakın, dev-set (doğal dağılımdan)
bu yüzden YANILTICI bir seçim ölçütü. Bu, projenin "dev F1 artışı Çavuşoğlu genellemesini
garanti etmez" dersinin TERS yönünü de doğruluyor: düşük dev metriği yüksek gerçek-dünya
performansıyla bir arada olabilir, kaynak dağılımı hedefe yakınsa.

**Ölçeklendirme turu (kullanıcı kararıyla) — 750 deyim daha (30 alt-ajan dispatch edildi,
oturum API rate-limit'i 20'sinde vurdu, 20'si "failed" statüsüyle bile dosyasını YAZMIŞ
çıktı — kullanıcı "elimizdekilerle devam" dedi, 26/36 parça = 650 deyim, 1866 kayıt (1354 D /
512 L) ile devam edildi):**

| metrik | v7 (taban) | 150-deyim pilot | **650-deyim (ölçekli)** |
|---|---|---|---|
| Çavuşoğlu doğru-ayırt (tam) | 58.1% | 62.1% | **65.2% (+7.1pp, yeni rekor)** |
| Çavuşoğlu doğru-ayırt (unseen) | 57.1% | 61.6% | **65.0% (+7.9pp, yeni rekor)** |
| Çavuşoğlu doğru-ayırt (seen) | 66.7% | 66.7% | **66.7% (üçüncü kez birebir aynı)** |
| yanlış-poz | 28.3% | 25.8% | **21.2% (monoton iyileşme)** |
| GLU vaka (35) | 20/35 | 24/35 | **25/35 (en iyi)** |
| PARSEME ALL F1 | 66.02 | 64.27 | 63.48 (kabul edilebilir, ≤3pp bekçi içinde) |

**Kazanç veri hacmiyle MONOTON büyüdü** (150→650 deyim, her eksende aynı yönde ilerleme),
seen dilimi HER ÜÇ turda da birebir 66.7% kaldı — bu, ezber değil gerçek genelleme iddiasının
en güçlü kanıtı (rastgele bir turda tesadüf olabilirdi, üç bağımsız ölçümde aynı sabit değer
tesadüf değil).

**Sızıntı kontrolü (yayından önce, kullanıcı talebiyle):** eğitimde fiilen kullanılan 503
deyim (650 seçilenin 1866 kayda dönüşenleri) ile Çavuşoğlu'nun 198 eval deyimi (`_bench_seen.
json` + `_bench_unseen.json`) arasında kesişim **sıfır** — doğrulandı, iki ayrı liste tam
kümesi karşılaştırılarak.

## KRİTİK DÜZELTME (2026-09-20) — `_bench_seen.json`/`_bench_unseen.json` BAYATLAMIŞTI,
## Deney A/O/X/Z'nin "unseen dilimi" iddiaları için düzeltildi (asıl bulgular AYAKTA KALDI)

HF model kartı için harici bir düzenleme turu sırasında ("Leipzig cümleleri Çavuşoğlu'nun
177 unseen deyimiyle kesişmiyor mu, bir daha doğrula" sorusu) şu bulundu: `_bench_seen.json`/
`_bench_unseen.json` (21/177) **Deney A'da (2026-09-14) BİR KEZ hesaplanmış ve o günden beri
HİÇ YENİDEN HESAPLANMAMIŞTI.** Aradan geçen sürede (Aşama 2/3, Deney N, Deney J/K'nin ingest
turları) frozen deyim havuzu 1151→6928'e büyüdü — yani o zaman "unseen" olan birçok deyim
SONRADAN gerçekten eğitim verisine girdi ama statik split dosyası hiç güncellenmedi.

**Ölçüm:** vL'nin gerçek eğitim havuzunda (idx≤11070, D/L etiketli, vL'nin kendi held-out'u
hariç) 4616 deyim var; bunun **70'i** (`_bench_unseen.json`'daki 177'nin) ile örtüşüyor.
Kümülatif frozen havuz + TDK train split ile daha muhafazakâr bir birleşim yapılınca: 177
"unseen" deyimin **121'i** aslında eğitime-maruz kalmış çıktı — yalnız **57-59'u** gerçekten
hiç görülmemiş.

**Düzeltme:** `data/prepare_synthetic_stage2_pairs.py`'nin candidate-exclusion mantığı
kullanılarak (frozen ∪ TDK-train) yeni bir muhafazakâr split hesaplandı, `_bench_seen.json`/
`_bench_unseen.json` bu sürümle DEĞİŞTİRİLDİ (eskisi `_bench_{seen,unseen}_STALE_
pre20260920.json` olarak arşivlendi).

**Asıl bulgu ayakta mı?** EVET — v8'i düzeltilmiş, gerçekten-unseen 57-59 deyimlik dilimde
yeniden ölçünce: doğru-ayırt %67.8 (v7 aynı dilimde %59.3) — yaklaşık +8.5pp, ORİJİNAL
bulgudan (177'lik stale dilimde +7.9pp) BİLE BÜYÜK. Yani v8'in genelleme iddiası bağımsız
olarak doğrulandı; bayat olan yalnız RAPORLANAN alt-küme tanımıydı, asıl etki değil.

**Etkilenen geçmiş iddialar:** Deney A/O/X'in "seen/unseen" kırılımları da bu bayat split'e
dayanıyordu (Deney A'nın kendi %39.0 vs %66.7 orijinal bulgusu dahil). Bu deneylerin ANA
sonuçları (tam-198 doğru-ayırt rakamları) etkilenmez — yalnız "kazanç unseen dilimde
yoğunlaştı" biçimindeki YAN kanıt zayıflar. Geçmiş deney günlüğü metinleri düzeltilmedi
(tarihi kayıt olarak kalıyor) ama BU NOTA bakılmalı; yeni deneylerde seen/unseen kırılımı
gerekiyorsa `_bench_seen.json`/`_bench_unseen.json`'ın artık düzeltilmiş sürüm olduğu
unutulmamalı — ve büyüme devam ettikçe (yeni ingest turları) bu dosyalar YİNE bayatlayabilir,
periyodik olarak bu yöntemle (frozen ∪ TDK-train birleşimi) yeniden hesaplanmalı.

**Ders (genel, proje-ötesi):** "seen/unseen" ya da "train/test disjoint" gibi statik bir
kontrol dosyası, ALTINDAKİ VERİ HAVUZU BÜYÜMEYE DEVAM EDERKEN sabit kalamaz — append-only
ingest yapan her projede bu tür kontrol dosyalarının bir SON GEÇERLİLİK TARİHİ olduğunu
varsaymak gerekir, "bir kez hesapladım, bitti" değil.

## İKİNCİ DÜZELTME (2026-09-20) — Space smoke-test'leri YANLIŞ endpoint'i çağırıyordu,
## şimdiye kadarki tüm "gradio_client smoke-test: ... hâlâ geçiyor" notları GEÇERSİZ

Eşik güncellemesi (0.5→0.3) sonrası Space'te doğrulama yaparken şu bulundu:
`iatagun/dizge-demo` çok-sekmeli bir Gradio uygulaması (hece/g2p, bağımlılık, sözdizim, deyim
sekmeleri) ve BİRDEN ÇOK sekme aynı imzalı (`analyze(text, ...)`) fonksiyon tanımlıyor. Gradio
bunları otomatik numaralandırıyor (`/analyze`, `/analyze_1`, ... `/analyze_6`) — ve **bu
projenin şimdiye kadarki TÜM `gradio_client` smoke-testleri `api_name="/analyze"` kullanmıştı,
ki bu aslında HECE sekmesine ait**, deyim sekmesine değil (gerçek deyim endpoint'i
`/analyze_4`/`_5`/`_6`). Basit bir alt-dize kontrolü ("'yol al' in sonuç") her iki durumda da
GİRİŞ METNİNİN kendisini içerdiği için YANLIŞ POZİTİF doğrulama üretiyordu — model hiç
çalıştırılmadan "test geçti" görünüyordu.

**Etkilenen geçmiş iddialar:** v5→v8 arası HER "Space güncellendi, `gradio_client` smoke-test:
idyomatik doğru işaretlendi, bilinen literal ... hâlâ geçiyor" notu (Deney O, W, X, Z'nin
yayın adımlarında tekrarlanan bir cümle kalıbı) muhtemelen HİÇ gerçek modeli çağırmadı. Bu,
modelin o dönemlerdeki gerçek davranışını YANLIŞLAMIYOR (yerel `--hf-repo` round-trip
doğrulamaları HER ZAMAN doğruydu, gerçek modeli kullanıyordu) — yalnız Space'in CANLI
davranışının o smoke-testlerle hiç gerçekten doğrulanmadığı anlamına geliyor.

**Doğru kullanım:** `gradio_client.Client(...).view_api()` ile gerçek endpoint adları HER
ZAMAN önce listelenmeli; deyim sekmesi için doğru işaret CSS sınıfı kontrolüdür
(`"idiom-vid" in sonuç or "idiom-lvc" in sonuç`), alt-dize eşleşmesi DEĞİL (girdi metni zaten
aranan alt-diziyi içerebiliyor). Bu yöntemle 2026-09-20'de v8+0.3 eşiği canlıda DOĞRU şekilde
doğrulandı: idyomatik cümle işaretlendi, "otobüs yol aldı" artık işaretlenmiyor.

**YAYINLANDI v8 (2026-09-19).** Aşama 1 (v7, 3-gövde ensemble) HİÇ değişmedi; yalnız Aşama 2
yeniden eğitildi (`best_idiomaticity_clf_vSynthOnly650.pt`, doğal veri sıfır, yalnız sentetik).
- **HF push:** `iatagun/DizgeBERT-Idiom` commit `e11407f` (~1.76GB, stage-1 aynı, yalnız
  stage-2 tensörleri değişti, ~114MB delta). Round-trip hem yerel hem uzaktan (`config.json`
  indirilip `ensemble`/`ensemble_extra2`/`stage2` doğrulandı) kontrol edildi.
- **Space güncellendi:** "v7·3-gövde ensemble" → "v8", Aşama 2 açıklaması sentetik-veri
  anlatımına güncellendi, tam rebuild + `gradio_client` smoke-test (idyomatik doğru
  işaretlendi; bilinen "otobüs yol aldı" örneği hâlâ geçiyor — %21.2 kalıntı yanlış-pozitifin
  beklenen bir örneği, deploy hatası değil).
- **MODEL_CARD.md v7→v8:** yeni karşılaştırma tablosu (v3..v8), Aşama 2 eğitim anlatımı
  tamamen yeniden yazıldı, Kısıtlar bölümü güncellendi.
- **Kalıcı kod:** `data/prepare_synthetic_stage2_pairs.py` (`--select`/`--build`,
  `find_span_lenient`, kümülatif tur takibi — `_synthetic_candidates.json` sonraki turların
  aynı deyimi tekrar seçmesini önler), `training/train_idiomaticity_clf.py`
  (`load_synthetic`, `--synthetic-file`, `--synthetic-only`).

**Sonraki oturum için not (güncellendi 2026-09-20):** 650→900 deyim turu tamamlandı (aşağı
bak) — kazanç yönü hâlâ pozitif ama artık doygunlaşmaya başlamış olabilir (PARSEME/Çavuşoğlu
net iyi, ama 900-deyim stage-2'nin PARSEME/CASES rakamları terim-düzeltmeli sürüm için hâlâ
ÖLÇÜLMEDİ — bir sonraki oturumda önce bu tamamlanmalı). Kalan ~10061 dokunulmamış TDK deyimi
hâlâ mevcut, `_synthetic_candidates.json` kümülatif tutuluyor. Ayrıca **"eli kolu bağlanarak"
homonym-literal vakası hâlâ çözülmedi** (terim-negatifinden FARKLI bir kategori — deyimin
KENDİSİNİN literal/fiziksel okunuşu, bitki/terim adı değil; kendi hedefli hard-negative'i
gerekebilir).

## Faz 3 sondası (2026-09-23) — yüklem düşmesi hipotezi ÇÜRÜTÜLDÜ, ama yerine
## çok daha büyük bir kapsam kaybı bulundu: katı `find_span` TDK örneklerinin %73'ünü atıyor

Salt ölçüm — hiçbir boru hattı değiştirilmedi, eğitim yok. Yer-gerçeği TDK'nın KENDİ örnek
cümleleri (örnek, tanımı gereği o deyimin kullanımıdır; `find_span` bulamıyorsa KAÇIRMA'dır).
Leipzig derlemi yerelde olmadığı için bu kaynak kullanıldı — aslında daha temiz, çünkü
her cümlenin hangi deyime ait olduğu kesin.

**(1) Yüklem düşmesi hipotezi ÇÜRÜTÜLDÜ — madde kapatıldı.** TDK'nın %19.5'i (2188 deyim)
düşebilir bir yardımcı fiille bitiyor ve `find_span` tüm gövdeleri şart koştuğu için bunları
yüklemi düşmüş hâlde YAPISAL olarak bulamıyor (GLU eval'deki "Proje aylardır rölantide."
vakası bunu doğruluyor). AMA pratikte neredeyse hiç ısırmıyor:

| | örnek | kaçan | oran |
|---|---|---|---|
| yardımcı fiille biten deyimler | 1120 | 834 | %74.5 |
| diğer deyimler | 4068 | 2865 | %70.4 |

Fark yalnız **4.1pp** — yani yardımcı-fiilli deyimler diğerlerinden anlamlı ölçüde daha çok
kaçmıyor. Kaçan 834'ün içinde gerçek yüklem-düşmesi imzası (ad-kısmı var + yüklem konumunda)
taşıyan yalnız **15 vaka** (%1.8; tüm örneklerin %0.3'ü). Sözlük örnekleri deyimi tam biçimde
kullanıyor. **Opt-in eşleştirici yazmaya değmez, madde kapandı.**

**(2) Asıl bulgu — katı eşleşme kapsamın üçte ikisini atıyor.** Aynı ölçümde ortaya çıktı:

| eşleştirici | eşleşen örnek | oran | kapsanan farklı deyim |
|---|---|---|---|
| `find_span` (şu anki, katı) | 1400 / 5196 | %26.9 | 1325 |
| `find_span` + `max_gap=2` | 1497 / 5196 | %28.8 | 1421 |
| **`find_span_lenient`** (Deney Z'den, önek-toleranslı) | **3772 / 5196** | **%72.6** | **3546** |

**2.69× kayıt, 2.68× deyim kapsamı.** Doğrulama: gerçek boru hattı çıktıları
(`tdk_examples_train/dev/test.json` = 1394+43+53 = 1490) sondanın rakamıyla birebir uyuşuyor,
yani ölçüm sadık.

Kök neden Deney Z'de ZATEN bulunmuştu: Türkçe snowball stemmer çekim eklerinin çoğunu
atmıyor, katı gövde-eşitliği gerçek kullanımda tutmuyor. Deney Z bunu sentetik havuzda
çözmüş (%17 → %87) ama düzeltmeyi bilerek O SCRIPT'E hapsetmişti ("paylaşılan `find_span`
dokunulmadı"). Aynı kusur paylaşılan boru hattında hâlâ duruyor. Örnek yeni yakalananlar
(20/20 elle kontrol, hepsi doğru span): `zayıf düşmekten`, `gözüne kestirdiği`,
`çaba harcıyordu`, `payidar kalacaktır`, `göklere çıkarıyorlardı`, `ortaklık kurarsak`.

**ÖNERİ (yapılmadı, ayrı bir deney olmalı):** TDK boru hattına opsiyonel gevşek eşleşme
(`--lenient`) ekleyip Aşama 1'i yeniden eğitmek. **Ama peşinen kazanç varsayılmamalı** —
Deney N'de Leipzig 3M→8M genişletmesi REGRESYON vermişti; bu projede daha çok veri otomatik
olarak daha iyi değil. Ayrıca Aşama 1 yeniden eğitimi bu turun tüm stage-2 koşularından
pahalı. Gerekirse önce küçük bir pilot.

**Yan düzeltme — bayat kayıt:** proje notlarındaki "TDK 2.629 train / 313+313" rakamı
BAYAT (TDK sitesi yeniden tasarlanıp yeniden tarandıktan sonra güncellenmemiş). Güncel
gerçek: train 1394 / dev 43 / test 53.

## Deney AC (2026-09-23) — gevşek TDK eşleşmesi ile Aşama 1: tek gövdede ANLAMLI
## (+4.55pp, 3 tohum), yayındaki 3-gövde ensemble'da ERİYOR (+0.84pp) — promote YOK

Faz 3'ün önerisi uygulandı. `find_span_lenient` paylaşılan `prepare_tdk_idiom_examples.py`'e
taşındı (sentetik betik oradan içe aktarıyor, `--report` birebir aynı çıktı → davranış
korundu). `--lenient` yalnız `tdk_examples_lenient.json` yazar; frozen dev/test ve kanonik
dosyalar md5 ile doğrulandı, dokunulmadı. Frozen dev/test cümleleri train'den atılır; aynı
cümle birden çok deyimin örneğiyse span'ler birleştirilir (eskiden ikinci deyim O'ya
düşüyordu). TDK train **1394 → 3632 kayıt / 3452 deyim**. Yeni sınır genişlemesi YOK:
yeni span'lerin %94'ü deyim uzunluğunda, kalanı gerçek araya-söz kullanımları. Nadir
önek hatası var (`düzelttim` → "çulu düzmek").

`train_idiom_bert.py`'ye `--tdk-file`, `--seed`, `--out` eklendi (stage-1 o güne dek hiç
tohumlanmıyordu). vL reçetesi (`--class-weights --tdk-examples --corpus-glu --epochs 10`),
baz = katı TDK, tohum 1/2/3 × 2 koşul, hepsi v8 stage-2 ile:

| eksen | baz (ort) | gevşek (ort) | fark | tohum başına |
|---|---|---|---|---|
| **Çavuşoğlu doğru-ayırt, tek gövde** | %55.4 | %59.9 | **+4.55pp (GA +1.18/+8.25, ANLAMLI)** | +8.1 / +2.5 / +3.0 |
| Çavuşoğlu yanlış-poz | %14.6 | %14.1 | −0.5 | — |
| PARSEME ALL F1 | 61.2 | 61.9 | +0.7 | −0.7 / +1.5 / +1.4 |
| **TDK-test F1** | 58.5 | 55.6 | **−2.8 (3 tohumda da aynı yön)** | −1.9 / −5.0 / −1.6 |
| CASES / GLU | 13.3 · 28.7 | 13.7 · 29.0 | ~eşit | — |

Baz tohum 1 = %54.0, belgelenmiş vL ile birebir (reprodüksiyon tuttu). Sızıntı değil:
gevşek verinin TDK train'e yeni soktuğu 57 Çavuşoğlu deyiminde +7.0pp, dokunulmayan
141'de +8.5pp (tohum 1). Recall-skew deseni YOK (yanlış-poz artmadı). Tek tohum yine
yanılttı: tohum 1 tek başına +8.1 "anlamlı"ydı, gerçek büyüklük ~yarısı.

**Yayındaki ensemble'da (vE + vL→vLen_sN + vX3, v8 stage-2):** prod %65.2 (birebir
yeniden üretildi) → %65.7 / %66.2 / %66.2; 3-tohum ort **+0.84pp (GA −1.01/+2.86,
ANLAMSIZ)**, net kazanılan çift yalnız +1/+2/+2. Ensemble'ın birleşik duyarlılığı zaten
%84.8 — diğer iki gövde gevşek verinin getirdiği kapsamı çoktan sağlıyor. Yanlış-poz
%21.2'de sabit kalıyor.

**Karar: promote YOK.** Kanonik vL, v8 paketi, HF/Space dokunulmadı. Checkpoint'ler
arşivde: `best_idiom_tagger_vLen_s{1,2,3}.pt`, `best_idiom_tagger_vLbase_s{1,2,3}.pt`.
**Ders:** ensemble'da stage-1 recall artık darboğaz değil (%85); kalan hata yanlış-poz
(%21) = stage-2 tarafı. Gevşek veri yalnız tek-gövde (ucuz, 1× gecikme) bir sürüm
hedeflenirse değerli: tek gövde %59.9 vs ensemble %65.2, fark hâlâ büyük. Sonraki
stage-1 turlarında TDK verisi gevşek kurulabilir (tek gövdede zararsız-pozitif) ama
TDK-test'teki −2.8 nedeni bilinmiyor.

## Yanlış-poz analizi + hedef-konumlu metrik (2026-09-23)

Prod ensemble'ın 42 Çavuşoğlu yanlış-pozu tek tek döküldü (scratchpad betiği, commit
edilmedi). Elle sınıflama: **~34 gerçek stage-2 hatası** (hedef VID, stage-2 idyomatik
dedi), 3 metrik artefaktı (cümlede başka GERÇEK eşdizim bulunmuş: "rahat et", "dikkat
et", "koşu yap"), 2 hedef-dışı gerçek FP, 2 hedef-LVC (stage-2 LVC'ye bakmıyor), ~3
şüpheli altın ("Alıcı çıktı geldi" deyimin kendi anlamı, "anası ağladı" iki okumalı,
"ağzının lokması yok").
- Stage-2 bu literal vakaları SIRALIYOR ama eşik altında bırakıyor: hedef p(literal)
  medyanı literalde 0.22 (max 0.45 — hiçbiri 0.5'i geçmiyor), idyomatikte 0.06. Eşik
  0.3: 23 FP'nin 8'i yakalanır, 126 idyomatiğin 19'u kaybedilir — eşik yolu yine kapalı.
- İpucu (anlamlı DEĞİL, n=42): vücut-parçalı deyimler FP'lerin %52'si, doğru elenenlerin
  %37'si; sentetik stage-2 havuzunda L cümlesi olan deyimlerin yalnız %21'i vücut-parçalı.
  Literal okumaları fiziksel ("başımı ağrıttı", "boğazımı sıkıyor"). Sonraki aday: havuzu
  yeni veri üretmeden bu türe doğru yeniden dengelemek (3 tohum şart).

**Hedef-konumlu metrik (`eval_idiom --mode external`, üçüncü satır):** eski "sıkı" satır
katı `find_span` yüzünden 198 çiftin yalnız **11**'ini konumlayabiliyordu (çekim eki,
parantezli deyim metni, "ağladı." gibi yapışık noktalama). Yeni satır `find_span_lenient`
+ noktalama ayıklama ile **136/198** konumluyor (12/12 rastgele kontrol doğru; kalan 62 =
kısa gövde `al-` önek güvenliği, parantezli alternatifler, farklı fiil biçimi — güvenlik
kuralı bilerek gevşetilmedi). Eski iki satır birebir aynı (geçmişle kıyas korunur).
`--dump-hits` artık `sample_t/literal_t/both_t` da yazar; `compare_runs --field both_t`.

Prod ensemble, aynı 136 çiftte: yanlış-poz gevşek %19.1 → hedef-konumlu **%16.9**, ama
duyarlılık da %83.8 → %81.6 (gevşek metrik idyomatik cümlede başka bir span'i de "bulundu"
sayıyordu) → doğru-ayırt %66.2 → **%65.4**. **İki şişme birbirini götürüyor; başlık
metriği (gevşek doğru-ayırt) pratikte yansız — geçmiş kararlar etkilenmez.**

## Deney AB (2026-09-22) — minimal-çift SAFLIĞI: havuzu %40'a budamak Çavuşoğlu'nda
## +6.1pp (ANLAMLI) getirdi — PROMOTE ADAYI (v9), henüz yayınlanmadı

GLU kılavuzunun yeniden incelenmesinden çıktı: kılavuz minimal çiftleri "Aşama-3'ün TEK
sinyali" sayıyor, ama havuz bunu tutmuyordu. **Yeni veri ÜRETİLMEDİ** (kullanıcı kısıtı:
alt-ajanla veri üretimi yasak) — her şey mevcut veriden seçim/filtreleme.

**Teşhis (`--report`, kalıcı):** span doğrulamasından sonra 920 deyimin **594'ü (%65)**
D-only; kabul edilen D:L = 2.77:1; ölçekleme turu eğriliği ARTIRDI (tur1 2.64:1 → tur2
4.30:1, tur2'de deyimlerin %53'ü L-siz). Span doğrulayıcı L'yi D'den sert eliyor (%41.5 vs
%31.6) — yeniden sözcüklenen literal cümleler sıralı eşleşmeyi bozuyor, yani kalite filtresi
havuzu KENDİSİ D'ye eğiyor. Bu eğrilik hiç ölçülmemişti.

**AB-1 — yalnız çift oluşturan deyimler** (`--build --only-paired`, saf seçim):
920→**326 deyim**, 2515→**1498 kayıt**, D:L 2.77:1→**1.25:1**. Stage-2 v8'in kazanan
reçetesiyle yeniden eğitildi (`best_idiomaticity_clf_vPairedOnly.pt`, dev best macro 70.5).

**AB-2 — `--natural-l-only`** (sentetik havuzun tamamı + doğal derlemin YALNIZ L kayıtları,
doğal D atıldı, + `--exclude-eval-idioms` ile 156 eval deyimi çıkarıldı): D:L 0.41:1.
(`best_idiomaticity_clf_vNatLOnly.pt`, dev best macro 55.5.)

| eksen | v8 (yayında) | **AB-1 (aday)** | AB-2 |
|---|---|---|---|
| Çavuşoğlu doğru-ayırt | %65.2 | **%71.2** | %53.0 |
| Çavuşoğlu yanlış-poz | %21.2 | **%14.6** | %21.2 |
| PARSEME ALL F1 | 63.48 | **64.09** | 50.24 |
| PARSEME VID F1 | — | 55.88 | **9.14 (recall %5.5)** |
| CASES | 14/16 | **14/16 (eşit)** | 11/16 |
| GLU vaka | **29/35** | 28/35 | 24/35 |

**Eşleştirilmiş bootstrap (yeni altyapı: `eval_idiom --dump-hits` + `benchmark/compare_runs.py`):**
İlk (TOHUMSUZ) koşu: AB-1 − v8 = +6.1pp, 95% GA +1.0/+11.1 → anlamlı görünüyordu.

**⚠ 3-TOHUM REPRODÜKSİYONU BUNU ÇÜRÜTTÜ (aynı gün, kullanıcı talebiyle).** `--seed` bayrağı
eklendi (eğitim o güne dek HİÇ tohumlanmıyordu — geçmiş tüm tek-koşu kıyaslarında ölçülmemiş
bir tohum varyansı var demektir) ve aynı reçete 3 tohumla tekrarlandı:

| tohum | Çavuşoğlu doğru-ayırt | yanlış-poz | v8'e göre eşleştirilmiş fark |
|---|---|---|---|
| (tohumsuz ilk koşu) | %71.2 | %14.6 | +6.1pp (GA +1.0/+11.1) — ANLAMLI |
| 1 | %68.7 | %16.7 | +3.5pp (GA −1.5/+8.6) — **ANLAMSIZ** |
| 2 | %68.2 | %13.6 | +3.0pp (GA −2.5/+8.6) — **ANLAMSIZ** |
| 3 | %69.7 | %14.6 | +4.5pp (GA −0.5/+9.6) — **ANLAMSIZ** |
| **ortalama** | **%68.9 ± 0.8** | — | **+3.0/+4.5pp, hiçbiri anlamlı değil** |

**Yorum:** ilk koşunun %71.2'si dağılımın üst ucuydu; gerçek seviye %68.9 ± 0.8. Yön ÜÇ
tohumda da pozitif ve üçünde de kazanılan çift kaybedilenden fazla (16/9, 18/12, 18/9), ama
n=198 çift bu büyüklükteki (~+3.5pp) bir farkı sertifikalamaya YETMİYOR. Deney AA'daki
4-gövde adayıyla (GA sıfırı içerdiği için reddedilmişti) aynı durumdayız — dev macro üç
tohumda 70.5/70.4/70.4 ile çok sıkı olmasına rağmen.

**v8 REÇETESİ DE 3 TOHUMLA ÖLÇÜLDÜ (kullanıcı talebiyle) — asıl fark +6.1pp DEĞİL, +2.0pp.**
v8'in eğitim havuzu Deney AA'da kanonik dosyanın üzerine yazıldığı için repoda artık yoktu;
ham parçalardan `--raw-max-index 25` ile BİREBİR geri kuruldu (1866 kayıt, 1354 D / 512 L —
belgelenen rakamla aynı).

| reçete | tohum 1 | 2 | 3 | **ortalama ± sd** | tek-koşu kaydı |
|---|---|---|---|---|---|
| v8 (650 deyim, D:L 2.64:1) | %66.2 | %67.2 | %67.2 | **%66.87 ± 0.58** | yayındaki ckpt %65.2 |
| AB-1 (326 çift, D:L 1.25:1) | %68.7 | %68.2 | %69.7 | **%68.87 ± 0.76** | tohumsuz koşu %71.2 |

Tohum-eşleşmeli bootstrap: +2.5 / +1.0 / +2.5 pp — **üçü de ANLAMSIZ** (GA sıfırı içeriyor).
Üç tohumun ortalamasıyla çift-düzeyi bootstrap: **+2.02pp, 95% GA −0.51/+4.71 → ANLAMSIZ.**

**İlk ölçümün +6.1pp'si ~3 kat şişmiş çıktı:** yayındaki v8 checkpoint'i KENDİ dağılımının
ALT ucunda (%65.2 vs reçete ortalaması %66.87), AB-1'in ilk koşusu ise ÜST ucunda (%71.2 vs
%68.87). İki uç birbirine karşı ölçülmüş.

**METODOLOJİK DERS (bu projedeki geçmiş kararları da etkiliyor):** stage-2 tohum varyansı
(sd ~0.6-0.8, uçtan uca ~1.5pp) bu projenin karar verdiği fark büyüklükleriyle AYNI MERTEBEDE.
Deney AA'da v9 ±1-1.5pp'lik farklarla değerlendirilmişti — o farklar gürültü bandının içinde.
Bundan sonra ~2pp altındaki tek-koşu farkları KARAR SAYILMAMALI. Ayrıca yayındaki v8'in kendi
reçete ortalamasının 1.7pp altında olması şu tuzağı doğuruyor: v8'i AYNI reçeteyle yeniden
eğitmek bile ~1.7pp "kazandırır" — yöntem hakkında hiçbir şey söylemeden.

**Kritik: AB-1, v9'u öldüren recall bedelini ÖDEMİYOR.** CASES'teki 2 hatası v8'inkiyle
birebir aynı (`söz verdi` stage-1 kategori hatası, `dili uzundur` bilinen artefakt) — yani
`kafa tuttu` kaçmıyor. Deney AA'nın terim-negatifleri karar sınırını temkinli yöne itip
recall'a mal olmuştu; havuzu çift-saflığına göre BUDAMAK aynı precision kazancını
(yanlış-poz %21.2→%14.6) bu bedel olmadan veriyor. GLU'daki −1 `eli kolu bağlanarak`
FP'si — hiçbir sürümde çözülmemiş homonym-literal vakası, yeni gerileme değil.

**Üç nokta aynı eksende temiz bir eğri çiziyor (asıl bulgu):**

| havuz | D:L | Çavuşoğlu doğru-ayırt |
|---|---|---|
| AB-2 (L-ağır) | 0.41:1 | %53.0 |
| **AB-1 (dengeli)** | **1.25:1** | **%71.2** |
| v8 (D-ağır) | 2.77:1 | %65.2 |

Kazanç "daha az/daha çok veri" değil **dengeye yakınlık**. Ters-frekans sınıf ağırlıklandırması
(`train_idiomaticity_clf.py`) ÜÇ koşuda da açıktı — sorun kayıpta değil, hangi bağlamların hiç
GÖRÜLMEDİĞİnde: ağırlık, olmayan çifti yaratamıyor. **Bu, Deney AA'nın "TERİM zor-negatifi
eksikti" teşhisinin yerine geçen daha basit ve daha genel bir açıklama** — ve "10061 TDK
deyimine ölçekle" fikrinin yönünü değiştiriyor: kapsam değil, ÇİFT SAFLIĞI.

**GENELLEME KIRILIMI (kullanıcı sorusu — `--seen-idioms-file`, düzeltilmiş split):**

| model | seen (143 çift) | unseen (59 çift) | unseen yanlış-poz |
|---|---|---|---|
| v8 | %63.6 | %67.8 | %15.3 |
| AB-1 (tohumsuz ckpt) | %70.6 | %72.9 | **%8.5** |

İki gözlem: (a) kazanç HER İKİ dilimde de var (seen +7.0, unseen +5.1) — ezberin keskinleşmesi
değil. (b) İKİ modelde de unseen > seen — yani "görülmüş deyim" avantajı YOK, hatta ters.
Stage-2 açısından zaten tamamı görülmemiş: AB-1'in 326 eğitim deyimiyle Çavuşoğlu'nun 198
eval deyiminin kesişimi TAM SIFIR (`select()` bunu bilerek sağlıyor), yani %68.9'un tamamı
stage-2'nin hiç görmediği deyimlerde. UYARI: bu kırılım tohumsuz (üst uç) checkpoint'le
ölçüldü, mutlak değerler ~2pp şişkin; model-içi seen-vs-unseen kıyası etkilenmez.

**DURUM: promote EDİLMEDİ (2026-09-23 kesin karar).** Çift-saflığına göre budama küçük,
TUTARLI ama KANITLANMAMIŞ bir iyileşme veriyor: altı ölçümün (3 tohum × 2 reçete) hepsinde
yön pozitif, üç tohum eşleşmesinde de kazanılan çift kaybedilenden fazla (9/4, 11/9, 10/5),
ama n=198 çift +2pp'yi sertifikalamıyor. Deney AA'daki 4-gövde adayıyla aynı gerekçeyle
arşive alındı. Diğer eksenlerdeki avantaj (PARSEME +0.61, yanlış-poz −6.6pp, CASES eşit)
tek tohumla ölçüldüğü için aynı çekinceyi taşıyor.
Ayrıca Çavuşoğlu ~20 deneyde
model SEÇİMİ için kullanıldı ("dev-seti gibi okunmalı" çekincesi MODEL_CARD'da) — ama
PARSEME (+0.61) ve CASES (eşit) bağımsız eksenler ve kazancı çürütmüyor. Aşama 1 (v7 3-gövde
ensemble) DEĞİŞMEDİ; bu saf bir stage-2 takası, paket boyutu aynı. Kanonik
`best_idiom_tagger.pt` (vL), HF ve Space DOKUNULMADI.

Kalıcı kod: `prepare_synthetic_stage2_pairs.py` (`--report`, `--build --only-paired`),
`train_idiomaticity_clf.py` (`--natural-l-only`, `--exclude-eval-idioms`, `drop_eval_idioms`),
`eval_idiom.py` (`--dump-hits`), `benchmark/compare_runs.py` (eşleştirilmiş bootstrap).

## Eval düzeltme turu (2026-09-22) — GLU kılavuzunun yeniden incelenmesi:
## 2 puanlama hatası + 1 rubrik çelişkisi, v9 kararı yeniden ölçüldü (değişmedi)

Kılavuz (`glu_karar_cercevesi.md`) kodla/veriyle satır satır karşılaştırıldı. Faz 1 = yalnız
eval düzeltmesi + yeniden puanlama, hiçbir yeniden eğitim yok, hiçbir checkpoint'e dokunulmadı.

**Düzeltme 1 — eşdizim zor-negatifinde LVC hata sayılıyordu.** `prepare_glu_examples.py`
HARD_NEG_DIAG yorumu "LVC kabul edilebilir, VID = hata" diyordu, ama `glu_diagnostic_cases()`
bu vakaları `(…, None, None)` diye veriyor ve `eval_idiom._check` `phrase is None` için
HERHANGİ bir span'i fail sayıyordu — kod kendi yorumuyla çelişiyordu. Kılavuzun eşleme tablosu
da eşdizimliliği zaten B/I-LVC'ye eşliyor. `"!VID"` işareti eklendi (`_check`'te yalnız VID
span hata). Terim / tek-sözcük vakaları katı kaldı.

**Düzeltme 2 — `söz vermek` rubrik çelişkisi.** Kılavuzun zor-negatif bölümü "söz almak, söz
vermek, yol göstermek, ön ayak olmak = DEYİM" diyor; eşleme tablosu "söz/yol gibi anlam-
aktarımlı ad + fiil → B/I-LVC" diyor. Aynı örnekler, zıt etiket — ve bizim iki eval setimiz
iki farklı tarafı seçmişti (CASES: LVC, GLU: VID). PARSEME-TR altınıyla kesildi: `söz ver` =
LVC.full ×10 / VID ×1 → GLU eval **LVC**'ye düzeltildi. `ön ayak ol` PARSEME'de VID (kılavuzla
uyumlu), `söz al`/`yol göster` PARSEME'de hiç geçmiyor (kılavuzun VID'i korundu). Çözüm kuralı
kılavuz dosyasına yazıldı.

**Düzeltme 3 — `--glu-examples` ayak kapanı.** `glu_hard_examples.json`, GLU tanı setiyle AYNI
`PAIRS` listesinden üretiliyor; `prepare_glu_examples.main()` bu flag'le eğitmeyi ÖNERİYORDU.
Hiçbir yayınlanmış sürümde kullanılmamış (kontrol edildi), öneri kaldırıldı + iki tarafa uyarı.

**Yeniden puanlama (v7 3-gövde ensemble sabit, yalnız stage-2 değişiyor):**

| aday | GLU (eski) | **GLU (düz.)** | CASES (eski) | **CASES (düz.)** | GLU minimal-çift doğru-ayırt |
|---|---|---|---|---|---|
| **v8 (yayında, vSynthOnly650)** | 25/35 | **29/35** | 13/16 | **14/16** | **%67** |
| 900+terim, thresh=0.5 | 24/35 | 28/35 | 12/16 | 13/16 | %56 |
| 900+terim, thresh=0.6 | 24/35 | 28/35 | 12/16 | 13/16 | %56 |

**Sonuç: v9'u reddetme kararı hatalı ölçüme dayanmıyordu.** Düzeltme herkesi eşit kaldırdı
(+4), fark ve sıralama aynen korundu. Kazanılan asıl bilgi:
- GLU'nun gerçek seviyesi 29/35'ti; "başarısız" sayılan 6 vakanın 4'ü model doğru LVC
  üretirken yanlış puanlanıyordu. Geçmiş tüm GLU rakamları (Deney Z/AA dahil) bu yüzden
  ~4 puan düşük kayıtlı — sürümler arası KIYAS geçerli (hepsi aynı hatayı taşıyor), MUTLAK
  seviye değil.
- v8 ile adayın farkı tam olarak **2 kaleme** indi: `kafa tuttu` (CASES) ve `söz aldım` (GLU).
  İkisinin de kökü aynı ve zaten bilinen: `--synthetic-only` bu eval-deyimlerinin doğal-derlem
  kapsamını attı, `select()` de sızıntı önlemek için onları sentetik havuza sokmuyor.
- Aday eşdizim precision'ında GERÇEKTEN daha iyi (`görüş aldık` FP'si adayda yok, v8'de var) —
  Deney AA'nın terim-negatif düzeltmesinin ölçülmemiş bir kazancı.

**Düzeltme 4 (kullanıcı kararı) — CASES `muayene etti` altını.** `Doktor gözünü muayene
etti .` "serbest" kategorisindeydi ve altın span YOK'tu; model `muayene etti:LVC` buluyor.
Kılavuzun Aşama-2 karar tablosuna göre (bileşimsel EVET / anlam aktarımı HAYIR / kalıplaşma
EVET) "muayene etmek" tam olarak **EŞDİZİMLİLİK → B/I-LVC**, yani altın yanlıştı. Vakanın
ASIL amacı ("gözünü" çeldiricisine karşı göz-DEYİMİ ateşlenmemesi) korunacak şekilde aynı
`"!VID"` mekanizması verildi — VID hata, LVC kabul. İkisi de +1 aldı, fark yine 1.

**Kalan tek artefakt (düzeltilmedi, her iki adayı EŞİT etkiliyor):** `Çocuk küçük yaştan beri
dili uzundur .` — altın öbek `dili uzun`, model `dili uzundur` buluyor; `_check`'in alt-küme
testi çekim eki yüzünden fail veriyor. En küçük çözüm altın öbeği yüzey biçime (`dili
uzundur`) çekmek olur; `_check`'i gevşetmek DEĞİL — katılık, sınır hatalarını gizlememek için
bilerek konmuştu (bkz. fonksiyonun kendi yorumu).

## Deney AA (2026-09-20) — 4-gövde ensemble ablasyonu (mimari vs veri izolasyonu) + sentetik
## havuzu 650→900 deyime ölçekleme + stage-2 GLU-terim regresyonu kök-neden + düzeltme

Dördüncü düzeltme turunun kapattığı "karışık-değişken" sorununu (28.8→56.6→68.2 basamağında
mimari VE veri aynı anda değişmişti) temiz izole eval'lerle (yeni eğitim yok, mevcut
checkpoint'lerle) çözdü:

| kurulum | stage-2 | doğru-ayırt (gevşek) |
|---|---|---|
| vL tek, sentetik yok | kapalı | %43.9 |
| vAblationSynth tek (+sentetik, aynı mimari) | kapalı | **%56.6 (+12.7pp — temiz veri-etkisi)** |
| union(vE,vL,vX3) 3-gövde, sentetik yok | kapalı | **%28.8 (mimari TEK BAŞINA zararlı — union stage-2'siz sadece FP getiriyor)** |
| vAblationSynth tek + v8 stage-2 | açık | %61.1 (v8'in %65.2'sinden 4.1pp altında, 3× ucuz) |
| union(vE,vL,vX3,vAblationSynth) 4-gövde + v8 stage-2 | açık | %65.7 |

**Sonuç:** sentetik verinin katkısı gerçek ve mimariden bağımsız; ensemble'ın katkısı stage-2
olmadan aslında NEGATİF (union daha çok FP getiriyor, temizleyecek şey yok). 4-gövde adayı
tam doğrulandı (PARSEME F1 62.50 vs v8 63.48 [-0.98], CASES 13/16=13/16 eşit, GLU 25/35=25/35
eşit, eşleştirilmiş Çavuşoğlu farkı +0.5pp [95% GA -1.0/+2.5, **anlamsız**]) → **PROMOTE
EDİLMEDİ**, v7 (3-gövde) DEĞİŞMEDEN kaldı. `best_idiom_tagger_vAblationSynth.pt` arşivde.

**Sentetik havuz ölçekleme:** önceki turda seçilmiş ama rate-limit yüzünden hiç üretilmemiş
10 parça (batch 26-35, 250 deyim) Claude Code alt-ajanlarıyla (ücretsiz, aynı yöntem)
tamamlandı → sentetik havuz 650→**900 deyim, 1866→2475 kayıt (1848 D / 627 L)**. Stage-2
(v8'in kazanan reçetesi: `--synthetic-only`, doğal veri sıfır) bu genişletilmiş havuzla
yeniden eğitildi (`best_idiomaticity_clf_vSynth900.pt`, best macro 70.0):

| metrik | v8 (650-deyim stage-2, yayında) | 900-deyim stage-2 (aday) |
|---|---|---|
| PARSEME ALL F1 | 63.48 | **65.73 (+2.25)** |
| Çavuşoğlu doğru-ayırt | %65.2 | %66.7 (+1.5pp, GA -3.5/+6.6 — anlamsız ama yön tutarlı) |
| Çavuşoğlu yanlış-poz | %21.2 | %20.7 |
| CASES (16) | 13/16 | 13/16 (eşit) |
| **GLU vaka (35)** | **25/35** | **21/35 (-4, gerileme)** |

**GLU regresyonunun kök nedeni (satır-satır diff ile bulundu):** 4 vaka bozuldu, hepsi FP —
"eli kolu bağlanarak" (literal), "sigorta attı" (literal), "aslan ağzı" ve "deve dikeni"
(glu-terim: bitki adı, deyim DEĞİL). Sebep: `--synthetic-only` rejimi (Deney Z'nin kazanan
reçetesi) doğal-derlem verisini TAMAMEN atıyor — ve doğal-derlem etiketleme sürecinin GLU
rubric'i (`glu_karar_cercevesi.md` / `filter_corpus_idiomaticity.py` TASK metni) "TERİM /
ÖZEL AD / bitki-canlı adı"nı açıkça N (deyim-değil) sayıyordu, bu yüzden eski doğal-veri-
karışık stage-2 modelleri dolaylı olarak terim-negatif görmüştü. Sentetik ÜRETİM süreci
(`prepare_synthetic_stage2_pairs.py`) ise YALNIZ aynı-deyimin D/L çiftlerini üretiyor — hiçbir
zaman "yapısal olarak deyime benzeyen ama aslında bambaşka kategori" negatifi üretmiyor. Havuz
büyüdükçe paylaşılan karar sınırı "kalıp deyime benziyorsa D" yönüne kaymış.

**Ucuz hedefli düzeltme:** elle derlenmiş 20 terim (bitki/hayvan/teknik terim, ör. "aslan
ağzı", "deve dikeni", "kırlangıç kuyruğu", "balık kılçığı") için 40 L-cümlesi (D=[] hep boş —
terim asla idyomatik okunmaz) `idiom_data/_synth_raw_100.json`'a elle yazılıp (`--select`
gerekmedi, TDK deyim havuzunun dışında) `--build`'e eklendi (900 deyim → 2515 kayıt, +40 L).
Stage-2 yeniden eğitildi (`best_idiomaticity_clf_vSynth900terim.pt`):

| metrik | v8 (yayında) | 900 (terim-fix yok) | **900+terim (düzeltme)** |
|---|---|---|---|
| GLU vaka (35) | 25/35 | 21/35 | **24/35 (+3, 4 hedeflenen vakanın 3'ü düzeldi)** |
| Çavuşoğlu doğru-ayırt | %65.2 | %66.7 | %65.7 (v8'in hâlâ üstünde) |
| Çavuşoğlu yanlış-poz | %21.2 | %20.7 | **%16.7 (en iyi, gerçek precision kazancı)** |
| PARSEME / CASES | 63.48 / 13/16 | 65.73 / 13/16 | **ÖLÇÜLMEDİ (sonraki oturumda tamamlanmalı)** |

Hedeflenen 4 vakadan 3'ü düzeldi (aslan ağzı, deve dikeni, sigorta attı) ama "eli kolu
bağlanarak" DÜZELMEDİ (bu bir terim değil, deyimin kendisinin homonym-literal okunuşu — farklı
kategori, kendi hard-negative'i gerekir) ve küçük bir YAN ETKİ oluştu: "Toplantıda ben de söz
aldım" (gerçek VID) artık kaçırılıyor (yeni FN) — bunu "Uzmandan görüş aldık" (yanlış FP)
düzelmesi dengeledi. Net +3 GLU, ama v8'in 25'ine hâlâ 1 eksik.

**PARSEME/CASES tamamlandı + eşik taraması yapıldı (900+terim, thresh=0.5):** PARSEME ALL F1
**64.07** (v8: 63.48, +0.59 — no-fix 900'ün 65.73'ünden düşük, terim-negatifin recall maliyeti
burada da görünüyor), CASES **12/16 (-1 vs v8)** — yeni FN: "Öğrenci öğretmenine kafa tuttu"
artık kaçırılıyor (terim-negatifi eklemek karar sınırını hafifçe tutucu yöne kaydırmış, aynı
mekanizma GLU'daki "söz aldım" FN'iyle aynı kök). Yani terim-fix üç ekseni (PARSEME, Çavuşoğlu
doğru-ayırt, yanlış-poz) düzeltirken iki ekseni (CASES, GLU) 1'er puan bedelle ödüyor — klasik
precision/recall takası.

**Eşik taraması (`--stage2-thresh`, 0.55/0.6/0.65/0.7, yalnız external+cases+glu, ucuz
eval-only):** Çavuşoğlu doğru-ayırt 0.5'te %65.7 → **0.6'da %66.2 (en iyi, yanlış-poz %17.2)**
→ 0.65'te %65.2 → 0.7'de %64.6 (düşüyor). AMA CASES/GLU'daki "kafa tuttu"/"söz aldım" FN'leri
0.6'da DA kaybolmuş durumda (12/16, 24/35 — DEĞİŞMEDİ) — bu iki vakanın p(literal) güveni
0.5-0.7 aralığının üstünde kalıyor, yani kayıp kalibrasyon artefaktı DEĞİL, gerçek bir
doğruluk bedeli.

**Final üç-yönlü tablo (2026-09-20 turu sonu):**

| aday | Çavuşoğlu doğru-ayırt | yanlış-poz | PARSEME F1 | CASES | GLU |
|---|---|---|---|---|---|
| v8 (yayında) | %65.2 | %21.2 | 63.48 | 13/16 | 25/35 |
| 900+terim, thresh=0.5 | %65.7 | %16.7 | 64.07 | 12/16 | 24/35 |
| **900+terim, thresh=0.6 (en dengeli aday)** | **%66.2** | %17.2 | ölçülmedi (0.5'e yakın beklenir) | 12/16 | 24/35 |
| 900+terim, thresh=0.3 (2026-09-20 ek test) | %63.1 | **%14.1** | 61.42 (−2pp vs v8) | 12/16 | **25/35** |

**KARAR (2026-09-20, kullanıcı): v8'de KAL, v9 yayınlanmadı.** thresh=0.6 900+terim adayının
CASES/GLU'daki 2 puanlık gerçek-deyim-kaçırma bedeli (kafa tuttu, söz aldım) kabul edilmedi —
Çavuşoğlu/yanlış-poz kazancı bunu telafi etmedi. Kanonik `best_idiom_tagger.pt` (vL) ve
yayındaki v8 paketi (HF/Space) DOKUNULMADI, tüm bu tur (Deney AA) tamamen yerel araştırma
olarak arşivde kaldı. Checkpoint'ler: `best_idiom_tagger_vAblationSynth.pt`,
`best_idiomaticity_clf_vSynth900.pt`, `best_idiomaticity_clf_vSynth900terim.pt`.

**(a) fikri araştırıldı ve KAPATILDI (2026-09-20, aynı gün takip).** "kafa tuttu" ve "söz
aldım" FN'lerinin kök nedeni bulundu: bu cümleler UYDURMA test-vakaları değil, **doğrudan
CASES (`benchmark/eval_idiom.py:42`) ve GLU (`data/prepare_glu_examples.py:76`) eval
setlerinin kendi metni.** `data/prepare_synthetic_stage2_pairs.py::select()` eval/frozen/
holdout'taki deyimleri sentetik havuzdan BİLEREK dışlıyor (test-sızıntısını önlemek için —
bu doğru davranış). Yani "kafa tutmak"/"söz almak" 900-deyimlik sentetik havuzda hiç yok;
grep ile doğrulandı (`_synth_raw_*.json` içinde "kafa tut"/"söz al" araması boş döndü).
v8'in stage-2'si bu iki deyimi doğal-derlem madenli `corpus_examples_glu.json`'dan (eval-
dışlama bu kadar sıkı uygulanmamış) öğrenmişti; `--synthetic-only` bu kaynağı tamamen
attığı için kapsamı kaybetti. **Sonuç: bu "hard-negative eklemekle" düzeltilecek bir veri
boşluğu değil — düzeltmenin tek yolu ya eval-dışlamayı gevşetmek (CASES/GLU'yu kirletir,
kabul edilemez) ya da doğal-derlem verisini sentetik-yalnız rejimine GERİ katmak (Deney Z'nin
temel önermesini bozar). Düşük-öncelik listesinden çıkarıldı, tekrar denenmeyecek.**

**(b) açık kaldı (düşük öncelik, henüz denenmedi):** sentetik havuzu kalan ~10061 dokunulmamış
TDK deyimine doğru ölçeklemeye devam.

**Ek eşik testi (thresh=0.3, aynı gün, kullanıcı isteğiyle):** daha agresif filtreleme "eli
kolu bağlanarak" GLU yanlış-pozitifini düzeltti ve yanlış-poz'u %14.1'e (en düşük ölçüm)
indirdi, GLU vaka skoru v8'i yakaladı (25/35) — ama Çavuşoğlu doğru-ayırt %63.1'e (v8'den
bile kötü) ve PARSEME F1 61.42'ye (v8'den −2pp, thresh=0.5'ten −2.65pp) geriledi. Hiçbir eşik
diğerlerini domine etmiyor, salt precision/recall eğrisinde farklı bir nokta. Karar değişmedi:
v8'de kalınıyor.

## Deney W (2026-09-19) — anlaşmazlık-ağırlıklı ensemble distilasyonu: TEK modelde ensemble'ın
## Çavuşoğlu doğru-ayırtını (%57.1) YAKALADI, PARSEME'de ensemble'ı bile geçti — PROMOTE ADAYI,
## KULLANICI KARARI BEKLİYOR

Fork araştırmasının 2. önerisi. Deney R'nin teşhisi: vE+vL öğretmen çiftinin softmax
ORTALAMASINI TÜM kayıtlara eşit KL kaybıyla damıtmak, öğretmenlerin ÇEŞİTLİLİĞİNİ (asıl
değerli kısmı — birbirini tamamlayan kör-nokta kapsamı) siliyor, öğrenciye yalnız ortalamayı
öğretiyor ("Agree to Disagree", NeurIPS 2020'nin genel teşhisiyle örtüşüyor). Çözüm: damıtım
kaybını öğretmenlerin GERÇEKTEN ayrıştığı (tamamlayıcı bilgi taşıyan) pozisyonlara yönlendirmek,
zaten anlaştıkları (gereksiz sinyal) pozisyonlara değil.

**Uygulama (kalıcı, repoda):** `scripts/distill_ensemble_labels.py` artık her token için
öğretmenlerin (vE/vL) per-token softmax'ının toplam-varyasyon uzaklığını (`0.5*Σ|p_a-p_b|`,
[0,1]) da hesaplayıp `disagree_tags`/`disagree_tags2` olarak kaydediyor. `training/train_idiom_bert.py`:
`_soft_kl` artık opsiyonel `disagree_weight` alıyor — verilirse KL'nin her pozisyondaki katkısı
o pozisyonun anlaşmazlık ağırlığıyla çarpılıp ağırlıklı ortalama alınır (verilmezse eskisiyle
BİREBİR aynı — `batchmean` = `kl.sum(-1).mean()`e matematiksel olarak eşit olduğu doğrulandı).
`--distill-weight-disagree` bayrağı (kapalıyken Deney R ile bit-birebir aynı davranış).

**Eğitim:** Deney R'nin BİREBİR aynı reçetesi (vL verisi, `--distill-lambda 1.0`) +
`--distill-weight-disagree`. Best epoch 10, dev F1 **67.37** (vL'nin 66.25'ini geçti;
Deney R'nin 68.83'ünden düşük ama Deney R'nin dev F1 kazancı "sahte" çıkmıştı — asıl ölçüt
Çavuşoğlu). `best_idiom_tagger_vW_disagree.pt` (kanonik `best_idiom_tagger.pt`'ye geçici
yazıldı, hash doğrulamasıyla vL'ye geri yüklendi — DOKUNULMADI).

**Tam boru hattı (vW + DEĞİŞMEMİŞ stage-2 v3), diğer turlarla kıyas:**

| metrik | vL (taban) | Deney R (düz distilasyon) | union(vE,vL) ensemble (yayında) | **Deney W (vW)** |
|---|---|---|---|---|
| PARSEME ALL F1 | 64.34 | 67.90 | 65.57 | **66.07** |
| **Çavuşoğlu doğru-ayırt (tam)** | 54.0% | 53.0% | **57.1%** | **57.1% (EŞİT)** |
| CASES (16) | 13/16 | — | 14/16 | **14/16 (eşit)** |
| GLU vaka (35) | 21/35 | — | 20/35 | 20/35 |
| Paket/gecikme bedeli | 1×, ~440MB | 1×, ~440MB | **2×, ~1.3GB** | **1×, ~440MB (ensemble bedeli YOK)** |

**Kesin kanıt — seen/unseen kırılımı (Deney A'nın dondurulmuş setiyle), ezber değil gerçek
genelleme olduğunu gösteriyor:**

| dilim | vL | vE (tek) | union(vE,vL) ensemble | **Deney W** |
|---|---|---|---|---|
| seen (21 çift) | 66.7% | 66.7% | 66.7% | **71.4% (en iyi)** |
| **unseen (177 çift)** | 39.0% | 54.2% | 55.9% | **55.4% (ensemble'a neredeyse eşit)** |

Deney W, ensemble'ın unseen-deyim kazancının (39.0→55.9, +16.9pp) **%98'ini** (39.0→55.4,
+16.4pp) TEK modelde yakalıyor — Deney R'nin başaramadığı tam olarak bu eksendi (ensemble'ın
tamamlayıcı kapsamını tek gövdeye aktarma). Fork'un teşhisi doğru çıktı: sorun distilasyonun
KENDİSİ değil, damıtım kapasitesinin nasıl DAĞITILDIĞIydı.

**Durum: YAYINLANDI v6 (2026-09-19).** Şimdiye kadarki tüm turlardan (S/T/U/V dahil) farklı
olarak bu, hem asıl karar metriğinde (Çavuşoğlu) ensemble'ı yakalıyor HEM maliyet eksenindeki
bedeli tamamen ortadan kaldırıyor — Deney O'nun ensemble kararının (maliyet kabul edilebilir
çünkü doğruluk kazancı onu haklı çıkarıyor) gerektirdiği takas bile yok. Kullanıcı onayıyla
yayınlandı:
- **HF push:** `iatagun/DizgeBERT-Idiom` (commit `fd002bd`, ~134MB delta, 880MB paket — v5'in
  1.32GB'ının yarısı). Round-trip doğrulandı (`--hf-repo` yerel `--checkpoint vW` sayılarıyla
  birebir eşleşti). MODEL_CARD.md v5→v6 güncellendi (yeni tablo, "ensemble" yerine
  "anlaşmazlık-ağırlıklı distilasyon" anlatımı, 2×→1× gecikme notu).
- **Space (`iatagun/dizge-demo`) güncellendi:** `idiom_tab.py`/`app.py`'deki "v5 · ensemble"
  rozetleri/metinleri "v6 · tek gövde" olarak güncellendi (scratchpad'de klonlanıp commit+push
  edildi, kod değişikliği olduğu için restart değil tam rebuild tetiklendi: BUILDING→
  APP_STARTING→RUNNING). `gradio_client` smoke-test: idyomatik "yol aldık" doğru işaretlendi,
  bilinen literal yanlış-pozitif ("otobüs yol aldı") hâlâ geçiyor — ölçülmüş/belgelenmiş %23.7
  yanlış-pozitifin canlıda doğrulanması, deploy hatası değil.
- **Süreç notu (mühendislik dersi):** ilk push denemesi (`... &` ile hem shell arka-planına hem
  `run_in_background`'a aynı anda alma) sessizce yarıda kesildi — HF repo'sunda değişiklik
  olmadı ama komut "exit 0" ile "tamamlandı" göründü. `list_repo_commits`/`model_info` ile
  gerçek durumu doğrulamadan "push bitti" denmemeli; log dosyasının son satırını (`✓ push
  tamam`) görmeden başarı varsayılmamalı.
- **Yerel:** kanonik `idiom_data/best_idiom_tagger.pt` hâlâ **vL** (deney disiplini gereği —
  yalnız HF paketi v6, yerel tek-checkpoint iş akışı v5/Deney O'daki gibi değişmedi).
  Checkpoint arşivde: `best_idiom_tagger_vW_disagree.pt`. Stage-2 DEĞİŞMEDİ (v3).

## Deney X (2026-09-19) — 3. bağımsız öğretmen ensemble'da YENİ REKOR, distilasyonda ÇÖKÜŞ:
## "bağımsız veri" ensemble'a ve distilasyona FARKLI davranıyor

Deney N'in reddedilen Leipzig 3M→8M genişlemesini Deney W çerçevesiyle (çoklu-öğretmen +
anlaşmazlık-ağırlıklı distilasyon) yeniden denemek için önce bir ÖN-KOŞUL doğrulandı: gerçekten
bağımsız 3. bir öğretmen bulunabilir mi? `_corpus_sample_labels.tsv`'de vL'nin eğitiminden
(idx≤11070) SONRA eklenmiş ama hiç kullanılmamış 2383 kayıt (1130 yeni deyim kimliği, Deney N'in
ingest turu, %84.8 tam-oybirliği) keşfedildi — vM checkpoint'i silinmiş olsa da ham etiket havuzu
append-only olduğu için hâlâ diskteydi. `data/filter_corpus_idiomaticity.py`'e `--min-idx`/
`--out-suffix` eklendi (idx-filtreli dilim çıkarma, canonical dosyaları ezmeden) — bu dilim
(1924 kayıt, 958 D-span/966 L→hepO) vF/vJ/vK/vL zincirinden TAMAMEN AYRI: aynı soyun ardışık
sürümü değil, gerçek bağımsız veri. `vX3` (bu dilim + standart TDK/PARSEME taban) eğitildi:
dev F1 69.81 — tek başına vE/vL'den bile güçlü.

**Ön-koşul testi (yalnız çıkarım-zamanı birleştirme, Deney O'nun union/agree/majority
taramasının 3'lü hâli) — GEÇTİ, yeni rekor:**

| kombinasyon | Çavuşoğlu doğru-ayırt (tam) | (unseen) |
|---|---|---|
| vL tek | 54.0% | 52.0% |
| vE tek | 55.6% | 54.2% |
| union(vE,vL) (Deney O, yayında v5 ensemble mantığı) | 57.1% | 55.9% |
| Deney W tek-gövde distilasyon (yayında v6) | 57.1% | 55.4% |
| majority(vE,vL,vX3) 2/3 | 49.0% | 45.8% (Deney O deseni: karma soy zarar veriyor) |
| agree(vE,vL,vX3) 3/3 | 31.3% | 27.1% (çöktü) |
| **union(vE,vL,vX3) min_votes=1** | **58.1% (yeni rekor)** | **57.1% (yeni rekor)** |

PARSEME ALL F1 66.02 (union(vE,vL)'nin 65.57'sine ve Deney W'nin 66.07'sine yakın), CASES
13/16 (14/16'dan hafif düşük, n=16 gürültü bandında), GLU 20/35 (düz). Ön-koşul net geçti —
kullanıcı onayıyla distilasyon adımına geçildi.

**Distilasyon (Deney X asıl denemesi) — TEMİZ NEGATİF, PARSEME F1 en iyisi ama Çavuşoğlu
vL'den de kötü.** `scripts/distill_ensemble_labels.py` 2→N öğretmene genelleştirildi
(`--teachers ck1,ck2,...`; anlaşmazlık ağırlığı artık öğretmen ÇİFTLERİNİN ortalama ikili
toplam-varyasyon uzaklığı — N=2 için eski formülle birebir aynı sonucu verir, geriye uyumlu).
`train_idiom_bert.py`'nin distilasyon yolu zaten öğretmen sayısından bağımsızdı (yalnız
`soft_tags`/`disagree_tags` tüketiyor), değişiklik gerekmedi. vE+vL+vX3 ile Deney W'nin
BİREBİR aynı reçetesi (`--distill-lambda 1.0 --distill-weight-disagree`, 10 epoch) → **vY**,
best epoch 10, dev F1 67.07 (Deney W'nin 67.37'sine yakın).

| metrik | vL | Deney W (2-öğretmen distill, yayında) | union(vE,vL,vX3) | **vY (3-öğretmen distill)** |
|---|---|---|---|---|
| PARSEME ALL F1 | 64.34 | 66.07 | 66.02 | **67.09 (en iyi F1)** |
| CASES (16) | 13/16 | 14/16 | 13/16 | 13/16 |
| **Çavuşoğlu doğru-ayırt (tam)** | 54.0% | 57.1% | 58.1% | **51.5% (vL'den de kötü)** |
| **Çavuşoğlu doğru-ayırt (unseen)** | 52.0% | 55.4% | 57.1% | **49.2% (vL'den de kötü)** |
| seen (21 çift) | 66.7% | 71.4% | 66.7% | 71.4% |
| GLU vaka (35) | 21/35 | 20/35 | 20/35 | 20/35 |

**Ders — bu, projenin "PARSEME/dev-F1 artışı Çavuşoğlu genellemesini garanti etmez" meta-
dersinin şimdiye kadarki EN ÇARPICI örneği:** vY en yüksek PARSEME F1'i verdi AMA asıl karar
ölçütünde vL TABANINDAN BİLE kötü — yalnız "ensemble'ın gerisinde kalmak" değil, "distilasyon
hiç yapılmamışından daha kötü" durumu ilk kez görüldü. **Mekanizma:** saf çıkarım-zamanı
union'da fazladan bir öğretmen yalnız EKLEME yapabilir (min_votes=1 ile hiçbir doğru span
kaybolmaz, stage-2 fazladan yanlış-pozitifi sonradan süzer) — bu yüzden vX3'ün kendi
gürültüsü zararsız, yalnız tamamlayıcı kapsamı işe yarıyor. Distilasyonda ise TÜM öğretmen
görüşleri TEK sürekli hedef dağılıma karışıyor; vX3 çok daha küçük/dar bir dilimde
(1924 kayıt, vL'nin 7938'ine karşı) eğitildiği için token-düzeyi anlaşmazlıkları çoğu zaman
GERÇEK tamamlayıcı sinyal değil, az-veriden kaynaklanan gürültü — anlaşmazlık-ağırlıklı kayıp
tam da bu gürültülü noktalara ağırlık veriyor, öğrenciyi yanlış yöne itiyor. **Sonuç: "bağımsız
öğretmen" ensemble için yeterli olsa da distilasyon için YETERLİ DEĞİL — teacher'ın kendi
eğitim hacmi/kalitesi de belli bir eşiği geçmeli, yoksa disagreement-weighting gürültüyü
sinyal sanıp damıtır.** Kanonik `best_idiom_tagger.pt` vL'ye geri yüklendi (hash doğrulandı),
`corpus_examples_glu.json` vL'nin 7938 kaydına geri yüklendi. Checkpoint'ler arşivde:
`best_idiom_tagger_vX3_slice3.pt`, `best_idiom_tagger_vY_3teacherdistill.pt`. HF/Space
DOKUNULMADI, hâlâ v6 (Deney W). **union(vE,vL,vX3)'ün ham %58.1/%57.1 rekoru yayınlanmadı**
(3× çıkarım maliyeti + distilasyon denemesi başarısız olduğu için tek-gövde alternatifi yok) —
kullanıcı kararı bekliyor: ya Deney O'nun kabul ettiği türden bir maliyet-takasıyla 3'lü
ensemble'ı yayınla, ya da mevcut v6'da kal.

**Kod kalıcı:** `data/filter_corpus_idiomaticity.py --min-idx/--out-suffix` (bağımsız veri
dilimi çıkarma, ileride başka deneyler için de kullanılabilir), `scripts/distill_ensemble_labels.py --teachers`
(N-öğretmenli, 2 ile geriye uyumlu).

**GÜNCELLEME — YAYINLANDI v7 (2026-09-19, kullanıcı kararıyla).** Kullanıcı, 3× çıkarım
maliyetini kabul edip ham union(vE,vL,vX3) ensemble'ını yayınlamayı tercih etti (distilasyon
başarısız olduğu için tek-gövde alternatifi yoktu). Paketleme kodu Deney O'nun tek-ikinci-gövde
desenini (`config.ensemble`/`encoder_b`/`tag_head_b`/`tag_head2_b`) 3. bir gövdeye genelleştirdi:
`configuration_dizgebert_idiom.py` → `ensemble_extra2: bool`, `modeling_dizgebert_idiom.py` →
`encoder_c`/`tag_head_c`/`tag_head2_c` + `predict_spans()` artık `merge_ensemble_spans()`'a
üç span listesi birden veriyor (fonksiyon zaten N-way genel, değişiklik gerekmedi),
`train_idiom_bert.py --export-hf` → `--ensemble-ckpt2` (üçüncü checkpoint).
- **HF push:** `iatagun/DizgeBERT-Idiom` commit `3672edf` (primary=vE, ensemble b=vL,
  ensemble c=vX3, stage2=v3, ~1.76GB safetensors). Round-trip doğrulandı (`--hf-repo` yerel
  union ölçümleriyle birebir: PARSEME F1 66.02, Çavuşoğlu doğru-ayırt tam %58.1, CASES 13/16,
  GLU 20/35) VE ayrıca uzak repo'dan `model_info`/`config.json` indirilip `ensemble_extra2=True`
  olduğu bağımsız doğrulandı (Deney W'nin "exit 0'a güvenme" dersi uygulandı).
- **Space (`iatagun/dizge-demo`) güncellendi:** `idiom_tab.py`/`app.py` "v6·tek gövde" →
  "v7·3-gövde ensemble" rozetleri/metinleri, kod değişikliği olduğu için restart değil tam
  rebuild (BUILDING→APP_STARTING→RUNNING, `space_info().runtime.stage` ile doğrulandı).
  `gradio_client` smoke-test: idyomatik "yol aldık" doğru işaretlendi, bilinen literal
  yanlış-pozitif ("otobüs yol aldı") hâlâ geçiyor — ölçülmüş %28.3 yanlış-pozitifin canlıda
  doğrulanması, deploy hatası değil.
- **Yerel:** kanonik `idiom_data/best_idiom_tagger.pt` hâlâ **vL** (deney disiplini —
  yalnız HF paketi v7, yerel tek-checkpoint iş akışı değişmedi). `corpus_examples_glu.json`
  vL'nin 7938 kaydına geri yüklü. MODEL_CARD.md v6→v7 güncellendi (yeni tablo, "ham ensemble,
  distilasyon değil" notu, v6 hâlâ tek-gövde alternatifi olarak belgelendi).

## Deney V (2026-09-19) — leksikal/bağlamsal uyumluluk mimarisi (Zeng&Bhat 2021): şimdiye
## kadarki en dengeli tek-model sonuç ama YİNE vL/ensemble'ın altında — REDDEDİLDİ

Fork araştırmasının 3. önerisi: stage-2 head'ini TEK birleşik `[CLS]⊕span-ilk⊕son` yerine,
span'in BAĞLAMSAL (cümledeki hali) ve LEKSİKAL (bağlamsız — yalnız span kelimeleri, ayrı
küçük dizi olarak kodlanmış) temsillerini AYRI çıkarıp `[bağlamsal, leksikal, fark]` (6H)
olarak birleştirmek. Önceki turların (v5ctx, Deney H sonrası) hep BAĞLAMSAL tarafı
zenginleştirmesinden (`[CLS]⊕ortalama⊕ilk⊕son`) FARKLI bir eksen — hiç bir LEKSİKAL/bağlamsız
referans temsili denenmemişti.

**Uygulama (kalıcı, repoda):** `training/train_idiomaticity_clf.py`: `ClfDS` artık span
kelimelerini AYRICA bağlamsız tokenize edip (`lex_input_ids`/`lex_attention_mask`/`lf`/`ll`)
saklıyor; `IdiomaticityClf(compat_gap=True)` ikinci bir küçük forward ile leksikal temsili
çıkarıp head'e `[ctx, lex, ctx-lex]` veriyor (`compat_gap=False` = eskisiyle bit-birebir aynı,
head 2H girdi alır). Checkpoint'e `compat_gap` bayrağı gömülüyor. **Gerçek boru hattı da
güncellendi** (yalnız eğitim değil): `dizgebert_idiom/modeling_dizgebert_idiom.py::span_p_literal_gap`
(yeni fonksiyon, `span_p_literal`in compat-gap ikizi) + `wrap_stage2()` artık checkpoint'in
`compat_gap` bayrağını okuyup gerekirse her aday span için span-kelimelerini AYRICA bağlamsız
kodluyor (cümle başına ekstra küçük forward, yalnız aday sayısı kadar). Uçtan uca smoke-test
ile doğrulandı (eğitim + `wrap_stage2` çıkarımı).

**Eğitim:** v3'ün BİREBİR aynı reçetesi (`--freeze 8 --dropout 0.3 --weight-decay 0.05
--epochs 14`, AYNI frozen havuz — 9694 train/3031 test) + `--compat-gap`. Best epoch 3
(macro 74.2, sonraki epoch'lar düşüş/aşırı-öğrenme). `best_idiomaticity_clf_vGap.pt`.

**Tam boru hattı (vL span modeli + vGap stage-2), diğer turlarla kıyas:**

| metrik | vL (taban, v3) | Deney S | Deney T (PCGrad) | Deney U (freeze8) | **Deney V (vGap)** |
|---|---|---|---|---|---|
| PARSEME ALL F1 | 64.34 | 67.38 | 65.82 | 60.87 | **62.46** |
| **Çavuşoğlu doğru-ayırt** | **54.0%** | 42.9% | 34.3% | 45.5% | **47.5%** |
| Çavuşoğlu yanlış-poz | 18.7% | 37.9% | 46.5% | 32.8% | **19.7%** |
| CASES (16) | 13/16 | — | 12/16 | 13/16 | 12/16 |
| GLU vaka (35) | 21/35 | — | 17/35 | 22/35 | 21/35 |

**Sonuç: REDDEDİLDİ ama şimdiye kadarki en dengeli tek-model sonuç.** Deney U'yu doğru-ayırtta
geçti (%47.5>%45.5) VE yanlış-pozitifte ÇOK daha iyi (%19.7, vL'nin %18.7'sine neredeyse eşit —
S/T/U'nun hepsi %33-46 aralığındaydı). Leksikal/bağlamsal FARK sinyali gerçek bir şey ölçüyor
gibi görünüyor (mekanizma doğru yönde) ama tek başına vL'yi (%54.0) geçmeye yetmiyor — muhtemelen
çünkü span'i "bağlamsız" kodlamak da kendi başına belirsiz (2-3 kelimelik izole bir dizi hâlâ
ELECTRA'nın kendi ön-eğitim önyargılarını taşıyor, "gerçek" bir sözlük/lemma referansı değil).
Kanonik `best_idiomaticity_clf_v3.pt` DEĞİŞMEDİ. Checkpoint arşivde (`best_idiomaticity_clf_vGap.pt`).
**Not (denenmeden bırakılan takip):** `compat_gap` + `freeze` birlikte (bu turda freeze zaten
kullanıldı, ama vGap'in kendi mimarisi lex-encoder'ı da mı dondurmalı yoksa serbest mi bırakmalı
sorusu ayrı bir eksen) — düşük öncelikli.

## Deney U (2026-09-19) — Deney S'e freeze=8 (v3'ün reçetesi, ortak gövdede): büyük iyileşme
## ama YİNE vL/ensemble'ın altında — REDDEDİLDİ

Deney T'nin teşhisi (PCGrad gradyan-YÖNÜ çakışmasını doğru çözdü ama asıl sorun stage-2'nin
KORUMASIZLIĞI/aşırı-öğrenmesiydi) doğrudan test edildi: `scripts/train_joint_stage2.py`'e
`--freeze N` eklendi (`training/train_idiomaticity_clf.py::IdiomaticityClf`'in AYNI dondurma
mantığı — embeddings + alttan N transformer katmanı — ama PAYLAŞILAN gövdede, `tag_head`/
`tag_head2`'yi de aynı anda korur). `--freeze 0` = Deney S/T ile bit-birebir aynı davranış.
Deney S'in BİREBİR aynı reçetesi + `--freeze 8`, PCGrad KAPALI (en temiz tek-değişkenli test —
Deney T'nin hangi ekseninin gerçekten işe yaradığını izole etmek için).

Doğrulama: freeze 8 → 110M→**28.4M trainable** (v3'ün kendi rakamıyla birebir). Yan fayda:
eğitim de **~6.5× hızlandı** (~1.1it/s→~7.2it/s, daha az katmanda geri-yayılım + PCGrad'ın ek
autograd.grad yükü yok) — 10 epoch ~80dk'da bitti (Deney T'nin ~8.5 saatine karşı).

Best epoch 9 (dev F1 62.70). Tam boru hattı (vJointF8):

| metrik | vL (taban) | Deney S | Deney T (PCGrad) | **Deney U (freeze8)** |
|---|---|---|---|---|
| PARSEME ALL F1 | 64.34 | 67.38 | 65.82 | **60.87 (−3.47)** |
| **Çavuşoğlu doğru-ayırt** | **54.0%** | 42.9% | 34.3% | **45.5%** |
| Çavuşoğlu yanlış-poz | 18.7% | 37.9% | 46.5% | **32.8%** |
| CASES (16) | 13/16 | — | 12/16 | **13/16 (vL'yle eşit)** |
| GLU vaka (35) | 21/35 | — | 17/35 | **22/35 (o ana kadarki en iyisi)** |

**Sonuç: REDDEDİLDİ ama üç ortak-gövde varyantının en iyisi.** Hipotez doğrulandı — freeze,
Deney S'in %42.9'unu VE Deney T'nin %34.3'ünü büyük farkla geçti. Ama PARSEME'de gerçek bir
bedel var (kapasite kısıtlaması, stage-1'in kendi öğrenmesini de yavaşlatıyor) ve doğru-ayırt
hâlâ vL'nin (%54.0) altında. **Ortak-gövde ekseni (Deney S/T/U, üç farklı mekanizma —
korumasız/PCGrad/freeze) artık tükenmiş görünüyor** — hiçbiri AYRI iki gövdenin (vL span +
v3 stage-2) toplamını geçemedi. Checkpoint arşivde (`best_idiom_tagger_vJointF8.pt`,
`best_idiomaticity_clf_vJointF8.pt`), kanonik değişmedi.

## Deney T (2026-09-18/19) — Deney S'e PCGrad gradyan cerrahisi: gradyan çakışması gerçekten
## VARDI (doğrulandı) ama Çavuşoğlu düz-Deney S'ten de KÖTÜ — REDDEDİLDİ, farklı bir kök neden

Deney S'in teşhisi ("paylaşılan gövde, korumasız, stage-1'in büyük gradyanı stage-2'yi
boğuyor") literatürdeki gradyan-çakışması/görev-dengesizliği problemiyle örtüşüyordu —
PCGrad (Yu et al., NeurIPS 2020) tam bunu hedefliyor: iki görev gradyanı PAYLAŞILAN
parametrelerde çakışırsa (kosinüs negatif), çakışan bileşen projekte edilip silinir; head'ler
(`tag_head`/`tag_head2` ↔ loss1, `stage2_head` ↔ loss2) zaten göreve özel, yalnız `encoder`
parametrelerinde cerrahi uygulandı (`scripts/train_joint_stage2.py::pcgrad_step`, `--pcgrad`
bayrağı, kapalıyken Deney S ile bit-birebir aynı davranış). Aynı vL reçetesi, `--epochs 10
--mu 1.0 --pcgrad --batch-size 8` (bu makinede 4GB VRAM zorunluluğuyla batch 16→8 — Deney S ile
tek-değişkenli kıyas tam saf değil, ama sonuç zaten net negatif olduğundan bu ikincil).

**Süreç notu (mühendislik dersi):** ilk implementasyonda iki gerçek bug vardı, ikisi de
düzeltildi ve smoke-test'le doğrulandı: (1) `loss1.backward(inputs=..., retain_graph=True)`
olması gerekirken `True` bırakılmıştı — grafiği hiç serbest bırakmıyordu, GPU'da adım başına
bellek birikip epoch içinde 1s/it'ten 5.6s/it'e çıkıyordu; (2) PCGrad'ın simetrik
projeksiyonunda ikinci satır `flat1`'i BİRİNCİ satırda zaten güncellenmiş haliyle kullanıyordu
(orijinal değerler cache'lenmeliydi) — matematiği sinsice bozan bir referans hatası. Ayrıca
epoch1→epoch2 geçişinde tekrarlayan bir yavaşlama (~1.1s/it→~5s/it, sonra platoya oturuyor)
gözlendi; checkpoint kaydını CPU'ya taşımak (D2H trafiğini yarıya indirmek) ÇÖZMEDİ — asıl
sebep, eval'den önce çağrılan `torch.cuda.empty_cache()` imiş: 4GB'lık kartta cache'i zorla
boşaltmak allocator'ı her epoch başında sıfırdan (daha parçalı) yeniden ısınmaya zorluyordu.
Bu çağrıyı (Deney S'ten kalma) kaldırmak sorunu tamamen çözdü. **İkisi de mekanizma
doğrulamasından SONRA, gerçek 10-epoch koşusundan ÖNCE yakalandı** — deney sonucunun kendisini
etkilemedi.

**Mekanizma doğrulandı:** `pcgrad_conflict_rate` epoch başına ölçüldü, adımların **%46-62'sinde
gerçek çakışma vardı** (epoch1 %46 → epoch10 %62, model ezberledikçe çakışma ARTIYOR) —
Deney S'in "stage-1/stage-2 gradyanları paylaşılan gövdede çakışıyor" teşhisi bağımsız olarak
doğrulandı, PCGrad'ın müdahale edecek gerçek bir şeyi vardı.

Tam boru hattı (vJointPC, epoch6 en iyi dev-F1 66.51'de seçildi), Deney S ve vL ile kıyas:

| metrik | vL (taban) | Deney S (vJoint, korumasız) | **Deney T (vJointPC, PCGrad)** |
|---|---|---|---|
| PARSEME ALL F1 | 64.34 | 67.38 | 65.82 |
| **Çavuşoğlu doğru-ayırt (tam, gevşek)** | **54.0%** | **42.9%** | **34.3% (Deney S'ten de KÖTÜ)** |
| Çavuşoğlu yanlış-poz | 18.7% | 37.9% | **46.5% (en kötü)** |
| CASES (16) | 13/16 | — | 12/16 |
| GLU vaka (35) | 21/35 | — | 17/35 |
| minimal-çift doğru-ayırt (stage2-iso) | ~59% (v3) | — | **11% (çöküş)** |

**Sonuç: REDDEDİLDİ, Deney S'ten de kötü.** PCGrad gradyan-YÖNÜ çakışmasını doğru şekilde
çözdü ama bu, sorunun yanlış ekseniydi. Gerçek teşhis: stage-2, ~1212 batch'lik küçük havuzu
her epoch'ta `itertools.cycle` ile 2.8× tekrar tekrar görüyor ve hiçbir dondurma/düzenlileştirme
olmadan hızla ezberliyor (loss2 epoch1 0.33 → epoch10 0.001) — bu Deney S'te de vardı, ama
PCGrad bunu DAHA KÖTÜLEŞTİRDİ: çakışma anlarında PCGrad, stage-2'nin gradyan YÖNÜNÜ stage-1'in
büyük ama iyi-koşullu gradyanının "sulandırmasından" bilinçli olarak KORUYOR — sorun stage-2'nin
yönü değil, o yönün zaten aşırı-ezberlenmiş/gürültülü olmasıydı, PCGrad kötü bir sinyali
korumuş oldu. İkinci, ayrı bir kusur: seçim ölçütü (`selection_score`) YALNIZ stage-1 span-F1'e
bakıyor, stage-2'nin kendi genellemesini hiç izlemiyor — epoch6 stage-1 için en iyi ama stage-2
zaten epoch4'ten beri (loss2 0.006) ağır ezberlemiş durumdaydı, "en iyi" checkpoint stage-2 açısından
gelişigüzel bir noktada seçildi. **Deney S'in notundaki takip fikri ("--freeze ile stage2_head
öncesi katmanları dondurarak ortak-gövdeyi tekrar dene") hâlâ denenmedi ve şimdi daha güçlü bir
öncelik kazandı** — PCGrad'ın gösterdiği kadarıyla sorun gradyan YÖNÜ değil, stage-2'nin
KORUMASIZLIĞI (regularizasyon eksikliği); freeze bunu hem gradyan-hacim hem overfit ekseninde
birden çözebilir, PCGrad tek başına yetersiz kaldı. Checkpoint'ler arşivde
(`best_idiom_tagger_vJointPC.pt`, `best_idiomaticity_clf_vJointPC.pt`), kanonik değişmedi.

## Deney S (2026-09-18) — ortak-gövde çok-görevli eğitim (multi-task): PARSEME iyi, Çavuşoğlu
## ÇÖKTÜ — REDDEDİLDİ, kök neden stage-2 head'in korumasız paylaşımı

Öneri #2 — stage-1 (BIO) ve stage-2 (idiomatiklik) TAMAMEN AYRI iki ELECTRA gövdesi yerine
TEK paylaşılan gövde + üç head (`tag_head`, `tag_head2`, `stage2_head`), aynı adımda birlikte
eğitim (`scripts/train_joint_stage2.py`, yeni `JointIdiomTagger(IdiomTagger)`). Stage-1 verisi
vL'nin birebir aynı reçetesi, stage-2 verisi `train_idiomaticity_clf.py::load_pairs()`'ın aynı
frozen havuzu (9694 kayıt) — PARSEME altına dokunulmadı. Her adımda `loss1 + mu*loss2`
(mu=1.0), TEK backward. Değerlendirme kolaylığı için en iyi epoch iki mevcut-format-uyumlu
checkpoint'e bölündü (`best_idiom_tagger_vJoint.pt` + `best_idiomaticity_clf_vJoint.pt`,
`best_idiom_tagger.pt` kanonik DOKUNULMADI — hash doğrulandı).

Maliyet notu: her adımda 2. bir forward+backward (stage-2 batch) olduğundan eğitim ~3× yavaş
(~30dk/epoch, öncekiler ~10dk).

| metrik | vL (taban) | vJoint (mu=1) |
|---|---|---|
| dev F1 (seçim ölçütü, epoch 7) | 66.25 | 67.25 (+1.0) |
| PARSEME ALL F1 | 64.34 | **67.38 (+3.04, P=62.01 dengeli)** |
| **Çavuşoğlu doğru-ayırt** | **%54.0** | **%42.9 (−11.1, ÇÖKTÜ)** |
| Çavuşoğlu yanlış-poz | 18.7% | **37.9% (ikiye katlandı)** |

**Kök neden, izole `--mode stage2-iso` ile doğrulandı:** joint stage2_head'in kendi başına
ayırt gücü v3'ten belirgin düşük — çift-içi sıralama **%84.4** (v3: ~%93), thresholded
doğru-ayırt **%43.6** (v3: ~%58.7). Yani sorun stage-1 span kalitesinde DEĞİL (PARSEME iyi,
hatta vFocal/vDistill'den daha dengeli precision'la) — **paylaşılan gövde, korumasız
(freeze yok) ve her adımda çok daha büyük hacimli stage-1 gradyanının baskısı altında,
stage-2'nin ince idiomatik/literal ayrımını öğrenecek representasyonu koruyamıyor.** v3'ün
`--freeze 8` reçetesi tam olarak bu sorunu (küçük/hassas stage-2 görevinin büyük gövdede
kaybolması) çözmek için vardı — ortak-gövde deneyi bu korumayı kaldırınca aynı sorun daha
şiddetli geri geldi.

**Sonuç: REDDEDİLDİ.** Ensemble (Deney O, çalıştı) ve distilasyon (Deney R, kısmi/negatif)
ile birlikte üçüncü "iki modeli birleştirme" ekseni de kapandı — hepsi farklı mekanizmalarla
denendi (tahmin birleştirme / bilgi damıtma / gövde paylaşımı), üçü de gerçek ensemble'ın
(%57.1) gerisinde kaldı. **Not (denenmeden bırakılan takip):** `--freeze` ile stage2_head
öncesi katmanları dondurarak aynı ortak-gövde fikri tekrar denenebilir (v3'ün reçetesini
paylaşılan gövdeye taşımak) — düşük öncelikli, mekanizma teşhisi zaten net. Checkpoint'ler
arşivde (`best_idiom_tagger_vJoint.pt`, `best_idiomaticity_clf_vJoint.pt`), kanonik değişmedi.

## Deney R (2026-09-18) — ensemble distilasyonu (vE+vL → tek gövde): dev/PARSEME YÜKSELDİ,
## Çavuşoğlu düştü — REDDEDİLDİ, önemli bir metodoloji dersiyle

Deney O'nun ensemble kazancını (Çavuşoğlu doğru-ayırt tam %57.1 / unseen %55.9) 2× çıkarım
maliyeti ödemeden tek gövdeye aktarma denemesi (bilgi damıtma / knowledge distillation).
Weak-supervision kayıtlarının (PARSEME altın verisi HARİÇ — yalnız `tdk_examples.json` +
`corpus_examples_glu.json`, 9332 kayıt) etiketlerini DEĞİŞTİRMEDEN, öğretmen (vE+vL ortalama
softmax) sinyalini EK bir KL kaybı olarak eğitime kattık — sabit etiketi silmek yerine
yumuşak-hedef ekleme, bilinçli tercih (deyim kimliklerini bilerek etiketlenmiş kayıtları
modelin kendi eksik tahminleriyle EZMEMEK için).

**Altyapı (kalıcı, repoda):** `scripts/distill_ensemble_labels.py` (öğretmen çifti → her
kayda `soft_tags`/`soft_tags2` alanı ekler, `*_distill.json` üretir; truncation'a takılan
kayıtlar atlanır, hizalama karmaşıklığından kaçınmak için). `training/train_idiom_bert.py`:
`IdiomDataset` opsiyonel `soft_tags`/`soft_tags2`/`has_soft` taşır (yoksa eskisiyle birebir
aynı), `compute_loss`'a `distill_lambda` (varsayılan 0.0=kapalı) — `--distill-lambda`
bayrağı hem KL terimini açar hem `--tdk-examples`/`--corpus-glu`'yu otomatik `_distill.json`
varyantına yönlendirir.

vL'nin BİREBİR aynı verisiyle (`--class-weights --tdk-examples --corpus-glu --epochs 10`,
tek fark `--distill-lambda 1.0`) eğitildi. **Dev seçim metriği bu kez GERÇEKTEN yükseldi**
(vFocal'ın aksine): best epoch 5, span F1 ALL **68.83** — vL'nin 66.25'inin +2.58 üstünde,
Aşama 3'ten beri stage-1'in gördüğü en yüksek dev F1.

Tam boru hattı (vDistill + DEĞİŞMEMİŞ stage-2 v3, gap=2), vL ile bire bir aynı eval komutuyla:

| metrik | vL (taban) | vDistill (λ=1) |
|---|---|---|
| dev F1 (seçim ölçütü) | 66.25 | **68.83 (+2.58)** |
| PARSEME ALL F1 | 64.34 | **67.90 (+3.56)** |
| PARSEME ALL P/R | — | 59.05 / 79.88 (dengeli — vFocal'ın recall-skew'ü YOK) |
| **Çavuşoğlu doğru-ayırt** | **54.0%** | **53.0% (−1.0)** |
| (referans: union(vE,vL) ensemble) | — | 57.1% (tam) |

**Ders — bu deneyi öncekilerden ayıran şey:** dev F1 VE PARSEME F1 GERÇEKTEN yükseldi (vFocal'daki
gibi bir recall-skew artefaktı değil — precision de sağlam), yani bu "iyi görünüp asıl testte
kötü çıkan" bir yanlış-pozitif ölçüm değil. Yine de Çavuşoğlu'nda ENSEMBLE'IN kendisinin (57.1%)
çok altında kaldı, hatta vL tekten bile hafif düştü (53.0 < 54.0). **Teşhis: distilasyon
öğretmenin kendi eğitim-cümleleri üzerindeki davranışını damıtıyor — bu, modelin zaten gördüğü
TDK/corpus-glu dağılımına daha iyi uymasını sağlıyor (dev F1 aynı dağılımdan, PARSEME'e yakın
üslup) ama ensemble'ın GERÇEK kazancı (Deney O'da kanıtlanmıştı: vE ve vL'nin FARKLI kör
noktalarının birleşimi, özellikle unseen-deyim genellemesinde) tek-gövde damıtmayla aktarılamıyor
— öğretmen sinyali aynı eğitim cümlelerinde iki modelin ORTALAMASI, iki modelin
TAMAMLAYICI kapsamının BİRLEŞİMİ değil.** Bu, projenin "dev F1/PARSEME artışı Çavuşoğlu
genellemesini garanti etmez" meta-dersini (v6, v9-13, vFocal'dan sonra) dördüncü kez, bu kez
en temiz biçimde doğruluyor (çünkü burada precision de gerçekten iyiydi, saf recall artefaktı
değildi). **Kanonik `best_idiom_tagger.pt` vL'ye geri yüklendi.** `best_idiom_tagger_vDistill_lambda1.pt`
arşivde. Altyapı (`--distill-lambda`, `scripts/distill_ensemble_labels.py`) kalıcı — belki
farklı bir λ veya sıcaklık (T>1, öğretmen dağılımını yumuşatma) ile tekrar denenebilir ama
düşük öncelikli (mekanizma teşhisi net: aynı-cümle damıtımı ensemble'ın kapsam-birleşimini
yakalayamıyor).

## Deney Q (2026-09-17) — stage-1 focal loss (γ=2): F1 düz, Çavuşoğlu düz/hafif kötü — REDDEDİLDİ

Precision tavanı için en başından "denenmedi" diye işaretli kalan tek stage-1 kaldıracı: focal
loss (Lin et al. 2017). Class-weight'ten mekanik olarak FARKLI bir eksen — sınıf SIKLIĞI yerine
ÖRNEK GÜÇLÜĞÜNE (kendi p_t'sine) göre dinamik ağırlıklandırır. `train_idiom_bert.py` içine
`--focal-gamma` eklendi (`_ce_or_focal`, `compute_loss`) — γ≤0 iken davranış eskisiyle birebir
aynı (tek `F.cross_entropy` çağrısı, kod yolunda hiç dallanma yok), γ>0 iken `--class-weights`
ile birlikte kullanılabilir (weight=alpha).

vL'nin BİREBİR aynı verisiyle (`--class-weights --tdk-examples --corpus-glu --epochs 10`,
tek değişken `--focal-gamma 2.0`) eğitildi. **Dev seçim metriği zaten uyarı verdi:** best epoch
9, span F1 ALL **64.40** — vL'nin dev F1'i (66.25) altında, ilk kez bir stage-1 varyantı dev
setinde bile vL'yi geçemedi (v6/vE/Aşama-3 hepsi dev'de de iyileşme göstermişti).

Tam boru hattı (vFocal + DEĞİŞMEMİŞ stage-2 v3, gap=2), vL ile bire bir aynı eval komutuyla:

| metrik | vL (taban) | vFocal (γ=2) |
|---|---|---|
| PARSEME ALL F1 | 64.34 | 64.55 (düz) |
| PARSEME ALL P/R | — | 54.28 / **79.62** |
| Çavuşoğlu duyarlılık | 70.7% | **77.3% (+6.6)** |
| Çavuşoğlu yanlış-poz | 18.7% | **25.3% (+6.6)** |
| **Çavuşoğlu doğru-ayırt** | **54.0%** | **53.5% (−0.5, gürültü içinde)** |

**Ders:** focal loss, mekanik olarak class-weight'ten bambaşka bir eksen olmasına rağmen,
BİREBİR AYNI hata imzasını üretti — v6-v13'ün "recall-skew" deseni (F1 düz/hafif iyi ama
duyarlılık VE yanlış-poz birlikte yükseliyor, doğru-ayırt net kazanç vermiyor). Bu, projenin
"precision tavanı bir mekanizma sorunu değil, weak-supervision etiket kalitesi sorunu" tanısını
(bkz. "Çekirdek ders" bölümü) BAĞIMSIZ bir üçüncü kez doğruluyor — artık class-weight (orijinal
tanı), UPOS-enjeksiyonu (Deney I) VE focal loss (bu deney) hepsi aynı duvara çarptı. Kanonik
`best_idiom_tagger.pt` vL'ye geri yüklendi (hash doğrulandı). `best_idiom_tagger_vFocal_gamma2.pt`
arşivde (gitignore'lu). `--focal-gamma` altyapısı kalıcı (varsayılan 0.0 = etkisiz).

## Deney P (2026-09-17) — stage-2 ENSEMBLE: Deney O'nun stage-2 karşılığı, 3 varyant, ÜÇÜ DE NEGATİF

Deney O stage-1'de ensemble çeşitliliğinin işe yaradığını gösterince, aynı fikir stage-2'de
denendi. Altyapı eklendi (kalıcı, repoda): `training/train_idiomaticity_clf.py::wrap_stage2`
artık virgülle ayrılmış birden çok checkpoint alıyor (p(literal) ortalaması — stage-1'in
`--ensemble` deseniyle aynı), `dizgebert_idiom/modeling_dizgebert_idiom.py::span_p_literal`
opsiyonel `temperature` parametresi aldı (varsayılan 1.0 = eskisiyle birebir aynı).

**Varyant 1 — ensemble(v2, v3), diskte hazır checkpoint'lerle (ücretsiz):** Çavuşoğlu tam
(198), doğru-ayırt v3 tek %54.0 → ensemble **%48.5 (düştü)**. v2/v3 AYNI SOYDAN (aynı 2173
kayıtlık veri, yalnız `--freeze` derinliği farklı) — Deney O'nun "aynı soyu ensemble'lamak
zarar veriyor" (vL+vF) dersinin stage-2 karşılığı.

**Varyant 2 — sıcaklık kalibrasyonu bunu düzeltir mi?** Held-out (3031 çift, pool artık Aşama
2/3 ingest'leriyle büyümüş) üzerinde v2/v3 için NLL-minimize eden T uyduruldu (T̂=6.9 / 3.7 —
v2 gerçekten aşırı-güvenli). Kalibre edilmiş ortalama held-out macro'da bile v3 tekten geçemedi
(70.55 vs 70.73). **Teşhis: sorun ölçek uyumsuzluğu değil, v2'nin gerçekten v3'ten daha az
bilgili bir model olması — kalibrasyon bunu düzeltemez.**

**Varyant 3 — GERÇEKTEN bağımsız veri ile ikinci stage-2 (vE/vL mantığının stage-2 karşılığı):**
`--min-idx 2173` bayrağı eklendi (`train_idiomaticity_clf.py`) — yalnız v3'ten SONRA eklenmiş
(idx≥2173, LLM/ajan-etiketli, 11281 kayıt, v3'ün orijinal 2173 elle-etiketinden AYRIK)
kayıtlarla eğit. `best_idiomaticity_clf_postv3.pt` (freeze 8, aynı reçete, best epoch 2,
held-out macro 72.7). Çavuşoğlu tam (198):

| | v3 tek | postv3 tek | ensemble(v3,postv3) |
|---|---|---|---|
| doğru-ayırt | **%54.0** | %51.5 | %53.5 |

Bağımsız veri, v2+v3'ten (48.5) daha az kötü ama yine de v3 tekten (54.0) DÜŞÜK — postv3 tek
başına da v3'ten zayıf (LLM/ajan etiketleri stage-2 için hep daha gürültülü çıktı, bkz. stage-2
v4/v4b). **Sonuç: 3 varyantın ÜÇÜ de v3'ü geçemedi.** Stage-1'de ensemble işe yaradı çünkü
vE ve vL AYRI AYRI GÜÇLÜYDÜ (ikisi de tek başına %52-55 doğru-ayırt) ve farklı kör noktaları
vardı; stage-2'de v3'ten başka hiçbir varyant (v2, postv3) tek başına v3'e yakın değil —
ensemble'ın işe yaraması için önce iki eşit-güçte-ama-bağımsız model gerekir, ki stage-2'de
elimizde öyle bir ikinci model yok (post-v3 verisi kalite olarak yapısal biçimde daha zayıf).
**Stage-2 ensemble ekseni artık kapalı** — bu, stage-2'nin 10. bağımsız negatifi (7 negatif +
2 pilot + bu = 10). Kanonik `best_idiomaticity_clf_v3.pt` DEĞİŞMEDİ. `best_idiomaticity_clf_postv3.pt`
silindi (gitignore'lu, tekrar üretilebilir: `--min-idx 2173`). Altyapı (`--stage2` çoklu-checkpoint,
`temperature` param) kalıcı — ileride GERÇEKTEN eşit-güçte iki stage-2 kaynağı bulunursa hazır.

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
