---
name: idiom
description: Türkçe deyim (VID) / eşdizim (LVC) span tespiti — DizgeBERT-Idiom. GLU çok-ölçütlü etiketleme karar çerçevesi (bir yapı bu bağlamda deyim mi, eşdizim mi, terim mi, literal mi?) + deney günlüğü (v6-v14 tek-model kaldıraçları TÜKENDİ; iki-aşamalı detect→filter ÇALIŞIYOR ve YAYINLANDI v3) + stage-2 idyomatiklik sınıflandırıcısı iş akışı. Trigger — DizgeBERT-Idiom üzerinde çalışırken, deyim/MWE eğitim verisi hazırlarken/etiketlerken, idyomatik-literal ayrımı, stage-2 sınıflandırıcı, `find_span`/`prepare_*idiom*`/`filter_corpus_idiomaticity`, `/deyim` ya da `/idiom`.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
user-invokable: true
---

# DizgeBERT-Idiom — Deyim / Eşdizim Span Tespiti

`lemma-rule-based` reposunda ELECTRA tabanlı Türkçe deyim (VID) / eşdizim (LVC.full)
BIO span etiketleyici. **Yayınlandı** (`huggingface.co/iatagun/DizgeBERT-Idiom`).

- **Stage-1 (span modeli) = v5**, değişmedi: `idiom_data/best_idiom_tagger_v5_bigappy.pt`,
  ELECTRA (`dbmdz/electra-base-turkish-cased-discriminator`), bigappy-unicrossy 2-katman + Viterbi.
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

**Ders:** stage-2 tavanı (a) daha çok veri, (b) ikili çerçeve, (c) cümle-bağlamı mimarisi ile
kırılmıyor — tek-model tavanının (v6-v14) stage-2 karşılığı. Mimari değişikliği geri alındı.

`data/filter_corpus_idiomaticity.py` yeni bayraklar (`stage2-llm-labels` dalı, main'e merge
kararı açık): `--gate` (LLM'i frozen sette kıyasla), `--new-idioms-only` (kapsam örneklemesi),
`--ingest-llm` (append-only). GLU prompt'u İKİLİ (D/N) + `sözlük biçimi: {idiom}` satırda.
anthropic.com uçları: `temperature` at + `thinking:{type:disabled}` (yoksa boş içerik).

### Hâlâ açık (düşük beklenen değer)
- Stage-2 ~1k elle-etiketli örnekte overfit; literal kullanımların ~%16-18'i geçiyor.
- Denenmemiş: contrastive margin kaybı, elle/LLM etiket karışım ağırlığı, stage-1 p(literal)
  özelliği. Meta-örüntü net — tavan yapısal.
- **Minimal çiftler / hard negative** (GLU `glu_karar_cercevesi.md`) hâlâ en keskin eval sinyali.

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
- Eşleştirme kuralı her yerde AYNI olmalı: sıkı ardışık GÖVDE alt-dizisi (`find_span`,
  `data/prepare_tdk_idiom_examples.py`; `stem()` artık snowballstemmer). Gevşetme
  (fuzzy/threshold) DENEME — v7/v8 dersi.
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
`data/prepare_tdk_idiom_examples.py` (+frozen-split, snowball stemmer),
`data/fetch_leipzig_tr.py`, `data/prepare_tdk_corpus_examples.py` (stem-map cache + sıkı eşleşme),
`data/filter_corpus_idiomaticity.py`, `data/prepare_glu_examples.py`,
`training/train_idiom_bert.py`, `training/train_idiomaticity_clf.py`,
`dizgebert_idiom/` (config+modeling+MODEL_CARD, kökte),
`benchmark/eval_idiom.py`, `inference/predict_idiom.py`, `inference/push_idiom_hf.py`,
`tests/test_idiom_labelspace.py`. **Arşiv/kullanılmıyor**: `data/prepare_idiom_arc_data.py`,
`training/train_idiom_arc_bert.py` (reddedilen arc-classification).

Kaynak kılavuz: `GLU_Deyim_Etiketleme_Ogretmen_Kilavuzu.pdf` (repo kökü).
