---
name: idiom
description: Türkçe deyim (VID) / eşdizim-yardımcıfiil (LVC) span tespiti — DizgeBERT-Idiom. GLU çok-ölçütlü etiketleme karar çerçevesi (bir yapı bu bağlamda deyim mi, eşdizim mi, terim mi, literal mi?) + projenin gerçek deneylerinden çıkan dersler (weak-supervision idyomatikliği ayırt edemez, precision tavanı, v6/v7/v8'de nelerin işe YARAMADIĞI). Trigger — DizgeBERT-Idiom üzerinde çalışırken, deyim/MWE eğitim verisi hazırlarken/etiketlerken, idyomatik-literal ayrımı, `find_span`/`prepare_*idiom*` dosyaları, `/deyim` ya da `/idiom`.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
user-invokable: true
---

# DizgeBERT-Idiom — Deyim / Eşdizim Span Tespiti

`lemma-rule-based` reposunda ELECTRA tabanlı Türkçe deyim (VID) / eşdizim (LVC.full)
BIO span etiketleyici. **Yayınlandı** (`huggingface.co/iatagun/DizgeBERT-Idiom`).
Kanonik model = **v5** (`idiom_data/best_idiom_tagger_v5_bigappy.pt`), ELECTRA
(`dbmdz/electra-base-turkish-cased-discriminator`), bigappy-unicrossy 2-katman + Viterbi.

## Bu skill ne için

1. **`glu_karar_cercevesi.md`** — bir yapının bu bağlamda deyim mi olduğuna KARAR VERME
   çerçevesi (Gülsün Leylâ Uzun, Ankara Üniv. öğretmen etiketleme kılavuzundan damıtıldı).
   Eğitim verisi hazırlarken, örnekleri gözden geçirirken, eval seti kurarken **bunu esas al**.
2. Aşağıdaki **deney dersleri** — nelerin denendiği ve neden işe yaramadığı. Tekrar deneme.

## Çekirdek ders: precision tavanı bir VERİ/ENCODER sorunu DEĞİL

v5 span-F1: PARSEME test ~69.6, TDK held-out ~72.8, **precision ~%64-71**. Dış bağlam-
bağımlılık testinde (Çavuşoğlu & Çöltekin) idyomatik/literal doğru-ayırt yalnız ~%37.

**Denendi, HEPSİ reddedildi (2026-09-07):**

| deney | ne | sonuç |
|---|---|---|
| **v6** | ConvBERTurk-mC4 encoder (aile-uyumlu, 242GB ön-eğitim) | recall'a kaydı; TDK held-out −1, dış doğru-ayırt −10 |
| **v7** | +TDK crawl verisi (+144 örnek, frozen-split) | TDK held-out −3 |
| **v8** | +3M Leipzig derleminden madenlenen 29.7k örnek (sıkı gövde-eşleşmesi) | precision çöküşü 63.8→53.5, epoch 4'te kesildi |

**Neden:** her weak-supervision yöntemi (TDK stem-match, derlem sıkı-eşleşme) tek ölçüte
bakıyor — YÜZEY BİÇİM. GLU çerçevesinin ilk kuralı: *tek ölçüt yetmez*. Model "bu kelimeler
bu sırada = span" öğreniyor, idyomatik/literal ayrımını (asıl zaaf) öğrenmiyor. Daha fazla
böyle etiket = daha fazla gürültü = precision'ı recall'a takas.

## Doğru yön: idyomatiklik sinyali

- **Minimal çiftler** (aynı yüzey biçim, biri literal biri deyim) — GLU kılavuzu ~20 tane
  elle-seçilmiş veriyor (`glu_karar_cercevesi.md` §minimal-çiftler). Küçük ama yüksek sinyal:
  hem diagnostik eval hem "deyim-biçimi → O" eğitim örneği.
- **Hard negative**: bileşen kelimeleri ardışık geçen ama deyim OLMAYAN kullanımlar
  (bitki adları "aslan ağzı/kuşburnu" = terim; "karar vermek" = eşdizim; literal "yol aldı").
- **v8 derlem verisini idyomatiklik için filtrele** (LLM'e GLU §5 rubric'i ver → temiz %65-70).
- **İki aşama**: yüksek-recall BIO (mevcut) → span+cümle → {VID, LVC, literal/O} sınıflandırıcı.
  GLU'nun kendi yapısı da iki aşamalı (eleme → üç ölçüt).

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
  `dizgebert_idiom/modeling_dizgebert_idiom.py` içinde değil — `prepare_tdk_idiom_examples.py`).
  Gevşetme (fuzzy/threshold) DENEME — v7/v8 dersi.

## Komutlar

```bash
# eğitim (v5 reçetesi)
python train_idiom_bert.py --class-weights --tdk-examples --epochs 10

# değerlendirme — 4 eksen
python benchmark/eval_idiom.py --local --checkpoint idiom_data/best_idiom_tagger.pt --mode all
python train_idiom_bert.py --eval --checkpoint <ckpt> --eval-file idiom_data/tdk_examples_test.json

# TDK verisi yeniden üret (frozen-split otomatik)
python prepare_tdk_idiom_examples.py

# Leipzig derlem madenciliği
python fetch_leipzig_tr.py
python -u prepare_tdk_corpus_examples.py --cap 8

# HF export + push
python train_idiom_bert.py --checkpoint idiom_data/best_idiom_tagger.pt --export-hf dizgebert_idiom_hf/
python push_idiom_hf.py
```

## Dosya envanteri

`fetch_parseme_tr.py`, `prepare_idiom_data.py`, `fetch_tdk_deyim.mjs`,
`prepare_tdk_idiom_examples.py` (+frozen-split), `fetch_leipzig_tr.py`,
`prepare_tdk_corpus_examples.py` (stem-map cache + sıkı eşleşme taraması),
`train_idiom_bert.py`, `dizgebert_idiom/` (config+modeling+MODEL_CARD),
`benchmark/eval_idiom.py`, `predict_idiom.py`, `push_idiom_hf.py`,
`tests/test_idiom_labelspace.py`. **Arşiv/kullanılmıyor**: `prepare_idiom_arc_data.py`,
`train_idiom_arc_bert.py` (reddedilen arc-classification).

Kaynak kılavuz: `GLU_Deyim_Etiketleme_Ogretmen_Kilavuzu.pdf` (repo kökü).
