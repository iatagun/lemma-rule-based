---
language: tr
license: cc-by-sa-4.0
library_name: transformers
pipeline_tag: token-classification
tags:
- token-classification
- dependency-parsing
- morphological-analysis
- universal-dependencies
- turkish
- electra
datasets:
- universal_dependencies
---

# DizgeBERT-Joint

Tek ELECTRA geçişinde Türkçe için **UPOS + XPOS + FEATS + HEAD + DEPREL** — yani
morfolojik etiketleme **ve** bağımlılık ayrıştırması birlikte.

- **Gövde:** [`dbmdz/electra-base-turkish-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)
- **Kelime temsili:** ilk subword ⊕ son subword
- **Etiketleme:** UPOS/XPOS + kategori-başına FEATS head'leri (bkz. [`iatagun/DizgeBERT-Morph`](https://huggingface.co/iatagun/DizgeBERT-Morph))
- **Ayrıştırma:** Dozat & Manning (2017) deep biaffine + root vektörü + Chu-Liu/Edmonds MST
- **Çok-treebank:** UD_Turkish-Kenet + BOUN + IMST native şemalarında; `treebank_id` embedding'i,
  çıkarımda `scheme` ile seçilir (`kenet` varsayılan)

## Neden joint?

UD'de bazı UPOS ayrımları kelimenin ağaçtaki işlevine göre *tanımlıdır*: bir isme `det`
bağlı demonstratif → `DET`, kendisi nominal ise → `PRON` (`o çocuk` vs `onu`); genitive
tamlayan → nominal. HEAD/DEPREL ortak öğrenildiğinde bu vaka sınıfı çözülür — yalnız-token
bir etiketleyicinin (DizgeBERT-Morph) yapısal olarak yapamadığı şey.

Karşılık: saf morfolojik doğruluk `DizgeBERT-Morph`'tan ~3 puan düşük. En iyi sonuç için
`hybrid.py` (Morph etiketleme + Joint ayrıştırma + güvenli UPOS düzeltmesi) —
[repo](https://github.com/iatagun/lemma-rule-based).

## Sonuçlar — held-out test

| treebank | UPOS | XPOS | UFeats F1 | FEATS exact | UAS | LAS |
|---|---|---|---|---|---|---|
| Kenet | 93.4 | – | 92.6 | 86.5 | 89.0 | 75.6 |
| BOUN | 92.9 | 84.8 | 90.5 | 79.7 | 84.0 | 75.5 |
| IMST | 92.5 | 92.9 | 92.9 | 83.5 | 86.0 | 76.1 |

## Kullanım

```python
from transformers import AutoModel, AutoTokenizer

m = AutoModel.from_pretrained("iatagun/DizgeBERT-Joint", trust_remote_code=True).eval()
tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Joint")

words = ["Çocukların", "geleceği", "bizim", "elimizde", "."]
for form, upos, xpos, feats, head, deprel in m.predict(words, scheme="kenet", tokenizer=tok):
    print(form, upos, feats, f"head={head}", deprel)
# geleceği NOUN Case=Nom|... head=4 nsubj
```

Ham logit'ler: `m(input_ids, attention_mask, treebank_id, first_pos, last_pos)` →
`ModelOutput(logits_upos, logits_xpos, logits_feats, arc, lab)`.

## Kısıtlar

- **Ön-token'lanmış** girdi bekler; ham metin için harici tokenizer + MWT bölücü gerekir.
- Saf morfoloji `DizgeBERT-Morph`'tan düşük; ayrıştırma uzman bir parser'ın altında (LAS ~76).
- IMST genitive/adlaşmış tamlayanları ADJ etiketler (BOUN/Kenet NOUN) — `scheme` seçimi önemli.

## Eğitim

- 10 epoch (best epoch 11'de kilitlendi), batch 12, encoder LR 1e-5 / head LR 5e-5
- Etiketleme kaybı ×2.5 (arc + label ×1) — aksi halde ayrıştırma gradyanı etiketlemeyi bastırıyor
- train: Kenet (15.398) + BOUN (7.803) + IMST (3.435) + ~1.3k sentetik belirsizlik minimal-çifti
- 45 deprel (sayım ≥ 20; nadir alt-türler → `dep`)

## Lisans & atıf

Eğitim verisi UD_Turkish-Kenet + BOUN + IMST (CC BY-SA 4.0) → model **CC BY-SA 4.0**.
Treebank atıfları için [`iatagun/DizgeBERT-Morph`](https://huggingface.co/iatagun/DizgeBERT-Morph#atıf).
