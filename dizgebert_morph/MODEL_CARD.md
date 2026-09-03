---
language: tr
license: cc-by-sa-4.0
library_name: transformers
pipeline_tag: token-classification
tags:
- token-classification
- morphological-analysis
- morphological-disambiguation
- universal-dependencies
- turkish
- electra
datasets:
- universal_dependencies
---

# DizgeBERT-Morph

UD-uyumlu, ELECTRA tabanlı Türkçe **morfolojik belirsizlik gidericisi**
(*morphological disambiguator*). Ön-token'lanmış bir cümle verildiğinde her kelime için
**UPOS + XPOS + FEATS** (Universal Dependencies morfolojik özellikleri) tahmin eder.

- **Gövde:** [`dbmdz/electra-base-turkish-cased-discriminator`](https://huggingface.co/dbmdz/electra-base-turkish-cased-discriminator)
- **Kelime temsili:** ilk subword ⊕ son subword — Türkçe'de çekim bilgisi son eklerde olduğu
  için (`evlerinden` → `ev ##ler ##inden`), yalnızca ilk subword suffiks bilgisini kaybeder.
- **Head'ler:** UPOS, XPOS ve her UD FEATS kategorisi için ayrı softmax (kategori-başına
  çoklu head — Stanza / UDPipe-2 / Trankit tarzı).
- `iatagun/DizgeBERT-Dep` bağımlılık ayrıştırıcısını beslemek üzere birlikte tasarlandı.

## Çok-treebank / şema seçimi

**UD_Turkish-Kenet + BOUN + IMST** üzerinde, her treebank kendi native şemasında eğitildi.
Bu üç treebank yapısal olarak farklı işaretlenir:

| | Kenet | BOUN | IMST |
|---|---|---|---|
| Çekimli fiil | `VerbForm=Fin` + `Mood` | `Evident` + MWT bölme (VERB+AUX) | `Mood`, MWT bölme |
| Delillilik | – | `Evident=Fh/Nfh` | `Evident=Nfh` |

Model bir `treebank_id` embedding'i taşır (encoder'a `token_type_ids`, head'lere concat);
çıktı şeması `scheme` argümanıyla seçilir:

- `scheme="kenet"` (varsayılan) — `iatagun/DizgeBERT-Dep` ile uyumlu
- `scheme="boun"` — BOUN şeması
- `scheme="imst"` — IMST şeması (UD literatürünün referans treebank'i)

## Sonuçlar — held-out test

| treebank | UPOS acc | XPOS acc | UFeats F1 | FEATS exact-match |
|---|---|---|---|---|
| **IMST** | 95.98 | 95.77 | **96.60** | 92.93 |
| **Kenet** | 93.73 | – | **93.49** | 89.73 |
| **BOUN** | 93.21 | 86.62 | **92.12** | 82.87 |

UFeats F1 = CoNLL-2018 UFeats metriği. Referans (tr, yayınlanmış): UDPipe-2 ~94, Stanza ~92-94,
Trankit ~93-95 (çoğunlukla in-domain / IMST).

## Kullanım

```python
from transformers import AutoModel, AutoTokenizer

m = AutoModel.from_pretrained("iatagun/DizgeBERT-Morph", trust_remote_code=True).eval()
tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Morph")

words = ["Yarın", "İstanbul'a", "gideceğim", "."]
for upos, xpos, feats in m.predict(words, scheme="imst", tokenizer=tok):
    print(upos, xpos, feats)
```

Ham logit'ler için: `m(input_ids, attention_mask, treebank_id, first_pos, last_pos)` →
`ModelOutput(logits_upos, logits_xpos, logits_feats: dict)`.

## Kısıtlar

- **Ön-token'lanmış** girdi bekler (kelime listesi). Ham metin için harici bir tokenizer +
  çok-kelimeli token (MWT) bölücü gerekir — model bunu yapmaz.
- **Sözdizimsel-tanımlı ayrımlar.** UD'de bazı UPOS ayrımları kelimenin ağaçtaki işlevine göre
  *tanımlanır*: bir isme `det` bağlı demonstratif → `DET`, kendisi nominal ise → `PRON`
  (`o çocuk` vs `onu gördüm`); adlaşmış sıfat-fiil `amod` ise `ADJ`, nominal baş ise `NOUN`
  (`gelecek hafta` vs `çocukların geleceği`). HEAD/DEPREL görmeyen yalnız-token bir modelin
  bu sınıflarda tavanı vardır. → [`iatagun/DizgeBERT-Joint`](https://huggingface.co/iatagun/DizgeBERT-Joint)
  (tek geçişte etiketleme + ayrıştırma) veya repo'daki `hybrid.py` (Morph etiketleme + Joint
  ayrıştırma + güvenli UPOS düzeltmesi) bu vakaları çözer.
- BOUN'da nadir kategoriler (`Abbr`, `Reflex`, `Polite`) seyrek anotasyon nedeniyle zayıf.
- Üleştirme sayıları (`beşer` → `NumType=Dist`) eğitimde seyrek; `Card` tahmin edilebilir.
- XPOS yalnızca BOUN ve IMST'te öğrenilir; `scheme="kenet"` iken XPOS ≈ `_`.
- Eğitim setinde çok seyrek görülen sözlüksel okumalarda hata olabilir
  (ör. `beni` = "leke" NOUN, IMST'te ~2 örnek).

## Eğitim

- `dbmdz/electra-base-turkish-cased-discriminator` + kelime-düzeyi first⊕last pooling
- 10 epoch, batch 16, LR 2e-5, AdamW, linear warmup 0.1, dropout 0.15
- UPOS/XPOS loss ağırlıkları ×3 / ×2 (FEATS kategorileri ×1)
- train: UD_Turkish Kenet (15.398) + BOUN (7.803) + IMST (3.435) + morfolojik belirsizlik için
  sentetik minimal-çiftler (~1.3k, gerçek taşıyıcı cümlelere yerleştirilmiş eş-yazımlar)

## Atıf

Bu modeli kullanıyorsanız lütfen eğitim verisini oluşturan üç UD treebank'ini atıflayın.

### UD_Turkish-IMST

```bibtex
@article{SulubacakEryigit2018,
  title   = {Implementing Universal Dependency, Morphology and Multiword Expression
             Annotation Standards for {Turkish} Language Processing},
  author  = {Sulubacak, Umut and Eryi{\u{g}}it, G{\"u}l{\c{s}}en},
  journal = {Turkish Journal of Electrical Engineering \& Computer Sciences},
  year    = {2018},
  pages   = {1--23},
  doi     = {10.3906/elk-1706-81}
}
@inproceedings{SulubacakEtAl2016,
  title     = {Universal Dependencies for {Turkish}},
  author    = {Sulubacak, Umut and G{\"o}k{\i}rmak, Memduh and Tyers, Francis and
               {\c{C}}{\"o}ltekin, {\c{C}}a{\u{g}}r{\i} and Nivre, Joakim and
               Eryi{\u{g}}it, G{\"u}l{\c{s}}en},
  booktitle = {Proceedings of COLING 2016},
  address   = {Osaka, Japan},
  year      = {2016}
}
```

### UD_Turkish-BOUN

```bibtex
@article{marsan2022enhancements,
  title   = {Enhancements to the {BOUN} Treebank Reflecting the Agglutinative Nature of {Turkish}},
  author  = {Mar{\c{s}}an, B{\"u}{\c{s}}ra and Akkurt, Salih Furkan and {\c{S}}en, Muhammet and
             G{\"u}rb{\"u}z, Merve and G{\"u}ng{\"o}r, Onur and {\"O}zate{\c{s}}, {\c{S}}aziye Bet{\"u}l and
             {\"U}skudarl{\i}, Suzan and {\"O}zg{\"u}r, Arzucan and G{\"u}ng{\"o}r, Tunga and {\"O}zt{\"u}rk, Balk{\i}z},
  journal = {arXiv preprint arXiv:2207.11782},
  year    = {2022}
}
@article{TurkEtAl2022,
  title   = {Resources for {Turkish} Dependency Parsing: Introducing the {BOUN Treebank}
             and the {BoAT} Annotation Tool},
  author  = {T{\"u}rk, Utku and Atmaca, Furkan and {\"O}zate{\c{s}}, {\c{S}}aziye Bet{\"u}l and
             Berk, G{\"o}zde and Bedir, Seyyit Talha and K{\"o}ksal, Abdullatif and
             Ba{\c{s}}aran, Balk{\i}z {\"O}zt{\"u}rk and G{\"u}ng{\"o}r, Tunga and {\"O}zg{\"u}r, Arzucan},
  journal = {Language Resources and Evaluation},
  volume  = {56},
  number  = {1},
  pages   = {259--307},
  year    = {2022},
  doi     = {10.1007/s10579-021-09558-0}
}
```

### UD_Turkish-Kenet

```bibtex
@misc{KuzgunEtAl2022Kenet,
  title  = {{UD} {Turkish}-{Kenet}},
  author = {Kuzgun, Asl{\i} and Cesur, Neslihan and Y{\i}ld{\i}z, Olcay Taner and
            Kuyruk{\c{c}}u, O{\u{g}}uzhan and San{\i}yar, Ezgi},
  note   = {Universal Dependencies},
  howpublished = {\url{https://universaldependencies.org/treebanks/tr_kenet/}},
  year   = {2022}
}
@inproceedings{KuzgunEtAl2020,
  title     = {On Building the Largest and Cross-Linguistic {Turkish} Dependency Corpus},
  author    = {Kuzgun, Asl{\i} and Cesur, Neslihan and Ar{\i}can, Bilge Nas and
               {\"O}z{\c{c}}elik, Merve and Mar{\c{s}}an, B{\"u}{\c{s}}ra and Kara, Neslihan and
               Aslan, Deniz Baran and Y{\i}ld{\i}z, Olcay Taner},
  booktitle = {2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  address   = {Istanbul, Turkey},
  publisher = {IEEE},
  year      = {2020},
  doi       = {10.1109/ASYU50717.2020.9259799}
}
```

## Lisans

Eğitim verisi UD_Turkish-Kenet + UD_Turkish-BOUN + UD_Turkish-IMST — üçü de **CC BY-SA 4.0**.
Model bu nedenle **CC BY-SA 4.0** (ShareAlike) altında yayınlanır.
