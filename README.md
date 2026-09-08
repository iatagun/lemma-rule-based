# DizgeBERT — Türkçe NLP fine-tuning stack

Türkçe için ELECTRA (`dbmdz/electra-base-turkish-cased`) tabanlı ince-ayar modelleri,
veri pipeline'ları, Hugging Face paketleme ve benchmark altyapısı.

> **Not:** Repo adı (`lemma-rule-based`) tarihsel — proje başlangıçta kural-tabanlı bir
> morfolojik çözümleyiciydi. O katman kaldırıldı (git geçmişinde duruyor); repo artık
> yalnızca DizgeBERT fine-tuning işleri için kullanılıyor.

## Modeller

| Model | Görev | Hugging Face |
|---|---|---|
| **DizgeBERT-Morph** | UPOS + XPOS + FEATS morfolojik etiketleme (çok-treebank) | [`iatagun/DizgeBERT-Morph`](https://huggingface.co/iatagun/DizgeBERT-Morph) |
| **DizgeBERT-Joint** | Morfoloji + bağımlılık (HEAD/DEPREL) tek modelde | [`iatagun/DizgeBERT-Joint`](https://huggingface.co/iatagun/DizgeBERT-Joint) |
| **DizgeBERT-Dep** | Bağımlılık ayrıştırma (ELECTRA + Biaffine, Dozat & Manning 2017) | [`iatagun/DizgeBERT-Dep`](https://huggingface.co/iatagun/DizgeBERT-Dep) |
| **DizgeBERT-Idiom** | Deyim (VID) / eşdizim (LVC) span + 2-aşamalı idyomatiklik filtresi | [`iatagun/DizgeBERT-Idiom`](https://huggingface.co/iatagun/DizgeBERT-Idiom) |

Modeller `trust_remote_code=True` ile yüklenir; her biri `dizgebert_<x>/` altında kendi
`configuration` + `modeling` + `MODEL_CARD` dosyalarıyla paketlenir.

## Dizin yapısı

```
training/     train_{morph,joint,dep,idiom}_bert.py · train_idiomaticity_clf.py
data/         fetch_*.py · prepare_*.py · filter_corpus_idiomaticity.py · generate_synthetic_morph.py
              treebanks/UD_Turkish-Kenet/   (vendored UD veri)
inference/    predict_{morph,idiom}.py · push_{morph,idiom}_hf.py
benchmark/    eval_{idiom,morph,ambiguity}.py · REPORT.md
dizgebert_{morph,joint,idiom}/   HF paketleri (kökte — ada göre import + export'a kopyalanır)
{morph,idiom,idiom_arc}_data/    gitignore'lu veri (yalnız label_space.json izlenir)
tests/        pytest label-space / çözümleme değişmezleri
```

Tüm scriptler **repo kökünden** çalıştırılır:

```bash
python training/train_idiom_bert.py --class-weights --tdk-examples --epochs 10
python benchmark/eval_idiom.py --local --checkpoint idiom_data/best_idiom_tagger.pt --mode all
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Veri pipeline'ları

- **Morph/Joint:** `data/fetch_boun.py` + vendored Kenet → `data/prepare_morph_data_ud.py`
  → `morph_data/{train,dev,test}.json`. Belirsizlik için `data/generate_synthetic_morph.py`.
- **Idiom:** `data/fetch_parseme_tr.py` (PARSEME-TR) + `data/fetch_tdk_deyim.mjs` (TDK sözlük)
  + `data/fetch_leipzig_tr.py` (Leipzig derlemi) → `data/prepare_idiom_data.py`,
  `data/prepare_tdk_*.py` (snowball gövde-eşleştirme), `data/filter_corpus_idiomaticity.py`.

## Testler

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

## Lisans

Kod: bkz. `LICENSE`. Modeller eğitim verilerinin lisansını miras alır —
DizgeBERT-Idiom PARSEME-TR (CC BY-NC-SA 4.0) nedeniyle **ticari kullanıma kapalı**;
diğerleri için ilgili `dizgebert_<x>/MODEL_CARD.md`'ye bakın.
