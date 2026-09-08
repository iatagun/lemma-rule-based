---
name: dizgebert-stack
description: Turkish NLP fine-tuning stack — DizgeBERT ELECTRA models (Morph / Joint / Dep / Idiom)
allowed-tools: Bash, Read, Edit, Write
user-invokable: true
---

# DizgeBERT fine-tuning stack

Türkçe için ELECTRA (`dbmdz/electra-base-turkish-cased`) ince-ayar modelleri, veri
pipeline'ları, HF paketleme ve benchmark. (Eski kural-tabanlı morfoloji/dep katmanı
kaldırıldı — git geçmişinde.)

## Modeller

| Model | Görev | HF repo |
|---|---|---|
| **DizgeBERT-Morph** | UPOS + XPOS + FEATS morfolojik etiketleme | `iatagun/DizgeBERT-Morph` |
| **DizgeBERT-Joint** | Morph + bağımlılık (HEAD/DEPREL) birlikte | `iatagun/DizgeBERT-Joint` |
| **DizgeBERT-Dep** | Bağımlılık ayrıştırma (ELECTRA + Biaffine) | `iatagun/DizgeBERT-Dep` |
| **DizgeBERT-Idiom** | Deyim (VID) / eşdizim (LVC) span + idyomatiklik (2-aşama) | `iatagun/DizgeBERT-Idiom` |

## Dizin haritası

```
training/     train_{morph,joint,dep,idiom}_bert.py, train_idiomaticity_clf.py
data/         fetch_*.py, prepare_*.py, filter_*.py, generate_synthetic_morph.py
              data/treebanks/UD_Turkish-Kenet/  (vendored UD data)
inference/    predict_{morph,idiom}.py, push_{morph,idiom}_hf.py
benchmark/    eval_{idiom,morph,ambiguity}.py
dizgebert_{morph,joint,idiom}/   HF trust_remote_code paketleri (config + modeling + MODEL_CARD)
{morph,idiom,idiom_arc}_data/    gitignore'lu veri (yalnız label_space.json izlenir)
```

Scriptler repo kökünden çalıştırılır: `python training/train_idiom_bert.py ...`

## Komutlar

```bash
# Idiom
python training/train_idiom_bert.py --class-weights --tdk-examples --epochs 10
python benchmark/eval_idiom.py --local --checkpoint idiom_data/best_idiom_tagger.pt --mode all

# Morph / Joint
python training/train_morph_bert.py --epochs 10
python benchmark/eval_morph.py                          # BOUN test, neural

# Testler
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

## Alt-skill'ler

- **idiom** — DizgeBERT-Idiom durumu, deney günlüğü, GLU karar çerçevesi
- **dep_parsing** — ELECTRA + Biaffine bağımlılık ayrıştırma
- **karpathy** — kodlama disiplini (think before coding, simplicity, surgical changes)
