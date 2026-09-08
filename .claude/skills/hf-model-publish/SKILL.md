---
name: hf-model-publish
description: Bir DizgeBERT modelini (Morph / Joint / Dep / Idiom) Hugging Face'e trust_remote_code paketi olarak yayınlama ve companion Space'i (iatagun/dizge-demo) güncelleme prosedürü. Trigger — "modeli push et / yayınla", HF export, MODEL_CARD güncelleme, Space güncelleme, yeni bir model sürümü (vN) hazır olduğunda.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob
user-invokable: true
---

# DizgeBERT modelini HF'ye yayınlama

Her DizgeBERT modeli `trust_remote_code=True` ile yüklenen bir paket: `dizgebert_<x>/`
altında `configuration_dizgebert_<x>.py` + `modeling_dizgebert_<x>.py` + `MODEL_CARD.md`.
Export → round-trip test → push → (gerekirse) Space restart → smoke test.

## 1. Export

`training/train_<x>_bert.py` içindeki `export_hf(model, tokenizer, ls, out_dir, metrics, [stage2_ckpt])`:
- `config.json` (`auto_map` ile `AutoConfig`/`AutoModel` → `.py` dosyalarına işaret eder), `model.safetensors`, tokenizer dosyaları
- `configuration_*.py` + `modeling_*.py` `dizgebert_<x>/`'ten **verbatim kopyalanır** → modeling dosyasını düzenledikten sonra mutlaka yeniden export
- `README.md` `dizgebert_<x>/MODEL_CARD.md`'den üretilir → kart değişikliği bir sonraki export'ta yansır

```bash
python training/train_idiom_bert.py --checkpoint idiom_data/best_idiom_tagger_v5_bigappy.pt \
    --stage2-ckpt idiom_data/best_idiomaticity_clf_v3.pt --export-hf dizgebert_idiom_hf
```

`dizgebert_<x>_hf/` gitignore'lu (build çıktısı). `--stage2-ckpt`: ikinci ELECTRA gövdesini
`stage2_encoder.` / `stage2_head.` önekiyle aynı safetensors'a katar → ~440MB → ~880MB.

## 2. Round-trip doğrulama (PUSH ÖNCESİ ZORUNLU)

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('dizgebert_idiom_hf', trust_remote_code=True)"
python benchmark/eval_idiom.py --hf-repo dizgebert_idiom_hf --mode all   # yerel .pt sayılarını BİREBİR üretmeli
```
`--hf-repo <yerel-klasör>` çalışır; eval'ın HF yolu `m.predict_spans(...)`'i config varsayılanıyla
çağırır → `config.stage2=True` ise stage-2 filtresi otomatik. Sayılar `--local --checkpoint`
ile ölçülenlerle eşleşmiyorsa export bozuk.

## 3. Push

```bash
python inference/push_idiom_hf.py --message "DizgeBERT-Idiom v3 — <özet>"
```
HF içerik-adresli: 880MB paket olsa da yalnız **değişen tensörler** (~delta) yüklenir.
Push arka planda uzun sürer → `nohup ... &` + Monitor ile "push tamam" satırını bekle.

## 4. MODEL_CARD güncelle

`dizgebert_<x>/MODEL_CARD.md` — sonuç tablolarını, sürüm rozetini (vN), "deneysel" notlarını
güncelle. Dürüst ol: bilinen zaaflar, hangi metriğin neyi ölçtüğü (bkz. [[finetune-iteration]]).
Sonra **adım 1'i tekrarla** (README.md karttan üretiliyor).

## 5. Companion Space (`iatagun/dizge-demo`)

Space, modeli **çalışma zamanında** çeker (`idiom_tab.py::_load()` → `AutoModel.from_pretrained`).
**Model repo'su değişince Space KENDİLİĞİNDEN yenilenmez** — bellekteki eski sürüm kalır.

- **Yalnız model değişti (Space kodu aynı):** `python -c "from huggingface_hub import HfApi; HfApi().restart_space('iatagun/dizge-demo')"` → cold start yeni modeli çeker.
- **Space kodu da değişti:** ayrı repo. scratchpad'e klonla (`GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/iatagun/dizge-demo`), düzenle, commit+push → Space otomatik BUILD eder. `__pycache__/` yanlışlıkla commit'lenmesin (Space repo'sunda `.gitignore` yoksa ekle).
- Space stage'i: `HfApi().space_info('iatagun/dizge-demo').runtime.stage` — `RUNNING` olana kadar bekle (`BUILDING` → `RUNNING_APP_STARTING` → `RUNNING`).

## 6. Space smoke test

```python
from gradio_client import Client
c = Client("iatagun/dizge-demo")
c.view_api(all_endpoints=True)          # api_name'ler otomatik numaralı: /analyze, /analyze_1, ...
c.predict("Projede yol aldık.", True, api_name="/analyze_4")   # deyim sekmesi handler'ı
```
`opentelemetry` ImportError'u zararsız (bkz. [[finetune-iteration]] — env notları).

## 7. Kayıt

`memory/dizgebert-<x>-project.md` + `MEMORY.md`: yayınlanan sürüm, sayılar, commit hash'i,
Space durumu. git commit (kod + kart), gerekirse GitHub'a push.

## Promote yolu (yeni encoder / mimari)

`ENCODER_MODEL` sabiti (`training/train_<x>_bert.py`), `configuration_dizgebert_<x>.py` default,
`prepare_*_data*.py` sabiti, `label_space.json`. Bunları değiştirmeden önce
[[finetune-iteration]] kıyas disiplinini uygula — kazanç doğrulanmadan promote yok.
