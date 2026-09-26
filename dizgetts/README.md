# DizgeTTS — Türkçe metinden konuşma

Antalia (CC-BY-4.0) + Matcha-TTS + `dizge` fonemleri + dilbilimsel vurgu. **Kalite iddiası: "en doğal ses" değil, "Türkçeyi en doğru okuyan açık TTS".**

## Nereden bakılır
| Ne | Nerede |
|---|---|
| Şu an ne yapılıyor / hangi deney | `experiments.yaml` -> `reports/EXPERIMENTS.md` (otomatik) |
| Eski (v1) dönem özeti | `reports/EXPERIMENT_LOG.md`, kod `_v1_archive/` |
| Sesbilim bilgisi (vurgu vb.) | `docs/turkish_phonology.md`, `tests/stress_gold.tsv` |
| Motor (metin -> token -> ses) | `engine.py` (Utterance/Engine), aşamalar `frontend/` |

## Komutlar (venv: `D:\dizgetts\venv`, paket `pip install -e dizgetts` ile kurulu; hepsi `-X utf8 -m ...`)
```
python -m dizgetts.engine "Merhaba, 3'te buluşalım."              # aşama çıktıları (normalize, fonemler, vurgu, token)
python -m dizgetts.engine "cümle" --ckpt <ep150.pt> -o out.wav   # ses
python -m dizgetts.train.train --config dizgetts/configs/<cfg>.yaml [--pilot]
python -m dizgetts.eval.evaluate --ckpt <ep150.pt> --label <etiket> [--extra dizgetts/eval/extra_sentences_ud.txt]   # varsayılan YALNIZ test bölümü: CER/WER + UTMOS
python -m dizgetts.eval.compare <etiket> <etiket> --splits test extra   # id ile eşleşmiş bootstrap; karar için val KULLANMA
python -m dizgetts.eval.report                                   # EXPERIMENTS.md üret
python -m dizgetts.tests.test_normalize | test_phonemize | test_stress | test_engine_stages | test_engine_parity | test_compare | test_packaging   # torch'suz (CI'da da koşar)
python -m dizgetts.tests.test_dpfeat | test_tagger               # torch/matcha/HF (+checkpoint) ister, yerelde
```

## Kurallar
1. Aynı anda en fazla 1 eğitim. Yeni koşu = önce `experiments.yaml`'a hipotez + karar kuralı.
2. Eğitim ve çıkarım aynı `Engine` koduyla token üretir (`test_engine_parity` bunu garanti eder).
3. Ağır dosyalar (venv, veri, koşular, ön-eğitimli ağırlıklar) `D:\dizgetts` altında, repoya girmez. Başka makinede `DIZGETTS_HOME` ortam değişkeni (`paths.py`); `configs/*.yaml` kendi yollarını taşır.
4. Ayar/seçim (epoch, eşik, `length_scale`, dp erken durdurma) `val`'de yapılır; sonuç ve karar kuralı `test` (+ `extra`) üzerinde. `val` üstünde raporlanan sayı karar sayısı değildir.
5. Kurulum: çekirdek ön uç `pip install dizgetts` (yalnız `dizge`); g2ptts için `pip install "dizgetts[g2ptts]"`; eğitim/değerlendirme yığını `requirements.txt`.
