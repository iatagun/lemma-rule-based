# DizgeTTS — Aşama 4: baseline (espeak-ng) eğitim hazırlığı (2026-09-23)

Eğitim **başlatılmadı**; onay bekliyor. Yazılan: config'ler, eğitim kodu, pilot ölçümü.

## Dosyalar
- `configs/train_base.yaml` (ortak) → `baseline_espeak.yaml` (`frontend: espeak`) ve `main_dizge.yaml` (`frontend: dizge`, Aşama 5). İki model aynı veri/split/mel/model/init/ayarı paylaşır; fark yalnız fonem kaynağı.
- `train/train.py`, `train/data.py`; `frontend/symbols_espeak.py` (46 espeak sembolü + PAD).
- Her koşu `D:\dizgetts\runs\<ad>_<zaman>\`: `config.resolved.yaml`, `env.json` (git commit + dirty bayrağı, tohum, sürümler, GPU), `symbols.json`, `metrics.csv`, tensorboard, `last.pt`, `best.pt`, `ep{25,50,…}.pt`.

## Kararlar (senin "önerdiğin neyse" yetkinle)
| Konu | Karar | Neden |
|---|---|---|
| Başlangıç | **LJSpeech ön-eğitimli Matcha'dan aktarım** (302 tensör; `encoder.emb` ve mel istatistiği hariç, yeniden başlar) | 4,2 saatlik veri sıfırdan için az; iki model aynı başlangıçla → adil. Checkpoint 219 MB, HiFi-GAN universal 56 MB indirildi (`D:\dizgetts\pretrained`). |
| Hassasiyet | **fp32, AMP yok** | Aşama 1: fp16+cuDNN bu kartta NaN. |
| Batch | **8 × grad_accum 2 (etkin 16)**, `out_size=172` | Pilotta bs=16 gerçek veride 4 GiB'ı aştı (reserved 5,5 GiB, paylaşımlı belleğe taşma → 1,31 s/adım). bs=8: tepe 1,8 GiB, 0,38 s/mini-batch. Smoke testteki 2,5 GiB rastgele-veri ölçümü gerçek uzun cümlelerde yetersiz kalmıştı. |
| Allocator | `expandable_segments:True` | parçalanmayı azaltıyor |
| LR | 1e-4 Adam, grad clip 5 | upstream |

## Pilot (30 güncelleme, gerçek veri)
0,38 s/mini-batch → **epoch ≈ 43 s** (114 mini-batch) + doğrulama 4 s / 5 epoch. Tepe bellek 1,81 GiB. Val kaybı 30 güncellemede 2,00 (dur 0,7 yüksek: yeni embedding).
**Tahmin: 400 epoch ≈ 4,9 saat / model** (baseline + ana ≈ 10 saat). 400 epoch bir tahmin: ön-eğitimli checkpoint LJSpeech'te 2.551 epoch × ~13.100 klip ≈ 33 M örnek gördü, 910 klip × 400 epoch ≈ 0,36 M örnek (%1). Aşırı uyum riski yüksek; `save_every: 25` ile ara checkpoint'ler tutulur, seçim Aşama 6'da ASR CER ile yapılır (val kaybı zayıf vekil).

## Bilinen karıştırıcılar (raporda dürüst yazılmalı)
1. **espeak çıktısı vurgu işareti taşıyor (`ˈ` 31.671 kez), dizge fonemleri (Aşama 5) taşımıyor.** Baseline bu yüzden haksız avantajlı olabilir. Config'te `espeak_strip_stress: true` ile vurgusuz baseline ablasyonu mümkün (ek 4,9 saat). Prosodi modelleri bağlanınca ana model de vurgu alacak.
2. Aynı tohum ve aynı başlangıç, ama iki koşu arasındaki gürültü (tek tohum) ölçülmedi; küçük farklar kanıt sayılmamalı.
3. Val kaybı 59 klip, rastgele-t CFM: gürültülü (3 tekrar ortalaması).

## Prosodi planı (iatagun modelleri; Aşama 5, henüz yapılmadı)
Duraklama: DizgeBERT-Dep öbek sınırı + noktalama. Vurgu: dizge-syllable + DizgeBERT-Morph (Türkçe vurgu kuralları: son hece; ön-vurgulu ekler, olumsuzluk, soru eki, yer adları/zarflar istisna). Her modelin doğruluğu kullanmadan önce ölçülecek; sonuç ayrı bir config'le (`main_dizge_prosody`) karşılaştırılır (ablasyon: yalnız noktalama vs +vurgu vs +öbek).
