# DizgeTTS deney günlüğü (v1 dönemi: 2026-09-23 → 09-24) — KAPATILDI

Kullanıcı kararıyla (2026-09-24 öğleden sonra) v1 dönemi kapatıldı: eğitimler durduruldu, tüm eğitilmiş checkpoint'ler (47 dosya, 9,6 GB) silindi. Aşağıdaki sayılar geriye dönük kayıttır; yeniden üretmek için `configs/*.yaml` + `runs/*/{config.resolved.yaml,metrics.csv,env.json}` duruyor. Silinmeyenler: kod, veri (`D:\dizgetts\data`), config'ler, raporlar, `docs/turkish_phonology.md`, ön-eğitimli Matcha/HiFi-GAN indirmeleri.

Ortak: Antalia (4,2 sa eğitim), Matcha-TTS, LJSpeech ön-eğitiminden aktarım, fp32, 150 epoch, TEK tohum. Değerlendirme: 143 eğitimde görülmeyen cümle, Whisper-small CER/WER + UTMOS (`eval/`).

| Koşu | Fikir | Sonuç (CER / WER, ep150) |
|---|---|---|
| baseline_espeak | espeak-ng fonemleri (vurgu işaretli) | **3,0 / 11,1** |
| baseline_espeak_nostress | espeak, vurgu atıldı | 4,3 / 12,4 |
| main_dizge | dizge 0.1.6 fonemleri, sözcük ayracı token'ı | 4,6 / 13,1 |
| main_dizge_feat | ayraç yok, sınır bilgisi fonem özniteliği (metin kuralı) | 5,9 / 17,5 |
| main_dizge_breaks | ayraç yok, sesten ölçülen B2/B3 token'ları | anlaşılmaz (%52–78) |
| main_dizge_nosep | ayraç yok | durduruldu (ep25: anlaşılmaz) |
| main_dizge_stress | dizge + espeak vurgusu aktarımı | durduruldu (ep~10) |

Ana bulgular: (1) vurgu işareti önemli (espeak vurgulu → vurgusuz +1,3 pp CER); (2) vurgu eşitlenince dizge ≈ espeak (fark anlamsız), dizge UTMOS +0,08; (3) sözcük ayracını tamamen atmak anlaşılırlığı çökertiyor; (4) sözcük-içi sınırların %90'ında duraklama yok (≤60 ms), noktalamalı sınırların %73'ü ≥250 ms (`reports/breaks_stats.json`); (5) espeak vurgusu kullanıcının kurallarında 9/28 doğru (`tests/stress_gold.tsv`), yani "espeak vurgusu = üst sınır" varsayımı geçersiz; (6) fp16 bu GPU'da (cuDNN) NaN, bs=8×accum2 sığıyor (`reports/stage1_audit.md`, `stage4_baseline.md`).

Detaylı raporlar: `stage1_audit.md`, `stage3_frontend.md`, `stage4_baseline.md`, `stage6_eval_v1.md`, `eval_v1_compare.txt`, `eval_v2_compare.txt`.

Bu dönemin sınırları: tek tohum, tek konuşmacı 4,2 saat, cümleler tek alan (sesli asistan), Whisper anlaşılırlığı ölçer (telaffuz doğruluğunu değil), UTMOS İngilizce ağırlıklı. Çok sayıda karşılaştırma yapıldı ve izlemesi zorlaştı; yeniden başlangıçta deney sayısı bilinçli olarak sınırlanacak.
