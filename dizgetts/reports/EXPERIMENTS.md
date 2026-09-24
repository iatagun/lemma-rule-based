# Deneyler (otomatik üretildi: `python -m dizgetts.eval.report`; kaynak `experiments.yaml`)

| id | durum | hipotez | CER % | WER % | UTMOS | not |
|---|---|---|---|---|---|---|
| v1-espeak | closed | espeak-ng fonemleri (vurgu işaretli) baseline | 3.0 | 11.1 | 2.90 | v1 dönemi kapatıldı; checkpoint silindi. En iyi CER, ama vurgu işareti taşıyor. |
| v1-espeak-nostress | closed | baseline'ın avantajı vurgu işaretinden mi? | 4.3 | 12.4 | 2.83 | Evet, büyük kısmı (CER +1,3 pp). |
| v1-dizge | closed | dizge 0.1.6 fonemleri (vurgusuz, sözcük ayracı token'lı) | 4.6 | 13.1 | 2.90 | vurgusuz espeak ile eşdeğer (fark anlamsız), UTMOS +0,08. |
| v1-dizge-feat | closed | sözcük sınırı token yerine fonem özniteliği | 5.9 | 17.5 | 2.85 | anlaşılırlık kötü; sözcük ayracını atmak (breaks/nosep) yıkıcıydı. |
| v1-dizge-breaks / v1-dizge-nosep / v1-dizge-stress | closed | ayraçsız denemeler; espeak vurgusu aktarımı |  |  |  | anlaşılmaz / durduruldu. Ayrıntı reports/EXPERIMENT_LOG.md. |
| v2-m2-first | blocked | dizge fonemleri + KURAL vurgusu + noktalama duraklaması, sözcük ayracı token'lı, motorun ilk modeli |  |  |  | vurgulu espeak (CER 3,0) ile fark anlamsız ya da daha iyi ise ilerle; değilse vurgu doğruluğuna (gold) dön |
| v2-m3-nostress | not-needed |  |  |  |  | v1-dizge (aynı token'lar, vurgusuz) zaten bu karşılaştırmayı veriyor (CER 4,6 / WER 13,1); yeniden koşturulmaz |
