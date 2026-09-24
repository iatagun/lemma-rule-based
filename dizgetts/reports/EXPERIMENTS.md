# Deneyler (otomatik üretildi: `python -m dizgetts.eval.report`; kaynak `experiments.yaml`)

| id | durum | hipotez | CER % | WER % | UTMOS | not |
|---|---|---|---|---|---|---|
| v1-espeak | closed | espeak-ng fonemleri (vurgu işaretli) baseline | 3.0 | 11.1 | 2.90 | v1 dönemi kapatıldı; checkpoint silindi. En iyi CER, ama vurgu işareti taşıyor. |
| v1-espeak-nostress | closed | baseline'ın avantajı vurgu işaretinden mi? | 4.3 | 12.4 | 2.83 | Evet, büyük kısmı (CER +1,3 pp). |
| v1-dizge | closed | dizge 0.1.6 fonemleri (vurgusuz, sözcük ayracı token'lı) | 4.6 | 13.1 | 2.90 | vurgusuz espeak ile eşdeğer (fark anlamsız), UTMOS +0,08. |
| v1-dizge-feat | closed | sözcük sınırı token yerine fonem özniteliği | 5.9 | 17.5 | 2.85 | anlaşılırlık kötü; sözcük ayracını atmak (breaks/nosep) yıkıcıydı. |
| v1-dizge-breaks / v1-dizge-nosep / v1-dizge-stress | closed | ayraçsız denemeler; espeak vurgusu aktarımı |  |  |  | anlaşılmaz / durduruldu. Ayrıntı reports/EXPERIMENT_LOG.md. |
| v2-m1a | done | dizge fonemleri + KURAL vurgusu M1a (kök sözlüğü + clitic + varsayılan son seslem), sözcük ayracı token'lı. Vurgu işaretinin yalnız kural-tabanlı kaba biçimi de CER'i vurgusuz dizge'den (4,6) düşürür. | 3.6 | 11.8 | 2.85 | M1b (vurgusuz ekler) YOK; espeak vurgusu kullanıcı kurallarında 9/28 doğruydu, son-seslem kuralı bunu kısmen düzeltir. Karşılaştırma: v1-dizge (aynı token'lar, vurgusuz). |
| v2-m1b | done | M1a'ya morfolojik vurgu katmanları (-Iyor önü, olumsuzluk, Case=Ins, kişi eki/koşaç; 11% sözcüğün vurgusunu değiştirir) eklemek CER'i M1a'dan düşürür | 3.0 | 12.4 | 2.83 | kuralların bir kısmı kullanıcı listesinden (olumsuzluk, Ins, kişi eki, koşaç), -Iyor benim eklemem (literatür; onaysız). espeak ile örtüşme: yor %95, kişi eki %74, koşaç %78, Ins %68, olumsuzluk %42 (espeak ölçüt değil). |
| v2-m3-nostress | not-needed |  |  |  |  | v1-dizge (aynı token'lar, vurgusuz) zaten bu karşılaştırmayı veriyor (CER 4,6 / WER 13,1); yeniden koşturulmaz |
