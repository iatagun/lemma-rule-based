# İlk karşılaştırma (150 epoch, tek tohum, 2026-09-24)

143 eğitimde görülmeyen cümle (84 test + 59 val), her sistemin FİNAL (ep150) checkpoint'i (seçim kuralı sabit, test'e bakılmadı). Whisper-small CER/WER + UTMOS22.
Ham çıktı: `reports/eval_v1_compare.txt`, `D:\dizgetts\eval_out\<etiket>\results.json`. Gerçek kayıtların Whisper tabanı (aynı cümleler): CER %4,7, WER %11,6.

| Sistem | CER % [%95] | WER % [%95] | UTMOS |
|---|---|---|---|
| espeak-ng baseline (vurgu işaretli) | **3,0** [2,3–4,0] | **11,1** [9,6–12,6] | 2,90 |
| dizge 0.1.6, ayraçlı | 4,6 [3,1–6,4] | 13,1 [11,1–15,3] | 2,90 |
| dizge 0.1.6, ayraç yok + fonem öznitelikleri | 5,9 [4,3–7,8] | 17,5 [15,4–19,8] | 2,85 |

Eşleşmiş farklar (espeak'e göre): dizge ayraçlı CER +1,6 pp [+0,3,+3,2], WER +2,0 pp [+0,2,+4,1] (anlamlı); öznitelikli CER +2,9, WER +6,4 (anlamlı); UTMOS farkı ayraçlıda ~0, öznitelikte −0,05.

## Yorum ve sınırlar
- Bu ayarda dizge fonemleri anlaşılırlıkta espeak'ten **daha iyi değil, hafif daha kötü**. Hatalar diffüz (sık sözcüklerde: "es em es", "ile", "randevu"); `ğ`/son-r gibi bilinen kural sorunlarına yığılmıyor.
- KARIŞTIRICI: espeak girdisi vurgu işareti (ˈ) taşıyor, dizge taşımıyor. Vurgusuz espeak ablasyonu bunu ayırır (config: espeak_strip_stress).
- Whisper'ın dil modeli yanlış telaffuzu düzeltebilir: CER anlaşılırlığı ölçer, telaffuz doğruluğunu değil. Telaffuz test seti (kullanıcı) gerekli.
- Öznitelikli model eğitimde geride (val kaybı 1,70 vs 1,64) ve hâlâ düşüyor; daha uzun eğitimle fark kapanabilir.
- Tek tohum, tek konuşmacı, 4,2 saat, in-domain (sesli asistan) cümleler.

# Güncelleme: vurgusuz espeak ablasyonu (aynı 143 cümle, ep150, 2026-09-24 11:20)

| Sistem | CER % [%95] | WER % [%95] | UTMOS | val kaybı |
|---|---|---|---|---|
| espeak (vurgulu) | **3,0** [2,3–4,0] | **11,1** [9,6–12,6] | 2,90 | 1,606 |
| espeak vurgusuz | 4,3 [3,0–5,9] | 12,4 [10,5–14,4] | 2,83 | 1,632 |
| dizge (ayraçlı, vurgusuz) | 4,6 [3,1–6,4] | 13,1 [11,1–15,3] | 2,90 | 1,636 |
| dizge öznitelikli | 5,9 [4,3–7,8] | 17,5 [15,4–19,8] | 2,85 | 1,699 |

Eşleşmiş farklar (bootstrap, cümle bazlı):
- Vurgu işareti etkisi (espeak vurgusuz − vurgulu): CER +1,3 pp [+0,2,+2,7] anlamlı; UTMOS −0,07 anlamlı; WER +1,3 anlamsız.
- **Adil karşılaştırma (ikisi de vurgusuz): dizge − espeak vurgusuz: CER +0,3 [−1,0,+1,6], WER +0,7 [−1,0,+2,5] anlamsız; UTMOS +0,08 [+0,05,+0,10] anlamlı (dizge lehine).**
- Öznitelikli model vurgusuz espeak'ten de anlamlı kötü (CER +1,6, WER +5,1).

Sonuç: dizge G2P (0.1.6, düzeltmesiz) vurgu eşitlenince espeak ile anlaşılırlıkta eşdeğer, doğallık vekilinde hafif iyi. İlk baseline farkının büyük kısmı espeak'in vurgu işaretinden. Espeak'in vurgusu son-hece dışı %24,6 (yalnız `-yor` öncesi, sayılar, bağlaçlar, yer/yabancı sözcükler gibi Kabak&Vogel istisnalarına benziyor); son hece %65, vurgusuz %10,5 (sözcük sayısı 35.325).
Sınırlar: tek tohum, 143 in-domain cümle, Whisper anlaşılırlığı ölçer (telaffuz doğruluğunu değil), UTMOS İngilizce ağırlıklı.
