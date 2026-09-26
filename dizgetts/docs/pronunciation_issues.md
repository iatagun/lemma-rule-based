# Telaffuz sorunları (kör AB dinleme notlarından), 2026-09-26

Kaynak: kullanıcının kör AB testlerindeki notları (`reports/ab/`). "dizge çıktısı" = `dizge==0.1.6` birincil okuma (engine'in kullandığı).
Karşılaştırılan modeller (v3a/v4/v4-400) AYNI fonemleri kullanır; iki modelde de yanlışsa kaynak çoğunlukla fonem (dizge) ya da eğitim verisidir.

| Sözcük | dizge çıktısı | Kullanıcı notu | Olası kaynak | Durum |
|---|---|---|---|---|
| sağlık | `s ɑː ł ɨ k` | v4 "salık" okudu | Model zamanlaması (fonem doğru) | açık |
| afiyet | `ɑ f iː ɛ t` | a çok kısa | **dizge**: uzunluk i'de, â'da değil (âfiyet) | kullanıcıya soruldu |
| kağıt / kâğıt | `cʰ aː ɨ t` | ikisi de yanlış | **dizge** | doğru söyleniş bekleniyor |
| mahzurlar | `m ɑ x z U ɾ ł ɑ ɣ` | ikisi de yanlış | **dizge** | doğru söyleniş bekleniyor |
| tahlil | `tʰ ɑ x l I l` | v4 daha iyi | model | – |
| Osmanlıca/Arapça sözcükler (UD cümlesi, AB-1 çift 26) | – | ikisi de yanlış | **dizge** (uzun ünlü sözlüğü yok) | açık |
| kanunda | `kʰ ɑ n U n d ɑ` | çok hızlı, sesler yutuluyor | Model zamanlaması (fonem doğru) | açık |
| hakim / hâkim | `x ɑː c I m` | a yaklaşık 1,5 kat uzun olmalı | Fonem uzun (ɑː) ama eğitim verisinde uzun ünlüler kısalardan uzun değil (aşağıda) | açık |
| uyruğuna | `Uː I ɾ uː n ɑ` | ğ uzatması yanlış | dizge: `uy` -> `Uː I` (yan ünlü deseni) + ğ uzatması; model | açık |
| iddia | `I d d I ɑ` | gündelik söyleyiş "iddaa" (a önceki sesi benzetir) | **dizge söyleyiş varyantı** (gündelik) eksik | öneri |
| klinik | `cʰ I l I n I c` | "kılinik" diye okunur | **dizge**: ünsüz öbeği başında ünlü türemesi (gündelik) | öneri |

## Ölçüm: uzun ünlüler (`ː`) gerçekten uzun mu?
v4-400 hizalaması, 143 test cümlesi (`eval_out/v4_e400_x543/prosody/align.json`):
gerçek kayıtta uzun ünlü medyanı 46 ms, kısa ünlü 46 ms (oran 1,00); sentezde 52 / 52 ms. Sentez/gerçek eşleşmiş uzun ünlü oranı medyan 1,12.
-> Model uzun ünlüleri ortalamada KISALTMIYOR; ama Antalia konuşmacısının kendi kaydında da dizge'nin uzun işaretlediği ünlüler kısalardan
uzun değil (ya konuşmacı bu uzatmaları yapmıyor, ya `ː` bazı bağlamlarda — ör. `ay` -> `ɑːI` — gerçek uzunluğa karşılık gelmiyor).
Model veride olmayan uzatmayı öğrenemez. Hizalama (MAS + boşluk paylaştırma) yaklaşık; sözcük düzeyinde kesin değildir.
