# Telaffuz sorunları (kör AB dinleme notlarından), 2026-09-26

Kaynak: kullanıcının kör AB testlerindeki notları (`reports/ab/`). "dizge çıktısı" = `dizge==0.1.6` birincil okuma (engine'in kullandığı).
Karşılaştırılan modeller (v3a/v4/v4-400) AYNI fonemleri kullanır; iki modelde de yanlışsa kaynak çoğunlukla fonem (dizge) ya da eğitim verisidir.

| Sözcük | dizge çıktısı | Kullanıcı notu | Olası kaynak | Durum |
|---|---|---|---|---|
| sağlık | `s ɑː ł ɨ k` | v4 "salık" okudu | Model zamanlaması (fonem doğru) | açık |
| afiyet | `ɑ f iː ɛ t` | a çok kısa | **dizge**: uzunluk i'de, â'da değil (âfiyet) | kullanıcıya soruldu |
| kağıt / kâğıt | `cʰ aː ɨ t` (kağıdı: `kʰ ɑː ɨ d ɨ`) | ikisi de yanlış | **dizge**: kök doğru yönde (ön `a`), ama çekimli biçim düzeltmeyi kaybediyor (`kağıdı` arka `ɑ`, arka `k`): dizge `lexicon.loans` yalnız TAM sözcük eşler | kullanıcı söylenişi verdi (aşağıda), düzeltme bekliyor |
| mahzur(lar) | `m ɑ x z U ɣ` | ikisi de yanlış | Kullanıcı tarifi (kalın a, h TAM sesli) dizge çıktısıyla UYUŞUYOR -> fonem doğru, sorun **model/veri** (nadir sözcük, `xz` öbeği, h zayıf) | açık |
| tahlil | `tʰ ɑ x l I l` | v4 daha iyi | model | – |
| Osmanlıca/Arapça sözcükler (UD cümlesi, AB-1 çift 26) | – | ikisi de yanlış | **dizge** (uzun ünlü sözlüğü yok) | açık |
| kanunda | `kʰ ɑ n U n d ɑ` | çok hızlı, sesler yutuluyor | Model zamanlaması (fonem doğru) | açık |
| hakim / hâkim | `x ɑː c I m` | a ~1,5 kat uzun VE tam arka ünlü DEĞİL | **dizge**: uzunluk var ama ünlü niteliği `ɑ` (arka) yerine `aː` (ön/orta) olmalı; ayrıca veride uzun ünlü uzun değil | kullanıcı söylenişi verdi, düzeltme bekliyor |
| uyruğuna | `Uː I ɾ uː n ɑ` | ğ uzatması yanlış | dizge: `uy` -> `Uː I` (yan ünlü deseni) + ğ uzatması; model | açık |
| iddia | `I d d I ɑ` | "iddaa": a güçlü, i'yi kendine benzetir | **dizge**: benzeşme yok (`iddiaya` -> `IddIɑːIɑ`); tek varyant | kullanıcı söylenişi verdi, düzeltme bekliyor |
| klinik | `cʰ I l I n I c` | "kılinik": İngilizce schwa gibi | **dizge**: ünlü türemesi VAR (kral -> `kʰɨɾɑł`) ama türeyen ünlü sonraki ünlüye uyuyor (ön `I`, ön `cʰ`); klinik için `ɨ` + arka `kʰ` gerekir (`tren` -> `tʰIɾɛn`, `kredi` -> `cʰIɾɛdI` de aynı kuralla ön) | kullanıcı söylenişi verdi, düzeltme bekliyor |

## Ölçüm: uzun ünlüler (`ː`) gerçekten uzun mu?
v4-400 hizalaması, 143 test cümlesi (`eval_out/v4_e400_x543/prosody/align.json`):
gerçek kayıtta uzun ünlü medyanı 46 ms, kısa ünlü 46 ms (oran 1,00); sentezde 52 / 52 ms. Sentez/gerçek eşleşmiş uzun ünlü oranı medyan 1,12.
-> Model uzun ünlüleri ortalamada KISALTMIYOR; ama Antalia konuşmacısının kendi kaydında da dizge'nin uzun işaretlediği ünlüler kısalardan
uzun değil (ya konuşmacı bu uzatmaları yapmıyor, ya `ː` bazı bağlamlarda — ör. `ay` -> `ɑːI` — gerçek uzunluğa karşılık gelmiyor).
Model veride olmayan uzatmayı öğrenemez. Hizalama (MAS + boşluk paylaştırma) yaklaşık; sözcük düzeyinde kesin değildir.

## Yumuşak ğ (kullanıcı tarifi 2026-09-26 + ölçüm)
Kullanıcı: ğ'nin kendi ses değeri YOKTUR; ya (1) komşu ünlüyü UZATIR (Türkçede normalde uzun ünlü yoktur) ya da (2) ünlüden ünlüye GEÇİŞ (diftong) sağlar.
dizge şu an ğ'yi yalnız `ː` olarak üretiyor: `dağlar` `dɑːłɑɣ`, `sağ` `sɑː`, `ağır` `ɑːɨɣ`, `soğuk` `soːUk`, `değil` (`dejIl` | `dɛːIl`, iki varyant).

Antalia (gerçek kayıt, 143 cümle, MAS hizalaması; `eval_out/v5s_x543/prosody/align.json`) ünlü süreleri:
| bağlam | n | medyan ms |
|---|---|---|
| kısa ünlü (tümü) | 11980 | 46 |
| ğ + ünsüz ya da sözcük sonu (dağlar, sağ) | 84 + 11 | **64** (x1,4) |
| ğ ünlüler arasında (ağır, kağıt): birinci ünlü | 83 | 46 (uzamıyor) |
| ...ve ondan SONRAKİ ünlü | 83 | 58 (x1,26) |
| `ay` -> `ɑːI` deseni (ğ'siz, `ː` işaretli) | 469 | 41 |
Yorum (kullanıcının iki tarifiyle tutarlı): ünsüz önünde/sözcük sonunda ğ gerçekten UZATMA (x1,4); ünlüler arasında ğ uzatma DEĞİL geçiş (ilk ünlü kısa, toplam uzun; `V ː V` gösterimi yanlış yerde uzatıyor);
`ay`'daki `ː` uzatma değil (dizge'nin yan ünlü deseni). Not: n küçük ve hizalama yaklaşık; kesin sayı için tüm 1053 klip hizalanmalı.

Kullanıcının verdiği söylenişler (kesin hedef):
- kağıt: `k a(æ karışımı, ARKA DEĞİL) ı t` = "kaeıt"; a güçlü olduğundan bazen "kaat" (a yakınını kendine benzetir) -> iki varyant.
- mahzur: a kalın (hala gibi), h TAM sesli (bazen h sesi kaybolur; burada kaybolmaz).
- hakim: a ~1,5 kat uzun, tam arka dil ünlüsü DEĞİL.
- iddia: "iddaa" (a i'yi benzetir).
- klinik: "kılinik" (schwa benzeri ünlü).

## Uygulama (2026-09-26): söyleyiş istisna sözlüğü
`resources/pronunciation_exceptions.tsv` + `frontend/pronounce.py`: dizge çıktısının üstüne, KÖK + EK ZİNCİRİYLE eşleşen kalıp değiştirme (dizge `lexicon.loans` yalnız tam sözcüğü eşliyordu).
Açmak: `Engine(pron_exceptions=True, register="özenli")` / `build_manifest --pron`; varsayılan KAPALI (eski checkpoint ve donmuş manifestler değişmesin). Eğitimde ÖZENLİ (kullanıcı kararı).
| sözcük | özenli (eğitim) | gündelik |
|---|---|---|
| kağıt / kâğıt (+ çekimler) | `cʰ aː ɨ` (kağıdı'da da; eskiden `kʰ ɑː ɨ`) | `cʰ aː` (kaat) |
| hakim / hâkim (+ çekimler) | `x aː c I m` | aynı |
| iddia | dizge'nin (`I d d I ɑ`) | `I d d ɑː` (iddaa); iddiaya (ay -> ɑːI) henüz kapsanmıyor, uyarı verir |
| klinik / kliniğ | dizge'nin | `kʰ ɨ l I ...` (kılinik) |
| mahzur | dizge'nin (tarifle uyuşuyor; sorun model/veri) | – |

**Engel ve çözümü (kullanıcı kararı 2026-09-26: gömmeleri ortalamayla başlat):** Antalia eğitim verisinde `a` atomu 0, `aː` atomu 1 kez geçiyor (`ɑ` 17.514, `ɛ` 10.322). Özenli kağıt/hakim `aː` üretir;
gradyan almayan gömme rastgele kalırdı (kağıt'ın AB'de "ikisi de yanlış" çıkmasını da açıklar: mevcut `cʰ aː ɨ t` aynı atomu kullanıyordu).
Çözüm `train/embed_alias.py`: `a` = (ɑ + ɛ)/2, `aː` = (ɑː + ɛː)/2 ("ae karışımı"). Yalnız başlangıçta bağlamak YETMEZ (ɑ/ɛ eğitimde değişir, `a` ilk rastgele ortalamada kalırdı): bağ her optimizer adımından
sonra yenilenir (`train.py`, pilotla doğrulandı: 1 + 30 çağrı) ve sentezde uygulanır (`Synth(embed_alias=...)`, cfg `model.embed_alias`).
Eğitim config'ine eklenecekler: `engine: {pron_exceptions: true, register: özenli}`, `model: {embed_alias: true}`; manifest: `build_manifest --pron`.
**Doğrulanmadı:** model ara vektörü eğitimde hiç görmedi. Mevcut v5s checkpoint'iyle (eğitimsiz bağ) 6 cümlede ASR belirsiz (kaba CER: eski fonem 16,3 | sözlük+bağsız 22,2 | sözlük+bağ 18,5; n çok küçük),
örnek sesler `D:/dizgetts/samples/alias_check/NN_{A,B,C}.wav` (A eski, B sözlük bağsız, C sözlük+bağ). Karar dinleyerek verilir; eğitim sonrası kör AB.
Denetim: `python -m dizgetts.frontend.pronounce` (eğitimde < 100 kez geçen atomları listeler; bağ bunları kapatmaz, yalnız işaretler).
