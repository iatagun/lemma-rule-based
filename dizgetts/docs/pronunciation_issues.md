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

## Uzun ünlü (`ː`) kuralları (2026-09-26, `frontend/pronounce.py` LengthRules; `Engine(length_rules=True)` / `build_manifest --length-rules`; varsayılan KAPALI)
Sorun: dizge `ː`'yi üç ayrı gerçek için kullanıyor; Antalia'da `Vː` atomunun yalnız %8'i gerçek uzama (ğ + ünsüz/son, x1,4 = 64 ms), %20'si ünlü arası ğ (geçiş, ilk ünlü 46 ms),
%63+'ü y yan ünlüsü (~41 ms). Model `ː`'nin ne demek olduğunu öğrenemez (ölçümde uzun ünlü = kısa ünlü, 46/46 ms).
Kurallar (dizge kaynağında y kalıpları `Vj -> VːI`, `ij -> iː` dize değiştirmeleridir; kural bunları harf olaylarıyla eşleyip geri alır):
| bağlam | önce | sonra |
|---|---|---|
| ğ + ünsüz / sözcük sonu (dağ, sağlık, doğru, öğle) | `ɑː` | **aynı** (gerçek uzama) |
| ğ farklı iki ünlü arası (ağır, soğuk, yapacağım, diğer) | `ɑː ɨ` | `ɑ ɨ` (geçiş) |
| ğ AYNI ünlü arası (yaptığı, çocuğu, olduğunu) | `ɨː`, `uː` | **aynı** (ğ iki ünlüyü uzatıp birleştirir; 143 klipte örnek yok, süresi ÖLÇÜLMEDİ) |
| y yan ünlüsü, i dışı (ayak, şey, kuyu, boyunca, hikâye) | `ɑː I` | `ɑ I` |
| i + y (iyi, geliyor, diye), **özenli** | `iː I`, `iː ɔ` | `i j I`, `i j ɔ` (özenli kullanımda y SESLENİR; `j` dizge'nin yapıyor/büyük yazımıyla tutarlı) |
| i + y, **gündelik** | `iː I` | **aynı** (gündelik kullanımda y seslenmez, "ii"; kullanıcı 2026-09-26) |
| e + ğ (eğlence, değil, eğer) | `e j l ɛ...` | **aynı** (dizge birincil okuması y'leşmiş: eylence). İSTİSNA: **eğri e UZAR** (`ɛː ɾ I`, sözlük satırı; eğlence ise y'leşir: aynı bağlam, sözcüğe bağlı) |
| değer, eğer (+ değerli, değerlendirme...) | `d e j ɛ ɣ` | `d ɛː ɣ`, `ɛː ɣ` (UZAMA, kullanıcı 2026-09-27; sözlük satırı) |
| değişik, değişim, eğitim | `d e j I ʃ...` | **aynı** (y'leşme: dizge'nin okuması; kullanıcı 2026-09-27) |
| -eceğim/-eceğiz (göndereceğim, edeceğiz) | `dʒ ɛ j I m` | `dʒ ɛ I m` (DİFTONG: ğ'nin `j`'si düşer, ünlüler bitişik; -acağım zaten `ɑ ɨ`) |
| -diği/-liği/-tiği (söylendiğinde, gönderdiğim, güvenliğiniz) | `d iː I n` | `d iː n` (UZAMA: i-ğ-i tek uzun `iː`'ye birleşir, uğu -> uː gibi) |
| diğer (tek y'leşen ğ; iğne/öğün/düğün y'leşmez) | `d Iː ɛ ɣ`, çekimde `d iː e ɾ...` | `d I j ɛ ɣ`, `d i j e ɾ...` (sözlük, iki alternatif satır) |
| ı/ü + y (yapıyor, büyük), dizge sözlüğü (nisan, itibaren, teminat, hakim) | – | dokunulmaz |
Hizalama: harften `ː` üretebilecek olaylar (ğ, y) ile `ː` atomları ünlü uyumuyla sırayla eşlenir; eşleşmeyen `ː`'li sözcük DOKUNULMAZ ve sayılır (`stats["uzun_hizalanamadi"]`).
Sıra: sözlük -> kural (kağıt: sözlük ön `a` koyar, kural ğ geçişinde `ː`'yi düşürür = `cʰ a ɨ t` "kaeıt"; gündelik `cʰ aː t` "kaat"). Test: `tests/test_length_rules.py`.

**Antalia'da etki** (sözlük + kural; 1053 klip, 7193 tekil sözcük, 35.338 geçiş): özenli: 1389 tekil sözcük (%19,3), 4396 geçiş (%12,4), **1037 klip (%98,5)** değişir; gündelik: 1085 (%15,1), 3545 (%10,0), 1007 klip -> eğitim yeniden fonemleştirme gerektirir.
`Vː` özenli 4945 -> 1008, gündelik 4945 -> 1870 (i+y'de `iː` kalır).
| kural | tekil sözcük | geçiş |
|---|---|---|
| y yan ünlüsü (`Vː I -> V I`) | 803 | 2660 |
| i + y (`iː -> i j`) | 311 | 858 |
| ğ ünlü arası (`ː` düşer) | 101 | 309 |
| diğer (y'leşme) | 3 | 15 |
`Vː` atomu (geçiş bazında): 4945 -> 954 (%81 azalır). Kalan: ğ + ünsüz/son 386, ğ aynı ünlü arası 539, diğer 29. Hizalanamayan: 6 tekil sözcük (nisan, itibaren, teminat: dizge sözlüğü, meşru). Geçersiz atom üretimi: 0.
Örnek sesler (mevcut v5s checkpoint, model bu diziyi eğitimde görmedi -> yalnız kulakla kontrol): `D:/dizgetts/samples/length_check/NN_{eski,yeni}.wav`.

**ğ'nin y'leşmesi (kullanıcı 2026-09-26):** `diğer` TEK y'leşen sözcük (diyer); iğne/öğün/düğün y'leşmez -> sözlükte. `eğlence` y'leşir (dizge zaten yapıyor), `eğri` UZAR (sözlük satırı). Kullanıcı: "şüpheli y'leşme potansiyeli görürsen haber ver".
**Kullanıcı kararları 2026-09-27:** değer/eğer UZAMA; değişik/eğitim y'leşme (dizge'nin okuması kalır); -eceğim DİFTONG; -diği/-liği/-tiği UZAMA. Etki (özenli): -diği/-liği 114 sözcük/228 geçiş, -eceğ 55/295, değer/eğer 7/53.
Hizalama sıkılaştırıldı: olay ile `ː` atomu yalnız KATI ÜNSÜZ sayısı tutarsa eşlenir (iyiliğinden'de iy olayı ğ'nin iː'sini almasın); hizalanamayan hâlâ 6 sözcük (nisan, itibaren, teminat).
**`değil` (71 geçiş): y'leşme, deyil** (kullanıcı 2026-09-27) = dizge'nin okuması `d e j I l`, DEĞİŞİKLİK YOK (tests/test_length_rules.py sabitler).
**Hâlâ açık:** kullanıcı `ğ` uzama süresini notlarından kontrol edecek; "şüpheli y'leşme görürsen haber ver" (kalıcı istek).

## Vurgu -> fonem eşlemesi ve köken izi (2026-09-27 düzeltmesi)
Hata (a198676'da push'landı, bu düzeltmeyle giderildi): `to_phone_index` yan ünlüyü `ː` işaretiyle tanıyordu; uzun ünlü kuralları `ɑː I -> ɑ I` yapınca `bayram`'ın ilk seslem vurgusu yan ünlü `I`'ya düşüyordu
(kurallar açıkken 3359 geçişte ilk seslem yanlış) ve -diği'de iki ünlü tek `iː`'ye birleşince (`iː I -> iː`) indeksler kayıyordu. Kural bayrakları kapalıyken etkilenmez (parity 1053/1053).
Çözüm: `pronounce.Atoms` her atomun KÖKENİNİ (dizge'nin ham atom indeksleri) izler; `engine._stress_index` eşlemeyi HAM atomlarda (kanıtlanmış yöntem) yapıp kökenle son diziye taşır. Ekleme (`j`) kökensiz, birleşen atom birden çok köken taşır, silinen ünlünün kökeni en yakın yeni ÜNLÜ atoma gider (değer: `d e j ɛ -> d ɛː`).
Ayrıca ham eşleme iyileşti: `öy` yan ünlüsü + `iğ` (söylendiğinde) birlikte olduğunda tüm `X ː I` atılmıyor, baştan m tane atılıyor (yalnız eskiden 'sondan' düşen sözcükleri etkiler).
Ölçüm (Antalia sözcük dağarı, ilk/son seslem eşleme hatası, tekil sözcük): kapalı 271, kurallar açık 271 (düzeltmeden önce açıkken +3359 geçiş). Test: `tests/test_stress_map.py`.
