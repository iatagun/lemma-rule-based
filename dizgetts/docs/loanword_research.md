# Arapça / Farsça / Batı alıntıları: söyleyiş araştırması (2026-09-29)

Amaç: dizge==0.1.6 + `pronounce.py` çıktısının alıntılarda nerede yanlış olduğunu ölçmek, kuralları/sözlüğü kullanıcıyla (anadili Türkçe, dilbilimci) kesinleştirmek.
Sonuç hem FONEM (dizge üstü düzeltme) hem VURGU (stress.py `weight_stress` uzun ünlüyü â/î/û'dan okur, dizge uzunluğu bilmez) katmanını besler.

## 1. Literatür özeti
- **TDK düzeltme işareti** üç iş görür ([TDK](https://tdk.gov.tr/icerik/yazim-kurallari/duzeltme-isareti/)): (a) eşyazımlıları ayıran UZUN ünlü (âdet, âlem, âşık, hâlâ),
  (b) Arapça/Farsça sözcüklerde İNCE k, g (ve özel adlarda l) sonrası a/u (dükkân, hikâye, kâğıt, rüzgâr, tezgâh, mahkûm, sükût),
  (c) nispet -î (askerî, resmî, dinî, millî). Yani şapka iki ayrı bilgi taşır: UZUNLUK ya da ÖNCEKİ ÜNSÜZÜN İNCELİĞİ; ikisi birden de olabilir (kâtip).
- **Clements & Sezer (1982)**: kök içi ince (damaksıl) l, k, g arka ünlülerle bir arada; eki ön uyumla alır (hal-ler, rol-ü, saat-ler, kalp-ler, idrak-i).
  Yani ek uyumu kökteki ünsüzün niteliğini YAZIDA gösterir -> otomatik aday çıkarımı mümkün ([Turkish palatalized consonants and vowel harmony](https://journals.linguisticsociety.org/proceedings/index.php/tu/article/download/4781/4477/8708)).
- **Al-Hashmi (2016)**, Arapça gırtlaksıllar (ʕ, ʔ, ħ): ünsüz önünde düşen ʕ/ʔ ÖRNEKLEME UZATMASI bırakır (meʔmur -> memur [meːmuɾ], teʔsir -> tesir, maʕna -> mana);
  ünlüler arasında boşluk (hiatus) ya da gırtlak vuruşu / birleşip uzun ünlü (saat, cemaat, maalesef, tabii) ([tez](https://etheses.whiterose.ac.uk/id/eprint/20807/)).
  Uzun ünlüler modern Türkçede KISMEN kısalır (ölçüm: Antalia'da uzun = kısa, `pronunciation_issues.md`).

## 2. Mevcut çıktı (özenli, sözlük + uzunluk kuralları açık) — sorun sınıfları
| Sınıf | Örnek: şimdi | Beklenen (taslak, kullanıcı onayı bekler) |
|---|---|---|
| Şapka yok sayılıyor (dizge `â -> a`, `loanAlert` kullanılmıyor) | kâr `kʰɑɣ` = kar; hâlâ `xɑłɑ` = hala; âdet = adet; resmî = resmi | kâr `cʰaɣ`; hâlâ `xaːlaː`? ; resmî `resmiː`? |
| Kök sonu ince l (ünsüz önü / sözcük sonu) | hal `xɑł`, halde `xɑłdɛ`, kalp `kʰɑłp`, rol `rɔł`, alkol, kontrol, usul, kabul, meşgul, hayal, ihtimal (ama gol `gɔl` doğru) | `l` (ön); ekli biçim (hali, rolü) zaten `l` |
| Kök sonu ince k | idrak `Idɾɑk` (idraki `IdɾɑcI` doğru), iştirak | `c` |
| İnce k/g + a (Farsça/Arapça) | dükkân `dYckɑn`, rüzgâr `rYzgɑɣ`, hikâye `çIkɑIɛ`, kâfir `kʰɑfIɣ`, mekân | `c/ɟ` + ön `a` (kağıt gibi) |
| Birleşen ünlü / ʕ-ʔ izi | saat `sɑɑt`, cemaat, maalesef `mɑɑlesɛf`, tabii `tʰɑbII`, teessüf `tʰeɛssYf`, iade, mesai | saat `sɑːt`? tabii `tʰɑbiː`? |
| Arapça uzun ünlü (yazıda işaretsiz) | zaten, tatil, tarih, adil, kâtip, şair, faiz, âlim; dizge sözlüğünde yalnız 19 sözcük (memur, temin, tesir, nisan...) | sözlük; özenli/gündelik ayrımı? |
| Batı alıntısı ünlü türemesi | klinik (sözlükle çözüldü), tren `tʰIɾɛn`, spor `sɨpɔɣ`, stres `sɨtɾɛs`, plan `pʰɨłɑn`, program | spor/stres: ı mı i mi (ıspor/ispor)? plan `pʰɨlɑn` (ince l?) |

## 3. Antalia eğitim verisindeki ağırlık (7305 tip, 32.884 geçiş)
Şapkalı yazım VAR ama tutarsız (hâlâ 11, hikâye 2 / hikaye 7+, kâr 2, resmî 2, dükkân 5, tezgâh 5, rüzgâr 3). Sık alıntılar: saat* (~100), dakika 75, lira 64, tabii 22,
zaten 17, rica 24, iade 25, dahil 7, halde 6, itibaren 8, nisan 14, tarih* 16, kahve 8, sükunet 7, vakit 8, misafir 8.
Uyarı: model yalnız kayıttaki sesi öğrenir; konuşmacının kendisi uzatmıyorsa (ölçüldü: uzun = kısa, 46/46 ms) uzunluk etiketi gürültü olur. İncelik (l/ł, c/k) spektral, öğrenilebilir.

## 4. Önerilen mekanizma (kullanıcı onayından sonra)
Tek sözlük `resources/loans.tsv`: `yazım<TAB>işaretli söyleyiş<TAB>kayıt<TAB>not`; işaretli biçim kullanıcının hızlı yazabileceği gösterim:
`â/î/û` = uzun, `ḱ ǵ ĺ` = ince ünsüz (örn. `saat -> sâat`?, `hal -> haĺ`, `dükkân -> dükḱan`, `kâtip -> ḱâtip`). Buradan hem fonem kalıbı (pronounce.py Atoms, kök + ek zinciri)
hem vurgu ağırlığı (stress.py zaten â okur) türetilir. Metindeki şapka da aynı yoldan okunur. Kök-sonu ince l/k adayları ek uyumundan (hal-ler, idrak-i) otomatik çıkarılır, kullanıcı onaylar.

## 5. Kullanıcı kararları ve uygulama (2026-09-29)
Kararlar: (1) zaten, tatil, tarih, adil, âlim, şair, faiz: a 1,5 kat uzun (şair/faiz'de a -> i geçişi); memur, tesir: e uzun (dizge zaten `eː`).
(2) saat, cemaat, maalesef, tabii, iade, mesai, teessüf: TEK uzun ünlü; cüret, sanat normal. (3) hal ("haal"), halde, kalp, ihmal, ihtimal, hayal, sual: ae karışımı a + ince l;
rol, alkol, kontrol, petrol (kabul, meşgul): oe karışımı ("neredeyse ö") + ince l; usul: ünlü normal, l ince. (4) ince k/g + a = ön a. (5) nispet î uzun. (6) hâlâ vurgusu ilk hecede.
(7) spor/stres: "sıpor", çok kısa schwa benzeri ı = dizge'nin mevcut `sɨ` okuması, DEĞİŞİKLİK YOK.
Uygulama: mekanizma olarak yeni sözlük dosyası AÇILMADI; mevcut `resources/pronunciation_exceptions.tsv` genişletildi (+ `-kök` dışlama gösterimi: hal satırı halı/hala/halk/Halil'i yakalamaz).
`frontend/pronounce.py` `_circumflex`: sözlükte olmayan şapkalı sözcük: k/g/l + â -> ince ünsüz + ön `a` (dergâh), başka â -> `ɑː` (âdet), î -> `iː` (resmî; sözlüklü hayalî'de de).
`LengthRules` artık yalnız dizge'nin kendi `ː`'lerini hizalar (sözlüğün koyduğu uzunluk harf olayıyla açıklanmaz). Vurgu: `stress_roots.tsv` hâlâ 0.
Denetim: UD treebank'ları + Antalia dağarı (51.955 tip) tarandı, yanlış pozitifler dışlandı. Test: `tests/test_pronounce.py` §7.
Antalia etkisi (özenli, HEAD'e göre): 378 klip, 511 geçiş, 107 tip (saat* ~180, kontrol* 92, iade 28, tabii 22, zaten 17, hâlâ 11, tarih* 20).
Parity (`test_engine_parity`, varsayılan bayraklar): 12 klip farklı, YALNIZ hâlâ vurgusu (xɑłˈɑ -> xˈɑłɑ); fonem değişiklikleri bayrak arkasında.
**Kullanıcı onayı 2026-09-29 (ikinci tur):** Arapça uzun a = `aː` (hakim gibi, tam arka değil; sözlük satırları + şapka kuralı `âdet -> aː`); hal ve TÜM çekimleri uzun ön `aː`
(hali'de de tam arka değil); kabul/meşgul `Y`; dahil, kâtip uzun; iade `I aː d`, mesai `s aː I`; şapkasız `resmi`'ye dokunulmaz. Açık varsayım kalmadı.
**Eğitim notu:** yeni kurallarla train'de `aː` 272 (önce 1), `a` 66 kez geçiyor. `model.embed_alias` her adımda `aː`'yı (ɑː+ɛː)/2'ye geri bağlıyor -> artık kendi gömmesini
öğrenebilecek `aː` için bağ yalnız BAŞLANGIÇTA olmalı (ya da yalnız `a` için sürmeli); sonraki eğitimden önce `train/embed_alias.py` buna göre ayarlanmalı.

## 6. g2ptts v1: tam ön uç + ek uyumundan alıntı kökleri (2026-09-29)
**Paket:** HF `DizgeBERT-G2PTTS` artık sesbirim de üretir: `tag()` -> `phonemes` (TTS'e hazır dizi) + sözcük başına `phones`, `stress_phone`. Kod dizgetts frontend'inden BİREBİR kopya
(stress/normalize/symbols/phonemize/pronounce/g2p + resources/, `export_hf.py`); vurgunun sesbirime eşlenmesi `stress.phone_stress_index`'e taşındı (Engine ve paket aynı fonksiyon).
Config: `phonemes`, `pron_exceptions`, `length_rules` (varsayılan açık), `register` (özenli). `tests/test_hf_g2ptts.py` (6): paket dizisi == `Engine(g2ptts, aynı etiketleyici, sözlük+kurallar)`
token'ları, 263 cümle birebir. Tuzak: transformers uzak kodda yalnız modelleme dosyasının DOĞRUDAN göreli içe aktarmalarını kopyalar -> sesbirim modülleri orada anılır.
**Madencilik:** hermitdave/FrequencyWords tr_full (OpenSubtitles 2018, 2 M tip; D:/dizgetts/data/lexicon/tr_full.txt): son ünlüsü kalın kök + ≥%80 ince ek + ≥3 ince ek türü -> 109 aday,
elle ayıklandı (yabancı/ASCII gürültü) -> `resources/loan_roots.tsv` (70+ kök). Kullanıcı grup kararları: Batı ve Arapça -al = hal gibi (ae + ince l); -ol = rol gibi (œ + ince l);
-ul = kabul gibi (Y); -at = son a ön, önündeki k ince; -aat = tek uzun aː. Kural `pronounce._loan_final`; sözlük satırları önceliklidir. Baştaki ünsüz öbeği türemesi hizalamada atlanır.
Yanlış pozitif taraması (sıklık ≥ 20 biçimler): hol çıkarıldı (holmes/holland...); gol/rol/metal/moral/adil/alim/dahil/sual/hayal/mekan dışlamaları eklendi.
Antalia etkisi (HEAD'e göre, özenli): 442 klip (%42), 677 geçiş, 138 tip. Train'de ön `a` 207 (önce 0), `aː` 272 (önce 1), `eː` 1 (teessüf).
**Açık:** (1) sesbirim için bağımsız kör ölçüm yok; (2) embed_alias `aː`/`a`'yı her adımda bağlıyor, artık yeterli veri var -> yalnız başlangıçta bağla; (3) TTS yeniden eğitimi
(manifest `--pron --length-rules`) ve kör AB; (4) HF push kullanıcı onayıyla.

## 7. Kör sesbirim değerlendirmesi, set 1 (2026-09-30) -> GELİŞTİRME seti
`scripts/make_phoneme_eval.py`: 120 sözcük (R 50 rastgele, G 15 ğ, A 25 kuralsız alıntı adayı, K 20 kuralların değiştirdiği, B 10 ünsüz öbeği), UD + Antalia dağarı, tohum sabit.
Kullanıcı etiketi `tests/phoneme_gold_dev.tsv`. **Önce: %83,3 (100/120)**; kaynağa göre dizge %91,1, sözlük/alıntı %78,9, uzun ünlü kuralı %60,0.
Hata sınıfları ve düzeltme (kök neden):
- y yan ünlüsü `I` (8): LengthRules `Vː I -> V I` yapıyordu; kullanıcı: y kendi sesiyle -> `V j` (hayata, baleyi, uyandığım, sağduyusu). Yan etki: kurallar açıkken vurgu eşleme hatası 271 -> 255.
- ö/ü (7): dizge sıradan ö/ü = `œ`/`Y`; kullanıcı: "direkt ö/ü" -> sözlük açıkken sıradan ö/ü `ø`/`y`; `œ` (oe) ve `Y` (ü benzeri) YALNIZ alıntılara kalır (rol, kabul). Yeni simge yok.
- hal uzunluğu (2): yalın hal ve ünlü önünde (hali) uzun, ünsüz önünde (halde, hâlleri, hâlbuki) kısa ön a (`kök=` tam sözcük gösterimi eklendi).
- sözcüğe özgü: madenî/maden, cami (uzun a), istikbal/istiklal/ikbal (ön a, loan_roots), not olarak gıyabi ve defa (uzun a).
Sonra: 19/20 hata düzeldi, "doğru" işaretli 9 sözcük yalnız kullanıcı kurallarıyla değişti (ö/ü, defa/gıyabi notları), beklenmedik gerileme yok. Açık: demiyor (not: "demiˈjoɾ", kullanıcıya soruldu).
Bu set kuralları düzeltmek için kullanıldı -> bağımsız sayı değil. **Set 2** (yeni kör, tohum 20260930, set 1 sözcükleri dışarıda): `reports/phoneme_eval_2.html`.
Eğitim sayıları (Antalia train, yeni kurallarla): ø 1629, y 3258, œ 81 (yalnız alıntı), Y 4 (yalnız alıntı), j 6181. Yeniden eğitimde (v6 checkpoint'inden sıcak başlangıç) gömme satırları
eski anlamlarından başlatılmalı: ø <- eski œ, y <- eski Y (sıradan ö/ü eskiden bu simgelerdeydi); œ, Y eski satırlarında kalır (alıntı sesleri ö/ü'ye yakın); a/aː `embed_alias: init`.
