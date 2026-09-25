# Sonraki oturum planı — cümle bazlı G2P / sesbilim motoru (2026-09-24 sonu)

## Durum (özet)
- v2 motoru (`Engine`: Normalize → Phoneme → [Morph] → Stress → Assemble) çalışıyor; branch `dizgetts-v2`, son commit 9f1d841.
- v2-m1a: CER 3,6 / WER 11,9 / UTMOS 2,85. v2-m1b: CER 3,0 / WER 12,4 / UTMOS 2,83. M1b − M1a: CER −0,60 [−1,76, +0,25], WER +0,59 [−0,86, +1,90] → anlamsız.
  Karar kuralı gereği M1b varsayılan OLMADI (M1a varsayılan; `tiers` opt-in). Referans: espeak vurgulu 3,0/11,1; dizge vurgusuz 4,6/13,1; gerçek kayıt tabanı 4,7/11,6.
- Ölçüm gücü sorunu: 143 cümle %11 sözcüğün vurgu farkını ayıramıyor → G2P'yi TTS eğitmeden, doğrudan (iç değerlendirme) ölçmek gerekiyor.

## Karar (kullanıcı onayladı): önce cümle bazlı G2P analizi, TTS eğitimi sonra
Cümle bazlı G2P = sözcüğü bağlamıyla fonemleyen; vurgu, sınır ve sözcükler arası etkiyi cümleden çıkaran aşama.

| İhtiyaç | Bugün | Hedef |
|---|---|---|
| Vurgu | sözcük + Morph etiketi | işlev sözcüğü vurgusuzlaşması, soru/seslenme, odak |
| Duraklama/sınır | yalnız noktalama | Dep (DizgeBERT-Dep) ile öbek sınırı |
| Sözcükler arası etki | yok (`" "` ayracı) | sözcük sonu/başı ses etkileşimi (kullanıcı kuralları) |
| Eşyazımlılar | BERT yedeği (`ğ`'de güvensiz) | cümleden ayrım |

## Adımlar
1. **Şema + `PhraseStage` iskeleti** (kural içermeyen): sözcük başına `{sözcük, fonemler, vurgu_hecesi, vurgu_kaynağı, sonraki_sınır: 0|ip|IP|cümle, notlar}`; Dep'ten sınır tahmini. Ölçüt: eski token akışıyla parity testi geçmeli (1053/1053).
2. **İç değerlendirme (eğitimsiz):** kullanıcı etiketli sözcük/cümle setine karşı doğruluk raporu: kural modülü vs espeak vs son-hece varsayılanı, katman katman. `tests/stress_gold.tsv` + 250 sözcük.
3. **Sözcükler arası kurallar:** kullanıcı yazdıkça `resources/*.tsv`'ye.
4. **TTS'e yansıma:** ölçülmüş kural setinden sonra tek koşu; değerlendirme setini büyüt (143 yetmedi).
- Öğrenilmiş cümle modeli (BERT) SONRA: etiket kural üretirse döngüsel olur; bağımsız etiket kaynağı gerekir (kullanıcı gold'u veya Antalia sesinden sınır/süre).

## Kullanıcıdan bekleyenler
1. `reports/stress_annotation_sheet.tsv` (250 sözcük; yalnız yanlışları düzelt).
2. Sınır ve sözcükler arası etki için 10–20 örnek cümle ("şurada şöyle okunur").
3. 4 soru: `nispeten` nis-PE-ten mi; `asosyal/kapkara/başbakan` ilk hece mi; `çocuklar!` seslenmesi; `yapsaydı` yap-SAY-dı mı.
4. `-Iyor` öncesi vurgu kuralı (benim eklemem, onaysız) onayı.
5. Sözlük listeleri (belirteç 167 tür, soru sözcükleri, yer adları, ödünçlemeler, bileşikler); işlev sözcükleri (bir, bu, ve, ile, için…) vurgusuz mu; kuralların `dizge` paketine taşınması kararı.

## Açık, onay bekleyen seçenekler (kullanıcı henüz seçmedi)
- **VoxCPM2** (openbmb, 2,4 B param, Apache-2.0, Türkçe destekli, düz metin girişi, ~8 GB VRAM, LoRA/SFT): yalnız 143 cümlede zero-shot REFERANS ölçümü (~5 GB indirme tahmini, onay gerekir). G2P'yi denemeye uygun değil (fonem girmiyor). GTX 1650 4 GB'a büyük olasılıkla sığmaz (denenmedi). Klonlama etiği: Antalia konuşmacısını taklit etme, sentetik olduğunu belirt.
  Türkçe çalışmalar: Trendyol/Trendyol-TTS (VoxCPM2 LoRA, özel 20+ sa veri, formal ölçüm yok), FreyaTTS arXiv 2607.09530 (183M, fonemleştirici yok, CER 3,0/WER 8,0 kendi raporu).
- İkinci veri (FLEURS-tr CC-BY-4.0, ISSAI TSC MIT; omersaidd/* KULLANMA; Common Voice kullanıcı dışladı): tek konuşmacı 4,2 sa sınırını gevşetir, G2P sorusunu değiştirmez.

## Çalışma kuralları (devam)
Aynı anda en fazla 1 eğitim; yeni koşu = önce `experiments.yaml`'da hipotez + karar kuralı; sayılar script çıktısından; onaysız büyük indirme/uzun eğitim yok; fp32 (GTX 1650 fp16 NaN); regex'i heredoc python ile değil Edit/Write ile yaz.
Komutlar: `python -X utf8 -m dizgetts.eval.status`, `python -X utf8 dizgetts/eval/compare.py <a> <b>`, `python -X utf8 -m dizgetts.tests.test_engine_parity`.

## İlerleme (2026-09-24, oturum 2)
- Adım 1 (iskelet) YAPILDI: `Word.stress_src / boundary (0|ip|IP|cümle) / notes`, `PhraseStage` (şimdilik yalnız noktalama; token'a dokunmaz). Parity 1053/1053.
  Dep öbek sınırı henüz YOK (sıradaki iş; bağımsız sınır etiketi: v1 `derive_breaks.py` sessizlik ölçümü -> `_v1_archive`).
- Adım 2 başladı: `python -X utf8 -m dizgetts.eval.stress_intrinsic` (son_hece / m1a / m1b / espeak). stress_gold.tsv (28): 14,3 / 71,4 / 89,3 / 28,6 %.
  DÖNGÜSEL: gold kökleri stress_roots.tsv'de de var -> yalnız regresyon kontrolü. Bağımsız sayı için annotation sheet gerekli (sheet okuyucu, sheet dolunca yazılacak).
- Kullanıcı onayı: stress_gold.tsv'deki kullanıcı örnekleri son-hece istisnasıdır. Konum (hangi seslem) 3+ seslemlilerde onaysız: pırasa, ufacık, semracığım, lokanta, kapkara, başbakan.
- Kullanıcı kuralı (seslem ağırlığı) UYGULANDI: `weight_stress` (-en belirteç: sondan 2. H ise o, L ise sondan 3.; alıntı/yer adı: güçlü-zayıf sözcük),
  stress_roots.tsv'de sıra yerine `ağırlık`; uzun ünlü şapkayla (esâsen). M1b'ye `pek` (pekiştirme) ve `cik` (-CIk türemiş sıfat) katmanları -> ilk seslem.
  Gold 35 sözcük: son_hece 11,4 / m1a 68,6 / m1b 94,3 / espeak 28,6 % (hâlâ döngüsel). Parity 1053/1053.
  Açık: uzun ünlü sözlüğü (dizge uzunluk işaretlemiyor); asosyal (A-sos-yal) alıntı kuralına aykırı -> önek istisnası mı?; semracığım, güzeldir kuralsız.
- 2026-09-25 İLK BAĞIMSIZ VURGU ÖLÇÜMÜ (kullanıcının kör etiketi, tests/stress_gold_random.tsv, 250 rastgele sözcük): son_hece 62,8 / m1a 64,4 / m1b 76,8 /
  g2ptts (v2b karma) 78,0 / espeak 69,2 %; g2ptts örnek cümle içinde 196/250 (bağlam şimdilik katkısız: etiketler bağlamsız kurallardan).
  Gold'un %37'si son-dışı. 54 hata: belirteç/bağlaç/edat (~25: böyle, için, gibi, hemen, sadece, çünkü, elbette...), alıntı/özel ad (~12: cumartesi,
  pazartesi, perşembe, fatura, merhaba, harika...), vurgusuz ek (~12: -ken, -(y)ince, -(y)ArAk, -(y)AlIm, emir, -sInlAr, -mIş/-dIr koşaç), model (yerleştirdik).
  B grubunda ve A'nın çoğunda sözcük istisna olarak İŞARETLENİNCE ağırlık kuralı konumu doğru veriyor -> darboğaz "hangi sözcük istisna".
  DİKKAT: bu 250'nin hataları incelendi -> artık GELİŞTİRME seti; buradan sözcük sözlüğe eklenirse skor döngüsel olur. Son rapor için yeni kör set gerekir.
