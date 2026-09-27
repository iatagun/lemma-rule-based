# Ezgi kuramları, sentez yöntemleri ve Türkçe için öneri (2026-09-27)

Amaç: DizgeTTS'in ezgi katmanını hangi kurama ve hangi modern yönteme dayandıracağımıza karar vermek.
Kısıtlarımız: 4,2 sa tek konuşmacı (Antalia), GTX 1650 4 GB, Matcha-TTS gövdesi, dilbilimsel ön uç (vurgu + sınır) zaten var.
Bu projede ölçtüklerimiz (v4-e400, reports/prosody_acoustic_v3a.md): F0 aralığı ve alçalma gerçeğe yakın; asıl açık **zamanlama**
(nPVI 27 vs 45, duraklama 310 vs 470 ms) ve **uç ezgi** (soru, devam yükselişi).

---

## 1. Ezgiyi betimleme kuramları

| Okul / model | Temel birim | Ezgi nasıl kurulur | Sentez için güçlü yan | Zayıf yan |
|---|---|---|---|---|
| **İngiliz okulu** (O'Connor & Arnold 1961, Halliday 1967, Crystal 1969) | Ton birimi: ön-baş, baş, **çekirdek** (nucleus), kuyruk | Bütüncül kontur (düşen, yükselen, düşen-yükselen…); Halliday'in üçlüsü: *tonality* (bölümleme), *tonicity* (çekirdek yeri), *tone* (kontur tipi) | Öğretimde sezgisel; "bölümle → çekirdeği bul → kontur seç" doğrudan kural hattına dönüşür | Konturlar bütün olarak etiketlenir, akustik hedefe çevirisi elle; ölçülebilir parametre yok |
| **Hollanda (IPO) okulu** ('t Hart, Collier & Cohen 1990) | Algısal olarak ayırt edilen **perde hareketleri** | *Close-copy stilizasyon*: F0 algısal eşdeğer düz çizgilere indirgenir; alçalma çizgileri (declination) üzerinde standart hareket envanteri | Algı temelli: "kulağın fark etmediği ayrıntıyı atma" ilkesi; kural tabanlı sentezde çok başarılı (Hollandaca, Almanca, Türkçe'ye de uyarlandı) | Dile özgü hareket envanteri elle çıkarılır; bağlam etkileşimleri zor |
| **Özerk-bölütsel ölçülü (AM) sesbilim** (Pierrehumbert 1980; Ladd 2008) | İki düzeyli tonlar: **L, H**; vurgu tonu (H\*, L+H\*), öbek tonu (L-, H-), sınır tonu (L%, H%) | Seyrek ton **hedefleri** + aralarında ara değerleme (interpolation); ezgi = ton dizisi | Sesbilimsel olarak ayrık, dilden dile taşınabilir; **ToBI** ile etiket standardı | Hedeflerin gerçekleşme (yükseklik, zamanlama) kuralları ayrıca gerekir |
| **ToBI** (Silverman vd. 1992) | AM'nin etiketleme sistemi: **ton katmanı** + **kırılma indisi** (0–4) + yorum katmanları | Etiket; sentez modeli değil | Kırılma indisi ↔ bizim `0 / ip / IP` sınıflarımız birebir; etiketçiler arası güvenilirlik raporlu | Dil başına ToBI gerekir; **Türkçe için yerleşik bir ToBI/etiketli derlem bulamadım** |
| **MOMEL** (Hirst & Espesser 1993) | F0'ın **hedef noktaları** (anchor) | F0 → mikro-ezgiyi (ünsüz etkisi) ayıklayıp ikinci dereceden eğrilerle (quadratic spline) hedef noktalarına indirger | Dilden bağımsız, otomatik, geri dönüşlü (noktalardan F0 yeniden kurulur) | Sesbilimsel yorum vermez, yalnız fonetik temsil |
| **INTSINT** (Hirst 1987; Hirst & Di Cristo 1998) | MOMEL noktalarının 8 simgeli kodu: **T H U S M D L B** | Mutlak (T/M/B: konuşmacı aralığına göre) ve göreli (H/L/U/D/S: önceki noktaya göre) | "Ezginin IPA'sı": etiketsiz dilde bile otomatik yazım (Momel-INTSINT Praat eklentisi) | Sesbilimsel kategori değil, yüzey kodu |
| **Fujisaki** (1984) | Üst üste bindirme (superposition): **öbek komutları** (dürtü) + **vurgu komutları** (basamak) | Taban F0 + iki ikinci dereceden süzgeç çıkışı, log-F0'da toplanır | Fizyolojik temelli, az parametre; **Türkçe TTS'te denendi** (Uslu & İlk 2009, ikili-ses birleştirmeli sistemde doğallık artışı) | Parametre çıkarımı gürültülü; nöral sistemlere doğrudan girmez |
| **Dalgacık (CWT) temsili** (Suni, Vainio vd. 2013/2017) | F0 + enerji + sürenin çok ölçekli ayrıştırması | Ölçekler ≈ hece → sözcük → öbek → söyleyiş; tepeler **belirginlik**, çukurlar **sınır** | Etiketsiz belirginlik/sınır çıkarımı; HMM sentezinde kullanıldı; açık araç (`wavelet_prosody_toolkit`) | Sentez hedefi olarak dolaylı |

**"Perde değerine dayalı ezgi sentezi"** (hedef tabanlı sentez) bu tabloda AM, MOMEL ve Fujisaki'nin ortak fikridir:
ezgi, seyrek **perde hedeflerinden** (belirli hecelere bağlı F0 değerleri) ve aralarındaki geçiş kuralından üretilir.
Pierrehumbert'in 1981 sentez kuralları, IPO'nun stilize hareketleri ve MOMEL'in geri dönüşlü noktaları bunun örnekleridir.
Modern karşılığı: fonem ya da hece başına F0 hedefi tahmin eden nöral modeller (FastPitch, FastSpeech 2).

## 2. Türkçe ezgisi: ne biliniyor

- **Ipek & Jun (2013)**, AM modeli, yansız odak: içerik sözcüklerinin çoğunun vurgulu hecesinde **H\***; her içerik sözcüğü bir
  ezgisel sözcük (PW), **sol kenarı L** ile işaretli. PW'nin üstünde iki birim: **ara öbek (ip)**, sonunda **LH** yükselişi
  (ağır sözdizimsel öbeğin sağ kenarı, uzunluktan bağımsız); **ezgi öbeği (IP)**, çeşitli sınır tonları. Üç birim kavşak derecesiyle ayrışır.
- **Kamalı (2011)** ve **Özge & Bozşahin (2010)**: Türkçe odağı vurgu **yerini** değiştirerek değil, **öbekleme** ile işaretler.
  Çekirdek-öncesi öbek yüksek sınır tonuyla biter; çekirdek-sonrası alan **sıkıştırılmış perde aralığı ve alçalma** taşır ve
  orada odaklı/vurgulu öğe bulunamaz. Sözcük sırası, bilgi yapısı ve ezgi birlikte değişir.
- **Kabak & Vogel (2001)**: vurgu, sesbilimsel sözcük düzeyinde; düzensiz vurgu önceden belirlenir; "PW kapatan" ekler (clitic benzeri).
  `stress.py` ve DizgeBERT-G2PTTS kuralları bu çerçeveye dayanıyor.
- **Güneş**: sözdizimi-ezgi arayüzü; tümce düzeyindeki araya girenler (parantetik) IP düzeyinde yalıtılır, öbek düzeyindekiler ip ile
  bütünleşir. `boundary` sınıflarımızın sözdizimsel gerekçesi.
- **Eksik:** Türkçe için açık, ToBI-benzeri etiketli bir ezgi derlemi bulamadım. Soru ezgisi (*mi*'li sorularda *mi*'den önceki sözcüğün
  vurgusu ve sonrası) literatürde betimlenmiş, ama Antalia'da çok az soru var.

## 3. Sentez yöntemlerinin kuşakları

| Kuşak | Nasıl çalışır | Ezgi nereden gelir | Türkçe örnek |
|---|---|---|---|
| **Birleştirmeli (concatenative)**, ikili-ses (diphone) | Her ses geçişinden bir örnek; PSOLA/MBROLA ile perde ve süre değiştirilir | **Tamamen dış model**: kural (IPO, Fujisaki) → hedef F0 → PSOLA | Fatih Üniv. TTTS; Fujisaki uygulaması (Uslu & İlk 2009); IPO tarzı algısal kural modeli |
| **Birim seçimi** (Hunt & Black 1996) | Büyük kayıt veritabanından hedef + birleştirme maliyetini en küçükleyen birim dizisi (Viterbi) | Hedef maliyetine giren ezgi öznitelikleri; ezgi büyük ölçüde **kayıttan** (değiştirilmeden) gelir | Bitişken dile uygun **ek birimli** melez Türkçe sistem (EURASIP 2016) |
| **İstatistiksel parametrik / HMM** (HTS; Tokuda, Zen vd.) | Bağlama bağlı HMM'ler; spektrum + log-F0 (MSD-HMM) + süre birlikte; karar ağaçlarıyla bağlam kümelenir | **Bağlam özniteliklerinden** (vurgu, öbek konumu, POS…) istatistiksel olarak; aşırı düzleşme (oversmoothing) klasik sorun | Türkçe HTS tabanlı sistemler; ek birimli **melez HTS + birim seçimi** dinleyicilerce HTS'ye tercih edildi |
| **Nöral, açık ezgi tahmini** (FastSpeech 2, FastPitch) | Süre + **fonem başına F0** (+ enerji) tahmincisi; gerçek değerlerle eğitilir, sentezde tahmin/elle ayar | Açık **fonem düzeyi perde hedefi**, yani hedef tabanlı sentezin nöral karşılığı | — |
| **Nöral, örtük** (Tacotron, VITS, **Matcha**) | Ezgi, akustik hedefin (mel) parçası olarak örtük öğrenilir | Girdideki ipuçlarından (bizde `ˈ` + sınır özniteliği) | **DizgeTTS**; FreyaTTS (2026, Türkçe öncelikli akış eşleme; 183 M, Apache-2.0) |
| **Nöral, ezgi odaklı** (ProsodyFM, AAAI 2025; PitchFlow 2024) | **Matcha tabanlı**: öbek kırılması kodlayıcısı + süre tahmincisi (kırılma süreleri) + **uç ezgi kodlayıcısı** (6 öğrenilmiş ezgi şekli token'ı) + perde işlemcisi; **etiketsiz** | Etiketsiz keşfedilen kırılma ve **uç ezgi** kalıpları | — |

## 4. Türkçe için öneri

Kuram tarafında seçim açık: **AM (Ipek & Jun) iskeleti + Kamalı / Özge & Bozşahin'in öbekleme gözlemi.** Türkçe'de ezgisel anlam,
vurgu tonlarının seçiminden çok **öbekleme ve sınır tonlarıyla** taşınıyor. Elimizdeki `0 / ip / IP / cümle` sınıfları tam bu eksen
(ToBI'nin kırılma indisi). Tonları elle etiketlemek yerine AM'nin **yapısını** (nerede H\*, nerede ip/IP sınırı) metinden, **gerçekleşmesini**
(ne kadar yüksek, ne kadar uzun) veriden öğrenen melez yol, 4 saatlik veriye en uygunu.

Yöntem tarafında, önem sırasına göre:

1. **Süre / öbekleme (en büyük kaldıraç, ölçümle kanıtlı).** Bu ProsodyFM'in "phrasing" yarısına denk: kırılma süresini ayrı modelle.
   Bizde v5 (doğrusal MSE süre tahmincisi) zamanlamayı gerçeğe getirdi ama CER'i anlamlı kötüleştirdi. Kör AB (v4-400 vs v5s) karar noktası.
   Doğrudan sıradaki adım: **yalnız sınır token'larının** (ayraç, noktalama) süresini doğrusal, fonemleri log ölçekte eğitmek. Bu, anlaşılırlığı
   koruyup duraklamayı düzeltebilir (hipotez).
2. **Uç ezgi (soru, devam yükselişi).** ProsodyFM'in "terminal intonation" yarısı. Önce ucuz yol: `?` ve *mi* için sınır sınıfına
   **cümle türü** (bildirme / soru / *mi*-soru) ekle, ip sonu **LH**'yi AM'ye göre `ip` sınıfının kendisi taşıyor. Veri sorunu var: Antalia'da
   soru az; soru ezgisi için ek veri ya da kural tabanlı F0 düzeltmesi gerekebilir.
3. **Açık F0 hedefi (FastPitch tarzı, hedef tabanlı sentezin nöral biçimi)**, ancak 1 ve 2 yetmezse. Fonem başına ortalama log-F0; hedefler
   v6'nın donmuş MAS hizalamasından (v5'teki `train_dp.py` düzeneği). Perde gömmesi `mu`'ya değil **hizalama sonrası** (çözücü koşulu) eklenmeli.
   v3 dersi: hizalamaya giren her yeni sinyal hizalamayı bozabilir. Kazanç: denetlenebilirlik (odak ve soru için perdeyi elle bükme).
   Mevcut ölçüm F0'ın zaten gerçeğe yakın olduğunu gösteriyor; bu yüzden üçüncü sırada.
4. **Ölçüm aracı olarak MOMEL/INTSINT ya da dalgacık araç takımı.** Sentezde değil **değerlendirmede**: gerçek ve sentez cümlelerde
   H\* tepelerinin vurgulu heceye düşüp düşmediğini, ip sonu LH'nin gerçekleşip gerçekleşmediğini otomatik say. Şu anki ölçütler
   (F0 aralığı, nPVI) AM olaylarını doğrudan görmüyor; bu araçlar Türkçe ToBI eksikliğini kısmen kapatır.

**Önermediklerim:** birim seçimi ve HMM. 4 saat tek konuşmacı veri için nöral gövde kalite olarak açık ara önde; bu yöntemlerin dersleri
(bağlam öznitelikleri, ek birimleri, aşırı düzleşme) zaten ön uç tasarımımıza girdi. Fujisaki: yorumlanabilir ama parametre çıkarımı
gürültülü ve nöral gövdeye ek kazancı belirsiz; analiz aracı olarak dalgacık daha pratik.

## Kaynaklar

- Ipek & Jun 2013, *Towards a model of intonational phonology of Turkish: Neutral intonation*, POMA 19. https://linguistics.ucla.edu/people/jun/Turkish%20intonation-ASA-POMA2013.pdf
- Ipek & Jun 2014, *Distinguishing phrase-final and phrase-medial high tone…*, Speech Prosody. https://linguistics.ucla.edu/people/jun/papers%20in%20pdf/Ipek_Jun-SpeechProsody-2014-Turkish.pdf
- Özge & Bozşahin 2010, *Intonation in the grammar of Turkish*, Lingua 120. https://www.sciencedirect.com/science/article/abs/pii/S0024384109001272
- Kamalı 2011, *Topics at the PF Interface of Turkish* (doktora tezi, Harvard). https://www.academia.edu/934759/Topics_at_the_PF_Interface_of_Turkish
- Kabak & Vogel 2001, *The phonological word and stress assignment in Turkish*, Phonology 18. https://www.researchgate.net/publication/231964604_The_phonological_word_and_stress_assignment_in_Turkish
- Güneş, *Constraints on syntax-prosody correspondence: parentheticals in Turkish*, Lingua. https://www.sciencedirect.com/science/article/abs/pii/S0024384114001752
- Uslu & İlk 2009, *Fujisaki intonation model in Turkish Text-to-Speech Synthesis*, IEEE SIU. https://ieeexplore.ieee.org/document/5136528
- *A rule based perceptual intonation model for Turkish TTS*. https://www.researchgate.net/publication/261247530_A_rule_based_perceptual_intonation_model_for_Turkish_text-to-speech_synthesis
- *Hybrid statistical/unit-selection Turkish speech synthesis using suffix units*, EURASIP JASMP 2016. https://link.springer.com/article/10.1186/s13636-016-0082-0
- *TTTS: Turkish text-to-speech system*. https://www.academia.edu/20696422/TTTS_Turkish_text_to_speech_system
- Hirst, *A Praat plugin for Momel and INTSINT*. https://hal.science/hal-03625441v1/document ; INTSINT: https://en.wikipedia.org/wiki/INTSINT
- Suni vd. 2017, *Hierarchical representation and estimation of prosody using CWT*, CSL 45. https://www.sciencedirect.com/science/article/abs/pii/S0885230816303527 ; araç: https://github.com/asuni/wavelet_prosody_toolkit
- He vd. 2025, *ProsodyFM*, AAAI. https://arxiv.org/abs/2412.11795
- Sadekova vd. 2024, *PitchFlow*, Interspeech. https://www.isca-archive.org/interspeech_2024/sadekova24_interspeech.pdf
- Łańcucki 2020, *FastPitch*. https://arxiv.org/pdf/2006.06873
- Pamuk vd. 2026, *FreyaTTS*. https://arxiv.org/abs/2607.09530
- Klasik kaynaklar (künyeden, çevrimiçi doğrulanmadı): O'Connor & Arnold 1961; Halliday 1967; Crystal 1969; 't Hart, Collier & Cohen 1990;
  Pierrehumbert 1980; Silverman vd. 1992 (ToBI); Ladd 2008; Fujisaki 1984; Hunt & Black 1996; Zen, Tokuda & Black 2009.
