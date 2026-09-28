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
  vurgusu ve sonrası) literatürde betimlenmiş. **Düzeltme (aynı gün, ölçüm):** Antalia'da soru AZ DEĞİL — 1053 klipte 664 `?`, 605 *mi*-benzeri
  sözcük (hasta-klinik diyalogları). İlk sürümde bunu ölçmeden yazmıştım.

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

## 5. Bizim yaptıklarımızla karşılaştırma (ölçüm: `eval/am_events.py`, v4-e400, 143 test+val cümlesi, gerçek vs sentez)

| Olay (kuram) | Gerçek | Sentez | Yorum |
|---|---|---|---|
| **H\***: çok heceli sözcükte F0 tepesi vurgulu ünlüde (AM) | %48 (sonraki ünlüde %6, başka %46) | %51 (%6, %42) | Model gerçeğin dağılımını kopyalıyor. Gerçekte bile tepe sözcüklerin yarısında vurgulu hecede değil: vurgusuzlaşma / çekirdek sonrası sıkışma (Kamalı) ya da vurgu etiketi hatası. Ayrıştırılmadı. |
| **ip sonu LH** (virgül öncesi son ünlü − önceki, yt) | medyan +0,9; yükselen %49, düz %20, düşen %31; std 6,2 | medyan +3,6; yükselen %61, düz %16, düşen %23; std 5,3 | **Kalıplaşma:** model virgülde fazla sık ve fazla yükseltiyor. Gerçekte virgülün yarısı yükselmiyor: sınır tonu yalnız noktalamadan çıkmıyor (Kamalı / Özge & Bozşahin: çekirdek-öncesi öbek H ile biter, sonrası sıkışır). |
| **mi-soru**: *mi*'den önceki sözcük tepesi − cümle ort. | +6,7 yt (medyan +8,6) | +7,4 (+9,6) | **Doğru.** Model örüntüyü veriden öğrenmiş (Antalia'da 605 *mi*). |
| *mi* ünlüsü − önceki tepe | −10,4 yt | −11,5 | Doğru (tepe → sert düşüş). |
| Soru sonu sözcük (son − önceki ünlü) | −0,8 | −1,0 | Doğru. |
| Nokta sonu (L%) | medyan −1,0 | −0,7 | Yakın. |
| Cümle F0 aralığı / alçalma (önceki ölçüm) | 15,8 yt / −0,76 | 15,6 / −0,79 | Doğru. |
| Duraklama / ritim (önceki ölçüm) | 470 ms / nPVI 45 | 310 ms / 27 | **En büyük açık.** |

Sınırlar: eşleşmemiş (gerçek ve sentezde ölçülebilen sözcükler farklı), tek konuşmacı, sözcük vurgusu etiketleri tahmin (altın değil), ünlü F0 = medyan (tepe zamanlaması kaba).

### Kuram → bizde karşılığı → yapabileceğimiz

| Kuram / yöntem | Bizde var mı | Yapabileceğimiz (maliyet) |
|---|---|---|
| AM yapısı (PW, ip, IP) | **Var:** vurgu `ˈ` + `0/ip/IP/cümle` süre özniteliği | Sınır sınıfını **ton tipiyle** zenginleştir (ip-H / ip-L). Etiket sesten otomatik (virgül öncesi yükseliş/düşüş), metinden g2ptts ile tahmin (orta) |
| ToBI ton katmanı | Yok (yalnız kırılma indisi) | Yukarıdakinin genellemesi; Momel-INTSINT ile otomatik T/H/L etiketi → yarı-otomatik Türkçe ton katmanı (orta-yüksek) |
| İngiliz okulu: *tonicity* (çekirdek yeri) | Yok | Türkçede varsayılan çekirdek = **fiil-önü öğe** (yansız odak). DizgeBERT-Dep ile fiil-önü sözcüğü işaretle, önce gerçekte çekirdek sonrası sıkışmayı ÖLÇ (düşük), sonra öznitelik (orta) |
| IPO: algısal stilizasyon, analiz-yoluyla-sentez | Yok | **Kahin (oracle) deneyleri** — hangi kaldıraç önemli: (a) Matcha'ya gerçek MAS süreleri verilerek sentez (süre kahini), (b) sentezin F0'ını gerçeğin stilize F0'ıyla değiştirme (Praat PSOLA; F0 kahini). UTMOS + kör dinleme. Eğitim yok (düşük) |
| MOMEL/INTSINT, CWT | Yok | Değerlendirmeye ekle (`am_events.py` yerine hedef noktası tabanlı ölçüm) (düşük-orta) |
| Fujisaki | Yok | Önerilmiyor (nöral gövdeye kazancı belirsiz) |
| Açık F0 hedefi (FastPitch, PitchFlow) | Yok (örtük) | Ancak F0 kahini büyük kazanç gösterirse (yüksek) |
| ProsodyFM (öbekleme + uç ezgi, Matcha tabanlı) | Kısmen (sınır → süre) | Öbekleme yarısı = süre modeli işi; uç ezgi zaten doğru (yukarıda) |
| HMM / birim seçimi | — | Önerilmiyor |

**Önerilen sıra:** (1) kahin deneyleri, çünkü süre mi F0 mı sorusunu eğitimsiz yanıtlar ve sonraki her kararı belirler; (2) süre modeli (en büyük ölçülen açık);
(3) ip ton tipi (virgül kalıplaşması); (4) çekirdek/odak ölçümü. Açık F0 modeli ancak (1) gerektirirse.

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

## 6. Kahin deneyleri (2026-09-27, `eval/oracle.py`, v6 ep150, 84 test klibi)

| Koşul | CER % (toplam) | klip medyanı CER % | UTMOS |
|---|---|---|---|
| A normal sentez | 2,2 | 1,5 | 2,89 |
| B süre kahini (gerçek MAS süreleri) | 5,6 | 3,1 | 2,87 (A'ya göre −0,02, anlamsız) |
| Bpsola (B, kendi F0'ıyla PSOLA) | 5,4 | — | 2,70 |
| C süre + F0 kahini (B + gerçek F0, PSOLA) | 9,4 | 2,6 | 2,38 (Bpsola'ya göre −0,32) |
| Rvoc gerçek mel → HiFi-GAN | 4,8 | 2,6 | **2,64** |
| R gerçek kayıt | 3,1 | 2,1 | **3,41** |

Toplam CER'i Whisper uydurmaları sürüklüyor (ör. klip 22'de B ve Rvoc için kayıtta olmayan bir cümle yazıyor); medyanlar daha güvenilir.
Bulgular:
1. **Vokoder darboğazı:** gerçek mel'i HiFi-GAN universal'dan geçirmek UTMOS'u 3,41'den 2,64'e düşürüyor (−0,77), sentezimiz (2,89) bundan bile yüksek.
   Gerçek kayıt ile sentez arasındaki 0,52'lik UTMOS açığının büyük kısmı akustik modelde değil vokoderde olabilir → Antalia üzerinde vokoder ince ayarı.
2. **Süre kahini doğallığı (UTMOS) artırmadı, anlaşılırlığı (CER) düşürdü:** gerçek zamanlama daha hızlı ve daha kaynaşık; Whisper yavaş/net sentezi
   daha kolay çözüyor. v5'teki "zamanlama iyileşti ama CER kötüleşti" örüntüsüyle aynı: CER doğal zamanlamayı cezalandırıyor.
3. **F0 kahini sonuçsuz:** gerçek F0'ı PSOLA ile sentez sesine aktarmak UTMOS'u düşürdü, ama bu büyük olasılıkla yöntem hatası (gerçek ve sentezin
   ötümlülük sınırları farklı; gerçek perde takibindeki oktav hataları; PSOLA bozulması). Perde katkısını bu yolla ölçemedik.
Sonuç: sıralama değişti → (1) vokoder ince ayarı, (2) süre için karar kör dinlemeyle (CER değil), (3) perde için kahin yerine doğrudan AB.

**Düzeltme + kör AB (aynı gece):** kullanıcı Rvoc'u "mükemmel" buldu → vokoder darboğazı hipotezi yanlış; UTMOS'un −0,77'si yanıltıcı (bu tür
kıyasta UTMOS'a güvenme). Açık akustik modelin mel'inde. Kör AB A (sentez) vs B (süre kahini), 30 çift, tek soru "hangisi daha doğal":
**B 29 / A 1 / fark yok 0 (işaret testi p < 0,001)** (`reports/ab/ab_oracle_A_vs_B.tsv`). **Kaldıraç süre modeli.** CER bu yönü cezalandırıyor
(B'nin medyan CER'i daha yüksek) → süre kararlarında CER üst sınırı karar kuralı olamaz; kör dinleme + gerçeğe uzaklık (duraklama, nPVI, hız) kullanılmalı.

## 7. Kullanıcı sınır etiketleri (2026-09-28; D:/dizgetts/user_prosody, 40 cümle, `tools/read_breaks.py`)

Kullanıcı (dilbilimci) 40 cümleyi telefonla okudu ve Praat'ta sözcük sonrası sınırı kulakla 0/1/2 işaretledi (324 sözcük arası konum; yalnız 5'i noktalamalı).
- **Kulak = akustik:** etiket 0 → sessizlik medyanı 40 ms, 1 → 60 ms [60–100] (çoğunlukla sessizliksiz: uzama/perde), 2 → 250 ms [200–281].
- **İki konuşmacı büyük ölçüde aynı yerde durur (ilk 24 Antalia cümlesi):** kullanıcı 2 dediğinde Antalia okuyucusu da ≥180 ms duruyor 19/29;
  Antalia ≥180 ms durduğunda kullanıcı 28/41'inde sınır işaretlemiş → sınırların çoğu dilsel, bir kısmı konuşmacıya özgü.
- **Model (g2ptts sınır başlığı) vs kullanıcı:** sınır var/yok duyarlılık 0,70, kesinlik 0,66. Güç: kullanıcının 38 belirgin (2) sınırından model
  yalnız 9'unu 2, 21'ini 1, 8'ini 0 veriyor → **güçlü sınırları zayıf tahmin ediyor** (= "duraklamalar kısa" ölçümünün kök nedeni).
- Kalıplar (kural adayları; bu 40 cümle artık GELİŞTİRME seti):
  1. Cümle başı söylem belirteci sonrası belirgin sınır: *Yani, Ayrıca, Sonunda, Bu arada, İstersen* (model çoğunlukla 0/1). *Öncelikle* 1.
  2. Bağlaç ÖNCESİ belirgin sınır, sonrası sınır yok: *…gelmişti ‖ ama*, *…istedim ‖ ama*, *…çıkmayacağım ‖ çünkü*, *…mı ‖ yoksa*, *…dolduruyor ‖ ya da*.
     Model bağlaçtan SONRA sınır koyuyor (*ama ‖ sofrada*, *ama ‖ hiçbir*: kullanıcı 0). İstisna: *Küçük bir şey ama ‖ ikimizi de*.
  3. Zarf-fiil / koşul sonrası: *-dığında, -meden, -rken, -se* çoğunlukla 2; kısa cümlelerde 1 (*gelince, bitirdiğinde*).
  4. Konu *ise* sonrası 2 (uzun konu öbeğinde); kısa *Ben ise* sonrası 0.
  5. Uzun özne sonu 1–2 (*tamamı, ağacı, tercihler*: 2; *komşumuz, hepsi*: 1).
  6. Model yanlış-pozitifleri çoğunlukla ad öbeği İÇİNDE: *Dükkânın ‖ hemen*, *kırk ‖ beş*, *katılamayan ‖ çalışanların*, *kaçırmamak ‖ için*.

### Sınır kuralları (frontend/boundary_rules.py, Engine(boundary_rules=True); eval/boundary_rules_eval.py)
- Geliştirme (kullanıcı 40 cümle): F1 0,66 → 0,74; güçlü sınır 6/33 → 18/33.
- Antalia TEST, ölçülen sessizlik (bakılmamış): F1 0,50 → 0,47. Kural kural (iyileştirdi/kötüleştirdi): K1 1/5, K2 2/3 (+sil 3/1), **K3 zarf-fiil 1/33**,
  K4 0/1, **K5 ad öbeği içi silme 10/3**. Antalia okuyucusu zarf-fiil sonrası çoğunlukla SESSİZLİK bırakmıyor; kullanıcı orada sınır DUYUYOR
  (sessizliksiz: uzama/perde). Sessizlik tabanlı ölçüt algısal sınırı göremez.
- Tasarım kararı: kurallar EĞİTİMDE değil yalnız SENTEZDE (dp Antalia'nın gerçek alışkanlığını öğrenir; kurallar sınır davranışını dinleyicinin
  duyduğu yerlere taşır). Yeniden eğitim yok: v6-dp + ep400_dp_mse_brules.pt (cfg.engine.boundary_rules=True). Karar: kör AB
  (kuralın değiştirdiği 119/484 cümleden 30 çift; D:/dizgetts/ab/v6dp_sinir_kurallari).
- **Kör AB sonucu (2026-09-28, reports/ab/ab_v6dp_sinir_kurallari.tsv):** kurallı 8 / kuralsız 8 / fark yok 14 (p = 1,0). Kullanıcı: "hepsi birbirine çok
  benziyordu". Kurallar yalnız sentezde uygulandığında ALGILANABİLİR fark yaratmıyor → VARSAYILAN KAPALI kalır (kod durur, Engine(boundary_rules=True)).
  Olası neden: süre tahmincisi sınır özniteliğine zayıf tepki veriyor (öznitelik yalnız dp girdisinde; sınır sınıfı değişince tahmin edilen süre az değişiyor).

## 8. Süre modeli tanısı ve v7 (akış eşlemeli süre) (2026-09-28)
- Ölçüm aracı hatası: `cum_round_logw` kare değil log(k−0,5) döndürür; ilk "tepki" ölçümleri (+11/+29 ms) GEÇERSİZ. Düzeltilmiş (`eval/dp_response.py`):
  kuralın yükselttiği sınırda +23 ms medyan (ort. +70); tüm sınırlar 0→IP zorlanınca sözcük sonu başına +71 ms (v6-dp) / +77 (v6m-dp).
- v6m-dp (dp_feat ÖLÇÜLEN sınırlardan, `scripts/build_manifest_measured.py`, `_phon_v6m`): noktalamasız gerçek IP'de (test, n=39, gerçek 263 ms)
  v6-dp 144 ms; v6m-dp kahin sınırla 209, tahmin sınırla 143 → öznitelik doğru gelirse işe yarıyor ama sınır tahmini bu yerleri bulamıyor. AB yapılmadı (fark duyulmaz).
- **Asıl tanı (test, token türüne göre, v6-dp):** ortalamalar doğru (ünlü 27/26 ms, ünsüz 27/26, noktalama sonrası ayraç 401/388, noktalama 186/174,
  noktalamasız ayraç 78/74) ama **tahmin std'si gerçeğin %58–65'i HER türde** → tekdüzelik (nPVI 35 vs 45). Hata payı: ses süreleri %49, sınırlar %51.
  Deterministik (MSE) süre tahmincisinin bilinen ortalamaya çekme huyu. Kahin AB'nin 29-1'i büyük olasılıkla bundan.
- v7: `train/flowdp.py` (OT-CFM, koşul = x_dp + deterministik logw) + `scripts/train_flowdp.py` (dondurulmuş v6-dp; yalnız FlowDP). Sentezde logw yerine örnek.
  Karar öncesi denetim: val'de örneklenen sürelerin std oranı ~1'e çıkmalı, ortalama korunmalı; sonra (fark ölçülebilirse) kör AB v6-dp vs v7.
- **v7 sonuçları (eğitimsiz iç ölçüm, test):** en iyi val ep90 (sonra aşırı öğrenme). Std oranı (tahmin/gerçek): ünlü 0,58→0,82, ünsüz 0,61→0,79,
  boşluk 0,62→0,83, ayraç 0,52→0,68; toplam süre 0,97→0,95. Log-süre korelasyonu 0,49→0,40 (beklenen: kusursuz bağımsız örnekleyici ρ²≈0,24 verirdi;
  0,40 çeşitliliğin çoğunun anlamlı olduğunu gösteriyor). Ünlü nPVI (token) 48,8→57,8 (gerçek 60,4). Sonraki: 484 cümle sentez + kör AB v6-dp vs v7.
