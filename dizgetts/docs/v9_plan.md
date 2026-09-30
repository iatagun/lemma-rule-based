# v9 hazırlığı: veri bilimi ve kod incelemesi (2026-09-30)

## 1. Durum
- En iyi model v6-dp. v8 (g2ptts v1 sesbirimleri) kör AB'de 12 / 8 / 10, kararsız; sentezde uzun ünlü çarpanı duyulmadı (8/12 fark yok).
- Val kaybı v4 ep150'den beri 1,60-1,61 (v4 -> v6 -> v8 zinciri): bu veriyle akustik model DOYMUŞ; aynı veride daha uzun eğitim kazanç getirmez.
- En büyük ölçülmüş kaldıraç SÜRE: gerçek (MAS) sürelerle sentez kör AB'de 29-1 (oracle_A_vs_B). v6-dp (doğrusal MSE ile yeniden eğitilmiş süre tahmincisi) 19-0.
  Kalan açık: tahmin sıralaması iyi (virgül bölgesinde log-korelasyon 0,72; token düzeyi 0,49) ama std gerçeğin %58-65'i HER token türünde (tekdüzelik).
  Denenip reddedilenler: sentezde moment eşleme (v4c: konuşma %22 yavaşladı, Jensen), akış örnekleyici (v7: çeşitlilik arttı, korelasyon 0,49 -> 0,40).

## 2. Etiket geçerliliği: ayrım kayıtta var mı? (`eval/label_validity.py`, 1053 klip, 151.852 bölüt, v8 MAS)
| Etiket | Kayıtta | Kanıt (klip bootstrap %95 GA) |
|---|---|---|
| Vurgu (ˈ) | VAR, güçlü | vurgulu - diğer ünlü (aynı sözcük): +4,05 dB [+3,94, +4,17], +1,94 yt [+1,77, +2,12], +4,2 ms (n=16.000 sözcük) |
| Alıntı ince l | VAR | F2: l/ön-a 1542, l/œ 1557 vs ł/arka 1241 Hz (sıradan ön l 1815) |
| Uzun aː | VAR | uzun/kısa 1,7-2,3 (eval/long_vowel.py) |
| ğ uzunluğu | ZAYIF | uzun/kısa 1,24-1,32 (hedef 1,5; konuşmacı yapmıyor) |
| Ön a (ae) | YOK | a - ɑ F2 +64 Hz [-4, +134] anlamsız; aː - ɑː +82 [-13, +161] anlamsız |
| Alıntı œ (oe) | o'ya yakın | œ - ɔ +71 [+13, +139]; œ - ø -225 [-282, -167] |
| Alıntı Y | ölçülemez | n=2 |

## 3. Kod / yöntem bulguları
1. **v8 gömme başlangıcı veriyle çelişiyor (hata):** œ sıradan ö'den (eski œ) başlatıldı, konuşmacı o'ya yakın söylüyor; 81 örnekle düzelmedi (v8'de cos(œ, ø) 0,93).
   a/aː (ɑ+ɛ)/2'den başlatıldı (öne çekti); konuşmacı arka söylüyor. Doğrusu: gömme başlangıcı AKUSTİĞE göre (œ <- ɔ, a <- ɑ, aː <- ɑː, Y <- U).
   Ders: sembol başlatma kararı dilbilimsel hedefe değil, eğitim kaydının akustiğine göre verilmeli (model kaydı öğrenir).
2. **Süre tahmincisi bağlamı dar:** Matcha DurationPredictor 2 conv (k=3) -> alıcı alan 5 interspersed token (~2-3 sesbirim); bağlam yalnız
   (detach'lı) kodlayıcı gizli durumundan ve sınır sınıfından (dp_feat) geliyor. Sözcük içi konum, hece yapısı, sözcük uzunluğu açık değil.
3. **Kontrol noktası seçimi:** best.pt val kaybıyla seçiliyor ama val 59 klip ve kayıp gürültü bandında (1,60-1,62) -> seçim anlamsız; son epoch kullanılıyor (doğru).
4. **Sabit lr 1e-4, ısınma/azalma yok; her sıcak başlangıçta Adam sıfırdan.** Doygunlukta sorun değil; yeni başlıklar eklenirse ısınma düşünülmeli.
5. **Değerlendirme gücü:** 30 çiftlik işaret testi yalnız büyük tercihleri (~%75+) yakalar; %65 tercih için ~85 karar (fark yok hariç) gerekir.
   Kahin (29-1) boyutunda bir etki 30 çiftle rahat görülür; sesbirim/uzunluk boyutundaki etkiler görülmez.

## 4. v9 önerisi (tek değişken ilkesi: aynı anda tek kaldıraç)
**v9 = daha güçlü, deterministik süre tahmincisi**, dondurulmuş akustik model üzerinde (train_dp yolu; eğitim dakikalar-1 saat):
- Girdi: detach'lı kodlayıcı gizli durumu + dp_feat + AÇIK öznitelikler: token türü, vurgu, sözcük içi hece konumu (ilk/orta/son), sözcük hece sayısı,
  açık/kapalı hece, ezgi öbeği içi konum, noktalamaya uzaklık.
- Ağ: bağlamı geniş (ör. 4 katman genişletilmiş conv ya da 2-3 katman küçük transformer), kayıp doğrusal MSE (v6-dp'de kanıtlandı).
- Taban: v6 ep150 (kanıtlanmış en iyi akustik model). v8 ayrıca değerlendirilmez (karışan değişken olmasın).
- Karar kuralı (sonuçtan ÖNCE): test'te token log-süre korelasyonu >= +0,05 (0,49 ->) VE std oranı >= 0,75 VE toplam süre oranı 0,97-1,03
  -> ancak o zaman kör AB (v6-dp vs v9-dp, 30 çift, tek soru "hangisi daha doğal"). İç ölçüt geçmezse AB yok.
**v9b (sonra, ayrı):** v8 tarifi + akustiğe göre gömme başlangıcı (œ <- ɔ, a <- ɑ, aː <- ɑː, Y <- U); yalnız v9'dan sonra ve ayrı AB ile.

## 5. Sonuç (2026-09-30): v9 RED (experiments.yaml v9-dp2)
Test: korelasyon 0,622 -> 0,575, std 0,692 -> 0,644 (ikisi de anlamlı kötü), toplam aynı; val'de %1 iyi (59 klip, gürültülü seçim), train'de güçlü aşırı öğrenme.
**Yapısal ders:** MSE-en iyi deterministik tahmincide std oranı ~ korelasyon; tekdüzelik, öngörülebilirlik sınırının doğal sonucu. Açık metin öznitelikleri
bilgi eklemedi. Süre kahininin (29-1) kazandığı kısım metinden öngörülemeyen varyans olabilir -> deterministik yol bu veriyle tavanda.
**Bulunan tuzaklar:** (1) ÖN UÇ SÜRÜMÜ - CHECKPOINT: Synth her zaman GÜNCEL ön uç kodunu kullanır; v6 (v6 dönemi token'larıyla eğitildi) bugün sentezlenirse
eğitimde görmediği token'lar alır (ø/y, j, ɾ). Yayına hazır DizgeTTS-Antalia paketi v6 -> güncel dizgetts ile YANLIŞ girdi alır. Önlem gerekir
(engine_versions denetimi ya da ön uç sürümünü checkpoint'e sabitlemek). (2) Ölçüm betiklerinde çift yuvarlama (düzeltildi).
**Seçenekler:** (a) daha çok / farklı veri (amaca özel kayıt; çok konuşmacılı ön eğitim) - öngörülemeyen varyansı azaltmaz ama konuşmacı sesletimini düzeltir;
(b) olasılıksal süre modeli ama korelasyonu KORUYAN (v7 akış örnekleyicisi korelasyonu düşürdü; örn. düşük sıcaklıkta örnekleme, ya da tahmin + ölçekli artık);
(c) süre yerine başka kaldıraç (vokoder ince ayarı - kahin deneylerinde önerilmişti).
