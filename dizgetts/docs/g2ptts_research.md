# dizge-g2p-tts: katkı ve geliştirme (2026-09-26)

## 1. Şimdiye kadar ölçülen katkı

| Katman | Ölçüm | Sonuç | Kanıt |
|---|---|---|---|
| **Vurgu** (ön uç doğruluğu) | Kör test seti (97 sözcük, kurallar buna bakılmadan) | g2ptts **%92,8** / son hece %83,5 / espeak %73,2; g2ptts − son hece +9,3 pp [+2,1, +17,5] | `tests/stress_gold_test.tsv` |
| **Vurgu** (TTS'e etkisi) | v2-m1a (≈son hece) vs v3a (g2ptts vurgusu), aynı reçete, 543 cümle | CER 4,0 = 4,0 (anlamsız); Whisper vurguya duyarsız | experiments.yaml v3a |
| **Sınır** (ön uç doğruluğu) | Antalia test, sesten ölçülen duraklama | F1 0,50 (P 0,42 R 0,62) vs Dep kuralı 0,35, fark [+0,09, +0,21] | experiments.yaml g2ptts-v2b/v3 |
| **Sınır** (TTS'e etkisi) | Token olarak (v3) | Hizalama oturmadı -> başarısız | v3-g2ptts-tts |
| **Sınır** (TTS'e etkisi) | Süre tahmincisine öznitelik (v4) vs v3a | "Sözcük sözcük okuma" hissi gitti (kullanıcı); kör AB 16-9 (p=0,23); duraklama +20 ms | v4-dpfeat |

**Açık soru:** v4'teki öznitelik noktalama + g2ptts sınır modelini BİRLİKTE taşıyor. Sınır MODELİNİN (noktalamasız ip tahminleri) payı ayrıca ölçülmedi -> §2.

## 2. Ablasyon (eğitim yok; v5 modeli, 143 test+val cümlesi) — `eval/ablate_frontend.py`
Futamata vd. (2021) tasarımı: tahmin edilen sınır, yalnız noktalama ve gerçek kayıttan KEHANET sınırı karşılaştırılır.

| Varyant | Sınır | Vurgu | Ölçtüğü |
|---|---|---|---|
| default | g2ptts (model + noktalama) | g2ptts | – |
| punct | yalnız noktalama | g2ptts | sınır modelinin katkısı (default − punct) |
| oracle | gerçek kayıttaki duraklama (+ noktalama) | g2ptts | kalan pay (oracle − default) |
| m1a | g2ptts | M1a (≈ son hece) | vurgu katmanının etkisi (CER + kör AB) |

Sonuçlar (2026-09-26, v5 = `ep400_dp_mse.pt`, 143 cümle; eşleşmiş klip bootstrap; gerçek kayıt referans):

| Ölçü | Gerçek | default (g2ptts) | punct (yalnız noktalama) | oracle (kehanet) |
|---|---|---|---|---|
| Noktalamasız sınırda duraklama (ms) | 17,4 | **12,9** | 5,0 (-7,9 [-9,2, -6,7]) | 12,5 (-0,5, anlamsız) |
| Noktalamalı sınırda duraklama (ms) | 469 | **468** | 419 (-49 [-56, -43]) | 457 (-10 [-15, -6]) |
| Noktalamasız son ünlü uzaması | 1,06 | **1,00** | 0,97 (-0,03 anlamlı) | 1,00 (0) |
| nPVI | 44,4 | **34,0** | 32,9 (-1,1 anlamlı) | 34,0 (0) |
| Cümle F0 aralığı (yt) | 15,8 | **15,6** | 15,5 (-0,16 anlamlı) | 15,6 (0) |
| CER (143) | – | 3,2 | +0,44 (anlamsız) | +0,40 (anlamsız) |

Bulgular:
1. **Sınır modelinin katkısı ölçülebilir** (default - punct): tüm zamanlama ölçütleri gerçeğe doğru; noktalamada bile duraklamayı 419 -> 468 ms
   yapıyor (model IP/ip ayrımıyla). CER'e etkisi anlamsız.
2. **Kehanet ≈ tahmin** (oracle - default ~0): bu konuşmacı ve TTS için sınır SINIFLANDIRMASINDA pay kalmamış (Futamata vd. bulgusuyla aynı).
   Kehanetle bile noktalamasız duraklama 12,5 ms (gerçek 17,4) -> kalan açık sınırın YERİNDE değil, SÜRE ÜRETİMİNDE (süre modeli).
3. **Vurgu** (m1a - default): CER 3,2 -> 5,2 (+2,0 [-1,1, +8,2], anlamsız, birkaç cümle sürüklüyor). Model g2ptts vurgusuyla eğitildiği için
   M1a girdisi dağılım dışı -> adil ölçüm değil. Vurgunun katkısı için KÖR AB gerekir (Whisper vurguya duyarsız).


## 3. Geliştirme yol haritası (kanıt ve literatürle)

1. **Vurgu kalan hatalar** (kör test + geliştirme seti): ad + koşaç -ydI/-ysA (+kişi) (neydi, hazırsanız), zamir istisnaları (sana, bana),
   alıntı ve gün adları (sözlük üyeliği; konumu ağırlık kuralı veriyor), uzun ünlü bilgisi (â, î, û; hem ağırlık kuralı hem dizge fonemi için).
   Etiket kaynağı: kullanıcı kuralları -> kural modülü -> damıtma. Test seti artık dokunulmaz; yeni kurallar geliştirme setinde ölçülür.
2. **Sınır modeli**: tek konuşmacı (Antalia) ve düşük kesinlik (0,42). Duraklama konuşmacıya bağlıdır (konuşmacı koşullu sınır tahmini:
   [arXiv 2509.00675](https://arxiv.org/pdf/2509.00675)). Seçenekler: (a) ikinci konuşmacı (kullanıcının kendi kaydı) ile iki konuşmacı arası uyum = TAVAN;
   (b) çok konuşmacılı açık veri (FLEURS-tr CC-BY, ISSAI TSC MIT) ile zorunlu hizalamadan sınır etiketi; (c) sınıf yerine duraklama SÜRESİ tahmini
   (v5'in doğrusal süre tahmincisi zaten süre kullanıyor).
3. **Belirginlik / odak (prominence)**: cümlede hangi sözcüğün öne çıktığı (çekirdek) — Türkçede çekirdek öncesi / sonrası ayrımı ve çekirdek sonrası
   daralma (Kamalı 2011; Özge & Bozşahin 2010). Metinden BERT ile tahmin edilebilir ([Talman vd. 2019](https://arxiv.org/abs/1908.02262);
   odak kontrolü [arXiv 2207.01718](https://arxiv.org/pdf/2207.01718)). Etiket kaynağı: gerçek kayıtlardaki F0'dan otomatik (Antalia).
4. **Ezgi etiketleri (AM)**: PW/ip/IP kenar tonları (Ipek & Jun 2013). Sözdizimine dayalı öbek tespiti Türkçe TTS'te önceden önerilmiş
   ([Külekci & Oflazer](https://www.semanticscholar.org/paper/An-infrastructure-for-Turkish-prosody-generation-in-K%C3%BClekci-Oflazer/983926ac334a002e3f96d9518c39095006f5eec3)).
5. **Fonem (dizge)**: gündelik söyleyiş varyantları (iddaa, kılinik), uzun ünlü sözlüğü, Arapça/Farsça alıntılar — `docs/pronunciation_issues.md`.
6. **Sözcükler arası ses etkileşimi** (yeniden heceleme, ötümsüzleşmenin ünlü önünde kalkması): kullanıcı kuralları bekleniyor.
7. **Değerlendirme**: her ön uç değişikliği için (a) ön uç doğruluğu (kör set), (b) ablasyon, (c) kör AB. Whisper CER vurguya duyarsız.

## Kaynaklar
- Futamata vd. 2021, Phrase break prediction with bidirectional encoder representations in Japanese TTS — [arXiv 2104.12395](https://arxiv.org/abs/2104.12395)
- Speaker-conditioned phrase break prediction (2025) — [arXiv 2509.00675](https://arxiv.org/pdf/2509.00675)
- Talman vd. 2019, Predicting prosodic prominence from text — [arXiv 1908.02262](https://arxiv.org/abs/1908.02262)
- Predicting and controlling prominence in neural TTS using a language model — [arXiv 2207.01718](https://arxiv.org/pdf/2207.01718)
- Külekci & Oflazer, An infrastructure for Turkish prosody generation in TTS — [Semantic Scholar](https://www.semanticscholar.org/paper/An-infrastructure-for-Turkish-prosody-generation-in-K%C3%BClekci-Oflazer/983926ac334a002e3f96d9518c39095006f5eec3)
- Ipek & Jun 2013 — [POMA](https://linguistics.ucla.edu/people/jun/Turkish%20intonation-ASA-POMA2013.pdf); Özge & Bozşahin 2010 — [Lingua](https://www.sciencedirect.com/science/article/abs/pii/S0024384109001272)
