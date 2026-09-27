---
language: tr
license: cc-by-4.0
library_name: pytorch
pipeline_tag: text-to-speech
tags:
- text-to-speech
- turkish
- matcha-tts
- flow-matching
- prosody
- experimental
datasets:
- cloud0day3/antalia-voice-corpus
---

# DizgeTTS-Antalia (deneysel, v0)

Türkçe metinden konuşma. [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) akustik modeli, **dilbilimsel bir ön uçla** birlikte:
fonemler kural tabanlı [`dizge`](https://pypi.org/project/dizge/) G2P'sinden, sözcük vurgusu ve duraklama sınırları
[`iatagun/DizgeBERT-G2PTTS`](https://huggingface.co/iatagun/DizgeBERT-G2PTTS)'den gelir. Tek konuşmacı (Antalia derleminin okuyucusu), 22,05 kHz.

> **Deneysel araştırma sürümüdür.** 4,2 saatlik tek bir okuma derleminden, 4 GB'lık bir dizüstü GPU'sunda eğitildi. Ticari kalitede bir ses değildir.
> Sınırları aşağıda açıkça yazılı.

## Dinleyin

Aşağıdaki cümleler eğitim verisinde **yok** (seçilmeden, ilk üretim; tohum 0):

1. *Merhaba, ben DizgeTTS. Türkçe metinleri sesli okuyabilirim.*

   <audio controls src="https://huggingface.co/iatagun/DizgeTTS-Antalia/resolve/main/samples/00.wav"></audio>

2. *Kağıt, kalem ve silgiyi masanın üstüne bıraktım; akşam yeniden bakacağım.*

   <audio controls src="https://huggingface.co/iatagun/DizgeTTS-Antalia/resolve/main/samples/01.wav"></audio>

3. *Yarın sabah saat dokuzda İzmir'e gidiyoruz, değil mi?*

   <audio controls src="https://huggingface.co/iatagun/DizgeTTS-Antalia/resolve/main/samples/02.wav"></audio>

4. *Bu sistem, sözcüklerin vurgusunu ve cümle içindeki duraklamaları kendisi tahmin ediyor.*

   <audio controls src="https://huggingface.co/iatagun/DizgeTTS-Antalia/resolve/main/samples/03.wav"></audio>

## Ne farklı: ön uç

Türkçe TTS'te sorunların çoğu akustik modelde değil, **metnin nasıl okunacağında**. Bu model şu ön ucu bekler (aynı ön uç eğitimde de kullanıldı):

| Aşama | Ne yapar |
|---|---|
| Normalleştirme | Sayı, tarih, saat, para, kısaltma, kodları açar (`15:30` → *on beş otuz*, `3,5 TL` → *üç lira elli kuruş*) |
| Fonemleştirme | `dizge==0.1.6`: ince/kalın `k g l`, sözcük sonu `r`, ünlü uzunluğu (`ğ`), ödünçleme sözcükler |
| Söyleyiş sözlüğü | Özenli okuma: *kağıt, hakim, değer, eğri* (uzun ünlü), *-eceğim* (geçiş), *iyi* (seslenen `y`) |
| Uzun ünlü kuralları | Ünlü arası `ğ` ve `y` yan ünlüsünde uzunluk düşer; `ğ`+ünsüz / sözcük sonu uzatır |
| Vurgu | Sözlük öncelikli karma: düzensiz kökler, yer adları, vurgusuz ekler (`-DI`, `-mIş`, `-ken`, `-(y)lA`...), sonra model |
| Sınırlar | Sözcük sonrası duraklama düzeyi (`0` / ara öbek / ezgi öbeği) **süre tahmincisine** öznitelik olarak verilir |

Vurgu işaretleri fonem dizisine girer (`ˈ`); sınır bilgisi ise yalnızca süre tahmincisine gider — ayrı bir sınır belirteci (`|`) eklemek
hizalamayı bozdu, sözcük ayracını atmak konuşmayı anlaşılmaz yaptı (ayrıntılar kaynak depoda, `EXPERIMENT_LOG`).

## Kullanım

```bash
pip install "matcha-tts==0.0.7.2" soundfile "transformers>=4.44,<5" "huggingface_hub>=0.28,<1"
pip install "git+https://github.com/iatagun/lemma-rule-based@dizgetts-antalia-v0#subdirectory=dizgetts"
```

```python
import soundfile as sf
from dizgetts.eval.synth import Synth

tts = Synth.from_hub("iatagun/DizgeTTS-Antalia", device="cpu")   # "cuda" da olur
wav, norm, phonemes, sec = tts("Yarın sabah saat dokuzda İzmir'e gidiyoruz.")
sf.write("cikti.wav", wav, 22050)
print(phonemes)   # modelin gerçekte okuduğu fonem dizisi (vurgu ˈ ile)
```

- `tts(metin, steps=10, temperature=0.667, length_scale=None)`: `steps` akış eşleme adımı (10 yeterli), `length_scale` > 1 yavaşlatır.
- CPU'da bir cümle birkaç saniye; ilk çağrıda DizgeBERT-G2PTTS (~440 MB) indirilir.
- Metin cümle cümle işlenir; çok uzun paragrafı cümlelere bölüp sırayla okutun.

## Değerlendirme

Ölçümler, eğitimde ve model/ayar seçiminde **kullanılmayan** 84 Antalia test klibi + 400 UD Türkçe cümlesi (484 cümle) üzerinde.
Anlaşılırlık otomatik: sentezi Whisper-small ile yazıya döküp karakter/sözcük hata oranı (CER/WER). Doğallık: UTMOS (otomatik MOS tahmini).
Köşeli parantezler 2.000 tekrarlı bootstrap %95 güven aralığıdır.

| Sistem | CER % | WER % | UTMOS |
|---|---|---|---|
| Gerçek kayıt (yalnız 84 test klibi; Whisper tabanı) | 3,1 | 9,9 | — |
| v4-e400 (önceki en iyi) | 2,8 [2,6–3,2] | 13,0 [11,8–14,2] | 3,05 |
| **Bu model (v6, ep150)** | **2,7** [2,4–3,0] | **12,6** [11,4–13,8] | **3,10** |

Eşleşmiş fark (v6 − v4): CER −0,13 [−0,37, +0,11] ve WER −0,40 [−1,29, +0,48] anlamsız; UTMOS **+0,04 [+0,02, +0,07] anlamlı**.
Yayın kararı önceden yazılmış kuralla verildi: CER farkının üst sınırı +0,5 puanın altında (sağlandı: +0,11). Bu sürümün asıl
kazancı söyleyişte (aşağıda); Whisper bu farkları büyük ölçüde görmez.

Ön uçtaki değişikliğin sese yansıyan örnekleri (v4 → bu sürüm): *kağıdı* `kʰɑːɨdˈɨ` → `cʰaɨdˈɨ`, *hakim* `xɑːcˈIm` → `xaːcˈIm`,
*değeri* `dejɛɾˈI` → `dɛːɾˈI`, *vereceğim* `veɾɛdʒˈɛjIm` → `veɾɛdʒˈɛIm`, *iyi* `iːˈI` → `ijˈI`.

**Ölçütün sınırı:** Whisper-small Türkçe'de kusursuz değil — **gerçek insan kayıtlarında** bile Antalia testinde CER %3,1. Yani ~%3'lük sentez
CER'i "Whisper'ın insan sesinde yaptığı kadar hata" düzeyidir, "insan kadar iyi" anlamına gelmez. Vurgu hatalarını CER hemen hiç görmez;
vurgu doğruluğu ön uçta ayrıca ölçüldü: kör etiketlenmiş 97 sözcükte %92,8 (bkz. DizgeBERT-G2PTTS kartı).

## Eğitim

- **Veri:** [Antalia](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus) (Patientdesk.ai, CC-BY-4.0), tek konuşmacı, okuma;
  910 eğitim / 59 doğrulama / 84 test klibi (kaynak kayda göre bölünmüş), 4,2 saat eğitim.
- **Başlangıç:** LJSpeech üzerinde eğitilmiş Matcha-TTS → 400 epoch Antalia (v4) → yeni ön uçla 150 epoch daha (bu sürüm; kontrol noktası doğrulama kümesinde CER + UTMOS ile seçildi).
- **Mimari:** Matcha-TTS (18,2 M parametre; RoPE kodlayıcı, U-Net akış eşleme çözücüsü), süre tahmincisine sözcük sınırı özniteliği eklendi.
  Eğitimde hiç görülmeyen ön `a` / `aː` sesleri (*kağıt*, *hakim*) için gömme, `ɑ` ile `ɛ`'nin ortalamasına bağlıdır.
- **Vokoder:** HiFi-GAN universal v1 ([jik876/hifi-gan](https://github.com/jik876/hifi-gan), MIT), değiştirilmeden.
- **Donanım:** tek GTX 1650 (4 GB), fp32 (fp16'da cuDNN NaN), etkin batch 16, Adam 1e-4.

## Sınırlar

- **Tek konuşmacı, tek üslup** (haber/metin okuma). Soru ezgisi, heyecan, diyalog iyi değil.
- **Ritim hâlâ fazla düzenli** (nPVI ~27, gerçek ~45): süre tahmincisi uzun/kısa ayrımını daraltıyor. Duraklamalar gerçekten kısa.
- 4,2 saat veri: nadir ses dizileri, yabancı adlar ve ödünçlemeler yanlış okunabilir. `phonemes` çıktısına bakarak sorunun ön uçta mı
  (yanlış fonem/vurgu) yoksa seste mi olduğunu ayırabilirsiniz.
- Normalleştirici her biçimi bilmez (ör. karmaşık birimler, formüller); bilinmeyen karakterler uyarı verip atlanır.

## Sorumlu kullanım

Bu ses, izinli ve açık lisanslı (CC-BY-4.0) bir derlemin gerçek okuyucusuna aittir. Onu **taklit etmek, adına konuşturmak ya da
yanıltıcı içerik üretmek için kullanmayın.** Sentetik ses kullandığınızı dinleyiciye belirtin.

## Lisans ve atıf

Model ağırlıkları **CC-BY-4.0** (Antalia derleminin lisansı). Kullanırken Antalia derlemini (Patientdesk.ai) ve bu modeli anın.
Matcha-TTS ve HiFi-GAN MIT lisanslıdır. Ön uç: `dizge` (MIT), DizgeBERT-G2PTTS (kendi kartındaki lisans).

Kaynak kod: [github.com/iatagun/lemma-rule-based](https://github.com/iatagun/lemma-rule-based/tree/dizgetts-antalia-v0/dizgetts)
