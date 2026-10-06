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
- cloud0day3/alania-synthetic-speech-tr
---

# DizgeTTS-Antalia (deneysel, v2)

Türkçe metinden konuşma. **Bu model tek başına çalışmaz: metni [`iatagun/DizgeBERT-G2PTTS`](https://huggingface.co/iatagun/DizgeBERT-G2PTTS) okur,
bu model seslendirir.** G2PTTS her sözcük için sesbirimleri, vurgulu ünlüyü ve sözcükten sonraki duraklama düzeyini üretir;
[Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) akustik modeli bu girdiyle eğitildi ve aynı girdiyi bekler. espeak kullanılmaz.

```
metin  ->  DizgeBERT-G2PTTS (normalleştirme, sesbirim, vurgu, duraklama)  ->  DizgeTTS-Antalia (Matcha-TTS)  ->  HiFi-GAN  ->  ses
```

Tek konuşmacı (Antalia derleminin okuyucusu), 22,05 kHz.

> **Deneysel araştırma sürümüdür.** 4,2 saatlik tek bir okuma derleminden (v2'de öncesinde 20 saat sentetik Türkçe konuşmayla ön eğitim), 4 GB'lık bir
> dizüstü GPU'sunda eğitildi. Ticari kalitede bir ses değildir.
> Sınırları aşağıda açıkça yazılı. Ürettiği ses **sentetiktir**; gerçek kayıt gibi sunulamaz (bkz. Sorumlu kullanım).

## Dinleyin

Aşağıdaki cümleler eğitim verisinde **yok** (seçilmeden, ilk üretim; tohum 0):

1. *Merhaba, ben Dizge. Türkçe metinleri sesli okuyabilirim.*

   <audio controls src="https://huggingface.co/iatagun/DizgeTTS-Antalia/resolve/main/samples/00.wav"></audio>

2. *Kağıt, kalem ve silgiyi masanın üstüne bıraktım; akşam yeniden bakacağım.*

   <audio controls src="https://huggingface.co/iatagun/DizgeTTS-Antalia/resolve/main/samples/01.wav"></audio>

3. *Yarın sabah saat dokuzda İzmir'e gidiyoruz, değil mi?*

   <audio controls src="https://huggingface.co/iatagun/DizgeTTS-Antalia/resolve/main/samples/02.wav"></audio>

4. *Bu sistem, sözcüklerin vurgusunu ve cümle içindeki duraklamaları kendisi tahmin ediyor.*

   <audio controls src="https://huggingface.co/iatagun/DizgeTTS-Antalia/resolve/main/samples/03.wav"></audio>

Tarayıcıda denemek için: [iatagun/dizge-demo](https://huggingface.co/spaces/iatagun/dizge-demo) (TTS sekmesi).

## Ön uç: DizgeBERT-G2PTTS

Türkçe TTS'te sorunların çoğu akustik modelde değil, **metnin nasıl okunacağında**. Bu modelin girdisini G2PTTS üretir (eğitimde de aynı ön uç kullanıldı):

| Aşama | Ne yapar |
|---|---|
| Normalleştirme | Sayı, tarih, saat, para, kısaltma, kodları açar (`15:30` → *on beş otuz*, `3,5 TL` → *üç lira elli kuruş*) |
| Sesbirim | Kural tabanlı `dizge==0.1.6` + söyleyiş sözlüğü (alıntılar, şapka) + uzun ünlü (ğ / y) kuralları. Kör testte %96,7 (120 sözcük) |
| Vurgu | Sözlük öncelikli karma: düzensiz kökler, yer adları, clitic'ler kuraldan; gerisi ELECTRA başlığından. Kör testte %92,8 (97 sözcük) |
| Duraklama | Sözcük sonrası sınır düzeyi (`0` / ara öbek / ezgi öbeği), ELECTRA başlığı + noktalama. Ölçülen duraklamaya göre F1 0,50 |

Vurgu işareti sesbirim dizisine girer (`ˈ`, vurgulu ünlünün önünde). Duraklama düzeyi ise yalnızca **süre tahmincisine** öznitelik olarak verilir:
ayrı bir sınır belirteci (`|`) eklemek hizalamayı bozdu, sözcük ayracını atmak konuşmayı anlaşılmaz yaptı (ayrıntılar kaynak depoda, `experiments.yaml`).

**Ön uç sürümü önemlidir.** Model, eğitildiği ön ucun ürettiği dizileri tanır. Paket, eğitim girdisinden 30 örnek cümle taşır; `Synth` yüklenirken
güncel ön ucun çıktısını bunlarla (sesbirim dizisi ve duraklama özniteliği) karşılaştırır, sistematik fark varsa çalışmayı reddeder.

## Kullanım

`matcha-tts` paketinin bağımlılık listesi eski sürümlere sabitlidir; bağımlılıksız kurulur (C derleyicisi gerekir):

```bash
pip install torch numpy Cython setuptools
pip install matcha-tts==0.0.7.2 --no-deps --no-build-isolation
pip install soundfile matplotlib hydra-core lightning gdown wget librosa tensorboard conformer diffusers einops transformers huggingface_hub safetensors
pip install "git+https://github.com/iatagun/lemma-rule-based@4bf562f8781c77e6e6cef94f52d646a1dc18f932#subdirectory=dizgetts"
```

```python
import soundfile as sf
from dizgetts.eval.synth import Synth

tts = Synth.from_hub("iatagun/DizgeTTS-Antalia", device="cpu")   # DizgeBERT-G2PTTS'i de indirir (~440 MB); "cuda" da olur
wav, norm, phonemes, sec = tts("Yarın sabah saat dokuzda İzmir'e gidiyoruz.")
sf.write("cikti.wav", wav, 22050)
print(phonemes)   # G2PTTS'in ürettiği, modelin gerçekte okuduğu sesbirim dizisi (vurgu ˈ ile)
```

- `tts(metin, steps=10, temperature=0.667, length_scale=None)`: `steps` akış eşleme adımı (10 yeterli), `length_scale` > 1 yavaşlatır.
- CPU'da bir cümle birkaç saniye.
- Metin cümle cümle işlenir; çok uzun paragrafı cümlelere bölüp sırayla okutun.
- Kurulum Windows'ta Python 3.12, transformers 4.57 ve 5.x ile denendi.

## Değerlendirme

Ölçümler, eğitimde ve model/ayar seçiminde **kullanılmayan** 84 Antalia test klibi + 400 UD Türkçe cümlesi (484 cümle) üzerinde.
Anlaşılırlık otomatik: sentezi Whisper-small ile yazıya döküp karakter/sözcük hata oranı (CER/WER). Doğallık: UTMOS (otomatik MOS tahmini;
İngilizce ağırlıklı eğitildiği için mutlak değeri Türkçede güvenilir değildir, yalnız sistemler arası fark anlamlıdır).
Köşeli parantezler bootstrap %95 güven aralığıdır.

| Sistem | CER % | WER % | UTMOS |
|---|---|---|---|
| Gerçek kayıt (yalnız 84 test klibi; Whisper tabanı) | 3,1 | 9,9 | — |
| Önceki sürüm (v1) | 3,1 [2,8–3,4] | 14,1 [12,9–15,3] | 3,16 [3,14–3,19] |
| **Bu model (v2)** | **2,6** [2,4–2,9] | **12,3** [11,3–13,4] | **3,09** [3,07–3,12] |

v2, v1'e göre **daha anlaşılır ama otomatik doğallık puanı biraz daha düşük**; eşleşmiş farkların üçü de anlamlı: CER −0,50 [−0,74, −0,24],
WER −1,77 [−2,79, −0,81], UTMOS −0,07 [−0,09, −0,05]. Süre tahmininin gerçeğe uyumu değişmedi (test, log-süre korelasyonu 0,62 / 0,62).
**v2 dinleme testinden geçmedi:** UTMOS düşüşünün duyulup duyulmadığı ölçülmedi; değişiklik yalnızca otomatik ölçütlere dayanıyor. İki şey birlikte
değişti (ön eğitim verisi ve 60 ek epoch), kazancın hangisinden geldiği ayrıştırılmadı. v1'i yeğlerseniz:
`Synth.from_hub("iatagun/DizgeTTS-Antalia", revision="8c717a33e769da8529b77c92c8a9427337597f10")`.

**Dinleme testleri** (önceki sürümlerde; kör, 30 çift, tek dinleyici): süre tahmincisinin yeniden eğitilmesi (aynı tarif, bir önceki akustik modelde) 19–0 tercih edildi (11 fark yok);
bu, projedeki en büyük duyulabilir kazançtır. G2PTTS v1 sesbirimlerine geçiş (v8) ise 12 / 8 / 10 ile **kararsız** kaldı:
okunuş iyileşmesinin sese geçtiği bu veriyle gösterilemedi.

**Ölçütün sınırı:** Whisper-small Türkçede kusursuz değil; gerçek insan kayıtlarında bile CER %3,1. Yani ~%3'lük sentez CER'i
"Whisper'ın insan sesinde yaptığı kadar hata" düzeyidir, "insan kadar iyi" anlamına gelmez. WER, gerçekçi zamanlamayı cezalandırır:
gerçek sürelerle yapılan sentez WER'de daha kötü çıktığı halde kör dinlemede 29–1 tercih edildi.

## Eğitim

- **Veri:** [Antalia](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus) (Patientdesk.ai, CC-BY-4.0), tek konuşmacı, okuma;
  910 eğitim / 59 doğrulama / 84 test klibi (kaynak kayda göre bölünmüş), 4,2 saat eğitim.
- **Ön eğitim verisi (v2):** [Alania Turkish Synthetic Speech](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr) (Patientdesk.ai),
  `cc-by` bölümü (CC BY 4.0); **tamamı yapay zekâ üretimi** ses. 20 saat / 13.105 klip: kadın sesler, düz okuma, 2–15 sn, derlemin kendi kalite
  ölçütleriyle süzülmüş. Konuşmacı koşullaması yok (sesler havuzlandı); çıkan ses yine Antalia okuyucusunun sesi.
- **Girdi:** her klip DizgeBERT-G2PTTS ön ucundan geçirildi (sesbirim + vurgu işareti; duraklama düzeyi süre tahmincisine).
- **Zincir:** LJSpeech üzerinde eğitilmiş Matcha-TTS → 400 epoch Antalia → söyleyiş sözlüğü ve uzun ünlü kurallarıyla 150 epoch →
  G2PTTS v1 sesbirimleriyle 150 epoch (v1) → 60 epoch Alania ön eğitimi → 150 epoch Antalia (bu sürüm). Ardından akustik model dondurulup yalnızca süre tahmincisi doğrusal kare hatasıyla yeniden eğitildi;
  sentezde süreler birikimli yuvarlanır.
- **Mimari:** Matcha-TTS (18,2 M parametre; RoPE kodlayıcı, U-Net akış eşleme çözücüsü), süre tahmincisine sözcük sınırı özniteliği eklendi.
- **Vokoder:** HiFi-GAN universal v1 ([jik876/hifi-gan](https://github.com/jik876/hifi-gan), MIT), değiştirilmeden.
- **Donanım:** tek GTX 1650 (4 GB), fp32, etkin batch 16, Adam 1e-4.

## Sınırlar

- **Tek konuşmacı, tek üslup** (metin okuma). Heyecan, diyalog, güçlü soru ezgisi iyi değil.
- **Ritim fazla düzenli:** tahmin edilen sürelerin çeşitliliği gerçeğin yaklaşık %70'i (test, std oranı 0,71). Uzun/kısa ayrımı daralıyor.
- **ğ kaynaklı uzun ünlüler kısa:** model uzun ünlüyü gerçeğin yaklaşık %85'i kadar uzatıyor (önceki sürümlerde ölçüldü; v2'de yeniden ölçülmedi).
- 4,2 saat veri: nadir ses dizileri, yabancı adlar ve alıntılar yanlış okunabilir. `phonemes` çıktısına bakarak sorunun ön uçta mı
  (yanlış sesbirim/vurgu) yoksa seste mi olduğunu ayırabilirsiniz.
- Normalleştirici her biçimi bilmez (ör. karmaşık birimler, formüller); bilinmeyen karakterler uyarı verip atlanır.

## Sorumlu kullanım

Bu ses, açık lisanslı (CC-BY-4.0) bir derlemin gerçek okuyucusuna, profesyonel bir seslendirme sanatçısına aittir. Sanatçı kayıtların yeniden
dağıtımına şu anlayışla izin verdi; bu modeli kullanırken aynı koşullara uyun:

- Konuşmacıyı ya da gerçek bir kişiyi **taklit etmek**, birine söylemediği sözü söyletmek için kullanmayın.
- Sentetik sesi **gerçek insan kaydı gibi sunmayın**; dinleyicinin yanılabileceği yerde sentetik olduğunu belirtin.
- Dolandırıcılık, sosyal mühendislik ve siyasi otomatik aramalarda kullanmayın.

## Lisans ve atıf

Model ağırlıkları **CC-BY-4.0** (Antalia derleminin lisansı). Kullanırken Antalia derlemini (Patientdesk.ai), ön eğitimde kullanılan
Alania Turkish Synthetic Speech derlemini (Patientdesk.ai, CC BY 4.0) ve bu modeli anın.
Matcha-TTS ve HiFi-GAN MIT lisanslıdır. Ön uç: `dizge` (MIT), DizgeBERT-G2PTTS (CC BY-SA 4.0, kendi kartına bakın).

Kaynak kod: [github.com/iatagun/lemma-rule-based](https://github.com/iatagun/lemma-rule-based/tree/dizgetts-v2/dizgetts)
