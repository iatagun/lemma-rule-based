# DizgeTTS — Aşama 1 raporu (2026-09-23)

Sayılar `scripts/audit_dataset.py`, `scripts/g2p_probe.py`, `scripts/smoke_vram.py` çıktılarından (`reports/*.json`). Eğitim başlatılmadı; commit yapılmadı.

## 1. Lisans ve veri
| | Sonuç |
|---|---|
| **Commencis** | HF API'de `Repository not found` (anonim ve token'lı). Doğru id verilmedi → **kapsam dışı**, denetlenmedi. |
| **antalia** (`cloud0day3/antalia-voice-corpus`) | **CC-BY-4.0**, ticari kullanıma açık, gated değil. Atıf: "Antalia (Patientdesk.ai)" + repo bağlantısı. Konuşmacı rızası kısıtları (taklit/impersonation, sentetiği gerçek kayıt diye sunma, dolandırıcılık/siyasi robocall yok) model kartına yazılmalı. Yalnız train split var (test saklı → kendi split'imizi ayıracağız). |
| İndirme | 1.073 WAV + metadata.jsonl, 830 MB → `D:\dizgetts\data\raw\antalia` |

**Sinyal (WAV'dan yeniden hesaplandı):** 5,008 saat, 1 konuşmacı, 24 kHz mono 16-bit (hepsi), clipping yok, tepe ≤ 0,891. Yüzdelik-tabanlı SNR medyan 35,1 dB (min 28,5); metadata'daki SNR ile korelasyon 0,993 → doğrulandı. Gürültü tabanı medyan −54,2 dBFS, RMS medyan −23,9 dBFS. Baş sessizlik medyan 0,08 sn, son sessizlik medyan 0,11 sn (max 1,45) → kırpma gereği düşük.

**Uzunluk — planla çakışıyor:** medyan **17,6 sn**, p5 10,6 sn, max 20,2 sn. 1–10 sn aralığında yalnız **41 klip (0,084 saat)**; 1.032 klip > 10 sn. Aşama 2'deki "1–10 sn dışını ele" kuralı uygulanırsa neredeyse veri kalmıyor. Seçenekler: (a) elemeyi kaldır, tam klip + `out_size` segment eğitimi (bkz. §4); (b) sessizlik/duraklama noktalarından böl (çerçeve enerjisi + noktalama; kelime zaman damgası yok).

**Metin:** 33.636 sözcük, 7.240 tür. 695 klipte `transcript ≠ normalized_transcript`, 275 klipte rakam var (normalizer şart; yayımlanan `normalized_transcript` normalizer'ımız için referans olabilir). Tekrar eden transkript: 39 (aynı script farklı parent kayıtlardan; split'te sızıntı riski → parent id'ye göre ayır, 65 parent kayıt var). Noktalama: `. , ; : ? ! " ' - – % /` + 860 satır sonu.

**Transkript–ses uyumu:** transkript vs Scribe ASR (metadata'daki) CER medyan 0, p95 0,169, max 0,30 (yayımcı ≤0,3 kapısı). CER>0,10 olan 153 klip; CER'i en yüksek 6 klip elle incelendi (40'lık şüpheli listesinin başı); neden **gerçek uyumsuzluk değil, yazım farkı** ("8 gigabayt" ↔ "8 GB", "yüzde 53" ↔ "%53", "www nokta" ↔ "www.", Scribe tekrar halüsinasyonu). Bağımsız Whisper doğrulaması **yapılmadı** (Aşama 6'ya ertelendi).

## 2. dizge-g2p etiket seti
- **Wrapper kaynağı:** `D:\playground\bert_turkish_finetune.py` (`DizgeBERT.g2p`). Model **karakter düzeyinde**: sözcük karakterlere bölünür, her karaktere **bir** etiket (1–2 fonemlik dizge), etiketler birleştirilir. Etiket sınırı fonem sınırı değil (`bI`, `Iɾ`, `ːt`… DP hizalama artıkları).
- 87 etiketin **hepsi** `frontend/symbols.py` atomlarına ayrışıyor (2 yalnız-değiştirici etiket `ʰ`, `̥` önceki atoma yapışıyor). Atomlar: 65 (`symbols.PHONES`), toplam sembol 73 (PAD, boşluk, 5 duraklama, vurgu ˈ rezerv).
- **`I`, `U`, `Y` anlamı** (dizge `tools/phonology.py`'den doğrulandı): normal `i/u/ü` = gevşek ünlüler (ɪ/ʊ/ʏ benzeri, ASCII kısayol); küçük harf `i/u/y` = `ğ` öncesi / sözcük sonu (gergin). Yani standart dışı sembol değil, dizge'nin kendi sistemi. `ɣ` = **sözcük sonu r** (baş r→r, son r→ɣ, diğer→ɾ); dilbilimsel olarak tartışmalı bir seçim, senin kararın.
- `dizge.g2p` 48.715 sözcükte bilinmeyen sembol üretmiyor; 715 sözcükte çok-varyantlı (tuple) çıktı veriyor (ilki alındı). Kullanılmayan atomlar: `o, u, y, ø, Yː, ɔː, ː` (bare hali yok, hep uzunlukla).
- **Ciddi bulgu:** dizge-g2p (BERT) öğretmen `dizge.g2p` ile: eğitim listesinde %98,9 uyuşuyor (%99,1 iddiasıyla tutarlı), ama antalia sözcük dağarcığında (7.240 tür) **%95,7, eğitim listesi dışındaki 5.430 türde %94,6**. Kırılma noktası **`ğ`**: `ğ` içeren 632 türde **%69,0**, içermeyenlerde 6.580 türde %98,3. Örnekler: `aldığım` → model `ʰłdʒːoːm` (doğrusu `ɑłdɨːm`), `aradığında` → `ʰɾddʒːoːndɑ`. `-dığ-/-lığ-` ekleri Türkçede çok sık → bu G2P olduğu gibi eğitim verisine girerse telaffuz hedefimizi doğrudan bozar. Ayrıca `ala-` köklerinde sistematik `ɑłaːdʒ…` (uzun a) hatası.
- Wrapper `str.lower()` kullanıyor: `İstanbul` → bozuk hizalama, `IŞIK` → `işik`. `frontend/g2p.py` Türkçe-duyarlı `tr_lower` kullanıyor.
- **Öneri (karar sende):** birincil fonemleştirici = **kural tabanlı `dizge.g2p`** (deterministik, 7.240 türden 28'inde hata veriyor → onlar için BERT yedek/sözlük); BERT dizge-g2p'yi ya yalnız OOV yedeği ya da `ğ` verisiyle yeniden eğit. İki model de bağlamsız (sözcük düzeyi, vurgu yok, sözcükler arası ses olayı yok).

## 3. Ortam (D:\dizgetts\venv, Python 3.12.5)
Çalışan: torch 2.8.0+cu128 (CUDA, GTX 1650), Matcha-TTS 0.0.7.2 (`--no-deps`, gradio/notebook vb. atlandı; `monotonic_align` Cython uzantısı pip ile derlendi, `.pyd` oluştu ve smoke testte çalıştı), lightning, hydra, librosa, soundfile, pyloudnorm, diffusers, conformer, phonemizer(python paketi).
**Kurulmadı:** `espeak-ng` (baseline Aşama 4 için gerekli; henüz istenmedi/kurulmadı), `ffmpeg` (WAV'lar zaten PCM16, resample için torchaudio/librosa yeterli görünüyor), Whisper, HiFi-GAN checkpoint'i.
Not: pip önbelleği C:'de (~8 GB); C: 20 GB boş idi.

## 4. VRAM smoke testi (rastgele veri, 20 sn klip = 1740 mel karesi, 600 token; eğitim değil)
- **fp16 autocast + cuDNN AÇIK → NaN.** Kök neden: GTX 1650 (TU117, sm_75) + cuDNN 9.10.02 (torch 2.8+cu128): `Conv1d(768→192)` fp16'da NaN veriyor (Matcha text encoder FFN `conv_2`); cuDNN kapatınca doğru. Bağımsız conv testiyle doğrulandı.
- fp16 + cuDNN kapalı çalışıyor ama **yavaş**: bs=4, out_size=172: 0,69 sn/adım; **fp32 + cuDNN açık 0,27 sn/adım** ve daha güvenilir. Tensor core yok → fp16 kazandırmıyor. **Karar önerisi: fp32, AMP yok.** (Plandaki "mixed precision" varsayımı bu donanımda geçerli değil.)
- Bellek (fp32, tepe): tam 20 sn klip bs=8 → 3,14 GiB (sınırda), bs=16 → 6,1 GiB (4 GiB'ı aşıyor, WDDM paylaşımlı belleğe taşıp 7 sn/adıma çıkıyor). **`out_size=172` (≈2 sn decoder segmenti):** bs=8 → 1,30 GiB, bs=16 → 2,49 GiB, 0,75 sn/adım. Önerilen: **fp32, out_size=172, bs=16** (ya da bs=8 + accumulation).
- Sınırlama: sabit-uzunluk sentetik girdi; gerçek uzunluk dağılımı ve gerçek veride dur/prior kayıplarının davranışı doğrulanmadı. Adım süresi ~1073 klip/epoch için epoch başına ~50 sn (bs16) civarı; 5 saatlik veri Matcha için az, kalite beklentisi düşük tutulmalı (ön-eğitim/transfer seçeneği ayrıca konuşulmalı).

## 5. Açık kararlar / riskler
1. Commencis için doğru id var mı, yoksa yalnız antalia mı (5 saat, tek kadın konuşmacı)?
2. Klip uzunluğu: 1–10 sn kuralı vs tam klip + `out_size` vs bölme.
3. Fonemleştirici: kural tabanlı `dizge.g2p` birincil + BERT yedek (öneri) mi, BERT'i `ğ` için yeniden eğitme mi?
4. `ɣ` (sözcük sonu r) ve `I/U/Y` sistemi mevcut haliyle mi kullanılsın, IPA'ya dönüştürülsün mü?
5. fp32 eğitim (öneri). 5 saatlik veri için ön-eğitimli Matcha checkpoint'ten (İngilizce LJSpeech/VCTK) transfer denensin mi, yoksa sıfırdan mı?
6. `espeak-ng` (baseline) ve Whisper `small` kurulumu için onay.

## Oluşturulan dosyalar
`dizgetts/frontend/{symbols.py,g2p.py}`, `dizgetts/scripts/{g2p_probe.py,audit_dataset.py,smoke_vram.py}`, `dizgetts/reports/{stage1_audit.md,antalia_audit.json,g2p_probe.json}`, `dizgetts/.gitignore`. Ağır her şey `D:\dizgetts` (venv, ham veri, cache).
