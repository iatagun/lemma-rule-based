# DizgeTTS — Aşama 3 raporu: metin ön ucu (2026-09-23)

Sayılar `reports/frontend_report.json`, `reports/normalizer_decisions.json`, `reports/asr_floor.json` dosyalarından. Eğitim yok, commit yok.

## Ne yapıldı
| Bileşen | Dosya | Durum |
|---|---|---|
| Normalizasyon (sayı, kesme ekleri, tarih/saat, yüzde, para, birim, kısaltma, harf harf okunan kısaltma, kod numarası, noktalama→duraklama) | `frontend/normalize.py` | çalışıyor, `tests/test_normalize.py` 12 sayı + 30 vaka |
| Fonemleştirme: `dizge==0.1.6` birincil (senin kararın: A yolu), hata olursa dizge-g2p BERT yedeği | `frontend/phonemize.py` | çalışıyor, `tests/test_phonemize.py` |
| espeak-ng baseline fonemleri (aynı normalize çıktısı üzerinde) | `frontend/espeak.py` | çalışıyor; espeak-ng 1.52.0 |
| Sembol tablosu (vurgu `ˈ` ve 5 duraklama token'ı ayrılmış) | `frontend/symbols.py` | Aşama 1'den |
| Toplu üretim + bilinmeyen sembol raporu | `scripts/build_phonemes.py` → `D:\dizgetts\data\processed\antalia\{train,val,test}_phon.jsonl` | 1.053 klip |
| Whisper hipotez puanlaması (normalizer kararlarını sesle sınamak) | `scripts/whisper_score.py`, `scripts/normalizer_decisions.py` | |
| Gerçek kayıtların ASR tabanı | `eval/asr_floor.py` | |

**Kurulumlar:** `espeak-ng` için choco/winget yok ve admin değilim; `espeakng-loader` paketi DLL+veriyi getiriyor (Windows'ta kurulum gerektirmiyor). Whisper: `openai/whisper-small` HF'den indirildi (transformers ile; fp32 GPU, GTX 1650 fp16 cuDNN NaN'ı nedeniyle). Bağımlılıklar `requirements.txt`'de sabitlendi.

## Sonuçlar
- **1.053 klip / 35.338 sözcük / 7.947 tekil sözcük** fonemleştirildi. `dizge.g2p` **0 hata** verdi; bilinmeyen sembol **0**. Kullanılan dizge atomu 56/65 (kullanılmayanlar `Yː a o u y œː ɔː ɱ ː`). BERT yedeği bu veride hiç devreye girmedi (ilk turda 9 tek-harfli sözcük hata veriyordu; normalizer'daki harf okuma düzeltilince kalmadı).
- **espeak-ng** sembol kümesi 46 karakter (`ˈ ˌ ː ɯ ɫ æ ɪ ʊ …`): baseline'ın sembol tablosu bu.
- **Normalizer, Antalia'nın `normalized_transcript`'iyle:** kanonik tam eşleşme %68,4 (720/1.053), CER %1,33. Fark **kasıtlı**: Antalia'nın referansı otomatik bir normalizer ve aşağıda 5 noktada sesle çelişiyor.
- **Whisper-small, gerçek val+test kayıtlarında (143 klip): CER %4,70, WER %11,55** → Aşama 6'da sentez çıktısı için pratik taban; TTS'in bunu geçmesi beklenmemeli.

## Normalizer kararları ve kanıtı (Whisper hipotez puanlaması: aynı sese farklı okunuşların NLL'i)
| Karar | Sonuç | Klip |
|---|---|---|
| Kod `PPV-3254-K` | **ikişer okunur** ("otuz iki elli dört"); Antalia kardinal ("üç bin iki yüz…") yazıyor | 64/65 |
| `yüzde 47'i`, `79'i` gibi yanlış yazılmış ekler | **okunuşa göre uyumlandırılır** (yedisi, dokuzu); kaynak metindeki ek yanlış | 58/58 |
| `389,50 TL` | "üç yüz seksen dokuz **lira elli kuruş**" (Antalia: "virgül elli türk lirası") | 12/16 |
| `SMS` | "es em es" (Antalia: "se me se") | 38/47 |
| `www` | "**vi vi vi**" ("ve ve ve"'ye karşı) | 13/13 |
| `PIN` | "pin" | 6/6 |
| `Wi-Fi` | kararsız: "vayfay" 6, "vifi" 8 → "vayfay" bırakıldı | 14 |
| 7+ haneli numara (`3006751942`) | ikişer | **1** klip, zayıf kanıt |

Yöntem sınırı: Whisper'ın dil modeli önyargısı her iki adayda da var, yalnız aynı biçimli adaylar kıyaslandı; `www` ve `Wi-Fi` gibi fonetik farkı küçük olanlar için **kulakla doğrulama** iyi olur.

## Normalizer'ın yakaladığı gerçek hatalar (ilk sürüm rakamları sessizce siliyordu)
`55x40x23` ve `VKT7M2` gibi harf-rakam bitişik dizilerde rakamlar **kayboluyordu**; şimdi ayrılıyor (`çarpı`, harf harf), ve kimsenin yakalamadığı rakam artık silinmek yerine rakam rakam okunuyor. Tek başına büyük harf (`A girişi`, `'H' yazın`), `A.Ş.`, `B'yi` de çözülüyor.

## Açık sorunlar (hepsi sende ya da sonraki adımda)
1. **`dizge.g2p` çok-varyantlı 195 sözcük.** Birincisini alıyorum. Aynı `ğ` sözcüğü iki farklı kural yolu: `değil` → `dejIl` / `dɛːIl`, `daha` → `dɑxɑ` / `dɑː`, `rahatlığı` → `rɑxɑtłɨː` / `rɑːtłɨː`. Hangisinin konuşmacının söylediği bilinmiyor; sen paketi düzeltirken burayı kararlaştır (liste `frontend_report.json: multi_variant_words`). İstersen Whisper puanlamasına benzer bir yöntemle varyantları sesten seçebilirim (fonem düzeyinde değil, yalnız yazım düzeyinde ayrım yapabilir, sınırlı).
2. Son `r` → `ɣ` kuralı hâlâ dizge'de; paketi düzelttiğinde fonemler yeniden üretilecek (`build_phonemes.py` ~1 dk).
3. Normalizer bağlamsız: `3.` (sıra sayısı) yalnız arkasından küçük harfli sözcük gelirse çözülüyor; eş yazımlı sözcükler ayrılmıyor.
4. **Vurgu ve duraklama modelleri henüz kullanılmadı.** Girdi formatı hazır (`ˈ` ve 5 duraklama token'ı `symbols.py`'de, `tokenize` bunları geçiriyor). `iatagun` modellerini (Morph/Joint/Dep/syllable) vurgu ve öbek sınırı için bağlamak Aşama 3'ün kapsamı olarak konuşulmuştu ama bunu yapmadım; önce her birinin doğruluğunu ölçmek istiyorum. Şu an duraklama yalnızca noktalamadan geliyor.
5. BERT dizge-g2p'nin `ğ` zayıflığı hâlâ geçerli; yedek olarak yalnız `dizge` hata verirse devreye giriyor (bu veride 0 kez).
6. Test setinde kaydedilen cümleler normalizer'ın geliştirildiği veriyle aynı kaynaktan; genelleme için senin telaffuz test setin (Aşama 6) ayrı ölçüm olacak.
