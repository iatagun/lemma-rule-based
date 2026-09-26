---
language: tr
license: cc-by-sa-4.0
library_name: transformers
pipeline_tag: token-classification
tags:
- token-classification
- text-to-speech
- word-stress
- prosodic-boundary
- turkish
- electra
- experimental
base_model: iatagun/DizgeBERT-Dep
datasets:
- universal_dependencies
---

# DizgeBERT-G2PTTS (deneysel, v0)

Türkçe TTS ön ucu için, ham metinden **sözcük başına** iki şey tahmin eden küçük bir etiketleyici:

- **Vurgu:** vurgulu seslem (sondan sıra) ya da vurgusuz.
- **Sınır:** sözcükten sonra duraklama var mı, hangi düzeyde (`0` / `ip` / `IP`, cümle sonu `cümle`).

> **Deneysel araştırma sürümüdür.** Üretim için tasarlanmadı, sınırları aşağıda açıkça yazılı. Beklenmedik çıktılar için lütfen sözlükleri (`resources/`) düzeltip geri bildirim verin.

## Bu bir HİBRİT hattır — yalnız ağırlıklar yetmez

| Parça | Ne yapar | Nerede |
|---|---|---|
| **Kural modülü** | Kök sözlüğü (düzensiz vurgulu kökler + ağırlık kuralı), clitic'ler, ünlüsüz sözcükler, sıfat sözlüğüne kapılı pekiştirme (`kıpkırmızı`) ve `-CIk` sıfat | `stress_rules.py` + `resources/*.tsv\|txt` |
| **ELECTRA başlıkları** | Kural "varsayılan son seslem" derse: damıtılmış biçimbilim (koşaç, kişi eki, olumsuzluk, `-dIr`...) ve **tüm sınır tahmini** | `model.safetensors` |

Vurgu kararı **sözlük öncelikli**: kural belirleyici bir sonuç veriyorsa o; değilse model. Model başlığı **tek başına** vurgu için yetersizdir (aşağıda). Bu yüzden `AutoModel` ile yüklerken `resources/` klasörü de indirilir ve kullanılır. Sözlükler düz metindir, düzeltme önerilerini bekleriz.

## Kullanım

```python
from transformers import AutoModel

m = AutoModel.from_pretrained("iatagun/DizgeBERT-G2PTTS", trust_remote_code=True).eval()
out = m.tag("Yarın İstanbul'a gideceğim, ama havanın nasıl olacağını bilmiyorum.")
for w in out["words"]:
    print(w["word"], w["stress_from_end"], w["stress_src"], w["boundary"], w["p_break"])
```

- `stress_from_end`: vurgulu seslemin sondan sırası (`0` = son seslem), `None` = vurgusuz.
- `stress_src`: `kural:<etiket>` (kök, clitic, ünlüsüz, pekiştirme...) ya da `model`.
- `boundary`: `0` sınır yok, `ip` orta, `IP` büyük, `cümle` cümlenin son sözcüğü. Noktalama ile birleştirilir (`,` → en az `ip`, `;` → en az `IP`).
- Girdi `normalize()` ile hazırlanır (sayı, kısaltma, tarih açılır; noktalama ayrılır). Uzun metin cümle sınırlarında 254 alt-sözcüklük parçalara bölünür.
- Fonem üretmez. Fonemler için [`dizge`](https://pypi.org/project/dizge/) (`==0.1.6`) kullanılır.

## Eğitim

- **Gövde:** [`iatagun/DizgeBERT-Dep`](https://huggingface.co/iatagun/DizgeBERT-Dep) (sabit revizyon; ELECTRA `dbmdz/electra-base-turkish-cased-discriminator` temelli), gömme + alt 6 katman dondurulmuş, iki doğrusal başlık.
- **Vurgu etiketleri:** UD Türkçe treebank'lerinde (BOUN, IMST, Kenet) ve Antalia cümlelerinde kural modülünden **damıtıldı** (UD'de altın UPOS/FEATS, Antalia'da DizgeBERT-Morph çıktısı). İnsan etiketi değildir.
- **Sınır etiketleri:** yalnız [Antalia](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus) (CC-BY-4.0, tek konuşmacı, ~4,2 saat okuma) kayıtlarında sözcükler arası **ölçülen sessizlik**: `< 60 ms` → 0, `60–250 ms` → ip, `≥ 250 ms` → IP.
- 4 epoch, öğrenme oranı 3e-5 (başlıklar 1e-3), Antalia x2 tekrarlı. Epoch seçimi: sınır F1'i en iyiye 0,02 yakın epoch'lar içinden UD son-dışı vurgu uyumu en yüksek olan (epoch 2). Eşik `tau=0,25` doğrulama kümesinden.

## Sonuçlar (dürüst okuma)

### Vurgu — KÖR test (97 sözcük, kural yazılırken bakılmadı)

Rastgele seçilmiş sözcükler, **tek başına** (cümle bağlamı yok), **tek etiketleyici** (proje sahibi). Hibrit hat:

| Sistem | Doğruluk |
|---|---|
| **DizgeBERT-G2PTTS (hibrit)** | **90/97 = %92,8** (%95 GA: %87,6–97,9) |
| Her zaman son seslem | 81/97 = %83,5 |
| espeak-ng (`tr`) | 71/97 = %73,2 |
| Kural + DizgeBERT-Morph hattı (M1b) | %94,8 (fark anlamsız; bu model o hattı Morph'suz, tek modelde damıtır) |

- Geliştirme setinde (kuralların yazıldığı 250 rastgele sözcük) %86,0. İlk 35'lik gold listesinde 33/35, ama bu liste kuralların yazıldığı örnekleri içerir; **bağımsız ölçüm değildir**.
- **Model başlığı tek başına** (kuralsız, ilk 35'lik gold): **18/35**. Kök sözlüğü ve istisnaları öğrenememiştir. Vurgunun büyük kısmını kural modülü ve sözlükler taşır.
- Model başlığının kural etiketleriyle uyumu (test): UD %96,6 (son-dışı %89,4), Antalia %98,3 (son-dışı %95,3).

### Sınır (test: 82 Antalia klibi, 2276 sözcük sınırı; noktalamasız sınırlar)

| Sistem | F1 | P | R |
|---|---|---|---|
| **DizgeBERT-G2PTTS** (`tau=0,25`) | **0,496** | 0,414 | 0,620 |
| Bağımlılık öbeği kuralı (DizgeBERT-Dep, K=2) | 0,347 | | |

Fark +0,096…+0,201 (klip bootstrap %95 GA). Baseline'ı anlamlı geçiyor, ama mutlak değer **mütevazı**: önerilen sınırların yarısından fazlası ölçülen duraklamayla örtüşmüyor.

## Sınırlar ve bilinen zaaflar

- **Küçük ve tek etiketleyicili değerlendirme.** Vurgu kör testi 97 sözcük, tek kişi; ikinci etiketleyici uyumu ölçülmedi. Bağımsız bir değerlendirme yok.
- **Sınır tek konuşmacıdan.** Antalia okuma sesi; spontan konuşma, haber okuma gibi tarzlarda davranış bilinmiyor. Etiket "≥60 ms sessizlik"tir, dilbilimsel öbek sınırı değildir.
- **Vurgu tek başına sözcük üzerinde ölçüldü**; cümle içi vurgusuzlaşma (işlev sözcükleri vb.) modellenmiyor.
- Kural kapsamı bilinçli dardır: seslenme, küçültme, ikileme, bileşikler için tam sözlük yok.
- Yer adı ayrımı büyük harfe bağlı (`Ordu` / `ordu`).
- Sıfat kapılı kurallar sözlüğe bağlı (UD ADJ lemmalarından türetildi); sözlük dışı sıfatlar kaçar.

## Lisans ve atıf

Kod/ağırlıklar `cc-by-sa-4.0` ile paylaşılıyor. Eğitim verisi: UD Türkçe treebank'leri (BOUN, IMST, Kenet; her birinin kendi lisansına tabidir), [Antalia](https://huggingface.co/datasets/cloud0day3/antalia-voice-corpus) (CC-BY-4.0; atıf: "Antalia (Patientdesk.ai)"), gövde `iatagun/DizgeBERT-Dep` / `dbmdz/electra-base-turkish-cased-discriminator`. Bu paketi kullanırken ilgili veri ve gövde lisanslarını da gözetin.

Bu model **ses üretmez ve ses kullanmaz**; Antalia'dan yalnız sözcükler arası ölçülen sessizlik süreleri (etiket) kullanıldı. Yine de veri sahibinin etik notu geçerlidir: konuşmacıyı taklit etmek, sentetik sesi gerçek kayıt diye sunmak, dolandırıcılık/siyasi robocall için kullanmayın.

## Kaynak

Eğitim, kural modülü, değerlendirme ve kör test setleri: [`iatagun/lemma-rule-based`, dal `dizgetts-v2`](https://github.com/iatagun/lemma-rule-based/tree/dizgetts-v2/dizgetts) (`dizgetts/`).
