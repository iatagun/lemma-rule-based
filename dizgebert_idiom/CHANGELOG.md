# DizgeBERT-Idiom — Sürüm Geçmişi ve Detaylı Doğrulama Notları

Bu dosya, ana [MODEL_CARD.md](MODEL_CARD.md)'nin kısa/güncel tutulması için oradan
taşınmış detaylı geçmiş ve istatistiksel doğrulama notlarını içerir. Güncel kullanım,
mimari ve headline sonuçlar için MODEL_CARD.md'ye bakın.

## Sürüm Geçmişi

Aşama 2 HER ZAMAN ayrı bir gövde/veri kaynağıyla eğitildi; hangi sürümde neyin değiştiği:

- **v3 — iki aşamalı boru hattına geçiş.** Aşama 1 tek-BIO modeli precision tavanına
  takılıyordu (yüzey biçim eşleşince bağlamdan bağımsız işaretliyordu). Aşama 2 eklendi:
  ayrı ELECTRA gövdesi, alt 8 transformer katmanı donduruldu (küçük veride tam fine-tune
  ağır overfit ediyordu).
- **v4 — Aşama 1 genişletildi, Aşama 2 değişmedi.** TDK eşleştiricisi `max_gap=2` oldu
  (araya en fazla 2 eşleşmeyen kelime girebilir) ve Leipzig'den madenlenip GLU rubriğiyle
  idyomatik-olarak etiketlenmiş binlerce doğal cümle Aşama 1'e eklendi. Sonuç: dış-kaynak
  doğru-ayırt %42→%56, kazanç tümüyle o zamanki görülmemiş-deyim diliminde.
- **v5 — Aşama 1'e ikinci gövde (ensemble), Aşama 2 değişmedi.** İki farklı veri diliminde
  eğitilmiş stage-1 checkpoint'inin aday span birleşimi, tek modelle kırılamayan bir tavanı
  aştı. Bedel: model iki katı (yaklaşık 1.3GB), yaklaşık 2× gecikme.
  PARSEME F1 hafifçe düştü, dış-kaynak doğru-ayırt yükseldi.
- **v6 — Aşama 1 ensemble'ı TEK gövdeye indirgendi (anlaşmazlık-ağırlıklı distilasyon).**
  Düz (ağırlıksız) damıtım öğretmenlerin ORTALAMASINI aktarıyordu, tamamlayıcı kapsamlarını
  değil; anlaşmazlık ağırlıklandırması (öğretmenlerin GERÇEKTEN ayrıştığı pozisyonlara
  damıtım kapasitesini yönlendirme) bu farkı kapattı. Sonuç: ensemble'ın doğruluğu tek
  gövdede korundu, yaklaşık 2× gecikme bedeli kalktı. **Hâlâ `revision="fd002bdd32..."` ile
  erişilebilir** (aynı HF repo'nun eski bir commit'i, ayrı model değil).
- **v7 — Aşama 1'e ÜÇÜNCÜ bağımsız gövde eklendi (ensemble'a dönüş, distilasyon değil).**
  v6'nın damıtımı iki öğretmene genelleşmişti ama üçe genelleşmedi (üçüncü öğretmenin küçük
  eğitim hacmi anlaşmazlık sinyalini gürültüleştirdi). Saf çıkarım-zamanı birleşim ise işe
  yaradı: dış-kaynak doğru-ayırt yeni bir rekor kırdı, bedelinde v6'nın kazandığı hız/boyut
  avantajı geri gitti (model tekrar yaklaşık 2GB/yaklaşık 3×).
- **v8 — Aşama 2 sentetik minimal çiftlerle yeniden eğitildi, Aşama 1 (v7) değişmedi.**
  Önceki tüm Aşama 2 turları doğal derlemden madenlenip LLM ile ETİKETLENMİŞ cümlelere
  dayanıyordu (literal kullanım doğal metinde nadir olduğu için hep dengesiz/gürültülüydü).
  v8, LLM'e doğal cümle sınıflandırtmak yerine TDK deyim listesinden seçilen yaklaşık 650 deyim için
  doğrudan dengeli idyomatik+literal cümle ÜRETTİRDİ, doğal-derlem verisini tamamen atıp
  yalnız bu sentetik havuzla eğitti (karışık deneme daha kötü sonuç verdi). Sonuç: dış-kaynak
  doğru-ayırt %58.1→%65.2 (varsayılan eşik 0.5'te).
- **v8 eşik güncellemesi (2026-09-20).** Model ağırlıkları DEĞİŞMEDİ; harici bir model-kartı
  incelemesi sonrası yapılan `stage2_thresh` taramasında 0.3'ün 0.5'ten ölçülebilir şekilde
  daha iyi olduğu görüldü (doğru-ayırt %65.2→%68.2, yanlış-poz %21.2→%14.6). Varsayılan
  `config.stage2_thresh` 0.3'e güncellendi (yalnız config.json değişti, ağırlıklar aynı).

## v7→v8 kıyası ve istatistiksel doğrulama

Kıyas, HER İKİ sürümün de FİİLEN YAYINLANDIĞI eşikle yapılıyor (v7: 0.5, v8: 0.3 — ikisi de
kendi varsayılanı):

| metrik | v7 (`thresh=0.5`) | **v8 (`thresh=0.3`)** |
|---|---|---|
| doğru-ayırt (n=198) | %58.1 (95% GA %51.0–%65.2) | **%68.2 (95% GA %61.6–%74.7)** |
| **eşleştirilmiş fark** (aynı 198 cümle, v8−v7) | — | **+10.1pp (95% GA +3.5pp – +16.7pp, 5000 bootstrap)** |
| McNemar dökümü | — | 103 ikisi de doğru, **32 yalnız v8 doğru**, 12 yalnız v7 doğru, 51 ikisi de yanlış |
| yanlış-pozitif | %28.3 | %14.6 |
| GLU tanı seti (35 vaka, küçük örneklem — bkz. not) | 20/35 | 25/35 |

Bağımsız ölçülen iki nokta (independent CI) tek başına bakılınca örtüşüyor gibi görünebilir,
ama bu 198 cümle HER İKİ modelde de AYNI olduğu için doğru karşılaştırma eşleştirilmiş
(paired) bootstrap'tır: fark %99.9 ihtimalle pozitif, %95 güven aralığı sıfırı içermiyor —
istatistiksel olarak anlamlı. **35-vakalık GLU tanı setindeki 20/35→25/35 gibi geçişler ise
küçük örneklem nedeniyle TEK BAŞINA güçlü bir kanıt sayılmamalı**, yalnız destekleyici sinyal.

**Bağımsız üçüncü doğrulama (2026-09-20, v7 kendi eşiğiyle taranarak — bkz. `benchmark/
eval_cavusoglu_threshold_sweep.py`):** v7'nin (Aşama-1 3-gövde ensemble değişmedi, Aşama-2=v3)
kendi eşik taraması yapıldığında en iyi eşiğin GERÇEKTEN 0.5 olduğu (yayınlanan varsayılanıyla
birebir örtüşüyor) doğrulandı: eşik=0.3→%55.6, **0.5→%58.1 (en iyi)**, 0.7→%57.1, 0.9→%51.5.
Yani v7↔v8 kıyası, v8'in eşiği taranıp v7'ninkinin taranmamasından kaynaklanan bir yanlılık
DEĞİLDİ — v7'nin kendi en iyi eşiği zaten 0.5 çıktı. Aynı 0.5 eşiğinde (her iki sürüm de) fark
+7.1pp (65.2 vs 58.1); her ikisi kendi en iyi eşiğinde ise +10.1pp — iki çerçeveleme de aynı
yönde, farklı büyüklükte.

## Sızıntı / "görülmemiş deyim" kontrolü — düzeltme notu

**v8'in eğitiminde kullanılan 503 sentetik deyimin** Çavuşoğlu'nun 198 eval deyimiyle sıfır
kesiştiği doğrulandı. Ayrıca, bu kartın önceki bir taslağı "görülmemiş deyim dilimi" diye bir
alt-küme (177/198) raporluyordu; bu alt-küme ilk kez erken bir deneyde (proje içi, HF dışı)
hesaplanmış ve sonraki veri-genişletme turlarında **yeniden hesaplanmamıştı** — denetlerken bu
fark edildi: o 177'nin aslında 121'i, sonraki turlarda eğitim havuzuna (Aşama 1'in çeşitli
gövdelerinden en az birine) girmiş çıktı. Düzeltilmiş, muhafazakâr bir "gerçekten görülmemiş"
kümeyle (59 deyim) yeniden ölçüldüğünde desen KORUNUYOR: v8 doğru-ayırt (`thresh=0.3`) **%69.5**
vs v7 (`thresh=0.5`) %59.3 (yaklaşık +10.2pp, genel bulguyla tutarlı) — yani asıl bulgu (v8'in
genellemesi) ayakta kalıyor, ama kart daha önce yanlış/güncellenmemiş bir "seen/unseen"
ayrımını rapor ediyordu.

## Aslantaş & Güngör kıyası — tam detay

Aslantaş, G. & Güngör, T. (Boğaziçi, SIGTURK 2026), "A Unified Turkic Idiom Understanding
Benchmark: Idiom Detection and Semantic Retrieval Across Five Turkic Languages"
([ACL Anthology 2026.sigturk-1.4](https://aclanthology.org/2026.sigturk-1.4/),
[github.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark](https://github.com/gozdeaslantas/Turkic_Idiom_Understanding_Benchmark)).
Kod ve veri açık olduğu için **çift yönlü, ölçülmüş** bir kıyas yapıldı (2026-09-20,
`benchmark/eval_aslantas_gungor.py`, `benchmark/train_ag_electra_baseline.py`,
`benchmark/analyze_ag_errors.py`).

**Sonuç, onların TR test setinde (131 cümle), onların KENDİ metriğiyle** (seqeval,
entity-düzeyi exact-match): DizgeBERT açıkça geride: F1=0.592 (tümü), **0.414 gerçekten
görülmemiş deyimlerde** — onların in-domain ELECTRA-tr/ConvBERT-tr'sinin 0.877/0.880'inin
belirgin altında. VID/LVC iki kategorimiz bu kıyas için TEK bir IDIOM sınıfına indirgendi.

| | tümü (n=131) | deyim-kimliği görülmüş (n=71) | gerçekten görülmemiş (n=60) |
|---|---|---|---|
| **entity-exact (seqeval, onların metriği), stage2=False** | F1=0.592 (95% GA 0.513–0.672) | F1=0.736 (0.639–0.827) | **F1=0.414** (0.300–0.537) |
| entity-exact, stage2=True (yayınlanan varsayılan) | F1=0.496 (0.414–0.580) | F1=0.608 (0.504–0.712) | F1=0.372 (0.252–0.496) |
| gevşek/token-örtüşmeli (kısmi sınır hatasına kısmi puan), stage2=False | P=0.887 R=0.721 F1=**0.795** | P=0.933 R=0.848 F1=0.888 | P=0.822 R=0.583 F1=0.682 |
| **onların ELECTRA-tr/ConvBERT-tr'si (in-domain, kendi eğitimleri)** | **F1=0.877 / 0.880** | — | — |

131 test idiomunun **71'i (%54) zaten bizim Aşama-1 eğitim havuzumuzda** — "sıfır-atış"
yalnız CÜMLE düzeyinde geçerli, DEYİM-kimliği düzeyinde değil.

**Entity-exact (0.592) ile gevşek/token-örtüşmeli (0.795) arasındaki fark ne kadarı sınır
konvansiyonundan geliyor?** Sınıflandırma yöntemi: `analyze_ag_errors.py` her gold/pred span
çiftini OTOMATİK bir örtüşme kuralıyla (EXACT/BOUNDARY/MISS/SPURIOUS, tanım script'in
docstring'inde) etiketliyor; bu depoyu tutan (proje sahibi + Claude Code) çıktıyı okuyup
örüntüleri elle doğruladı — bağımsız bir üçüncü-taraf insan değerlendirmesi DEĞİL. 131
cümledeki 153 span-olayının dökümü: **%50.3 tam eşleşme (EXACT), %20.3 doğru deyimi buldu ama
sınır farklı (BOUNDARY), %15.7 gerçek kaçırma (MISS), %13.7 gerçek fazladan-işaretleme
(SPURIOUS).** Yani fark KISMEN sınır-konvansiyonundan (EXACT+BOUNDARY=%70.6 "doğru yerde"),
ama önemli bir kısmı GERÇEK kaçırma/fazladan-işaretleme (%29.4). 31 BOUNDARY vakasının
**8/31'i** "Allah ..." ile başlayan dua/beddua kalıplarına ait (*Allah belasını versin*,
*Allah korusun*, *Allah mübarek etsin*) — onların anotasyonu TÜM cümleyi (vokatif+fiil) tek
span sayıyor, DizgeBERT (TDK sözlük-biçimli VID eğitiminden) yalnız çekirdek yüklemi/nesneyi
işaretliyor. Bu, iki farklı "deyim" tanımının (idiom-as-fixed-clause vs idiom-as-lexical-item)
sınır anlaşmazlığı — model hatası değil, tanım farkı.

**Reprodüksiyon, 3 tohum:** onların backbone'unu (ELECTRA-tr — bizimkiyle AYNI encoder,
checkpoint yayınlamadıkları için double-blind kendi TR verileriyle burada yeniden eğitildi)
3 farklı tohumla (`--seed 42/1/2`) eğitip onların KENDİ metriğiyle ölçtük: **F1 = 0.898 ±
0.031** (0.924/0.864/0.906) — onların bildirdiği 3-seed-ortalaması **0.877** bu aralığın
içinde, makul bir reprodüksiyon. **Ayrı bir not:** onların makale METNİ "token-level
precision/recall/F1/accuracy" diyor ama kodları (`src/modeling/metrics.py`) `seqeval` ile
ENTITY-düzeyi hesaplıyor — bu iki tanım farklı sayılar üretir (entity-exact daha katı).

Bu backbone'u (kendi TR verisiyle, tek tohumla eğitilmiş hâli) bizim Çavuşoğlu bağlam-ayrımı
benchmark'ımızda çalıştırdık: duyarlılık %87.4, yanlış-poz %85.9, **doğru-ayırt yalnız %9.6**
— bağlamdan bağımsız, saf yüzey-biçim eşleşmesi. Bu, "salt-pozitif eğitilmiş bir tagger
bağlam ayırt edemiyor" gözlemini destekler ama TEK BAŞINA "iki aşama şart" iddiasını
kanıtlamaz (onların modeli literal örnekle hiç eğitilmedi — kendi göreviyle tutarlı bir
sınırlama, "başarısızlık" değil).

## Dodiom kıyası — tam detay

Umut, Site, Arslan & Eryiğit (İTÜ, UBMK 2025), "Exploring Turkish Idiomaticity with LLMs"in
veri seti/kodu yayınlanmamış (IEEE Xplore, paywall) — doğrudan kıyas mümkün olmadı. Yerine,
aynı ITU NLP ekosisteminden halka açık bir kaynak kullanıldı: **Dodiom TR** (Eryiğit, Şentaş &
Monti, *Natural Language Engineering* 2022, [github.com/Dodiom/dodiom](https://github.com/Dodiom/dodiom),
6861 crowdsourced örnek / 36 deyim, idiom/nonidiom ikili etiket + hedef span). **Bu Umut et
al.'ın verisiyle AYNI DEĞİL** — yalnız benzer bir dış/insan-etiketli kaynak (`benchmark/
eval_dodiom.py`). "Dış kaynak, görülmemiş" demiyoruz — daha kesin: 36 deyimin **34'ü zaten
Aşama-1'in span eğitim havuzunda var** (Aşama-1 açısından neredeyse hiç görülmemiş-deyim testi
DEĞİL); Aşama-2'nin (bağlam-ayrımı) KENDİ etiketli eğitim havuzuyla örtüşme **1/36** (ilk
ölçümde bir sorgu hatası nedeniyle "0/36" raporlanmıştı, düzeltildi — bkz. aşağıdaki not) —
yani bu **Aşama-2 açısından görülmemiş deyim kimlikleri üzerinde, doğal/crowdsourced
cümlelerle** bir test.

| | duyarlılık | yanlış-poz | dengelenmiş doğruluk* |
|---|---|---|---|
| stage2=True (varsayılan) | %69.9 (küme-GA %62.5–76.3) | %15.1 (%12.1–18.1) | %77.4 |
| stage2=False | %85.5 (%81.5–89.1) | %35.6 (%30.1–41.3) | %74.9 |

*dengelenmiş doğruluk = (duyarlılık + (1−yanlış-poz))/2, EŞLEŞTİRİLMEMİŞ cümle-düzeyi bir
ölçüt — Çavuşoğlu'nun çift-düzeyi "ikisini de doğru ayırt etti" metriğiyle AYNI ölçek DEĞİL.
Güven aralıkları deyim-KÜMESİ (36 deyim) bootstrap'ı — etkin örneklem 36'dır, 6861 satır değil.

**Aşama 2'nin net etkisi burada Çavuşoğlu'ndaki kadar net DEĞİL.** Aşama 2 açıldığında
duyarlılık %85.5→%69.9 (**−15.6pp, gerçek recall bedeli**) düşerken yanlış-poz %35.6→%15.1
(**−20.5pp, gerçek kazanç**) düşüyor — net "dengelenmiş doğruluk" farkı görünürde yalnız
+2.5pp (%74.9→%77.4). Deyim-kümesi EŞLEŞTİRİLMİŞ bootstrap ile test edildiğinde: **fark
+2.4pp, 95% GA −1.5pp – +6.1pp — SIFIRI İÇERİYOR, bu ölçekte (n=36 deyim kümesi) istatistiksel
olarak ayırt edilemiyor.**

**Düzeltme notu (Aşama-2 örtüşme sayısı, 2026-09-20):** ilk ölçüm `corpus_examples_glu.json`'u
sorguluyordu — bu dosyada `idiom` alanı hiç YOK (yalnız words/tags), yani sorgu her zaman boş
küme döndürüyordu VE yanlış dosyaydı (Aşama-1'in verisi, Aşama-2'nin (v8) GERÇEK eğitim verisi
`_synthetic_stage2_records.jsonl`, 503 deyim). Düzeltilmiş sorguyla gerçek örtüşme 1/36
("göz yummak") — önceki "0/36" iddiasına çok yakın ama YÖNTEM güvenilir değildi, şimdi
doğrulanmış hâliyle raporlanıyor.

## Aslantaş & Güngör: v7/v8 threshold-sensitivity — v7'nin en iyi kendi eşiği

Bkz. yukarıdaki "v7→v8 kıyası" bölümündeki "Bağımsız üçüncü doğrulama" notu.
