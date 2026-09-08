# GLU Deyim Etiketleme Karar Çerçevesi (damıtılmış)

Kaynak: *Deyim Etiketleme Kılavuzu — Öğretmenler İçin Etiketleme İlkeleri*, Gülsün Leylâ
Uzun, Ankara Üniversitesi, "Öğrencilerin Söz Varlığının Tespiti, Geliştirilmesi ve İzlenmesi"
(ÖSP) projesi. Tam metin: `GLU_Deyim_Etiketleme_Ogretmen_Kilavuzu.pdf` (repo kökü, 49 s.).

Bu, DizgeBERT-Idiom eğitim verisi hazırlarken / örnek gözden geçirirken / eval seti kurarken
esas alınacak karar mantığıdır. Kuramsal dayanak: Subaşı Uzun (1991) *deyimleşme dereceleri*
(ikili değil, dereceli), + Nunberg/Sag/Wasow (1994), Moon (1998), Gibbs (1994), Kövecses (2010).

---

## Temel tanım

**Deyim** = bir kişi / durum / davranış / olay / niteliği **DOLAYLI** anlatan, **KALIPLAŞMIŞ**,
**BÜTÜNE AİT** yeni bir anlam taşıyan söz varlığı birimi. Sözlük biçiminde **EN AZ İKİ
sözlüksel öge** içerir ve en az biri (çoğu zaman ikisi) **anlam aktarımı** taşır.

## Çok ölçütlü karar ilkesi (kılavuzun bel kemiği)

Hiçbir birim tek özelliğine bakılarak etiketlenmez:
- Kalıplaşmış olmak ≠ deyim ("karar vermek", "ilişki kurmak" de kalıplaşmış)
- Mecaz taşımak ≠ deyim (tek sözcük mecazı: "tilki", "bomba")
- Sık kullanılmak ≠ deyim
- Sözcükleri değişmiş olmak ≠ varyant

---

## AŞAMA 1 — Eleme soruları (SIRALI; ilk "evet" kararı bitirir)

| # | Soru | Evet → | Örnek |
|---|---|---|---|
| 1 | Genel geçer yargı / öğüt bildiriyor mu? | **ATASÖZÜ** | "Sakla samanı gelir zamanı." |
| 2 | Belirli toplumsal durumda kullanılan hazır söz mü? | **KALIP SÖZ** | "geçmiş olsun", "kolay gelsin" |
| 3 | Bir nesne/kavram/uzmanlık teriminin adı mı? (§Terim 3 ölçütü) | **TERİM / BİRLEŞİK AD** | "kara delik", "dil ailesi" |
| 4 | Bir kişi/durum/davranış/olay/özelliği **DOLAYLI** betimliyor mu? | **Aşama 2'ye geç** | "gözden düşmek" |

**4 → Hayır** ("kitap okumak", "pencereyi açmak") → **SERBEST İFADE**, etiketlenmez.
Bu bir karar, kararsızlık değil.

## AŞAMA 2 — Deyim mi, Eşdizimlilik mi? (SIRALI DEĞİL — üç ölçüt BİRLİKTE)

| Ölçüt | Soru | Deyim yönü | Eşdizim yönü |
|---|---|---|---|
| **Bileşimsellik** | Anlam bütünüyle parçalardan çıkarılıyor mu? | HAYIR (gözden düşmek) | EVET (karar vermek) |
| **Anlam aktarımı** | Anlam başka bir şey/sahne üzerinden mi (mecazen) aktarılıyor? | EVET (küplere binmek) | HAYIR (karar vermek) |
| **Kalıplaşma** | Ögeler kolayca değiştirilebiliyor mu? | HAYIR = değişemez = kalıplaşma VAR | (kitabı/defteri masaya koymak = değişir = kalıplaşma YOK) |

### Karar tablosu (Bileşimsel? / Anlam aktarımı? / Kalıplaşma? → etiket)

| Bileşimsel | Anlam akt. | Kalıplaşma | Etiket | Not |
|---|---|---|---|---|
| Hayır | Evet | Evet | **DEYİM** | Standart: gözden düşmek, küplere binmek, pabucu dama atılmak |
| Evet | Hayır | Evet | **EŞDİZİMLİLİK** | Standart: karar vermek, ilişki kurmak, görüş almak |
| Evet | Hayır | Hayır | **SERBEST İFADE** | Etiketlenmez: kitabı masaya koymak |
| Evet | Evet | Hayır | **YARATICI BENZETME** | Etiketlenmez: mecaz var ama kalıplaşmamış, kişiye özgü |
| Hayır | Evet | Hayır | **TARTIŞMALI Tip 1** (kalıplaşma sürecinde) — ön eğilim DEYİM. Örn: "kafası güzel/iyi olmak" |
| Hayır | Hayır | Evet | **DEYİM (opak)** | Mecaz bağlantısı tarihsel/unutulmuş ama kalıplaşmış. Örn: "püsküllü bela" |
| Evet | Evet | Evet | **TARTIŞMALI Tip 2** (çelişkili bileşim). "beyaz yalan" → eşdizime yakın; "kara para" → deyime yakın |
| Hayır | Hayır | Hayır | **TARTIŞMALI Tip 3** (örnek hatası şüphesi) — muhtemelen etiketlenmemeli |

"DEYİM" ve "DEYİM (opak)" **aynı son karardır**; opaklık yalnız şeffaflık derecesini kayda geçer.
TARTIŞMALI = başarısızlık değil, güvenlik adımı → ekip görüşüne.

## AŞAMA 3 — Bağlamda kullanım: GERÇEK mi, MECAZİ mi? (DizgeBERT-Idiom için EN KRİTİK)

> Sözlükte deyim olan bir birim, otantik metinde **her zaman mecazi kullanılmaz**. Aynı yüzey
> biçim bağlama göre gerçek ya da mecazi geçebilir. Etiketlenen şey yapının kendisi değil,
> **O TÜMCEDEKİ kullanımıdır.**

Sorular:
- Tümcedeki özne/nesne **somut mu, soyut mu**? (Somut nesne + fiziksel eylem → çoğu zaman gerçek)
- Yapının **gerçek (birebir) okunuşu** bu bağlamda mantıklı ve olağan mı? Evetse gerçek ihtimali yüksek.
- Metnin geneli (konu, bağlam) hangi okumayı destekliyor?
- Tümceyi birimin **temel anlamıyla değiştirip** okuyunca anlam hâlâ tutarlı mı? Tutarlıysa gerçek.

---

## MİNİMAL ÇİFTLER — aynı yüzey biçim, biri literal biri deyim

Bunlar DizgeBERT-Idiom için altın: hard-negative eğitim örneği + idyomatik/literal diagnostik.

| Yapı | MECAZİ (etiketlenir) | GERÇEK (etiketlenmez / O) |
|---|---|---|
| eli kolu bağlanmak | "Yeni yönetmelik yüzünden öğretmenin eli kolu bağlandı." | "Hırsız, polis tarafından eli kolu bağlanarak götürüldü." |
| gözden düşmek | "Yalan söylediği ortaya çıkınca arkadaşlarının gözünden düştü." | "Kontak lensi gözünden düştü ve yere yuvarlandı." |
| yol almak | "Projede son aylarda önemli ölçüde yol aldık." | "Yürüyüş kolu / otobüs kısa sürede çok yol aldı." |
| el vermek | "Bu çalışmaya birçok kişi/kurum el verdi." | "Çocuğu kaldırmak için el verdi." |
| topu taca atmak | "Soruyu yine topu taca atarak geçiştirdi." | "Oyuncu topu taca attı." |
| rölantide olmak | "Proje aylardır rölantide." → **DEYİM** | "Motor rölantide çalışıyor." → **TERİM** |
| kısa devre yapmak | "Soruyu duyunca beynim kısa devre yaptı." → DEYİM | "Devrede kısa devre meydana geldi." → TERİM |
| devre dışı kalmak | "Sakatlığı nedeniyle sezonu devre dışı geçirdi." → DEYİM | "Arızalı ünite devre dışı bırakıldı." → TERİM |
| sigortası atmak | "Bir anda sigortaları attı." → DEYİM | "Aşırı yüklenince sigorta attı." → TERİM |

## HARD NEGATIVE listeleri

**Deyim sanılan EŞDİZİMLİLİK** (kalıplaşmış ama yeni bütünsel anlam YOK — anlamı ad-sözcüğün
temel anlamı belirler; VID değil, LVC ya da O):
karar vermek, görüş almak, izin vermek, yardım etmek, etki etmek, katkı sağlamak, dikkat
etmek, önem vermek, sonuç almak, not almak, bilgi vermek, zarar vermek, görev vermek/almak.

**Ama sınır — bunlar DEYİM** (ad-sözcük anlam aktarımı taşıyor): söz almak, söz vermek,
yol göstermek, ön ayak olmak. ("söz" ve "yol" mecazi.)

**Deyim sanılan TERİM** (bitki/kavram adı — O olmalı):
kara delik, kuşburnu, hanımeli, fare kulağı, aslan ağzı, deve dikeni.

**Deyim sanılan TEK SÖZCÜKLÜ MECAZ** (span değil — çok sözcüklü yapı yok):
tilki, bomba, melek, canavar, öküz, eşek.

---

## YÜKLEM DÜŞMESİ (çok sözcüklülük ölçütü)

Türkçede yardımcı eylem (olmak/kalmak/düşmek…) yüzeyde çoğu zaman düşer. "Proje iki aydır
rölantide." → yüzeyde tek sözcük, ama altta **"rölantide OLMAK"** iki öge; "olmak" anlamca
orada. Deyimi tek sözcüğe İNDİRMEZ.

**Karar testi:** yardımcı eylemi geri koyduğunda (rölantide OLDU / OLACAK / OLABİLİR) anlam ve
imgesel sahne değişmeden kalıyor mu? Evetse geçerli deyim örneği (yüklemi düşmüş yüzey biçim).
Yerine konacak ikinci öge YOKSA ("tilki") → tek sözcük mecazı, deyim değil.

## İMGESEL SAHNE

Deyimin zihinde canlandırdığı görüntü/olay. "ateş püskürmek" → ÖFKE=ATEŞ; "gözden düşmek" →
DEĞER=YÜKSEKLİK. Varyant kararında ve Tip 2 çözümünde kullanılır. Tek tümceyle yaz
("DEĞER=YÜKSEKLİK") — kararı somutlaştırır.

## VARYANT KARARI (SIRALI; her "hayır" bitirir)

1. İki deyimin **anlamı** aynı mı? Hayır → varyant değil.
2. Aynı **imgesel sahneyi** mi kullanıyor? Hayır → varyant değil.
3. Fark yalnızca sözcük / ek / söz dizimi düzeyinde mi? Hayır → ayrı deyim. Evet → **varyant**.

**Varyant = aynı deyim anlamı + aynı imgesel sahne + yalnız biçimsel fark.**

Türleri: sözcüksel (ateş saçmak/püskürmek), biçimbirimsel (gözden/gözünden düşmek),
sözdizimsel (dili tutulmak → dili tutulup kalmak), genişletilmiş, bölgesel, tarihsel
(kelamını esirgememek → sözünü esirgememek), yüklem düşmesi (rölantide olmak → "rölantide").

**Genişletme varyant sayılır ANCAK:** (1) eklenen öge çıkınca deyim yapısal+anlamsal eksiksiz
kalıyorsa VE (2) eklenen öge yeni imgesel sahne / bağımsız anlam katmanı oluşturmuyorsa.

**Yakın anlamlı ≠ varyant:** "küplere binmek" vs "tepesi atmak" (farklı sahne) = AYRI deyimler.
"havalara uçmak" vs "ağzı kulaklarına varmak" (yükselme vs yüz ifadesi) = AYRI deyimler.

---

## EN SIK 10 HATA

1. Kalıplaşma tek başına deyim göstermez.
2. Mecaz tek başına deyim göstermez.
3. Yardımcı eylem içermek tek başına eşdizim göstermez.
4. Tek sözcüklü mecazlar deyim değildir.
5. Aynı birim farklı tümcelerde farklı etiket alabilir (bağlam!).
6. Aynı anlamı taşıyan iki yapı her zaman varyant değildir.
7. Bir yapı hem terim hem deyim olabilir (alana göre).
8. Önce bağlama, sonra ölçütlere bak.
9. Emin değilsen TARTIŞMALI ÖRNEK / BELİRSİZ işaretle.
10. Hiçbir kararı tek ölçüte göre verme.

## DizgeBERT-Idiom etiket uzayına eşleme

| GLU kategorisi | DizgeBERT etiketi |
|---|---|
| DEYİM, DEYİM (opak), Tip 1 (ön eğilim deyim) | **B/I-VID** |
| EŞDİZİMLİLİK (söz/yol gibi anlam-aktarımlı ad + fiil) | **B/I-LVC** |
| SERBEST İFADE, YARATICI BENZETME, deyim-biçiminin LİTERAL kullanımı, TERİM, tek-sözcük mecaz, KALIP SÖZ, ATASÖZÜ | **O** |
| TARTIŞMALI Tip 2/3, BELİRSİZ | veri setinden çıkar / ayrı işaretle |

> Not: mevcut weak-supervision (TDK stem-match, `prepare_tdk_corpus_examples.py` sıkı eşleşme)
> yalnız yüzey biçme bakıyor → Aşama 3'ü (bağlam) hiç uygulamıyor. Precision tavanının kökü bu.
