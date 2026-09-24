# Türkçe sesbilimi notları (kaynak: kullanıcı, dilbilimci; DizgeTTS için)

Bu dosya kullanıcıdan gelen dilbilimsel bilgiyi ve **kodda uygulanıp uygulanmadığını** izler. Yorumlarım "(yorum)" ile işaretli; onay bekler.

## 1. Sözcük vurgusu (2026-09-24, kullanıcı açıklaması)

**Varsayılan:** vurgu son seslemdedir; sözcük ek alınca vurgu yeni oluşan son sesleme kayar (`kitap` → `kitaplar` → `kitaplarımızdan`, hep son hece).

**Bu örüntüye uymayanlar** (kullanıcının sıraladığı kategoriler ve örnekleri):

| Kategori | Örnek | Vurgu (yorum) |
|---|---|---|
| Olumsuzluk (-mA), düzensiz/ödünçleme örnekleri | `gelme`, `lokanta`, `İstanbul`, `pırasa` | sondan 2. hece (`GEL-me`, `lo-KAN-ta`, `is-TAN-bul`, `pı-RA-sa`) |
| Seslenme sözcükleri | `çocuklar!` | ilk hece (yorum: `ÇO-cuk-lar!`) |
| Küçültme eki alan | `semracığım` | küçültme ekinden önceki hece (yorum: `sem-RA-cı-ğım`) |
| Belirteçlerin çoğu | `şimdi`, `belki` | sondan 2. (`ŞİM-di`, `BEL-ki`) |
| Ödünçleme sözcükler | `futbol`, `lokanta` | sondan 2. (`FUT-bol`) |
| Yer adları | `Ordu`, `Bebek` | sondan 2. (`OR-du`, `BE-bek`) |
| Soru sözcükleri | `hangi`, `nasıl` | sondan 2. (`HAN-gi`, `NA-sıl`) |
| İkileme/pekiştirme, ödünçleme önekli | `kapkara`, `asosyal` | ilk hece (`KAP-ka-ra`, `A-sos-yal`) |
| Bileşik sözcükler | `bugün`, `başbakan` | ilk öğe (`BU-gün`, `BAŞ-ba-kan`) |
| Arapça kökenli belirteç `-en` | `nakil` → `naklen`, `nispet` → `nispeten` | vurgu `-en`'den öncesine (`NAK-len`, `nis-PE-ten`); (daha fazla örnek gelecek) |

**Vurgusuz sonekler ve biçimceler (clitic)** — bunlar vurguyu almaz, vurgu bunlardan ÖNCEKİ seslemde kalır:
- eylemcil koşaçlar / belirteç-benzeri biçimbirimler: `-ydı`, `-ymış`, `-ysa`, `-yken` (`yapsaydı`, `yapsaymış`)
- `-dır`, `-(y)la`, `-da`, `-ki`, `bile`
- türetim ekleri: `-cık` (`ufacık`), `-ca`, `-casına`, `-en`, `-(y)ın`, `-leyin`, `-ra`
- kişi ekleri: `-(y)ım`, `-sın`, `-(y)ız`, `-sınız`
- olumsuzluk ekleri (`-mA`)

Kullanıcı: "daha fazlası gelecek."

## 2. Uygulama durumu (2026-09-24)

| | Durum |
|---|---|
| Vurgu modülü M1a (`frontend/stress.py`, `resources/*.tsv`) | **VAR, dar kapsam:** clitic sözcükler (da, de, ki, bile, mı/mi/mu/mü), düzensiz vurgulu kök sözlüğü (kök + makul ek zinciri; yer adları yalnız BÜYÜK harfle), varsayılan son seslem. Korpusta 35.338 sözcükten: varsayılan 33.713, clitic 1.362, kök 211, kök+ek 47. |
| Vurgusuz ekler (M1b, `frontend/stress.py` TIERS; DizgeBERT-Morph UPOS+FEATS kapılı, ek sınırı yüzey biçimiyle) | **VAR, opt-in (varsayılan kapalı):** `-Iyor` önü (benim eklemem, literatür), olumsuzluk `-mA` önü (Polarity=Neg), Case=Ins `-(y)lA` önü, sonlu fiil kişi eki önü (Person 1/2; ek yalnız geçerli zaman/kip sonrası soyulur), koşaç `-ydı -ymış -ysa -yken` (yalnız zaman/kip ekinden sonra; düz `-dı` geçmiş zaman vurgulanır). Korpusta 3.944 sözcüğün (%11) vurgusunu değiştirir: yor 1.170, kişi eki 1.598, koşaç 433, olumsuzluk 469, Ins 274. espeak ile örtüşme: yor %95, kişi eki %74, koşaç %78, Ins %68, olumsuzluk %42 (espeak ölçüt değil). **Kullanıcı doğrulaması yok.** |
| Hâlâ YAZILMADI | `-dır`, `-cık`, `-ca`, `-casına`, `-en`, `-leyin`, `-ra`, `-(y)ın`, `-ki` (ek olarak), seslenme, küçültme, ikileme, bileşik/belirteç listeleri. Belirteçler Antalia'da 2.023 sözcük (167 tür; en sık: daha 130, en 113, şimdi 104, çok 71, önce 65, hemen 63, yeniden 50, yalnızca 45, sonra 44, yine 43…); hangilerinin düzensiz vurgulu olduğu kullanıcı listesi gerektirir. |
| Seslenme, küçültme, ikileme, bileşik (listesiz) | YAZILMADI; kök sözlüğüne yalnız kullanıcının verdiği örnekler girdi. |
| Fonem eşlemesi | Seslem = ünlü harfi; dizge fonem dizisindeki ünlü atomuna eşlenir. Ünlü sayıları %87 eşit, %7,4 `ay`→`ɑːI` yan ünlüsü atılarak, %1,5 sondan sayımla (ğ kaynaşması, ünlü türemesi). |
| Test verisi | `tests/stress_gold.tsv` (28 sözcük; son-hece istisnası olmaları kullanıcı onaylı 2026-09-24, sondan-sıra değerleri benim türetmem, konum **onaysız**) ve `reports/stress_annotation_sheet.tsv` (250 sözcük; kullanıcı etiketleyecek -> `tests/stress_gold_random.tsv`). |
| Karşılaştırma | espeak-ng gold listesinde 9/28 doğru; M1a bu 28'i büyük ölçüde kök sözlüğü (onların kendi örnekleri) sayesinde geçer, **bu bir ölçüm değildir**; gerçek doğruluk rastgele 250 sözcüğün etiketiyle ölçülecek. |
| Açık soru | İşlev sözcükleri (bir, bu, ve, ile, için…) sözcük vurgusu alıyor mu, yoksa cümle içinde vurgusuz mu? Şimdi hepsi son seslem alıyor (espeak %10,5 sözcüğü vurgusuz bırakıyordu). |

## 3. Plan
1. Kullanıcıdan kalan listeleri al (belirteçler, soru sözcükleri, yer adları, ödünçlemeler, bileşikler; "daha fazlası").
2. Hece bölücüyle her sözcüğe varsayılan son-hece vurgusu; ardından kategori kuralları (sözlük + sonek tabanlı) öncelik sırasıyla.
3. `stress_gold.tsv`'ye karşı doğruluk ölç (kural modülü vs espeak).
4. `dizge + kural vurgusu` koşusu, `dizge + espeak vurgusu` ile karşılaştır.
