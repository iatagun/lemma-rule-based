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

## 2. Uygulama durumu

| | Durum |
|---|---|
| Kural tabanlı vurgu modülü | **YAZILMADI.** Gereken: hece bölücü (dizge), sonek sınırları (DizgeBERT-Morph), sözlükler (belirteç, soru sözcüğü, yer adı, ödünçleme, bileşik). |
| Geçici vurgu | `frontend/stress.py`: espeak-ng vurgusunun dizge ünlülerine aktarımı (üst sınır deneyi). |
| Test verisi | `tests/stress_gold.tsv` (28 sözcük; sondan sıra değerleri BENİM türettiğim, **onaysız**). |

**espeak-ng bu listede 9/28 doğru** (ölçüldü, 2026-09-24). Yanlışlar tam bu kategorilerde: `şimdi`, `belki`, `hangi`, `nasıl` (son hece diyor), `bugün`, `başbakan`, `kapkara`, `asosyal`, `yapsaydı`, `ufacık`, `naklen`, `lokanta`, `futbol`, `bebek`. Bu sözcükler Antalia'da sık: `şimdi` 104, `bugün` 36, `hangi` 29, `nasıl` 20, `belki` 18 (35.561 sözcükten). Dolayısıyla "dizge + espeak vurgusu" deneyi bu sözcüklerde YANLIŞ vurgu öğreniyor; dilbilimsel kurallarla üretilen vurgu bunu düzeltmeli (aşağıdaki plan).

## 3. Plan
1. Kullanıcıdan kalan listeleri al (belirteçler, soru sözcükleri, yer adları, ödünçlemeler, bileşikler; "daha fazlası").
2. Hece bölücüyle her sözcüğe varsayılan son-hece vurgusu; ardından kategori kuralları (sözlük + sonek tabanlı) öncelik sırasıyla.
3. `stress_gold.tsv`'ye karşı doğruluk ölç (kural modülü vs espeak).
4. `dizge + kural vurgusu` koşusu, `dizge + espeak vurgusu` ile karşılaştır.
