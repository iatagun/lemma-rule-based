# Türkçe Dilbilgisi Kuralları Referansı

> Çözümleyicide kullanılan dilbilimsel kuralların özeti.

---

## Ünlü Uyumu

### Büyük Ünlü Uyumu (BÜU) — Kalınlık/İncelik

Türkçe'de kökteki son ünlünün kalınlık/incelik özelliği, ekteki ünlüyü belirler:

```
         KALIN          İNCE
       ┌─────────┐   ┌─────────┐
       │ a ı o u │   │ e i ö ü │
       └────┬────┘   └────┬────┘
            │              │
            ▼              ▼
    ek ünlüsü KALIN   ek ünlüsü İNCE
```

**Örnekler:**
- okul + lar → okullar ✓ (a→a)
- okul + ler → ✗ (a→e, BÜU bozulur)
- ev + ler → evler ✓ (e→e)
- ev + lar → ✗ (e→a, BÜU bozulur)

### Küçük Ünlü Uyumu (KÜU) — Düzlük/Yuvarlaklık

```
    Kök son ünlü          Ek ünlüsü
    ──────────────        ──────────
    DÜZ (a, e, ı, i) ───→ DÜZ (a, e, ı, i)

    YUVARLAK (o, ö, u, ü)
         │
         ├─ dar ek ünlüsü ───→ DAR YUVARLAK (u, ü)
         │
         └─ geniş ek ünlüsü ──→ GENİŞ DÜZ (a, e)
```

**Karar ağacı:**
```
kök son ünlü DÜZ mü?
├── EVET → ek ünlüsü DÜZ olmalı
└── HAYIR (yuvarlak)
    ├── ek ünlüsü DAR mı? → YUVARLAK olmalı
    └── ek ünlüsü GENİŞ mi? → DÜZ olmalı
```

---

## Ünsüz Benzeşmesi

```
Kök son ünsüz SERT (ç, f, h, k, p, s, ş, t)?
├── EVET → ek başı: d→t, c→ç
│          git + dir → git + tir ✓
│          git + dir ✗
└── HAYIR → ek başı değişmez
           gel + dir ✓
           gel + tir ✗
```

---

## Morfofonemik Ses Olayları

| Ses Olayı | Kural | Örnek |
|-----------|-------|-------|
| **Ünsüz yumuşaması** | p→b, ç→c, t→d, k→g/ğ + ünlü eki | kitap → kitab+ı |
| **Ünlü düşmesi** | 2. hece dar ünlüsü düşer + ünlü eki | burun → burn+u |
| **Ünlü daralması** | a→ı, e→i / __-yor | başla+yor → başlıyor |
| **Kaynaştırma y** | V-kök + V-ek arası | su+y+u |
| **Kaynaştırma n** | İşaret zamirleri | o+n+un, bu+n+a |
| **Kaynaştırma s** | İyelik 3T V-kök | araba+s+ı |

---

## Ek Hiyerarşisi (Slot Modeli)

### İsim Çekimi
```
KÖK → [Yapım] → [Çoğul] → [İyelik] → [Hal] → [-ki]
```

### Fiil Çekimi
```
KÖK → [Çatı] → [Olumsuz] → [Yeterlilik] → [Zaman/Kip] → [Kişi] → [Bildirme]
```

### Nominalizasyon Sıfırlaması
Sıfat-fiil veya isim-fiil ekinden sonra isim slotları yeniden başlar:
```
yaşadığını = yaşa + dığ(SIFAT_FİİL) + ı(İYELİK) + nı(HAL)
```

---

## Düzensiz Fiiller

Türkçe'nin yalnızca 2 gerçek düzensiz fiili var:

### demek
| Biçim | Kök | Açıklama |
|-------|-----|----------|
| dedi | de | Görülen geçmiş |
| di, diy | de | Şimdiki/geniş zaman |
| der | de | Geniş zaman |
| den | de | Edilgen |

### yemek
| Biçim | Kök | Açıklama |
|-------|-----|----------|
| yi, yiy | ye | Şimdiki/geniş zaman |
| yed | ye | Ettirgen |

### Yarı-Düzensiz (ünsüz yumuşaması)
- etmek: et→ed
- gitmek: git→gid

---

## Uyum Muaf Ekler

| Ek | Ünlü | Neden Muaf |
|----|------|-----------|
| -yor | o | Değişmeyen yuvarlak |
| -ken | e | Değişmeyen ince düz |
| -ki | i | Değişmeyen ince düz |

---

## BOUN Lemma Standardı

- **Fiiller:** Yalın gövde — `gel`, `yaz`, `iste` (mastar eki yok!)
- **İsimler:** Yalın kök — `ev`, `kitap`
- **Birleşik isimler:** Tamlama eki düşürülür — `cezaevi → cezaev`
- **Şapkalı biçimler:** Korunur — `hâl`, `âdet`, `kâr`
