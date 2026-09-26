"""python -X utf8 dizgetts/tests/test_normalize.py  (assert tabanlı; pytest gerekmez)."""
import os, sys
from dizgetts.frontend.normalize import cardinal, normalize  # noqa: E402

CARD = {0: "sıfır", 7: "yedi", 10: "on", 100: "yüz", 101: "yüz bir", 1000: "bin", 2000: "iki bin", 3254: "üç bin iki yüz elli dört",
        1990: "bin dokuz yüz doksan", 2026: "iki bin yirmi altı", 1_000_000: "bir milyon", 12_345_678: "on iki milyon üç yüz kırk beş bin altı yüz yetmiş sekiz"}
for n, w in CARD.items():
    assert cardinal(n) == w, (n, cardinal(n), w)

CASES = {
    "3'te buluşalım": "üçte buluşalım .",
    "4'ü aldım": "dördü aldım .",
    "%53'ü kullanılmış": "yüzde elli üçü kullanılmış .",
    "bu ayın yüzde 62'si": "bu ayın yüzde altmış ikisi .",
    "1.000 dakika ve 500 SMS": "bin dakika ve beş yüz es em es .",
    "ücret 389,50 TL.": "ücret üç yüz seksen dokuz lira elli kuruş .",
    "3679 lira 25 kuruş": "üç bin altı yüz yetmiş dokuz lira yirmi beş kuruş .",
    "saat 09.15'te": "saat dokuz on beşte .",
    "3 Ağustos 2026 Pazartesi": "üç Ağustos iki bin yirmi altı Pazartesi .",
    "12.05.2024 tarihinde": "on iki mayıs iki bin yirmi dört tarihinde .",
    "Kod PPV-3254-K.": "Kod pe pe ve otuz iki elli dört ke .",
    "no 4207-9916": "no kırk iki sıfır yedi doksan dokuz on altı .",
    "yüzde 47'i, yüzde 79'i": "yüzde kırk yedisi , yüzde yetmiş dokuzu .",
    "9'te, 5'te": "dokuzda , beşte .",
    "kod VKT7M2": "kod ve ke te yedi me iki .",
    "en fazla 55x40x23 cm": "en fazla elli beş çarpı kırk çarpı yirmi üç santimetre .",
    "A girişi, B'yi seçin, O geldi": "a girişi , beyi seçin , O geldi .",
    "Göksu A.Ş.'nin": "Göksu anonim şirketinin .",
    "3006751942 numaralı": "otuz sıfır altı yetmiş beş on dokuz kırk iki numaralı .",
    "www nokta": "vi vi vi nokta .",
    "'E' yazın, 'H' yazın": "e yazın , he yazın .",
    "USB ve PDF": "u se be ve pe de fe .",
    "25 GB'ın 18,4'ü": "yirmi beş gigabaytın on sekiz virgül dördü .",
    "Ağustos'ta 'Işınla' menüsü": "Ağustosta Işınla menüsü .",
    "Wi-Fi, e-posta; tamam": "vayfay , e posta ; tamam .",
    "Merhaba... Nasılsın?!": "Merhaba . Nasılsın ?",
    "Dr. Ali geldi": "doktor Ali geldi .",
    "Dedi ki - evet - sonra": "Dedi ki , evet , sonra .",
    "3. sınıf": "üçüncü sınıf .",
    "Bir\n\nİki": "Bir . İki .",
}
bad = [(k, normalize(k), v) for k, v in CASES.items() if normalize(k) != v]
for b in bad:
    print("FAIL", b)
assert not bad, f"{len(bad)}/{len(CASES)} başarısız"

# regresyon (2026-09-26 code review): normalize() çıktısındaki her belirteç harf ya da duraklama olmalı ve engine'in sözcük ayrıştırması (_TOK)
# ile tagger'ın str.isalpha()'sı AYNI sözcükleri görmeli. "m²" (No kategorisi: \w ama isalpha değil) g2ptts hizalamasını sessizce bozuyordu.
import random
from dizgetts.engine import _TOK

PUNCT = set(",.?!;")


def _check(s):
    n = normalize(s)
    toks = n.split()
    assert all(t.isalpha() or t in PUNCT for t in toks), (s, n)
    assert [t for t in toks if t not in PUNCT] == [m.group() for m in _TOK.finditer(n) if m.group() not in PUNCT], (s, n)
    return n


assert _check("Oda 12 m² büyüklüğünde, ama ışık yok.") == "Oda on iki metrekare büyüklüğünde , ama ışık yok ."
assert _check("3 m³ su, 5 km² alan") == "üç metreküp su , beş kilometrekare alan ."
assert _check("Ali² geldi ½ kez ① ﬁyat") .endswith(".")
assert _check("çay ve göz") == "çay ve göz ."  # NFD (ç = c + ◌̧) yazım: sözcük ikiye bölünmemeli
rng = random.Random(0)
POOLS = [(0x20, 0x24F), (0x300, 0x36F), (0x400, 0x4FF), (0x660, 0x6FF), (0x900, 0x97F), (0x2070, 0x209F), (0x2150, 0x218F), (0x2460, 0x24FF),
         (0xFB00, 0xFB06), (0xFF10, 0xFF5A), (0x4E00, 0x4E80)]
for _ in range(4000):
    _check("".join(chr(rng.randint(*rng.choice(POOLS))) for _ in range(rng.randint(1, 30))))
print(f"OK: {len(CARD)} sayı + {len(CASES)} normalize vakası + biçimsel değişmez (4000 rastgele girdi)")
