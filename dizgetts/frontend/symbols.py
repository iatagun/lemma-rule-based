"""DizgeTTS sembol tablosu.

Kaynak: `iatagun/dizge-g2p` task_config.json (87 etiket) + `dizge` paketinin
`tools/phonology.py` kuralları (I/U/Y anlamı oradan doğrulandı) + dizge.g2p'nin
48.563 sözcüklük çıktısındaki sembol sayımı (scripts/g2p_probe.py).

dizge-g2p karakter düzeyinde çalışır: her harfe TEK bir etiket verir ve etiket
1-2 fonemlik bir dizgedir ('ç'->'tʃ', 'ğ'->'ː'...). Sözcüğün fonem dizisi etiketlerin
birleştirilmesidir; etiket sınırı fonem sınırı DEĞİLDİR ('bI','ːt','Iɾ' gibi bileşik
etiketler var). Bu yüzden model girdisi için fonem dizisi `tokenize()` ile yeniden
atomlara ayrılır.
"""
from __future__ import annotations

PAD = "_"          # Matcha add_blank ile birlikte id 0
WORD_SEP = " "     # sözcük sınırı (g2p sözcük düzeyinde, sınırı biz koyarız)
PAUSES = (",", ".", "?", "!", ";")  # noktalama -> duraklama sınıfı (normalize.py bunlara indirger)
# Kırılma sınıfları (Ipek & Jun'un ip/IP ayrımından esinli; sesten ölçülen sözcük-sınırı sessizliğiyle etiketlenir, bkz. scripts/derive_breaks.py):
# B1 (sessizlik <60 ms, sözcükler arası bağ) = TOKEN YOK; B2 (60-250 ms) = "|" ; B3 (>=250 ms) = "‖".
BREAK_MID, BREAK_MAJOR = "|", "‖"
BREAKS = (BREAK_MID, BREAK_MAJOR)
STRESS = "ˈ"       # RESERVED: DizgeBERT-Stress gelene kadar kullanılmıyor; vurgulu ünlünün ÖNÜNE gelir

# atom -> (tür, not).  Notlar dizge kaynağından ya da sayımdan; emin olunmayanlar "?" ile.
_V, _C = "ünlü", "ünsüz"
PHONES: dict[str, tuple[str, str]] = {
    # ünlüler
    "a": (_V, "ön/nadir 'a' (n=24); loan/özel"),
    "ɑ": (_V, "a (arka, normal)"),
    "e": (_V, "e (kapalı; 'ğ' öncesi/tek heceli)"),
    "ɛ": (_V, "e (açık, normal)"),
    "i": (_V, "i, GEVŞEMEMİŞ: 'ğ' öncesi (n=1411)"),
    "I": (_V, "i, GEVŞEK (ɪ benzeri): normal 'i' (n=28990) -- dizge ASCII kısayolu"),
    "ɨ": (_V, "ı"),
    "o": (_V, "o, 'ğ' öncesi (kapalı)"),
    "ɔ": (_V, "o (normal)"),
    "ø": (_V, "ö, 'öğe/öğr..' özel (n=196)"),
    "œ": (_V, "ö (normal)"),
    "u": (_V, "u, 'ğ' öncesi / sözcük sonu (n=124)"),
    "U": (_V, "u, GEVŞEK (ʊ benzeri): normal 'u' -- dizge ASCII kısayolu"),
    "y": (_V, "ü, 'ğ' öncesi (n=52)"),
    "Y": (_V, "ü, GEVŞEK (ʏ benzeri): normal 'ü' -- dizge ASCII kısayolu"),
    # ünsüzler
    "b": (_C, ""), "p": (_C, ""), "pʰ": (_C, "aspire p"),
    "t": (_C, ""), "tʰ": (_C, "aspire t"),
    "d": (_C, ""),
    "k": (_C, "arka k"), "kʰ": (_C, "aspire k"),
    "c": (_C, "ön (palatal) k"), "cʰ": (_C, "aspire ön k"),
    "g": (_C, "arka g"), "ɟ": (_C, "ön (palatal) g"),
    "f": (_C, ""), "v": (_C, ""), "ʋ": (_C, "v (ünlü yanı, yaklaştırıcı)"),
    "s": (_C, ""), "z": (_C, ""), "z̥": (_C, "sessizleşmiş z (sözcük sonu)"),
    "ʃ": (_C, "ş"), "ʒ": (_C, "j"),
    "tʃ": (_C, "ç (affrikat, tek atom)"), "dʒ": (_C, "c (affrikat, tek atom)"),
    "x": (_C, "h (arka)"), "ç": (_C, "h (ön, damak)"),
    "ɣ": (_C, "SÖZCÜK SONU r (dizge phonology.py: baş r->r, son r->ɣ, diğer r->ɾ). Dilbilimsel olarak tartışmalı: gerçek son r sesi ɣ değil, sessizleşen ɹ/ɾ"),
    "m": (_C, ""), "ɱ": (_C, "dudak-diş m (n=3, nadir)"),
    "n": (_C, ""), "ŋ": (_C, "n (ön-'k/g'), 'ng'"),
    "l": (_C, "ön l"), "ł": (_C, "arka (velarize) l"),
    "r": (_C, "titrek r (nadir, n=754)"), "ɾ": (_C, "r (flap, normal)"),
    "j": (_C, "y"),
}
LENGTH = "ː"  # ünlü uzunluğu: 'ɑː','iː','oː','uː','yː','ɛː','øː','Uː' -> uzun ünlü atomları aşağıda
LONG_VOWELS = tuple(v + LENGTH for v in ("ɑ", "e", "ɛ", "i", "I", "ɨ", "o", "ɔ", "ø", "œ", "u", "U", "y", "Y", "a"))
for _lv in LONG_VOWELS:
    PHONES[_lv] = (_V, "uzun ünlü")
# ğ uzatması ünlü bulunmayan bağlamda tek başına 'ː' çıkabilir (model hataları + 'mʰːːzː' gibi)
PHONES[LENGTH] = ("mod", "bağımsız uzatma (ünlüsüz bağlam; genelde MODEL HATASI belirtisi)")

# En uzun eşleşme için uzunluğa göre azalan sıra
_ATOMS_BY_LEN = sorted(PHONES, key=len, reverse=True)

SYMBOLS: list[str] = [PAD, WORD_SEP, *PAUSES, STRESS, *sorted(PHONES), *BREAKS]  # BREAKS SONA eklendi: eski id'ler (73 sembollük koşular) değişmez
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
assert len(SYMBOLS) == len(SYMBOL_TO_ID)


class UnknownSymbol(ValueError):
    pass


def tokenize(phonemes: str, strict: bool = True) -> list[str]:
    """Fonem dizgesini atom listesine böler (en uzun eşleşme). Bilinmeyen karakter -> UnknownSymbol
    (strict) ya da atlanır (strict=False; çağıran bilinmeyenleri raporlamalı)."""
    out: list[str] = []
    i = 0
    while i < len(phonemes):
        for a in _ATOMS_BY_LEN:
            if phonemes.startswith(a, i):
                out.append(a)
                i += len(a)
                break
        else:
            ch = phonemes[i]
            if ch in (WORD_SEP, STRESS, *PAUSES):
                out.append(ch)
            elif strict:
                raise UnknownSymbol(f"{ch!r} (U+{ord(ch):04X}) in {phonemes!r}")
            i += 1
    return out


def to_ids(tokens: list[str]) -> list[int]:
    return [SYMBOL_TO_ID[t] for t in tokens]


if __name__ == "__main__":
    assert tokenize("tʃIkɔłɑtɑ") == ["tʃ", "I", "k", "ɔ", "ł", "ɑ", "t", "ɑ"]
    assert tokenize("ɑːtʃ") == ["ɑː", "tʃ"]
    assert tokenize("jɑłnɨz̥") == ["j", "ɑ", "ł", "n", "ɨ", "z̥"]
    assert tokenize("kʰɑxvɑłtɨ")[0] == "kʰ"
    print(f"{len(SYMBOLS)} sembol, {len(PHONES)} fonem atomu -- OK")
