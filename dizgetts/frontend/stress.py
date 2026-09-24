"""Kural tabanlı sözcük vurgusu (M1a). Dilbilimsel bilgi: docs/turkish_phonology.md; veri: resources/*.tsv.

Şimdiki kapsam (bilerek dar; ölçülmeden genişletilmez):
  1. ünlüsüz parça                  -> vurgu yok
  2. clitic sözcük (da, ki, bile, mI...) -> vurgu yok (resources/clitics.tsv)
  3. düzensiz vurgulu KÖK (resources/stress_roots.tsv; kök tam sözcük ya da "kök + makul ek zinciri") -> kökün belirtilen seslemesi
  4. varsayılan                     -> SON seslem
YAZILMADI (M1b, morfolojik çözümleme gerektirir): vurgusuz ekler (-ydı -ymış -ysa -yken -dır -(y)la -cık -ca ... kişi ekleri, olumsuzluk -mA),
seslenme, küçültme, ikileme. Yüzey biçimine bakarak ek soymak GÜVENİLMEZ (okul+a / -la, kesin / -sın); o yüzden yapılmadı.

Seslem = ünlü HARFİ (Türkçede her ünlü harf bir çekirdek). Vurgulu seslem indeksi sonra dizge fonem dizisindeki ünlü atomuna eşlenir
(dizge 'ay' -> 'ɑːI' gibi yan ünlü üretir; sayılar tutmazsa yan ünlü atılır, hâlâ tutmazsa sondan sayım).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .normalize import tr_lower
from .symbols import PHONES

RES = Path(__file__).resolve().parent.parent / "resources"
VOWEL_LETTERS = set("aeıioöuüâîû")
# "makul ek zinciri": (isteğe bağlı ünsüz) + ünlü + (en çok 3 ünsüz) tekrarı. 'ama'+'ç' gibi ünsüz-yalnız kalıntıyı reddeder.
_SUFFIX_CHAIN = re.compile(r"^(?:[lnrdtcçsşmyzkğbgp]?[aeıioöuüâîû][lnrdtcçsşmyzkğ]{0,3})+$")


def _rows(path: Path) -> list[list[str]]:
    return [l.rstrip("\n").split("\t") for l in path.read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]


def _n_vowels(w: str) -> int:
    return sum(c in VOWEL_LETTERS for c in w)


class StressRules:
    def __init__(self):
        self.roots: dict[str, tuple[int, str]] = {}
        self._cap_only: set[str] = set()  # yer adı kategorisi: yalnız BÜYÜK harfle başlayan yazımda (ordu/Ordu, bebek/Bebek ayrımı)
        for r in _rows(RES / "stress_roots.tsv"):
            root, idx, cat = tr_lower(r[0]), int(r[1]), r[2]
            if not 0 <= idx < _n_vowels(root):
                raise ValueError(f"stress_roots.tsv: '{root}' için seslem indeksi {idx} geçersiz (ünlü sayısı {_n_vowels(root)})")
            self.roots[root] = (idx, cat)
            if cat.startswith("yer adı"):
                self._cap_only.add(root)
        self.clitics = {tr_lower(r[0]) for r in _rows(RES / "clitics.tsv")}
        self._roots_longest_first = sorted(self.roots, key=len, reverse=True)

    @staticmethod
    def version() -> str:
        h = hashlib.sha1()
        for f in ("stress_roots.tsv", "clitics.tsv"):
            h.update((RES / f).read_bytes())
        h.update(Path(__file__).read_bytes())
        return h.hexdigest()[:8]

    def syllable(self, text: str) -> tuple[int | None, str]:
        """(vurgulu seslemin BAŞTAN indeksi ya da None, uygulanan kural)."""
        w = tr_lower(text)
        n = _n_vowels(w)
        if n == 0:
            return None, "ünlüsüz"
        if w in self.clitics:
            return None, "clitic"
        cap = text[:1].isupper()
        if w in self.roots and (cap or w not in self._cap_only):
            return self.roots[w][0], "kök"
        for r in self._roots_longest_first:
            if (cap or r not in self._cap_only) and w.startswith(r) and len(w) > len(r) and _SUFFIX_CHAIN.match(w[len(r):]):
                return self.roots[r][0], "kök+ek"
        return n - 1, "varsayılan_son"


def _is_vowel_atom(tok: str) -> bool:
    return tok in PHONES and PHONES[tok][0] == "ünlü"


def to_phone_index(text: str, phones: list[str], k: int) -> tuple[int | None, str]:
    """Yazımdaki k-ıncı ünlü harfe (seslem) karşılık gelen dizge ünlü atomunun `phones` içindeki indeksi."""
    w = tr_lower(text)
    n_l = _n_vowels(w)
    av = [i for i, p in enumerate(phones) if _is_vowel_atom(p)]
    if not av:
        return None, "atomsuz"
    if len(av) == n_l:
        return av[k], "eşit"
    if "y" in w:  # 'ay' -> 'ɑːI': uzun ünlüden hemen sonraki 'I' yan ünlüdür, seslem çekirdeği değil
        av2 = [i for i in av if not (phones[i] == "I" and i > 0 and phones[i - 1].endswith("ː"))]
        if len(av2) == n_l:
            return av2[k], "yan_ünlü_atıldı"
    r = n_l - 1 - k  # sondan sıra
    return (av[len(av) - 1 - r], "sondan") if 0 <= r < len(av) else (av[-1], "yedek_son")
