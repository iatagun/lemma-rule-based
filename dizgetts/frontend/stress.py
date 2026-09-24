"""Kural tabanlı sözcük vurgusu (M1a). Dilbilimsel bilgi: docs/turkish_phonology.md; veri: resources/*.tsv.

Kural sırası (ilk eşleşen kazanır; her kural kendi etiketiyle sayılır):
  1. ünlüsüz parça                        -> vurgu yok
  2. clitic sözcük (da, ki, bile, mI...)  -> vurgu yok (resources/clitics.tsv)
  3. düzensiz vurgulu KÖK                 -> kökün belirtilen seslemesi (resources/stress_roots.tsv; kök tam sözcük ya da "kök + makul ek zinciri")
  --- M1b: morfolojik özellik (DizgeBERT-Morph UPOS+FEATS) KAPILI, ek sınırı yüzey biçimiyle; `tiers` ile açılıp kapatılır ---
  4. -Iyor (ön-vurgulu)                   -> "yor"dan hemen önceki ünlü (literatür; kullanıcı listesinde YOK, onaysız)
  5. olumsuz fiil (Polarity=Neg)          -> olumsuzluk -mA'dan önceki seslem   (kullanıcı listesi)
  6. Case=Ins (-(y)lA)                    -> -(y)lA'dan önceki seslem          (kullanıcı listesi)
  7. sonlu fiil kişi eki (Person 1/2)     -> kişi ekinden önceki seslem        (kullanıcı listesi)
     + bileşik zaman/koşaç (-ydı -ymış -ysa -yken) -> koşaçtan önceki seslem   (kullanıcı listesi)
  8. varsayılan                           -> SON seslem
YAZILMADI: seslenme, küçültme, ikileme, belirteç ekleri (-cık -ca -casına -en -leyin -ra), -dır, -(y)ın. Bunlar için yüzey soyma tek başına
güvenilmez (okul+a / -la, kesin / -sın) ve kullanıcı etiketi/sözlüğü bekleniyor.

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

    def syllable(self, text: str, upos: str | None = None, feats: dict[str, str] | None = None, tiers=()) -> tuple[int | None, str]:
        """(vurgulu seslemin BAŞTAN indeksi ya da None, uygulanan kural). `tiers` boşsa (varsayılan) yalnız M1a kuralları; morfolojik katmanlar açıkça istenir (TIERS)."""
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
        mr = _morph_rule(w, upos, feats, tiers)
        if mr and mr[0] < n:
            return mr
        return n - 1, "varsayılan_son"


def _vowels_before(w: str, pos: int) -> int:
    """w[:pos] içindeki ünlü harf sayısı."""
    return _n_vowels(w[:pos])


# kişi ekleri (yüzey adayları). Ekin kendisi vurgusuzdur; vurgu ekten ÖNCEKİ seslemde. Aday, ancak kalan gövde geçerli bir zaman/kip ekiyle bitiyorsa
# seçilir ("geldim" -> "gel-di"+"m", "geld"+"im" DEĞİL; "gelirim" -> "gelir"+"im").
_PERSON = {
    ("1", "Sing"): ("ım", "im", "um", "üm", "yım", "yim", "yum", "yüm", "m"),
    ("2", "Sing"): ("sın", "sin", "sun", "sün", "n"),
    ("1", "Plur"): ("ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz", "k"),
    ("2", "Plur"): ("sınız", "siniz", "sunuz", "sünüz", "nız", "niz", "nuz", "nüz", "ınız", "iniz", "unuz", "ünüz"),
}
_TENSE_END = re.compile(r"(?:[dt][ıiuü]|m[ıiuü]ş|[ae]r|[ıiuü]r|[ae]c[ae][kğ]|yor|s[ae]|m[ae]kt[ae]|[ae]|z)$")


def _strip_person(w: str, person: str, number: str) -> str | None:
    for suf in sorted(_PERSON.get((person, number), ()), key=len, reverse=True):
        if w.endswith(suf):
            base = w[: -len(suf)]
            if _n_vowels(base) >= 2 and _TENSE_END.search(base):
                return base
    return None
# koşaç/bileşik zaman: -ydı -ymış -ysa -yken; zaman/kip ekinden SONRA gelirse koşaçtır (yoksa düz -dı geçmiş zaman ekidir ve vurgulanır)
_COPULA = re.compile(r"(?:m[ıiuü]ş|[ae]r|[ıiuü]r|[ae]c[ae][kğ]|s[ae]|m[ae]kt[ae]|yor)(y?(?:d[ıiuü]|t[ıiuü]|m[ıiuü]ş|s[ae]|ken))$")
_NEG = re.compile(r"m[ae](?=$|[zdtykmnlcrsğ]|[ıiuü])")
_YOR = re.compile(r"([aeıioöuü])yor")

TIERS = ("yor", "neg", "ins", "person")


def _morph_rule(w: str, upos: str | None, feats: dict[str, str] | None, tiers) -> tuple[int, str] | None:
    """M1b: (vurgulu seslem indeksi BAŞTAN, kural etiketi) ya da None. w = küçük harfli yazım."""
    verbal = upos in ("VERB", "AUX")
    if "yor" in tiers and (upos is None or verbal):
        m = _YOR.search(w)
        if m:
            return _vowels_before(w, m.start(1)), "yor_önü"
    if feats is None:
        return None
    if "neg" in tiers and verbal and feats.get("Polarity") == "Neg":
        for m in _NEG.finditer(w):
            if m.start() >= 1 and _vowels_before(w, m.start()) >= 1:
                return _vowels_before(w, m.start()) - 1, "olumsuzluk_önü"
    if "ins" in tiers and feats.get("Case") == "Ins":
        m = re.search(r"y?l[ae]$", w)
        if m and _vowels_before(w, m.start()) >= 1:
            return _vowels_before(w, m.start()) - 1, "ins_önü"
    if "person" in tiers and upos == "VERB" and feats.get("VerbForm") in (None, "Fin"):
        base, stripped = w, False
        if feats.get("Person") in ("1", "2"):
            sb = _strip_person(w, feats["Person"], feats.get("Number", "Sing"))
            if sb is not None:
                base, stripped = sb, True
        cm = _COPULA.search(base)
        if cm and _vowels_before(base, cm.start(1)) >= 1:
            return _vowels_before(base, cm.start(1)) - 1, "koşaç_önü"
        if stripped:
            return _n_vowels(base) - 1, "kişi_eki_önü"
    return None


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
