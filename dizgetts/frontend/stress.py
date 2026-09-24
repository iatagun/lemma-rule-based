"""Kural tabanlı sözcük vurgusu (M1a). Dilbilimsel bilgi: docs/turkish_phonology.md; veri: resources/*.tsv.

Kural sırası (ilk eşleşen kazanır; her kural kendi etiketiyle sayılır):
  1. ünlüsüz parça                        -> vurgu yok
  2. clitic sözcük (da, ki, bile, mI...)  -> vurgu yok (resources/clitics.tsv)
  3. düzensiz vurgulu KÖK                 -> kökün belirtilen seslemesi (resources/stress_roots.tsv; kök tam sözcük ya da "kök + makul ek zinciri")
     Sözlükte sıra yerine "ağırlık" yazılırsa seslem AĞIRLIĞINDAN hesaplanır (kullanıcı kuralı 2026-09-24, weight_stress):
       ağır (H) = uç ünsüzü dolu ya da çekirdeği uzun ünlü (â î û, ğ uzatması); hafif (L) = kısa ünlü, uç ünsüzsüz.
       -en belirteçleri (nak-len, e-SÂ-sen, NİS-pe-ten): sondan 2. seslem H ise o, L ise sondan 3.
       alıntı / yer adı (güçlü-zayıf sözcük): sondan 2. H ise o; değilse sondan 3. H ise o; ikisi de L (zayıf) ise sondan 2.
     Uzun ünlü yazıda görünmez (esasen) ve dizge de işaretlemez -> sözlük biçimi şapkayla yazılır (esâsen); eşleme şapkasız yapılır.
  --- M1b: morfolojik özellik (DizgeBERT-Morph UPOS+FEATS) KAPILI, ek sınırı yüzey biçimiyle; `tiers` ile açılıp kapatılır ---
  4a. pekiştirme sıfatı (kıp+kırmızı, ap+açık, ter+temiz)  -> İLK seslem   (kullanıcı kuralı)
  4b. -CIk türemiş sıfat (ince-cik, ufa-cık, küçü-cük)      -> İLK seslem   (kullanıcı kuralı; addaki küçültme -cık DEĞİL)
      4a/4b UPOS'a DEĞİL sıfat sözlüğüne kapılı (resources/adj_lemmas.txt, UD ADJ lemmaları): taban gerçek bir sıfat olmalı. UPOS kapısı UD'de
      arasında/emekli/usulca/serseri'yi pekiştirme sayıyordu (g2ptts-v1 etiket gürültüsü); sözlük kapısı Morph'suz da çalışır (g2ptts tagger).
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


_PLAIN = str.maketrans("âîû", "aiu")


def syllabify(w: str) -> list[str]:
    """Basit Türkçe hece bölme: her ünlü bir çekirdek; ünlüler arası tek ünsüz sonraki heceye, 2+ ünsüzde ilki öncekine."""
    v = [i for i, c in enumerate(w) if c in VOWEL_LETTERS]
    if len(v) <= 1:
        return [w]
    cuts = []
    for a, b in zip(v, v[1:]):
        gap = b - a - 1
        cuts.append(b if gap == 0 else b - 1 if gap == 1 else a + 2)  # ünlü-ünlü: araya; tek ünsüz sonraki heceye; 2+ ünsüzde ilki öncekine
    parts, start = [], 0
    for c in cuts:
        parts.append(w[start:c]); start = c
    parts.append(w[start:])
    return parts


def weight_stress(root: str, en: bool) -> int:
    """Seslem ağırlığından vurgulu seslemin BAŞTAN indeksi (kural metni: modül başı, madde 3). en=True: -en belirteç kuralı."""
    s = syllabify(root)
    n = len(s)
    if n < 2:
        return 0
    heavy = [s[i][-1] not in VOWEL_LETTERS or any(c in "âîû" for c in s[i]) or (i + 1 < n and s[i + 1][0] == "ğ") for i in range(n)]
    p = n - 2
    if n == 2 or heavy[p]:
        return p
    return p - 1 if en or heavy[p - 1] else p


class StressRules:
    def __init__(self):
        self.roots: dict[str, tuple[int, str]] = {}
        self._cap_only: set[str] = set()  # yer adı kategorisi: yalnız BÜYÜK harfle başlayan yazımda (ordu/Ordu, bebek/Bebek ayrımı)
        for r in _rows(RES / "stress_roots.tsv"):
            lex, cat = tr_lower(r[0]), r[2]
            idx = weight_stress(lex, "-en" in cat) if r[1] == "ağırlık" else int(r[1])
            root = lex.translate(_PLAIN)
            if not 0 <= idx < _n_vowels(root):
                raise ValueError(f"stress_roots.tsv: '{root}' için seslem indeksi {idx} geçersiz (ünlü sayısı {_n_vowels(root)})")
            self.roots[root] = (idx, cat)
            if cat.startswith("yer adı"):
                self._cap_only.add(root)
        self.clitics = {tr_lower(r[0]) for r in _rows(RES / "clitics.tsv")}
        self._roots_longest_first = sorted(self.roots, key=len, reverse=True)
        self.adj = {l for l in (RES / "adj_lemmas.txt").read_text(encoding="utf8").splitlines() if l and not l.startswith("#")}

    @staticmethod
    def version() -> str:
        h = hashlib.sha1()
        for f in ("stress_roots.tsv", "clitics.tsv", "adj_lemmas.txt"):
            h.update((RES / f).read_bytes())
        h.update(Path(__file__).read_bytes())
        return h.hexdigest()[:8]

    def syllable(self, text: str, upos: str | None = None, feats: dict[str, str] | None = None, tiers=()) -> tuple[int | None, str]:
        """(vurgulu seslemin BAŞTAN indeksi ya da None, uygulanan kural). `tiers` boşsa (varsayılan) yalnız M1a kuralları; morfolojik katmanlar açıkça istenir (TIERS)."""
        w = tr_lower(text).translate(_PLAIN)
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
        mr = _morph_rule(w, upos, feats, tiers, self.adj)
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

# pekiştirme: (ünsüz*)ünlü + p/m/r/s + taban; taban ilk seslemi tekrar eder (kıp+kırmızı, ap+açık, ter+temiz, mas+mavi)
_PEK = re.compile(r"^([^aeıioöuü]*[aeıioöuü])[pmrs](.+)$")
_CIK = re.compile(r"[cç][ıiuü][kğ]")

TIERS = ("pek", "cik", "yor", "neg", "ins", "person")


def _adj_base(s: str, adj: set[str]) -> bool:
    """s bir sıfat tabanı mı: tam sözcük ya da sıfat + makul ek zinciri (kıpkırmızı-ydı)."""
    return s in adj or any(s.startswith(a) and _SUFFIX_CHAIN.match(s[len(a):]) for a in adj if len(a) >= 4 and s[:4] == a[:4])  # dul+ar, ter+e değil


def _cik_stem_is_adj(stem: str, adj: set[str]) -> bool:
    """-CIk öncesi gövde bir sıfattan mı: ince(cik), küçü+k(cük), ufa+k(cık), dar+a(cık), genç~gence(cik), az+ı(cık)."""
    cands = {stem, stem + "k", stem[:-1], stem[:-1].replace("c", "ç")} if stem[-1:] in VOWEL_LETTERS else {stem}
    return any(len(c) >= 2 and c in adj for c in cands)


def _morph_rule(w: str, upos: str | None, feats: dict[str, str] | None, tiers, adj: set[str] = frozenset()) -> tuple[int, str] | None:
    """M1b: (vurgulu seslem indeksi BAŞTAN, kural etiketi) ya da None. w = küçük harfli yazım."""
    if "pek" in tiers:
        m = _PEK.match(w)
        g1, base = (m.group(1), m.group(2)) if m else ("", "")
        # taban ilk seslemi tekrar eder; araya giren ünsüz tabanın o noktadaki ünsüzünden FARKLI (ser+seri, sersem pekiştirme değil)
        ok_cons = w[len(g1)] == "p" if g1 in VOWEL_LETTERS else base[len(g1):len(g1) + 1] != w[len(g1)]  # ünlü başlı taban yalnız p alır: ap+açık, up+uzun (a+r+ada değil)
        if m and base.startswith(g1) and _n_vowels(base) >= 2 and ok_cons and _adj_base(base, adj):
            return 0, "pekiştirme"
    if "cik" in tiers:
        m = _CIK.search(w)
        if m and _vowels_before(w, m.start()) >= 2 and _cik_stem_is_adj(w[:m.start()], adj):  # küçük (kü+çük) -CIk DEĞİL
            return 0, "cık_sıfat"
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
