"""Metin -> DizgeTTS fonem token dizisi.

normalize -> sözcük/duraklama ayrıştırma -> sözcük başına fonem (birincil: kural tabanlı PyPI `dizge.g2p`; hata verirse
yedek: dizge-g2p BERT) -> symbols.tokenize. Sözcükler WORD_SEP ile, noktalama PAUSES token'ı olarak ayrılır.
Vurgu: symbols.STRESS ("ˈ") ünlünün ÖNÜNE eklenebilir; tokenize/to_ids bunu olduğu gibi geçirir (DizgeBERT-Stress gelince).
"""
from __future__ import annotations

import collections
import re
import unicodedata
import warnings

from .normalize import normalize, tr_lower
from .symbols import PAUSES, WORD_SEP, tokenize

# dizge.g2p Türkçe alfabe dışındaki harfleri SESSİZCE atıyor (xbox -> bɔ, café -> dʒɑf); kaba yakınsama tablosu. Kalan aksanlı harfler NFD taban harfine
# düşer (é -> e, ó -> o); hâlâ eşlenemeyen (Kiril, CJK...) harf atılır ve RuntimeWarning verilir.
FOREIGN = {"w": "v", "x": "ks", "q": "k", "ä": "e", "æ": "e", "ø": "ö", "œ": "ö", "å": "a", "ß": "ss", "ñ": "ny", "š": "ş", "č": "ç", "ž": "j",
           "ł": "l", "đ": "d"}
TURKISH = frozenset("abcçdefgğhıijklmnoöprsştuüvyzâîû")
_TOK = re.compile(r"[^\W\d_]+|[,.?!;]")


def fold_foreign(w: str) -> tuple[str, list[str]]:
    """Küçük harfli sözcük -> (dizge'nin işleyebileceği yazım, atılan karakterler)."""
    out, dropped = [], []
    for c in w:
        if c in TURKISH:
            out.append(c)
            continue
        c = FOREIGN.get(c) or unicodedata.normalize("NFD", c)[0]
        c = FOREIGN.get(c, c)
        if all(x in TURKISH for x in c):
            out.append(c)
        else:
            dropped.append(c)
    return "".join(out), dropped


class Phonemizer:
    def __init__(self, bert_fallback: bool = True, pron_exceptions: bool = False, register: str = "özenli", length_rules: bool = False, res_dir=None):
        import dizge
        from .pronounce import Atoms, Exceptions, LengthRules

        self._Atoms = Atoms

        self._dizge = dizge
        self._bert_ok = bert_fallback
        self._exc = Exceptions(register, res_dir) if pron_exceptions else None  # söyleyiş istisna sözlüğü (varsayılan KAPALI: eski checkpoint/manifestler)
        self._len = LengthRules(register) if length_rules else None            # uzun ünlü (ː) kuralları: ğ/y (varsayılan KAPALI)
        self._bert = None
        self.stats = collections.Counter()
        self.dropped: collections.Counter = collections.Counter()  # atılan (eşlenemeyen) karakterler
        self._trace: dict[str, object] = {}     # sözcük -> Atoms (yalnız sözlük/kural değiştirdiyse): atom kökeni, vurgu eşlemesi için
        self._cache: dict[str, str] = {}      # örnek başına (metot üstünde lru_cache self'i sonsuza dek tutar)
        self.failed: dict[str, str] = {}      # dizge hata verdi -> (BERT çıktısı ya da "")
        self.variants: dict[str, tuple] = {}  # dizge çok-varyantlı döndürdü
        self.unknown: collections.Counter = collections.Counter()
        self.unknown_syms: dict[str, list[str]] = {}  # sözcük -> okumasından atılan (sembol tablosunda olmayan) simgeler

    def _from_bert(self, w: str) -> str:
        if self._bert is None:
            from .g2p import G2P

            self._bert = G2P()
        return self._bert.g2p(w)

    def word(self, w: str) -> str:
        if w in self._cache:
            return self._cache[w]
        key = tr_lower(w)
        f, dropped = fold_foreign(key)
        for c in dropped:
            self.dropped[c] += 1
            warnings.warn(f"fonemleştirme: {c!r} (U+{ord(c[0]):04X}) {w!r} sözcüğünden atıldı (Türkçe alfabede ve eşleme tablosunda yok)", RuntimeWarning, stacklevel=2)
        r = ""
        if f:
            try:
                r = self._dizge.g2p(f)
                if not isinstance(r, str):
                    self.variants[key] = tuple(r)
                    r = r[0]
                self.stats["dizge"] += 1
            except Exception:
                self.stats["dizge_fail"] += 1
                r = self._from_bert(f) if self._bert_ok else ""
                self.failed[key] = r
        if r:  # sembol tablosunda olmayan simge konuşmadan düşer: sessiz bırakma (Atoms strict=False ile atar; sözlük açıkken iz kalmıyordu)
            bad: list[str] = []
            tokenize(r, strict=False, unknown=bad)
            if bad:
                self.unknown_syms[w] = bad
                warnings.warn(f"fonemleştirme: {w!r} okumasındaki ({r!r}) {bad} sembol tablosunda yok; atıldı", RuntimeWarning, stacklevel=2)
        if r and (self._exc is not None or self._len is not None):
            at = self._Atoms(r)
            if self._exc is not None:
                self._exc.apply_atoms(key, at)
            if self._len is not None:  # sıra: sözlük -> kural (kağıt: sözlük ön a'yı koyar, kural ğ geçişinde ː'yi düşürür)
                if self._len.apply_atoms(key, at) == "hizalanamadı":
                    self.stats["uzun_hizalanamadi"] += 1
            if at.changed:
                r = at.text()
                self._trace[w] = at
        if not r:  # sözcük konuşmadan düşer: sessiz bırakma
            self.stats["fonemsiz"] += 1
            warnings.warn(f"fonemleştirme: {w!r} için fonem üretilemedi; sözcük konuşulmayacak", RuntimeWarning, stacklevel=2)
        self._cache[w] = r
        return r

    def trace(self, w: str):
        """`word(w)` sözlük/kuralla DEĞİŞTİYSE Atoms (raw = dizge'nin ham atomları, origin = kökenler), değişmediyse None."""
        return self._trace.get(w)

    def __call__(self, text: str) -> tuple[str, list[str]]:
        """-> (normalize edilmiş metin, token listesi). Bilinmeyen karakterler atlanır ve self.unknown'a sayılır."""
        norm = normalize(text)
        toks: list[str] = []
        for m in _TOK.finditer(norm):
            t = m.group()
            if t in PAUSES:
                toks.append(t)
                continue
            if toks:
                toks.append(WORD_SEP)  # noktalamadan önce ayraç yok, sonra var: "a" "," " " "b"
            toks += tokenize(self.word(t), strict=False)
            self.unknown.update(self.unknown_syms.get(t, ()))
            self.stats["words"] += 1
        return norm, toks
