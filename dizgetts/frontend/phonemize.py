"""Metin -> DizgeTTS fonem token dizisi.

normalize -> sözcük/duraklama ayrıştırma -> sözcük başına fonem (birincil: kural tabanlı PyPI `dizge.g2p`; hata verirse
yedek: dizge-g2p BERT) -> symbols.tokenize. Sözcükler WORD_SEP ile, noktalama PAUSES token'ı olarak ayrılır.
Vurgu: symbols.STRESS ("ˈ") ünlünün ÖNÜNE eklenebilir; tokenize/to_ids bunu olduğu gibi geçirir (DizgeBERT-Stress gelince).
"""
from __future__ import annotations

import collections
import re
from functools import lru_cache

from .normalize import normalize, tr_lower
from .symbols import PAUSES, WORD_SEP, UnknownSymbol, tokenize

FOREIGN = {"w": "v", "x": "ks", "q": "k"}  # dizge.g2p bu harfleri SESSİZCE atıyor (xbox -> bɔ); kaba yakınsama
_TOK = re.compile(r"[^\W\d_]+|[,.?!;]")


class Phonemizer:
    def __init__(self, bert_fallback: bool = True):
        import dizge

        self._dizge = dizge
        self._bert_ok = bert_fallback
        self._bert = None
        self.stats = collections.Counter()
        self.failed: dict[str, str] = {}      # dizge hata verdi -> (BERT çıktısı ya da "")
        self.variants: dict[str, tuple] = {}  # dizge çok-varyantlı döndürdü
        self.unknown: collections.Counter = collections.Counter()

    def _from_bert(self, w: str) -> str:
        if self._bert is None:
            from .g2p import G2P

            self._bert = G2P()
        return self._bert.g2p(w)

    @lru_cache(maxsize=None)
    def word(self, w: str) -> str:
        w = tr_lower(w)
        for k, v in FOREIGN.items():
            w = w.replace(k, v)
        try:
            r = self._dizge.g2p(w)
            if not isinstance(r, str):
                self.variants[w] = tuple(r)
                r = r[0]
            self.stats["dizge"] += 1
            return r
        except Exception:
            self.stats["dizge_fail"] += 1
            r = self._from_bert(w) if self._bert_ok else ""
            self.failed[w] = r
            return r

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
            ph = self.word(t)
            try:
                toks += tokenize(ph)
            except UnknownSymbol:
                for ch in ph:
                    self.unknown[ch] += 1
                toks += tokenize(ph, strict=False)
            self.stats["words"] += 1
        return norm, toks
