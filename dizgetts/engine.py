"""DizgeTTS ses motoru: metin -> Utterance (aşama aşama) -> model token'ları -> (akustik model + vocoder) -> wav.

Aşamalar `Utterance -> Utterance` çalışır; eğitim manifestleri ve çıkarım AYNI kodu kullanır (parity testi: tests/test_engine_parity.py).
  1. normalize   : sayı/kısaltma/noktalama düzeni            (frontend/normalize.py)
  2. phonemize   : sözcük başına dizge fonemleri              (frontend/phonemize.py, PyPI dizge==0.1.6, BERT yedeği)
  3. stress      : sözcük vurgusu (frontend/stress.py, resources/*.tsv; M1a: kök sözlüğü + clitic + varsayılan son seslem)
  4. phrase      : cümle düzeyi sınır (Word.boundary: 0|ip|IP|cümle); iskelet, şimdilik yalnız noktalama, token'a dokunmaz
  5. assemble    : sözcükler + vurgu + noktalama -> token listesi (symbols.py)

  python -m dizgetts.engine "Merhaba, 3'te buluşalım."                     # aşama çıktıları
  python -m dizgetts.engine "Merhaba." --ckpt D:/.../ep150.pt -o out.wav   # ses (checkpoint gerekir)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import re
from dataclasses import dataclass, field

from dizgetts.frontend import normalize as _norm
from dizgetts.frontend.phonemize import Phonemizer
from dizgetts.frontend.stress import StressRules, to_phone_index
from dizgetts.frontend.symbols import PAUSES, PHONES, STRESS, SYMBOL_TO_ID, WORD_SEP, UnknownSymbol, tokenize

_TOK = re.compile(r"[^\W\d_]+|[,.?!;]")


@dataclass
class Word:
    text: str                                   # normalize çıktısındaki yazım
    phones: list[str] = field(default_factory=list)   # dizge atomları (vurgu simgesi HARİÇ)
    stress: int | None = None                   # vurgulu ünlünün `phones` içindeki indeksi (None: vurgusuz); M1'de dolar
    stress_src: str | None = None               # vurguyu belirleyen kural etiketi (frontend/stress.py: kök, clitic, varsayılan_son...)
    punct: list[str] = field(default_factory=list)    # sözcüğü izleyen noktalama token'ları
    boundary: str = "0"                         # sonraki sınır: "0" | "ip" | "IP" | "cümle" (PhraseStage)
    notes: list[str] = field(default_factory=list)    # sözcük düzeyi açıklamalar (hangi aşama neyi neden değiştirdi)
    upos: str | None = None                     # DizgeBERT-Morph (M1b; morfoloji aşaması açıksa)
    feats: dict | None = None


@dataclass
class Utterance:
    raw: str
    norm: str = ""
    words: list[Word] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)   # model girdisi
    meta: dict = field(default_factory=dict)          # uyarılar / sayaçlar (bilinmeyen sembol vb.)


class Stage:
    name = "stage"

    def version(self) -> str:
        return hashlib.sha1(inspect.getsource(type(self)).encode("utf8")).hexdigest()[:8]

    def __call__(self, u: Utterance) -> Utterance:
        raise NotImplementedError


class NormalizeStage(Stage):
    name = "normalize"

    def version(self) -> str:
        return hashlib.sha1(inspect.getsource(_norm).encode("utf8")).hexdigest()[:8]

    def __call__(self, u: Utterance) -> Utterance:
        u.norm = _norm.normalize(u.raw)
        return u


class PhonemeStage(Stage):
    name = "phonemize"

    def __init__(self, bert_fallback: bool = True):
        self.ph = Phonemizer(bert_fallback=bert_fallback)

    def version(self) -> str:
        return f"dizge=={md.version('dizge')}"

    def __call__(self, u: Utterance) -> Utterance:
        for m in _TOK.finditer(u.norm):
            t = m.group()
            if t in PAUSES:
                if u.words:  # normalize() baştaki noktalamayı zaten atar
                    u.words[-1].punct.append(t)
                continue
            ph = self.ph.word(t)
            try:
                atoms = tokenize(ph)
            except UnknownSymbol:
                u.meta.setdefault("unknown_chars", []).extend(c for c in ph if c not in "".join(PHONES))
                atoms = tokenize(ph, strict=False)
            u.words.append(Word(text=t, phones=atoms))
        return u


class MorphStage(Stage):
    """DizgeBERT-Morph UPOS+FEATS (frontend/morph.py) -> Word.upos/feats. Yalnız M1b vurgu katmanları için gerekir (Engine(morph=True)).
    `cache`: {normalize edilmiş cümle: [(UPOS, FEATS)...] belirteç başına} — toplu üretimde (build_manifest) önbellekten okur."""
    name = "morph"

    def __init__(self, cache: dict | None = None):
        self.cache = cache or {}
        self._an = None

    def version(self) -> str:
        from dizgetts.frontend.morph import MODEL_ID, SCHEME
        return f"{MODEL_ID}/{SCHEME}"

    def __call__(self, u: Utterance) -> Utterance:
        toks = u.norm.split()
        res = self.cache.get(u.norm)
        if res is None:
            if self._an is None:
                from dizgetts.frontend.morph import MorphAnalyzer
                self._an = MorphAnalyzer("cpu")
            res = self._an.analyze(toks)
        it = iter(w for w in u.words)
        for t, (pos, feats) in zip(toks, res):
            if t.isalpha():
                w = next(it, None)
                if w is None or w.text != t:
                    u.meta["morph_hizalama_hatasi"] = True
                    break
                w.upos, w.feats = pos, feats
        return u


class StressStage(Stage):
    """Kural tabanlı vurgu (frontend/stress.py, resources/*.tsv). Kural ve eşleme sayaçları u.meta["stress"]'e yazılır."""
    name = "stress"

    def __init__(self, tiers=()):
        self.rules = StressRules()
        self.tiers = tuple(tiers)  # M1b katmanları (frontend.stress.TIERS); varsayılan () = M1a (eğitilmiş v2_m1a modeliyle uyumlu)

    def version(self) -> str:
        return self.rules.version() + ("+" + ",".join(self.tiers) if self.tiers else "")

    def __call__(self, u: Utterance) -> Utterance:
        cnt = u.meta.setdefault("stress", {})
        for w in u.words:
            k, tag = self.rules.syllable(w.text, w.upos, w.feats, self.tiers)
            cnt[tag] = cnt.get(tag, 0) + 1
            w.stress_src = tag
            if k is None:
                continue
            idx, how = to_phone_index(w.text, w.phones, k)
            cnt["eşleme_" + how] = cnt.get("eşleme_" + how, 0) + 1
            w.stress = idx
        return u


class PhraseStage(Stage):
    """Cümle düzeyi: sözcükler arası sınır (Word.boundary). İSKELET: yalnız noktalamadan okur, token'lara dokunmaz (parity korunur).
    Sonraki katmanlar (Dep öbek sınırı, işlev sözcüğü vurgusuzlaşması, sözcükler arası ses etkisi) buraya gelir, her biri Word.notes'a yazar."""
    name = "phrase"
    _PUNCT_BOUNDARY = {",": "ip", ";": "IP", ".": "cümle", "?": "cümle", "!": "cümle"}
    _RANK = ("0", "ip", "IP", "cümle")

    def __call__(self, u: Utterance) -> Utterance:
        for w in u.words:
            w.boundary = max((self._PUNCT_BOUNDARY[p] for p in w.punct), key=self._RANK.index, default="0")
        if u.words:
            u.words[-1].boundary = "cümle"
        return u


class AssembleStage(Stage):
    name = "assemble"

    def __call__(self, u: Utterance) -> Utterance:
        toks: list[str] = []
        for i, w in enumerate(u.words):
            if i:
                toks.append(WORD_SEP)
            for j, p in enumerate(w.phones):
                if w.stress == j:
                    toks.append(STRESS)
                toks.append(p)
            toks.extend(w.punct)
        u.tokens = toks
        return u


class Engine:
    def __init__(self, bert_fallback: bool = True, morph: bool = False, tiers=(), morph_cache: dict | None = None):
        st: list[Stage] = [NormalizeStage(), PhonemeStage(bert_fallback)]
        if morph:
            st.append(MorphStage(morph_cache))
        self.stages: list[Stage] = st + [StressStage(tiers), PhraseStage(), AssembleStage()]
        self._acoustic = None

    # --- ön uç
    def frontend(self, text: str) -> Utterance:
        u = Utterance(raw=text)
        for s in self.stages:
            u = s(u)
        return u

    def ids(self, u: Utterance) -> list[int]:
        return [SYMBOL_TO_ID[t] for t in u.tokens]

    def versions(self) -> dict:
        return {s.name: s.version() for s in self.stages}

    # --- ses (checkpoint gerekir; eval/synth.py Synth'i sarar)
    def speak(self, text: str, ckpt: str, device: str = "cpu", **kw):
        from dizgetts.eval.synth import Synth

        if self._acoustic is None or self._acoustic[0] != (ckpt, device):
            self._acoustic = ((ckpt, device), Synth(ckpt, device, engine=self))
        return self._acoustic[1](text, **kw)


def _show(u: Utterance) -> str:
    return json.dumps(dict(norm=u.norm, tokens="".join(u.tokens).replace(" ", "␣"),
                           words=[dict(text=w.text, phones="".join(w.phones), stress=w.stress, stress_src=w.stress_src,
                                       punct="".join(w.punct), boundary=w.boundary, notes=w.notes) for w in u.words],
                           meta=u.meta), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--ckpt")
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    e = Engine()
    u = e.frontend(a.text)
    print(_show(u))
    print("sürümler:", e.versions())
    if a.ckpt:
        import soundfile as sf

        wav, *_ = e.speak(a.text, a.ckpt, a.device)
        sf.write(a.out, wav, 22050, subtype="PCM_16")
        print("->", a.out)


if __name__ == "__main__":
    main()
