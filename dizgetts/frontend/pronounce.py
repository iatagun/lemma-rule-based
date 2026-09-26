"""Söyleyiş istisna sözlüğü: dizge fonemlerinin üstüne, kök + ek zinciriyle eşleşen kalıp değiştirme (resources/pronunciation_exceptions.tsv).

Neden dizge'nin kendi `lexicon.loans` sözlüğü yetmiyor: yalnız TAM sözcüğü eşler; kağıt düzeltmesi kağıdı'da (t -> d, ünlü öncesi) kayboluyordu.
Satır: kökler(virgülle) <TAB> kayıt (özenli|gündelik) <TAB> kalıp (atomlar, boşlukla) <TAB> yerine <TAB> not. Kalıp sözcüğün BAŞINDAKİ atomlara uygulanır
(kökler sözcük başındadır). Özenli satırlar her zaman, gündelik satırlar yalnız register="gündelik" iken uygulanır (özenli sonucun üstüne; sıra: özenli, gündelik).
Kalıp bulunamazsa ve sonuç zaten `yerine` değilse (dizge sürümü/çekim beklenenden farklı) RuntimeWarning: sessiz uyuşmazlık olmaz.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

from .normalize import tr_lower
from .stress import _SUFFIX_CHAIN
from .symbols import PHONES, tokenize

TSV = Path(__file__).resolve().parent.parent / "resources" / "pronunciation_exceptions.tsv"
REGISTERS = ("özenli", "gündelik")


class Exceptions:
    def __init__(self, register: str = "özenli"):
        if register not in REGISTERS:
            raise ValueError(f"register {REGISTERS} içinden olmalı: {register!r}")
        self.register = register
        rows = []
        for n, line in enumerate(TSV.read_text(encoding="utf8").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            stems, reg, frm, to, *_ = line.split("\t") + [""]
            frm, to = tuple(frm.split()), tuple(to.split())
            if reg not in REGISTERS or not frm or not to or not all(a in PHONES for a in frm + to):
                raise ValueError(f"{TSV.name}:{n}: geçersiz satır {line!r}")
            rows.append((tuple(tr_lower(s) for s in stems.split(",")), reg, frm, to, line.split("\t")[-1] if line.count("\t") >= 4 else ""))
        self.rows = sorted(rows, key=lambda r: REGISTERS.index(r[1]))  # kararlı: özenli önce

    def version(self) -> str:
        h = hashlib.sha1(TSV.read_bytes())
        h.update(Path(__file__).read_bytes())
        return f"{h.hexdigest()[:8]}/{self.register}"

    def apply(self, word: str, phones: str) -> str:
        """word: yazım (herhangi büyüklük), phones: dizge okuması (dizge). Değişiklik yoksa girdiyi aynen döndürür."""
        w = tr_lower(word)
        atoms, changed = None, False
        for stems, reg, frm, to, _ in self.rows:
            if reg == "gündelik" and self.register != "gündelik":
                continue
            if not any(w.startswith(s) and (len(w) == len(s) or _SUFFIX_CHAIN.match(w[len(s):])) for s in stems):
                continue
            if atoms is None:
                atoms = tokenize(phones, strict=False)
            if tuple(atoms[:len(frm)]) == frm:
                atoms[:len(frm)] = to
                changed = True
            elif tuple(atoms[:len(to)]) != to:
                warnings.warn(f"söyleyiş istisnası: {word!r} için dizge çıktısı {' '.join(atoms)!r} beklenen kalıpla ({' '.join(frm)!r}) başlamıyor; satır atlandı",
                              RuntimeWarning, stacklevel=3)
        return "".join(atoms) if changed else phones


def coverage(min_count: int = 100, manifest: str | None = None) -> list[tuple[str, int, list[str]]]:
    """Sözlüğün ÜRETTİĞİ atomların eğitim manifestindeki sıklığı: [(atom, sayı, sözcük kökleri)] — sayı < min_count olanlar modelin öğrenemediği semboller
    (gradyan almayan gömme rastgele kalır; o atomu içeren sözcük bozuk çıkar). Manifest yoksa None döner."""
    import collections
    import json
    import os

    from dizgetts import paths

    path = manifest or f"{paths.ANTALIA}/train_phon.jsonl"
    if not os.path.exists(path):
        return None
    cnt = collections.Counter()
    with open(path, encoding="utf8") as f:
        for line in f:
            cnt.update(json.loads(line)["tokens"])
    used = collections.defaultdict(list)
    for stems, _, frm, to, _ in Exceptions().rows:
        for a in set(to) - set(frm):
            used[a] += [stems[0]]
    return sorted(((a, cnt.get(a, 0), s) for a, s in used.items() if cnt.get(a, 0) < min_count), key=lambda r: r[1])


if __name__ == "__main__":  # python -m dizgetts.frontend.pronounce  -> eğitim verisinde yetersiz atom uyarısı
    low = coverage()
    print("manifest yok" if low is None else "\n".join(f"UYARI: {a!r} eğitimde {n} kez (köklerde: {', '.join(s)})" for a, n, s in low) or "tüm atomlar yeterli")
