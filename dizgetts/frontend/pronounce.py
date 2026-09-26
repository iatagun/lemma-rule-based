"""Söyleyiş istisna sözlüğü: dizge fonemlerinin üstüne, kök + ek zinciriyle eşleşen kalıp değiştirme (resources/pronunciation_exceptions.tsv).

Neden dizge'nin kendi `lexicon.loans` sözlüğü yetmiyor: yalnız TAM sözcüğü eşler; kağıt düzeltmesi kağıdı'da (t -> d, ünlü öncesi) kayboluyordu.
Satır: kökler(virgülle) <TAB> kayıt (özenli|gündelik) <TAB> kalıp (atomlar, boşlukla) <TAB> yerine <TAB> not. Kalıp sözcüğün BAŞINDAKİ atomlara uygulanır
(kökler sözcük başındadır). Özenli satırlar her zaman, gündelik satırlar yalnız register="gündelik" iken uygulanır (özenli sonucun üstüne; sıra: özenli, gündelik).
Kalıp bulunamazsa ve sonuç zaten `yerine` değilse (dizge sürümü/çekim beklenenden farklı) RuntimeWarning: sessiz uyuşmazlık olmaz.
"""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

from .normalize import tr_lower
from .stress import _SUFFIX_CHAIN
from .symbols import PHONES, tokenize

TSV = Path(__file__).resolve().parent.parent / "resources" / "pronunciation_exceptions.tsv"
REGISTERS = ("özenli", "gündelik")


class Atoms:
    """Bir sözcüğün fonem atomları + her atomun KÖKENİ (dizge'nin ham atom indeksleri). Sözlük/kural düzenlemeleri `replace` ile yapılır; böylece vurgu eşlemesi (stress.to_phone_index)
    HAM atomlarda (kanıtlanmış yöntem, ː/y işaretleri sağlam) yapılıp kökenle son diziye taşınabilir. Birleşen atomlar (iː I -> iː) iki kökeni birden taşır."""

    def __init__(self, phones: str):
        self.a = tokenize(phones, strict=False)
        self.raw = list(self.a)
        self.origin = [[i] for i in range(len(self.a))]
        self.changed = False

    def copy(self) -> "Atoms":
        c = Atoms.__new__(Atoms)
        c.a, c.raw, c.origin, c.changed = list(self.a), self.raw, [list(o) for o in self.origin], self.changed
        return c

    def adopt(self, other: "Atoms") -> None:
        self.a, self.origin, self.changed = other.a, other.origin, other.changed

    def text(self) -> str:
        return "".join(self.a)

    def replace(self, start: int, old_len: int, new: list[str]) -> None:
        """a[start:start+old_len] yerine `new`. Kökenler eski/yeni atomların TABANLARI (ː çıkarılmış) hizalanarak devredilir (difflib): eşit atom kökenini korur, aynı uzunlukta
        değişen bölge sırayla devralır, eklenen atom (ör. `j`) kökensizdir. Silinen/artan atomun kökeni EN YAKIN yeni atoma eklenir: silinen ÜNLÜ, en yakın yeni ÜNLÜ atoma
        (değer: d e j ɛ -> d ɛː, silinen e ɛː'ye gider, d'ye değil); birleşen atom birden çok köken taşır."""
        from difflib import SequenceMatcher

        old, old_o = self.a[start:start + old_len], self.origin[start:start + old_len]
        base = lambda a: a.rstrip("ː")
        new_o: list[list[int]] = [[] for _ in new]
        pend = []  # (eski atom, köken, yeni dizideki konum)
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, [base(x) for x in old], [base(x) for x in new], autojunk=False).get_opcodes():
            n = min(i2 - i1, j2 - j1) if tag in ("equal", "replace") else 0
            for k in range(n):
                new_o[j1 + k] = list(old_o[i1 + k])
            pend += [(old[i], old_o[i], j1 + n) for i in range(i1 + n, i2)]
        isv = lambda a: a in PHONES and PHONES[a][0] == "ünlü"
        vnew = [j for j, a in enumerate(new) if isv(a)]
        for a, o, pos in pend:
            if new:
                cand = vnew if isv(a) and vnew else range(len(new))
                new_o[min(cand, key=lambda j: (abs(j - (pos - 0.5)), j))] += o
            elif start > 0:
                self.origin[start - 1] += o
            elif start + old_len < len(self.origin):
                self.origin[start + old_len] += o
        self.a[start:start + old_len] = new
        self.origin[start:start + old_len] = new_o
        self.changed = True


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
        """word: yazım (herhangi büyüklük), phones: dizge okuması. Değişiklik yoksa girdiyi aynen döndürür."""
        at = Atoms(phones)
        self.apply_atoms(word, at)
        return at.text() if at.changed else phones

    def apply_atoms(self, word: str, at: Atoms) -> None:
        w = tr_lower(word)
        atoms = at.a
        groups: dict = {}
        for row in self.rows:  # (kökler, kayıt) aynı olan satırlar ALTERNATİFTİR (dizge kökte `d Iː ɛ`, çekimde `d iː e` verebilir): biri yeterli
            groups.setdefault((row[0], row[1]), []).append(row)
        for (stems, reg), alts in groups.items():
            if reg == "gündelik" and self.register != "gündelik":
                continue
            if not any(w.startswith(s) and (len(w) == len(s) or _SUFFIX_CHAIN.match(w[len(s):])) for s in stems):
                continue
            hit = False
            for _, _, frm, to, _ in alts:
                if tuple(atoms[:len(frm)]) == frm:
                    at.replace(0, len(frm), list(to))
                    hit = True
                    break
                if tuple(atoms[:len(to)]) == to:  # zaten istenen okuma
                    hit = True
                    break
            if not hit:
                warnings.warn(f"söyleyiş istisnası: {word!r} için dizge çıktısı {' '.join(atoms)!r} beklenen kalıplarla ({' | '.join(' '.join(a[2]) for a in alts)}) başlamıyor; satır atlandı",
                              RuntimeWarning, stacklevel=4)



# ---------------------------------------------------------------- uzun ünlü (`ː`) kuralları
# dizge `ː`'yi üç ayrı gerçek için kullanır (Antalia ölçümü, docs/pronunciation_issues.md): (1) ğ + ünsüz/sözcük sonu = gerçek UZAMA (x1,4),
# (2) ünlüler arası ğ = GEÇİŞ (ilk ünlü uzamıyor), (3) y yan ünlüsü (ay, ey, iy...): dizge kaynağında `Vj -> VːI`, `ij -> iː` dize değiştirmeleri.
# Bu kural yalnız gerçek uzamayı `ː` bırakır; (2) ve (3)'te `ː` düşer, i+y için y `j` olarak geri konur (dizge zaten yapıyor/büyük'te `j` yazar).
# Hizalama: sözcüğün harflerinden `ː` üretebilecek OLAY listesi (ğ, y) ve dizge çıktısındaki `ː` atomları sırayla eşlenir: her `ː` atomu, ünlüsü ve komşusu UYUMLU
# ilk olayla (olayların hepsi `ː` üretmez: ö+y'de dizge bazen `j` bırakır; â/î/û `ː` üretmez, yalnız k/g'yi yumuşatır). Eşleşmeyen `ː` (dizge sözlüğünden gelen
# nisan/itibaren/hakim `iː`, `ɑː`) DOKUNULMAZ ve sayılır. e+ğ olay değildir: dizge birincil okumada zaten y'leştirir (eğlence -> e j l ɛ...).
VOWELS = "aeıioöuüâîû"


def _is_vowel_atom_name(a: str) -> bool:
    return a in PHONES and PHONES[a][0] == "ünlü"

_BASE = str.maketrans("âîû", "aiu")
_Y_LONG = frozenset("aeouöi")  # y'den önce bunlar gelirse dizge `ː` üretebilir (ı+y, ü+y'de `j` kalır: yapıyor, büyük)
_LETTER_ATOMS = {"a": "aɑ", "e": "eɛ", "ı": "ɨ", "i": "iI", "o": "oɔ", "ö": "øœ", "u": "uU", "ü": "yY"}


def _events(w: str) -> list[tuple[str, str, str, int]]:
    """(tür, önceki ünlü harf, sonraki harf, olaydan önceki 'katı ünsüz' harf sayısı); ğ ve y katı ünsüz sayılmaz (`j`/`ː` olarak değişken çıkar)."""
    ev = []
    for i, c in enumerate(w):
        p, n = (w[i - 1] if i else ""), (w[i + 1] if i + 1 < len(w) else "")
        cb = sum(x not in VOWELS and x not in "ğy" for x in w[:i])
        if c == "ğ" and p in VOWELS and p != "e":
            ev.append(("ğ", p.translate(_BASE), n, cb))
        elif c == "y" and p.translate(_BASE) in _Y_LONG:
            ev.append(("y", p.translate(_BASE), n, cb))
    return ev


def _compatible(ev: tuple[str, str, str, int], atom: str, nxt: str, cb: int) -> bool:
    """cb: `ː` atomundan önceki katı ünsüz ATOM sayısı (j hariç). Ünsüz sayısı tutmuyorsa olay bu `ː`'ye ait değildir (iyiliğinden: iy olayı ğ'nin iː'sini almasın)."""
    kind, p, _, ecb = ev
    if ecb != cb or atom[:-1] not in _LETTER_ATOMS[p]:
        return False
    return kind == "ğ" or p == "i" or nxt == "I"  # y: i+y -> iː | Iː; diğerleri Vː I


class LengthRules:
    """register: i+y için (kullanıcı 2026-09-26): özenli y SESLENİR (iyi -> i j I), gündelik y seslenmez, "ii" (dizge'nin `iː`'si aynen kalır)."""

    def __init__(self, register: str = "özenli"):
        if register not in REGISTERS:
            raise ValueError(f"register {REGISTERS} içinden olmalı: {register!r}")
        self.register = register

    def version(self) -> str:
        return f"{hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:8]}/{self.register}"

    def apply(self, word: str, phones: str) -> tuple[str, str]:
        """-> (fonemler, durum): durum 'aynı' | 'değişti' | 'hizalanamadı'. Değişiklik yoksa girdiyi aynen döndürür."""
        at = Atoms(phones)
        status = self.apply_atoms(word, at)
        return (at.text() if at.changed else phones), status

    def apply_atoms(self, word: str, at: Atoms) -> str:
        """`at` yerinde değişir (yalnız 'değişti' durumunda; 'hizalanamadı'da DOKUNULMAZ). -> 'aynı' | 'değişti' | 'hizalanamadı'."""
        from .stress import _is_vowel_atom

        work = at.copy()
        atoms = work.a
        w = tr_lower(word)
        diphthong = self._eceg(w, work)  # -eceğ + ünlü: ğ'nin j'si düşer (ɛ I bitişik, diftong)
        long_idx = [k for k, a in enumerate(atoms) if a.endswith("ː")]
        if not long_idx:
            if diphthong:
                at.adopt(work)
            return "değişti" if diphthong else "aynı"
        ev, edits, e = _events(w), [], 0
        for k in long_idx:
            a, nxt = atoms[k], (atoms[k + 1] if k + 1 < len(atoms) else "")
            cb = sum(PHONES[x][0] == "ünsüz" and x != "j" for x in atoms[:k] if x in PHONES)
            while e < len(ev) and not _compatible(ev[e], a, nxt, cb):
                e += 1
            if e == len(ev):
                return "hizalanamadı"  # bu `ː` hiçbir harf olayıyla açıklanamıyor: sözcüğe DOKUNMA
            kind, p, n, _ = ev[e]
            e += 1
            if kind == "ğ":
                if n in VOWELS and p.translate(_BASE) != n.translate(_BASE) and _is_vowel_atom(nxt):  # farklı ünlü arası: geçiş
                    edits.append((k, [a[:-1]]))
                elif p == n == "i" and a in ("iː", "Iː") and nxt in ("I", "i"):  # -diği/-liği/-tiği: uzama; dizge `iː I` (fazladan i) verir -> tek `iː`
                    edits.append((k, [a], 1))
            elif kind == "y":
                if p == "i":
                    if self.register == "özenli" and a in ("iː", "Iː"):
                        edits.append((k, [a[:-1], "j"]))
                elif nxt == "I":
                    edits.append((k, [a[:-1]]))
        for k, repl, *span in reversed(edits):
            work.replace(k, 1 + (span[0] if span else 0), repl)
        if edits or diphthong:
            at.adopt(work)
            return "değişti"
        return "aynı"

    @staticmethod
    def _eceg(w: str, work: "Atoms") -> bool:
        """-eceğ + ünlü (göndereceğim, edeceğiz): dizge `dʒ ɛ j V` verir; ğ'nin j'si düşer -> `dʒ ɛ V` (kullanıcı: diftong, 2026-09-27). Yerinde değiştirir."""
        if not re.search("eceğ[aeıioöuüâîû]", w):
            return False
        atoms = work.a
        for i in range(len(atoms) - 3, -1, -1):  # sondaki (ekteki) örüntü
            if atoms[i:i + 3] == ["dʒ", "ɛ", "j"] and _is_vowel_atom_name(atoms[i + 3] if i + 3 < len(atoms) else ""):
                work.replace(i + 2, 1, [])
                return True
        return False


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
    # üretilen atomlar: sözlük + uzun ünlü kuralları AÇIKKEN her kök için (eğitim yapılandırması), dizge'nin çıplak çıktısında olmayanlar (kağıt -> `a`, hakim -> `aː`)
    from .phonemize import Phonemizer

    bare = Phonemizer(bert_fallback=False)
    used = collections.defaultdict(list)
    for register in REGISTERS:
        ph = Phonemizer(bert_fallback=False, pron_exceptions=True, register=register, length_rules=True)
        for stems, *_ in Exceptions().rows:
            for a in set(tokenize(ph.word(stems[0]))) - set(tokenize(bare.word(stems[0]))):
                if stems[0] not in used[a]:
                    used[a].append(stems[0])
    return sorted(((a, cnt.get(a, 0), s) for a, s in used.items() if cnt.get(a, 0) < min_count), key=lambda r: r[1])


if __name__ == "__main__":  # python -m dizgetts.frontend.pronounce  -> eğitim verisinde yetersiz atom uyarısı
    low = coverage()
    print("manifest yok" if low is None else "\n".join(f"UYARI: {a!r} eğitimde {n} kez (köklerde: {', '.join(s)})" for a, n, s in low) or "tüm atomlar yeterli")
