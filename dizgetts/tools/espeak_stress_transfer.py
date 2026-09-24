"""Vurgu aktarımı (Aşama 5 deneyi): espeak-ng'nin sözcük vurgusunu dizge fonem token'larına taşır.

AMAÇ: "dizge fonemleri + vurgu"nun ÜST SINIRINI ölçmek. espeak'in vurgusu kusursuz değil (Kabak & Vogel istisnalarını kabaca uyguluyor),
ve çıkarımda espeak'e bağımlılık getirir; kalıcı çözüm DizgeBERT-Morph + hece + dilbilimsel kurallardır (bu modülün yerini alacak).

Yöntem (sözcük düzeyi, ilk sözcük = ilk sözcük):
  1. espeak çıktısı boşlukla sözcüklere ayrılır, yalnız noktalamadan oluşanlar atılır. dizge token listesi WORD_SEP'ten sözcüklere ayrılır.
  2. Sözcük sayıları tutmuyorsa (espeak bazen sözcük birleştirir/böler) TÜM CÜMLE için yedek kural: her sözcüğün SON ünlüsü vurgulu.
  3. Sözcükte espeak'in "ˈ"si yoksa (edat/bağlaç gibi vurgusuz) dizge sözcüğüne de vurgu konmaz.
  4. Vurgulu ünlü: espeak sözcüğünde "ˈ"den hemen sonraki ünlü. Ünlü sayıları eşitse aynı sıradaki ünlü; eşit değilse SONDAN sayılan sıra
     (sondan r-inci); o da sığmazsa son ünlü. Vurgu simgesi symbols.STRESS, vurgulu ünlü token'ının ÖNÜNE eklenir.
Not: yalnız birincil vurgu "ˈ" kullanılır; ikincil "ˌ" yok sayılır.
"""
from __future__ import annotations

import collections

from .symbols import PAUSES, PHONES, STRESS, WORD_SEP

ESPEAK_VOWELS = set("aeiouyæøœɔɛɪɯʊ")  # espeak-ng tr çıktısındaki ünlüler (reports/frontend_report.json espeak_symbols)
_PUNCT = set(PAUSES)


def _is_vowel(tok: str) -> bool:
    return tok in PHONES and PHONES[tok][0] == "ünlü"


def espeak_words(espeak_str: str) -> list[str]:
    """Yalnız noktalamadan oluşan parçalar atılır."""
    return [w for w in espeak_str.split() if not all(c in _PUNCT for c in w)]


def espeak_stress_index(word: str) -> tuple[int | None, int]:
    """(vurgulu ünlünün sözcük içi ünlü sırası ya da None, sözcükteki ünlü sayısı)."""
    vowels = [c for c in word if c in ESPEAK_VOWELS]
    if STRESS not in word:
        return None, len(vowels)
    after = word[word.index(STRESS) + 1 :]
    k = next((i for i, c in enumerate(after) if c in ESPEAK_VOWELS), None)
    if k is None:
        return None, len(vowels)
    before = word[: word.index(STRESS)]
    return sum(1 for c in before if c in ESPEAK_VOWELS), len(vowels)


def dizge_words(tokens: list[str]) -> list[list[int]]:
    """WORD_SEP'e göre sözcükler; her sözcük = token indeksleri (noktalama token'ları önceki sözcüğe yapışıktır). Yalnız noktalamadan oluşan
    parça (ör. cümle başı noktalama) sözcük SAYILMAZ, espeak_words ile aynı kural."""
    words, cur = [], []
    for i, t in enumerate(tokens):
        if t == WORD_SEP:
            if any(tokens[j] not in _PUNCT for j in cur):
                words.append(cur)
            cur = []
        else:
            cur.append(i)
    if any(tokens[j] not in _PUNCT for j in cur):
        words.append(cur)
    return words


def transfer_stress(tokens: list[str], espeak_str: str) -> tuple[list[str], collections.Counter]:
    """dizge token listesine (WORD_SEP'li) vurgu ekler. Dönen: (yeni token listesi, sayaçlar)."""
    st: collections.Counter = collections.Counter()
    dw = dizge_words(tokens)
    ew = espeak_words(espeak_str)
    mark: set[int] = set()  # vurgu simgesinin ÖNÜNE ekleneceği token indeksleri
    if len(dw) != len(ew):
        st["cümle_yedek_son_ünlü"] += 1
        for w in dw:
            v = [i for i in w if _is_vowel(tokens[i])]
            if v:
                mark.add(v[-1])
    else:
        for w, e in zip(dw, ew):
            s, n_e = espeak_stress_index(e)
            v = [i for i in w if _is_vowel(tokens[i])]
            if s is None:
                st["espeak_vurgusuz"] += 1
                continue
            if not v:
                st["dizge_ünlüsüz"] += 1
                continue
            if n_e == len(v):
                mark.add(v[s]); st["ünlü_sayısı_eşit"] += 1
            else:
                r = n_e - 1 - s  # sondan sıra
                if 0 <= r < len(v):
                    mark.add(v[len(v) - 1 - r]); st["sondan_sıra"] += 1
                else:
                    mark.add(v[-1]); st["yedek_son_ünlü"] += 1
    out: list[str] = []
    for i, t in enumerate(tokens):
        if i in mark:
            out.append(STRESS)
        out.append(t)
    st["vurgu_sayısı"] += len(mark)
    return out, st
