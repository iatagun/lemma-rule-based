"""Vurgu -> fonem indeksi eşlemesi, söyleyiş sözlüğü/uzun ünlü kuralları AÇIKKEN (2026-09-27 regresyonu).
  python -X utf8 -m dizgetts.tests.test_stress_map
Hata: to_phone_index yan ünlüyü `ː` işaretiyle tanıyordu; kurallar `ɑː I -> ɑ I` yapınca bayram'ın ilk seslem vurgusu yan ünlü `I`'ya düşüyordu (3359 geçiş);
-diği'de iki ünlü tek `iː`'ye birleşince (iː I -> iː) indeksler kayıyordu. Çözüm: atom kökeni izlenir; eşleme HAM dizge atomlarında yapılır, iz ile son diziye taşınır."""
import collections
import json
import os
import warnings

from dizgetts import paths
from dizgetts.engine import Engine, _stress_index
from dizgetts.frontend.normalize import tr_lower
from dizgetts.frontend.stress import _n_vowels
from dizgetts.frontend.symbols import PHONES

warnings.simplefilter("ignore")
E0 = Engine(bert_fallback=False)
E1 = Engine(bert_fallback=False, pron_exceptions=True, length_rules=True)


def word(e, w):
    return e.frontend(w).words[0]


def atom_at(w, k):
    i, how = _stress_index(w, k)
    return w.phones[i], how


# 1) yan ünlü: k0 = ilk ÜNLÜ (glide I değil), k1 = ikinci ünlü
for t, want in (("bayram", ["ɑ", "ɑ"]), ("ayak", ["ɑ", "ɑ"]), ("kuyu", ["U", "U"]), ("koyun", ["o", "U"]), ("sayı", ["ɑ", "ɨ"])):
    w = word(E1, t)
    assert [atom_at(w, k)[0] for k in range(len(want))] == want, (t, w.phones, [atom_at(w, k) for k in range(len(want))])

# 2) -diği (iğ+i): iki seslem tek `iː` atomuna işaret eder (birleşen atom)
w = word(E1, "gönderdiğim")
assert w.phones == ["ɟ", "œ", "n", "d", "ɛ", "ɾ", "d", "iː", "m"], w.phones
assert [w.phones[_stress_index(w, k)[0]] for k in range(4)] == ["œ", "ɛ", "iː", "iː"]
w = word(E1, "söylendiğinde")
assert [w.phones[_stress_index(w, k)[0]] for k in range(5)] == ["ø", "e", "iː", "iː", "ɛ"], (w.phones, [w.phones[_stress_index(w, k)[0]] for k in range(5)])

# 3) sözlük + kural birlikte (kağıt: ön a, ğ geçişi); i+y'de araya eklenen `j` ünlü indekslerini kaydırmaz; -eceğim'de düşen j
assert [atom_at(word(E1, "kağıt"), k)[0] for k in range(2)] == ["a", "ɨ"]
assert [atom_at(word(E1, "geliyor"), k)[0] for k in range(3)] == ["e", "i", "ɔ"]
w = word(E1, "göndereceğim")
assert [atom_at(w, k)[0] for k in range(5)] == ["œ", "e", "ɛ", "ɛ", "I"], (w.phones, [atom_at(w, k) for k in range(5)])
assert [atom_at(word(E1, "diğer"), k)[0] for k in range(2)] == ["I", "ɛ"] and [atom_at(word(E1, "eğri"), k)[0] for k in range(2)] == ["ɛː", "I"]

# 3b) köken izi (kök neden: ekleme/silmede konumla eşleştirme kökeni kaydırıyordu): eklenen `j` kökensiz, birleşen atom iki köken taşır
for tt, want in (("diğer", [[0], [1], [], [2], [3]]), ("eğri", [[0, 1], [2], [3]]), ("eğer", [[0, 1, 2], [3]]), ("gönderdiğim", None), ("geliyor", None)):
    w = word(E1, tt)
    if want is not None:
        assert [sorted(o) for o in w.origin] == want, (tt, w.phones, w.origin)
    assert all(len(w.origin) == len(w.phones) for _ in [0]) and sorted(x for o in w.origin for x in o) == list(range(len(w.raw_phones))), (tt, w.origin)  # her ham atom tam bir kez
w = word(E1, "gönderdiğim")
assert w.origin[-2] == [7, 8] or sorted(w.origin[-2]) == [7, 8], w.origin                                  # iː I -> iː: iki ham atom (7, 8) tek atomda

# 4) bayraklar KAPALIYKEN eski davranış birebir (iz yok, ham atomlar = son atomlar)
w = word(E0, "bayram")
assert w.origin is None and _stress_index(w, 0) == (1, "yan_ünlü_atıldı")

# 5) Antalia sözcük dağarı: ilk seslem = ilk ünlü atomu, son seslem = son ünlü atomu; hata sayısı kapalıyla ≤ (regresyon yok), kuralla açıkken de sıfıra yakın
ROOT = paths.ANTALIA
if os.path.exists(f"{ROOT}/train.jsonl"):
    from dizgetts.frontend.normalize import normalize

    voc = collections.Counter()
    for sp in ("train", "val", "test"):
        with open(f"{ROOT}/{sp}.jsonl", encoding="utf8") as f:
            for l in f:
                voc.update(tr_lower(x) for x in normalize(json.loads(l)["text"]).split() if x.isalpha())
    vow = lambda a: a in PHONES and PHONES[a][0] == "ünlü"
    bad = {}
    for name, e in (("kapalı", E0), ("açık", E1)):
        n_bad = 0
        for t in voc:
            w = word(e, t)
            nl = _n_vowels(t)
            va = [i for i, a in enumerate(w.phones) if vow(a)]
            if not nl or not va:
                continue
            for k, want in ((0, va[0]), (nl - 1, va[-1])):
                n_bad += _stress_index(w, k)[0] != want
        bad[name] = n_bad
    assert bad["açık"] <= bad["kapalı"], bad  # açıkken kapalıdan fazla hata OLMAMALI (kapalıdaki kalan hatalar dizge ham eşlemesinin kusurları)
    print("ilk/son seslem eşleme hatası (tekil sözcük): ", bad)
print("OK")
