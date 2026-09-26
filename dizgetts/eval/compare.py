"""evaluate.py sonuçlarını karşılaştır: cümle bootstrap'i (2000) ile %95 güven aralığı ve İLK etikete göre eşleşmiş fark.
  python -X utf8 -m dizgetts.eval.compare espeak_ep150 dizge_ep150 dizge_feat_ep150 [--splits test extra]
Öğeler `id` ile eşleşir (etiketlerde ortak olanlar, ilk etiketin sırasıyla); eskiden konumla eşleşiyordu ve farklı sıra/uzunlukta sonuçlar sessizce yanlış çiftlenirdi.
--splits: yalnız bu bölümler (test | val | extra). Karar için `val` KULLANMA: dp/epoch seçimi, tau ve length_scale val'de yapıldı (experiments.yaml v5, 2026-09-26 denetimi).
Gerçek kayıtların Whisper tabanı (aynı bölümler): reports/asr_floor.json."""
import argparse
import collections
import json
import os

import numpy as np

from dizgetts import paths

KEYS = ("cer_e", "n_chars", "wer_e", "n_words", "utmos")


def load_items(labels, splits=None, eval_root=None):
    """-> ({etiket: öğeler}, {etiket: ortak olmadığı için atılan sayı}); tüm etiketlerde bulunan id'ler, ilk etiketin sırasıyla."""
    root = eval_root or paths.EVAL_OUT
    R = {l: {x["id"]: x for x in json.load(open(f"{root}/{l}/results.json", encoding="utf8"))["items"] if not splits or x.get("split") in splits} for l in labels}
    ids = [i for i in R[labels[0]] if all(i in R[l] for l in labels[1:])]
    return {l: [R[l][i] for i in ids] for l in labels}, {l: len(R[l]) - len(ids) for l in labels}


def asr_floor(splits=None):
    """(CER, WER, n) gerçek kayıtlar için, istenen bölümlerde; per-klip kayıt yoksa (eski dosya) tüm bölümler ve n=None."""
    f = json.load(open(os.path.join(os.path.dirname(__file__), "..", "reports", "asr_floor.json"), encoding="utf8"))
    if "items" not in f:
        return f["cer"], f["wer"], None
    it = [x for x in f["items"] if not splits or x["split"] in splits]
    if not it:
        return None
    return sum(x["cer_e"] for x in it) / sum(x["n_chars"] for x in it), sum(x["wer_e"] for x in it) / sum(x["n_words"] for x in it), len(it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--splits", nargs="*", default=None)
    a = ap.parse_args()
    labels = a.labels
    R, dropped = load_items(labels, a.splits)
    n = len(R[labels[0]])
    comp = collections.Counter(x.get("split") for x in R[labels[0]])
    if any(dropped.values()):
        print("UYARI: ortak olmayan id'ler atıldı:", {l: d for l, d in dropped.items() if d})
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))
    arr = {l: {k: np.array([x[k] for x in R[l]], float) for k in KEYS} for l in labels}

    def corp(A, num, den, ix=None):
        ix = slice(None) if ix is None else ix
        return A[num][ix].sum() / A[den][ix].sum()

    def ci(f):
        return np.percentile(np.array([f(i) for i in idx]), [2.5, 97.5])

    try:
        fl = asr_floor(a.splits)
        if fl:
            print(f"gerçek kayıtlar (Whisper tabanı{'' if fl[2] is None else f', n={fl[2]}'}): CER {fl[0]*100:.1f}%  WER {fl[1]*100:.1f}%")
    except (OSError, KeyError, ValueError):
        pass
    print(f"bölümler: {dict(comp)}")
    print(f"{'sistem':22s} {'CER %':>16s} {'WER %':>16s} {'UTMOS':>16s}   (n={n}, %95 aralık)")
    for l in labels:
        A = arr[l]
        c, w, u = corp(A, "cer_e", "n_chars"), corp(A, "wer_e", "n_words"), A["utmos"].mean()
        cc, wc, uc = ci(lambda i: corp(A, "cer_e", "n_chars", i)), ci(lambda i: corp(A, "wer_e", "n_words", i)), ci(lambda i: A["utmos"][i].mean())
        print(f"{l:22s} {c*100:5.1f} [{cc[0]*100:4.1f},{cc[1]*100:4.1f}] {w*100:5.1f} [{wc[0]*100:4.1f},{wc[1]*100:4.1f}] {u:5.2f} [{uc[0]:4.2f},{uc[1]:4.2f}]")
    b = arr[labels[0]]
    print(f"\n{labels[0]} referansına göre eşleşmiş fark (pozitif = diğeri daha kötü CER/WER; UTMOS'ta pozitif = diğeri daha iyi):")
    for l in labels[1:]:
        A = arr[l]
        for name, f in (("CER", lambda X, i=None: corp(X, "cer_e", "n_chars", i)), ("WER", lambda X, i=None: corp(X, "wer_e", "n_words", i)),
                        ("UTMOS", lambda X, i=None: X["utmos"][slice(None) if i is None else i].mean())):
            d = f(A) - f(b)
            dc = ci(lambda i: f(A, i) - f(b, i))
            sig = "anlamlı" if dc[0] > 0 or dc[1] < 0 else "anlamsız (aralık 0'ı içeriyor)"
            sc = 100 if name != "UTMOS" else 1
            print(f"  {l:22s} {name}: {d*sc:+.2f}  [{dc[0]*sc:+.2f},{dc[1]*sc:+.2f}]  {sig}")


if __name__ == "__main__":
    main()
