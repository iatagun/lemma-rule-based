"""evaluate.py sonuçlarını karşılaştır: cümle bootstrap'i (2000) ile %95 güven aralığı ve İLK etikete göre eşleşmiş fark.
  python -X utf8 dizgetts/eval/compare.py espeak_ep150 dizge_ep150 dizge_feat_ep150
Gerçek kayıtların Whisper tabanı (aynı 143 cümle): reports/asr_floor.json."""
import json, os, sys

import numpy as np

labels = sys.argv[1:]
R = {l: json.load(open(f"D:/dizgetts/eval_out/{l}/results.json", encoding="utf8"))["items"] for l in labels}
n = min(len(v) for v in R.values())
rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(2000, n))
arr = {l: {k: np.array([x[k] for x in R[l][:n]], float) for k in ("cer_e", "n_chars", "wer_e", "n_words", "utmos")} for l in labels}


def corp(a, num, den, ix=None):
    ix = slice(None) if ix is None else ix
    return a[num][ix].sum() / a[den][ix].sum()


def ci(f):
    b = np.array([f(i) for i in idx])
    return np.percentile(b, [2.5, 97.5])


try:
    floor = json.load(open(os.path.join(os.path.dirname(__file__), "..", "reports", "asr_floor.json"), encoding="utf8"))
    print(f"gerçek kayıtlar (Whisper tabanı, aynı 143 cümle): CER {floor['cer']*100:.1f}%  WER {floor['wer']*100:.1f}%")
except Exception:
    pass
print(f"{'sistem':22s} {'CER %':>16s} {'WER %':>16s} {'UTMOS':>16s}   (n={n}, %95 aralık)")
for l in labels:
    a = arr[l]
    c, w, u = corp(a, "cer_e", "n_chars"), corp(a, "wer_e", "n_words"), a["utmos"].mean()
    cc, wc, uc = ci(lambda i: corp(a, "cer_e", "n_chars", i)), ci(lambda i: corp(a, "wer_e", "n_words", i)), ci(lambda i: a["utmos"][i].mean())
    print(f"{l:22s} {c*100:5.1f} [{cc[0]*100:4.1f},{cc[1]*100:4.1f}] {w*100:5.1f} [{wc[0]*100:4.1f},{wc[1]*100:4.1f}] {u:5.2f} [{uc[0]:4.2f},{uc[1]:4.2f}]")
b = arr[labels[0]]
print(f"\n{labels[0]} referansına göre eşleşmiş fark (pozitif = diğeri daha kötü CER/WER; UTMOS'ta pozitif = diğeri daha iyi):")
for l in labels[1:]:
    a = arr[l]
    for name, f in (("CER", lambda A, i=None: corp(A, "cer_e", "n_chars", i)), ("WER", lambda A, i=None: corp(A, "wer_e", "n_words", i)),
                    ("UTMOS", lambda A, i=None: A["utmos"][slice(None) if i is None else i].mean())):
        d = f(a) - f(b)
        dc = ci(lambda i: f(a, i) - f(b, i))
        sig = "anlamlı" if dc[0] > 0 or dc[1] < 0 else "anlamsız (aralık 0'ı içeriyor)"
        sc = 100 if name != "UTMOS" else 1
        print(f"  {l:22s} {name}: {d*sc:+.2f}  [{dc[0]*sc:+.2f},{dc[1]*sc:+.2f}]  {sig}")
