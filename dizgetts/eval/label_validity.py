"""Etiket geçerliliği: ön ucun işaretlediği ayrımlar (ön a / arka ɑ, alıntı œ / sıradan ø, ince l / kalın ł, vurgu) Antalia KAYDINDA akustik olarak var mı?
Model kayıtta olmayan ayrımı öğrenemez; bu ölçüm v9 kararından önce hangi etiketin veride karşılığı olduğunu söyler.
  python -X utf8 -m dizgetts.eval.label_validity --ckpt D:/dizgetts/runs/v8_dplin/ep400_dp_mse.pt --manifest _phon_v8 [--limit N]
Yöntem (prosody_acoustics ile aynı): hizalama = modelin kendi MAS'ı (gerçek mel), token karesi (boşluk hariç); Praat Burg formant ünlü/ünsüz ortasında,
F0 (AC, 75-500 Hz) ve şiddet ünlü boyunca ortanca. Vurgu: aynı sözcükte vurgulu ünlü vs diğer ünlüler (EŞLEŞMİŞ fark; konuşmacı/cümle etkisi düşer).
Güven aralığı: klip düzeyinde bootstrap (aynı klipteki ölçümler bağımsız değil).
"""
import argparse
import collections
import json
import random

import numpy as np
import parselmouth
import torch
from matcha.utils.model import normalize as mel_norm
from matcha.utils.utils import intersperse

from dizgetts import paths
from dizgetts.eval.prosody_acoustics import _load_model, mas_frames
from dizgetts.frontend.symbols import PHONES, STRESS, SYMBOL_TO_ID, WORD_SEP
from dizgetts.train.dpfeat import set_dp_feat

HOP = 256 / 22050
VOWEL = lambda t: t in PHONES and PHONES[t][0] == "ünlü"


def clip_measures(m, stats, r):
    toks = r["tokens"]
    ids = intersperse([SYMBOL_TO_ID[t] for t in toks], 0)
    set_dp_feat(m, None)
    fr, _ = mas_frames(m, ids, mel_norm(torch.load(f"{paths.ANTALIA}/mels/{r['id']}.pt"), stats["mel_mean"], stats["mel_std"]))
    ends = np.cumsum(fr) * HOP
    starts = ends - np.asarray(fr) * HOP
    snd = parselmouth.Sound(f"{paths.ANTALIA}/{r['wav']}")
    fmt = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)
    pit = snd.to_pitch_ac(pitch_floor=75, pitch_ceiling=500)
    inten = snd.to_intensity(minimum_pitch=75)
    out, word, stressed_next = [], 0, False
    for j, t in enumerate(toks):
        if t == WORD_SEP:
            word += 1; continue
        if t == STRESS:
            stressed_next = True; continue
        k = 2 * j + 1  # interspersed konum
        s, e = starts[k], ends[k]
        if e - s < HOP:  # 0 kare: ölçülemez
            stressed_next = False; continue
        mid = (s + e) / 2
        f1, f2 = fmt.get_value_at_time(1, mid), fmt.get_value_at_time(2, mid)
        ts = np.linspace(s, e, 5)
        f0 = np.array([pit.get_value_at_time(x) for x in ts], dtype=float)
        f0 = np.nanmedian(f0) if np.isfinite(f0).any() else np.nan
        db = float(np.median([inten.get_value(x) for x in ts]))
        prev = toks[j - 1] if j and toks[j - 1] not in (STRESS, WORD_SEP) else None
        out.append(dict(clip=r["id"], word=word, tok=t, dur=e - s, f1=f1, f2=f2, f0=f0, db=db, stressed=stressed_next, prev=prev))
        stressed_next = False
    return out


def boot(clips: dict, stat, n=1000, seed=0):
    """clips: klip -> değer listesi; stat: fonksiyon(birleşik liste) -> sayı. Klip düzeyinde bootstrap %95 GA."""
    keys = [k for k, v in clips.items() if v]
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        s = [x for k in (rng.choice(keys) for _ in keys) for x in clips[k]]
        vals.append(stat(s))
    return np.percentile(vals, [2.5, 97.5])


def report(rows):
    by = collections.defaultdict(list)
    for x in rows:
        by[x["tok"]].append(x)
    def f2(tok):
        v = [x["f2"] for x in by.get(tok, []) if np.isfinite(x["f2"])]
        return (np.mean(v), len(v)) if v else (np.nan, 0)
    print("== 1) Ünlü F2 ortalaması (Hz; ön ünlü = yüksek F2)")
    for grp in (("ɑ", "a", "ɛ"), ("ɑː", "aː", "ɛː"), ("ɔ", "œ", "ø"), ("U", "Y", "y")):
        print("  " + " | ".join(f"{t}: {f2(t)[0]:6.0f} (n={f2(t)[1]})" for t in grp))

    def diff_ci(ta, tb, key="f2"):
        ca, cb = collections.defaultdict(list), collections.defaultdict(list)
        for x in by.get(ta, []):
            if np.isfinite(x[key]): ca[x["clip"]].append(x[key])
        for x in by.get(tb, []):
            if np.isfinite(x[key]): cb[x["clip"]].append(x[key])
        keys = sorted(set(ca) | set(cb)); rng = random.Random(1); ds = []
        for _ in range(1000):
            smp = [rng.choice(keys) for _ in keys]
            a = [v for k in smp for v in ca.get(k, [])]; b = [v for k in smp for v in cb.get(k, [])]
            if a and b: ds.append(np.mean(a) - np.mean(b))
        m = np.mean([v for l in ca.values() for v in l]) - np.mean([v for l in cb.values() for v in l])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        return m, lo, hi
    print("== 1b) Etiket farkı (F2, Hz; klip bootstrap %95 GA)")
    for ta, tb in (("a", "ɑ"), ("aː", "ɑː"), ("œ", "ø"), ("œ", "ɔ"), ("a", "ɛ")):
        if by.get(ta) and by.get(tb):
            m, lo, hi = diff_ci(ta, tb)
            print(f"  {ta} - {tb}: {m:+6.0f} [{lo:+.0f}, {hi:+.0f}]  {'ANLAMLI' if lo > 0 or hi < 0 else 'anlamsız'}")

    print("== 2) l türleri F2 (Hz): ince l önceki ünlüye göre")
    lc = collections.defaultdict(list)
    for x in by.get("l", []) + by.get("ł", []):
        if np.isfinite(x["f2"]):
            kind = x["tok"] + ("/ön-a" if x["prev"] in ("a", "aː") else "/œ" if x["prev"] == "œ" else "/arka" if x["prev"] in ("ɑ", "ɔ", "U", "ɨ", "ɑː") else "/ön" if x["prev"] and VOWEL(x["prev"]) else "/ünsüz")
            lc[kind].append(x["f2"])
    for k in sorted(lc):
        print(f"  {k:10} F2 {np.mean(lc[k]):6.0f}  n={len(lc[k])}")

    print("== 3) Vurgu: aynı sözcükte vurgulu ünlü - diğer ünlüler (eşleşmiş, sözcük ortalaması)")
    words = collections.defaultdict(list)
    for x in rows:
        if VOWEL(x["tok"]):
            words[(x["clip"], x["word"])].append(x)
    for key, unit, fn in (("dur", "ms", lambda v: v * 1000), ("f0", "yt", None), ("db", "dB", lambda v: v)):
        per_clip = collections.defaultdict(list)
        for (clip, _), vs in words.items():
            s = [x for x in vs if x["stressed"]]; o = [x for x in vs if not x["stressed"]]
            if len(s) != 1 or not o:
                continue
            sv = s[0][key]; ov = [x[key] for x in o if np.isfinite(x[key])]
            if not np.isfinite(sv) or not ov:
                continue
            d = 12 * np.log2(sv / np.mean(ov)) if key == "f0" else fn(sv) - fn(np.mean(ov))
            per_clip[clip].append(d)
        allv = [v for l in per_clip.values() for v in l]
        lo, hi = boot(per_clip, np.mean)
        print(f"  {key:4} vurgulu - diğer: {np.mean(allv):+6.2f} {unit} [{lo:+.2f}, {hi:+.2f}]  n={len(allv)} sözcük  {'ANLAMLI' if lo > 0 or hi < 0 else 'anlamsız'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="_phon_v8")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="ölçümleri jsonl'e yaz (yeniden rapor için)")
    a = ap.parse_args()
    m, ck, stats = _load_model(a.ckpt)
    rows = []
    for sp in ("train", "val", "test"):
        for n, line in enumerate(open(f"{paths.ANTALIA}/{sp}{a.manifest}.jsonl", encoding="utf8")):
            if a.limit and n >= a.limit:
                break
            rows += clip_measures(m, stats, json.loads(line))
    if a.out:
        with open(a.out, "w", encoding="utf8") as f:
            for x in rows:
                f.write(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in x.items()}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} segment")
    report(rows)


if __name__ == "__main__":
    main()
