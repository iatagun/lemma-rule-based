"""Aşama 2: 22.05 kHz'e yeniden örnekleme, loudness normalizasyonu, baş/son sessizlik kırpma, süre süzgeci,
parent-bazlı train/val/test ayrımı, Matcha mel istatistikleri.

  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/scripts/preprocess.py [--config dizgetts/configs/data.yaml]

Çıktı (out_root): wavs/*.wav (PCM16), {train,val,test}.jsonl, stats.json (split, atılanlar, mel mean/std, config, git commit).
"""
import argparse, collections, json, os, random, subprocess, sys

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly

from matcha.utils.audio import mel_spectrogram


def trim(x, sr, thr_db, pad_s):
    hop = int(sr * 0.010)
    n = len(x) // hop
    rms_db = 10 * np.log10(np.maximum((x[: n * hop].reshape(n, hop) ** 2).mean(1), 1e-12))
    idx = np.flatnonzero(rms_db > thr_db)
    if len(idx) == 0:
        return x
    pad = int(pad_s * sr)
    return x[max(0, idx[0] * hop - pad): min(len(x), (idx[-1] + 1) * hop + pad)]


def process(path, c):
    x, sr0 = sf.read(path, dtype="float32")
    assert x.ndim == 1
    sr = c["audio"]["sample_rate"]
    g = np.gcd(sr, sr0)
    x = resample_poly(x, sr // g, sr0 // g).astype(np.float32)
    x = trim(x, sr, c["audio"]["trim_db"], c["audio"]["keep_pad_s"])
    lufs = pyln.Meter(sr).integrated_loudness(x)
    x = x * 10 ** ((c["audio"]["target_lufs"] - lufs) / 20)
    peak = float(np.abs(x).max())
    if peak > c["audio"]["peak_limit"]:  # ölçekle, kırpma yok
        x = x * (c["audio"]["peak_limit"] / peak)
    return x, sr, lufs


def choose_split(rows, c):
    rng = random.Random(c["seed"])
    par = collections.defaultdict(list)
    for r in rows:
        par[r["parent"]].append(r)
    by_cat = collections.defaultdict(list)
    for p, v in sorted(par.items()):
        by_cat[v[0]["category"]].append(p)
    cats = sorted(k for k, v in by_cat.items() if len(v) >= 4)  # her kategoride train'e >=3 parent kalsın
    rng.shuffle(cats)
    nt, nv = c["split"]["test_parents"], c["split"]["val_parents"]
    assert len(cats) >= nt + nv, "yeterli kategori yok"
    pick = lambda cat: rng.choice(by_cat[cat])
    test = {pick(k) for k in cats[:nt]}
    val = {pick(k) for k in cats[nt: nt + nv]}
    return {p: ("test" if p in test else "val" if p in val else "train") for p in par}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "data.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="hızlı deneme için ilk N klip")
    a = ap.parse_args()
    c = yaml.safe_load(open(a.config, encoding="utf8"))
    out = c["out_root"]
    os.makedirs(os.path.join(out, "wavs"), exist_ok=True)

    meta = [json.loads(l) for l in open(os.path.join(c["raw_root"], "metadata.jsonl"), encoding="utf8")]
    if a.limit:
        meta = meta[: a.limit]
    rows, dropped = [], collections.Counter()
    lufs_in = []
    for m in meta:
        x, sr, lufs = process(os.path.join(c["raw_root"], m["file_name"]), c)
        d = len(x) / sr
        if not (c["filter"]["min_dur_s"] <= d <= c["filter"]["max_dur_s"]):
            dropped["duration"] += 1
            continue
        wav = f"wavs/{m['clip_id']}.wav"
        sf.write(os.path.join(out, wav), x, sr, subtype="PCM_16")
        lufs_in.append(lufs)
        rows.append(dict(id=m["clip_id"], wav=wav, dur=round(d, 3), text=m["transcript"], text_norm_antalia=m["normalized_transcript"],
                         category=m["campaign_category"], parent=m["clip_id"].split("-script-seg-")[0], seg=m["segment_index"]))

    sp = choose_split(rows, c)
    for r in rows:
        r["split"] = sp[r["parent"]]
    key = lambda r: " ".join(r["text"].lower().split())
    if c["split"]["drop_text_leak"]:
        train_txt = {key(r) for r in rows if r["split"] == "train"}
        keep = []
        for r in rows:
            if r["split"] != "train" and key(r) in train_txt:
                dropped["text_leak_" + r["split"]] += 1
            else:
                keep.append(r)
        rows = keep
    for s in ("train", "val", "test"):
        with open(os.path.join(out, f"{s}.jsonl"), "w", encoding="utf8") as f:
            for r in rows:
                if r["split"] == s:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Matcha data_statistics: TRAIN mel'lerinin (log-mel, Matcha'nın kendi mel fonksiyonu) ortalaması/std'si
    au = c["audio"]
    n = s1 = s2 = 0
    for r in rows:
        if r["split"] != "train":
            continue
        y, sr = sf.read(os.path.join(out, r["wav"]), dtype="float32")
        mel = mel_spectrogram(torch.from_numpy(y)[None], au["n_fft"], au["n_feats"], sr, au["hop_length"], au["win_length"], au["f_min"], au["f_max"], center=False)
        mel = mel.double()
        n += mel.numel(); s1 += mel.sum().item(); s2 += (mel ** 2).sum().item()
    mean = s1 / n
    std = (s2 / n - mean ** 2) ** 0.5

    hours = {s: round(sum(r["dur"] for r in rows if r["split"] == s) / 3600, 3) for s in ("train", "val", "test")}
    stats = dict(
        n={s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}, hours=hours, dropped=dict(dropped),
        parents={s: sorted(p for p, v in sp.items() if v == s) for s in ("val", "test")},
        categories={s: dict(collections.Counter(r["category"] for r in rows if r["split"] == s)) for s in ("train", "val", "test")},
        mel_mean=mean, mel_std=std, lufs_in_quantiles={k: round(float(np.percentile(lufs_in, k)), 2) for k in (0, 5, 50, 95, 100)},
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        versions=dict(torch=torch.__version__, numpy=np.__version__), config=c,
    )
    json.dump(stats, open(os.path.join(out, "stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("config", "parents", "categories")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
