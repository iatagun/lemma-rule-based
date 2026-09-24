"""Antalia kayıtlarından sözcük-sınırı duraklama ölçümü (otomatik "kırılma dizini" için ham veri).

Fikir (Dalva vd. 2014 gibi bürünsel ipuçlarını sesten ölçmek; Ipek & Jun: ip/IP sınırları): eğitilmiş baseline Matcha'nın MAS hizalaması
her token'ın kare aralığını verir. Bir sözcük sınırı için "iki sözcüğün son/ilk FONEM'i arasındaki bölge" (blank/boşluk/noktalama token'ları)
alınır ve o bölgede WAV'daki düşük-enerjili (sessiz) çerçeveler ölçülür. Metin tarafı espeak token'larıdır (boşluk = sözcük sınırı; dizge
token'larında da k-inci ayraç aynı sınırdır; sözcük sayıları eşit mi kontrol edilir).

  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/scripts/derive_breaks.py --ckpt D:/dizgetts/runs/baseline_espeak_e150_*/ep150.pt
Çıktı: <out_root>/breaks.jsonl + reports/breaks_stats.json (dağılım). Sınıf eşikleri make_break_tokens'ta; burada yalnız ölçüm.
"""
import argparse, glob, json, math, os, sys

import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "dizgetts"))
import matcha.utils.monotonic_align as MA  # noqa: E402
from matcha.utils.model import sequence_mask  # noqa: E402

from frontend import symbols_espeak as S  # noqa: E402
from train.data import TTSDataset, collate, ensure_mels  # noqa: E402
from train.train import build_model  # noqa: E402

PUNCT = set(",.?!;")
HOP_MEL = 256 / 22050


def frame_energy_db(x, sr, hop=0.010):
    h = int(sr * hop)
    n = len(x) // h
    return 10 * np.log10(np.maximum((x[: n * h].reshape(n, h) ** 2).mean(1), 1e-12)), hop


@torch.no_grad()
def mas_durations(m, b):
    mu_x, logw, x_mask = m.encoder(b["x"], b["x_lengths"], None)
    y = b["y"]
    y_mask = sequence_mask(b["y_lengths"], y.shape[-1]).unsqueeze(1).to(x_mask)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    const = -0.5 * math.log(2 * math.pi) * m.n_feats
    factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype)
    lp = (torch.matmul(factor.transpose(1, 2), y ** 2) - torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
          + torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1) + const)
    attn = MA.maximum_path(lp, attn_mask.squeeze(1))
    return attn[0].sum(-1).numpy()  # token başına kare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=os.path.join(ROOT, "dizgetts", "configs", "data.yaml"))
    a = ap.parse_args()
    ck_path = glob.glob(a.ckpt)[0]
    dcfg = yaml.safe_load(open(a.config, encoding="utf8"))
    root = dcfg["out_root"]
    stats = json.load(open(os.path.join(root, "stats.json"), encoding="utf8"))
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert ck["cfg"]["frontend"] == "espeak"
    m = build_model(ck["cfg"], len(ck["symbols"]), stats)
    m.load_state_dict(ck["model"]); m.eval()
    out, n_words_mismatch = [], 0
    for split in ("train", "val", "test"):
        ds = TTSDataset(root, split, "espeak", stats, ck["cfg"]["espeak_strip_stress"])
        ensure_mels(root, ds.rows, dcfg["audio"])
        for i, row in enumerate(ds.rows):
            item = ds[i]
            b = collate([item])
            dur = mas_durations(m, b)
            c = np.concatenate([[0], np.cumsum(dur)])  # c[t]..c[t+1] = t. token'ın kare aralığı (interspersed indeks)
            toks = [S.SYMBOLS[t] for t in item["x"].tolist()[1::2]]
            x, sr = sf.read(os.path.join(root, row["wav"]), dtype="float32")
            e_db, hop = frame_energy_db(x, sr)
            thr = np.percentile(e_db, 95) - 30  # audit ile aynı sessizlik eşiği
            silent = e_db < thr
            # espeak dizgesinde noktalama iki boşlukla çevrili ("a ; b"), dizge token'larında ise "a; b": noktalamadan ÖNCEKİ boşluğu at
            # ki k-inci sınır iki frontend'de aynı sınır olsun.
            spaces = [j for j, t in enumerate(toks) if t == " " and j + 1 < len(toks) and toks[j + 1] not in PUNCT]
            n_words_espeak = len(spaces) + 1
            n_words_dizge = sum(1 for t in row["tokens"] if t == " ") + 1
            if n_words_espeak != n_words_dizge:
                n_words_mismatch += 1
            bnd = []
            for k, j in enumerate(spaces):
                a_i = max((q for q in range(j) if toks[q] not in PUNCT and toks[q] != " "), default=None)
                b_i = min((q for q in range(j + 1, len(toks)) if toks[q] not in PUNCT and toks[q] != " "), default=None)
                if a_i is None or b_i is None:
                    continue
                f0, f1 = c[2 * a_i + 2], c[2 * b_i + 1]  # a fonemin sonu .. b fonemin başı (mel karesi)
                t0, t1 = f0 * HOP_MEL, f1 * HOP_MEL
                i0, i1 = int(t0 / hop), int(math.ceil(t1 / hop))
                seg = silent[i0:i1]
                punct = "".join(t for t in toks[a_i + 1 : b_i] if t in PUNCT)
                bnd.append(dict(k=k, region_ms=round((t1 - t0) * 1000), silence_ms=int(seg.sum() * hop * 1000), punct=punct))
            out.append(dict(id=row["id"], split=split, n_words=n_words_espeak, n_words_dizge=n_words_dizge, boundaries=bnd))
        print(split, "bitti", flush=True)
    with open(os.path.join(root, "breaks.jsonl"), "w", encoding="utf8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sil = np.array([b["silence_ms"] for r in out for b in r["boundaries"]])
    sil_p = np.array([b["silence_ms"] for r in out for b in r["boundaries"] if b["punct"]])
    sil_n = np.array([b["silence_ms"] for r in out for b in r["boundaries"] if not b["punct"]])
    q = lambda a: {f"p{p}": int(np.percentile(a, p)) for p in (10, 25, 50, 75, 90, 95)} if len(a) else {}
    rep = dict(ckpt=ck_path, n_boundaries=len(sil), n_punct=len(sil_p), n_nopunct=len(sil_n), clips_word_count_mismatch=n_words_mismatch,
               silence_ms_all=q(sil), silence_ms_punct=q(sil_p), silence_ms_nopunct=q(sil_n),
               frac_nopunct_ge=dict({str(t): round(float((sil_n >= t).mean()), 3) for t in (30, 60, 100, 150, 250, 400)}),
               frac_punct_ge=dict({str(t): round(float((sil_p >= t).mean()), 3) for t in (30, 60, 100, 150, 250, 400)}))
    json.dump(rep, open(os.path.join(ROOT, "dizgetts", "reports", "breaks_stats.json"), "w", encoding="utf8"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
