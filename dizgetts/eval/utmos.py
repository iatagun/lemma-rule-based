"""Otomatik MOS tahmini (UTMOS22-strong, SpeechMOS). Sentez klasörleri ve gerçek val kayıtları için.
  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/eval/utmos.py DIR [DIR ...]
UTMOS İngilizce ağırlıklı eğitildi: MUTLAK değerler Türkçe için güvenilir değil; AYNI cümlelerin koşular arası GÖRECELİ kıyası anlamlı."""
import glob, json, os, sys

import numpy as np
import soundfile as sf
import torch

m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()


@torch.no_grad()
def score(paths):
    out = []
    for p in paths:
        x, sr = sf.read(p, dtype="float32")
        out.append(float(m(torch.from_numpy(x)[None], sr)))
    return out


if __name__ == "__main__":
    res = {}
    for d in sys.argv[1:]:
        fs = sorted(glob.glob(os.path.join(d, "*.wav")))
        s = score(fs)
        res[d] = dict(mean=round(float(np.mean(s)), 3), n=len(s), per_file=[round(v, 2) for v in s])
        print(f"{os.path.basename(os.path.dirname(d)) if d.endswith(tuple('0123456789')) else ''} {os.path.basename(d)}: UTMOS {res[d]['mean']} (n={len(s)})", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "reports", "utmos_last.json"), "w"), indent=1)
