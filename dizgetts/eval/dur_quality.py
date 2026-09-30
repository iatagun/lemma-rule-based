"""Süre tahmincisi kalitesi (v9 karar kuralı, docs/v9_plan.md): gerçek (MAS) vs tahmin (sentezdeki gibi birikimli yuvarlama), seslendirme yok.
  python -X utf8 -m dizgetts.eval.dur_quality D:/dizgetts/runs/v8_dplin/dp_mse.pt D:/dizgetts/runs/v9_dp2/dp2_mse.pt --split test
Birim = token + ardından gelen boşluk (ses ikisine yayılır). Ölçütler: log-süre Pearson korelasyonu (log(kare+1)), std oranı (tahmin/gerçek, doğrusal kare),
toplam süre oranı. İlk ckpt'ye göre eşleşmiş fark, klip düzeyinde bootstrap %95 GA. MAS hedefleri İLK ckpt'nin akustik modeliyle (aynı akustik taban varsayılır;
farklıysa uyarır).
"""
import argparse
import json
import random

import numpy as np
import torch
from matcha.utils.model import normalize as mel_norm
from matcha.utils.utils import intersperse

from dizgetts import paths
from dizgetts.eval.dp_response import frames
from dizgetts.eval.prosody_acoustics import _load_model, mas_frames
from dizgetts.frontend.symbols import SYMBOL_TO_ID
from dizgetts.train.dpfeat import set_dp_feat, token_types

TYPES = ("ünlü", "ünsüz", "ayraç", "noktalama", "vurgu")


def per_clip(models, stats, manifest, split):
    out = []
    ref = models[0]
    for line in open(f"{paths.ANTALIA}/{split}{manifest}.jsonl", encoding="utf8"):
        r = json.loads(line)
        ids = intersperse([SYMBOL_TO_ID[t] for t in r["tokens"]], 0)
        set_dp_feat(ref, None)
        real, _ = mas_frames(ref, ids, mel_norm(torch.load(f"{paths.ANTALIA}/mels/{r['id']}.pt"), stats["mel_mean"], stats["mel_std"]))
        real = np.asarray(real, dtype=float)
        preds = [np.asarray(frames(m, ids, r["dp_feat"], "cpu"), dtype=float) for m in models]
        ty = token_types(r["tokens"])[1::2]
        unit = lambda a: np.array([a[2 * j + 1] + a[2 * j + 2] for j in range(len(r["tokens"]))])
        out.append(dict(real=unit(real), preds=[unit(p) for p in preds], ty=np.array(ty)))
    return out


def metrics(clips, k, mask_fn=lambda c: slice(None)):
    real = np.concatenate([c["real"][mask_fn(c)] for c in clips]); pred = np.concatenate([c["preds"][k][mask_fn(c)] for c in clips])
    return dict(corr=float(np.corrcoef(np.log1p(real), np.log1p(pred))[0, 1]), std=float(pred.std() / real.std()), total=float(pred.sum() / real.sum()), n=len(real))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", default=None, help="varsayılan: ilk ckpt'nin manifesti")
    a = ap.parse_args()
    loaded = [_load_model(c) for c in a.ckpts]
    models = [m for m, _, _ in loaded]
    # set_round ÇAĞRILMAZ: frames() birikimli yuvarlamayı kendisi yapar (ikisi birlikte = çift yuvarlama, token başına ~-0,5 kare; 2026-09-30 bulundu)
    manifest = a.manifest or loaded[0][1]["cfg"]["manifest"]
    if len({ck["cfg"].get("manifest") for _, ck, _ in loaded}) > 1:
        print("UYARI: ckpt manifestleri farklı; MAS hedefi ilk ckpt'nin akustiğiyle")
    clips = per_clip(models, loaded[0][2], manifest, a.split)
    print(f"{a.split}: {len(clips)} klip, manifest {manifest}")
    for k, c in enumerate(a.ckpts):
        m = metrics(clips, k)
        by = {TYPES[t - 1]: metrics(clips, k, lambda cl, t=t: cl["ty"] == t) for t in (1, 2, 3, 4)}
        print(f"[{k}] {c}\n    TÜMÜ  korelasyon {m['corr']:.3f}  std oranı {m['std']:.3f}  toplam {m['total']:.3f}  (n={m['n']})\n    "
              + " | ".join(f"{t} r {v['corr']:.2f} std {v['std']:.2f} top {v['total']:.2f}" for t, v in by.items()))
    rng = random.Random(0)
    for k in range(1, len(a.ckpts)):
        d = {key: [] for key in ("corr", "std", "total")}
        for _ in range(1000):
            smp = [clips[rng.randrange(len(clips))] for _ in clips]
            m0, mk = metrics(smp, 0), metrics(smp, k)
            for key in d:
                d[key].append(mk[key] - m0[key])
        m0, mk = metrics(clips, 0), metrics(clips, k)
        print(f"[{k}] - [0]: " + " | ".join(f"{key} {mk[key] - m0[key]:+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]" for key, v in d.items()))


if __name__ == "__main__":
    main()
