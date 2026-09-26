"""v4c: süre kalibrasyonunu YALNIZ TRAIN kliplerinde uydur -> <run>/duration_calib.json (experiments.yaml v4c-calib).
  python -X utf8 -m dizgetts.scripts.fit_duration_calib --ckpt D:/dizgetts/runs/v4_g2ptts_dpfeat_e150_*/ep150.pt

Grup = (sınır sınıfı x token türü) (train/dpfeat.py). Hedef = modelin kendi MAS'ının gerçek mel'e verdiği kare sayısı; tahmin = süre tahmincisi.
İkisi de log1p(kare) alanında; grup başına ort/std eşlenir (moment eşleme: dağılımın genişliğini geri getirir; regresyon getirmez).
n < 200 ya da tahmin std'si ~0 olan grup özdeşlik kalır.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from matcha.utils.model import normalize as mel_norm
from matcha.utils.utils import intersperse

from dizgetts.engine import DP_FEAT
from dizgetts.eval.prosody_acoustics import _load_model, mas_frames
from dizgetts.frontend.symbols import SYMBOL_TO_ID
from dizgetts.train.dpfeat import TOKEN_TYPES, calib_groups, identity_table, intersperse_feat, set_dp_feat, token_types

ROOT = "D:/dizgetts/data/processed/antalia"
MIN_N = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    ckpt = glob.glob(a.ckpt)[0]
    m, ck, stats = _load_model(ckpt)
    assert ck["cfg"]["model"].get("dp_feat"), "kalibrasyon dp_feat'li (v4) model içindir"
    rows = [json.loads(l) for l in open(f"{ROOT}/train{ck['cfg']['manifest']}.jsonl", encoding="utf8")]
    ng = len(DP_FEAT) * len(TOKEN_TYPES)
    tgt, prd = [[] for _ in range(ng)], [[] for _ in range(ng)]
    for n, r in enumerate(rows):
        ids = intersperse([SYMBOL_TO_ID[t] for t in r["tokens"]], 0)
        mel = mel_norm(torch.load(f"{ROOT}/mels/{r['id']}.pt"), stats["mel_mean"], stats["mel_std"])
        set_dp_feat(m, None)  # önceki klibin özniteliği kalmasın (uzunluk assert'i yakalar)
        fr, _ = mas_frames(m, ids, mel)  # MAS yalnız mu'yu kullanır (öznitelikten bağımsız)
        feat = intersperse_feat(r["dp_feat"])
        set_dp_feat(m, torch.tensor([feat]))
        with torch.no_grad():
            _, logw, _ = m.encoder(torch.tensor([ids]), torch.tensor([len(ids)]))
        d_p = np.log1p(np.exp(logw[0, 0].numpy()))
        for g, t, p in zip(calib_groups(feat, token_types(r["tokens"])), np.log1p(fr), d_p):
            tgt[g].append(t); prd[g].append(p)
        if n % 200 == 0:
            print(f"  {n}/{len(rows)}", flush=True)
    table = identity_table()
    info = []
    inv_f = {v: k for k, v in DP_FEAT.items()}
    for g in range(ng):
        name = f"{inv_f[g // len(TOKEN_TYPES)]}/{TOKEN_TYPES[g % len(TOKEN_TYPES)]}"
        n = len(tgt[g])
        if n < MIN_N:
            info.append(dict(group=name, n=n, used=False)); continue
        T, P = np.array(tgt[g]), np.array(prd[g])
        if P.std() < 1e-3:
            info.append(dict(group=name, n=n, used=False)); continue
        table[g] = torch.tensor([P.mean(), P.std(), T.mean(), T.std()])
        info.append(dict(group=name, n=n, used=True, pred_mean_frames=float(np.expm1(P.mean())), tgt_mean_frames=float(np.expm1(T.mean())),
                         pred_std=float(P.std()), tgt_std=float(T.std()), stretch=float(T.std() / P.std()), corr=float(np.corrcoef(T, P)[0, 1])))
    out = os.path.join(os.path.dirname(ckpt), "duration_calib.json")
    json.dump(dict(ckpt=ckpt, min_n=MIN_N, table=table.tolist(), groups=info), open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"{'grup':18s} {'n':>7s} {'tahmin(kare)':>12s} {'hedef(kare)':>11s} {'germe':>6s} {'r':>5s}")
    for i in info:
        if i["used"]:
            print(f"{i['group']:18s} {i['n']:7d} {i['pred_mean_frames']:12.2f} {i['tgt_mean_frames']:11.2f} {i['stretch']:6.2f} {i['corr']:5.2f}")
    print("->", out)


if __name__ == "__main__":
    main()
