"""Uzun ünlü (ː) süreleri: gerçek kayıt (modelin kendi MAS hizalaması) vs süre tahmincisi -> sentezde `long_vowel_scale` çarpanı önerisi.
  python -X utf8 -m dizgetts.eval.long_vowel --ckpt D:/dizgetts/runs/v8_dplin/ep400_dp_mse.pt --manifest _phon_v8
Seslendirme yok. Birim = ünlü token'ı + ardından gelen boşluk token'ı (Matcha intersperse); iki tarafta aynı tanım.
Çarpan VAL'de seçilir (gerçek/tahmin ortalama oranı), test'te raporlanır (karar kuralı: val seçim, test doğrulama).
"""
import argparse
import collections
import json

import numpy as np
import torch
from matcha.utils.model import normalize as mel_norm
from matcha.utils.utils import intersperse

from dizgetts import paths
from dizgetts.eval.dp_response import HOP_MS, frames
from dizgetts.eval.prosody_acoustics import _load_model, mas_frames
from dizgetts.frontend.symbols import PHONES, SYMBOL_TO_ID
from dizgetts.train.dpfeat import set_dp_feat, set_round


def group(t: str) -> str | None:
    if t.endswith("ː"):
        return "uzun_aː" if t == "aː" else "uzun_diğer"   # aː: alıntı/Arapça uzun a; diğer: ğ uzatması, nispet iː, eː
    if t in PHONES and PHONES[t][0] == "ünlü":
        return "kısa"
    return None


def measure(m, stats, manifest: str, split: str) -> dict:
    real, pred = collections.defaultdict(list), collections.defaultdict(list)
    for line in open(f"{paths.ANTALIA}/{split}{manifest}.jsonl", encoding="utf8"):
        r = json.loads(line)
        ids = intersperse([SYMBOL_TO_ID[t] for t in r["tokens"]], 0)
        set_dp_feat(m, None)  # önceki cümlenin sınır özniteliği kodlayıcıda kalmasın (train_dp.load_split gibi)
        fr_real, _ = mas_frames(m, ids, mel_norm(torch.load(f"{paths.ANTALIA}/mels/{r['id']}.pt"), stats["mel_mean"], stats["mel_std"]))
        fr_pred = frames(m, ids, r["dp_feat"], "cpu")
        for j, t in enumerate(r["tokens"]):
            g = group(t)
            if g:
                u = slice(2 * j + 1, 2 * j + 3)  # ünlü BİRİMİ = token + ardından gelen boşluk (ses ikisine yayılır; yalnız token sayılırsa oran çarpılır)
                real[g].append(float(fr_real[u].sum())); pred[g].append(float(fr_pred[u].sum()))
    return {g: dict(n=len(real[g]), gerçek_ms=np.mean(real[g]) * HOP_MS, tahmin_ms=np.mean(pred[g]) * HOP_MS,
                    oran=np.mean(real[g]) / np.mean(pred[g])) for g in real}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="_phon_v8")
    a = ap.parse_args()
    m, ck, stats = _load_model(a.ckpt)
    set_round(m, ck["cfg"]["model"].get("dp_round"))
    for split in ("val", "test"):
        res = measure(m, stats, a.manifest, split)
        print(f"== {split}")
        for g, d in sorted(res.items()):
            print(f"  {g:12} n={d['n']:5}  gerçek {d['gerçek_ms']:5.1f} ms  tahmin {d['tahmin_ms']:5.1f} ms  gerçek/tahmin {d['oran']:.2f}")
        k = res["kısa"]
        for g in ("uzun_aː", "uzun_diğer"):
            if g in res:
                print(f"  {g}/kısa: gerçek {res[g]['gerçek_ms'] / k['gerçek_ms']:.2f}  tahmin {res[g]['tahmin_ms'] / k['tahmin_ms']:.2f}")


if __name__ == "__main__":
    main()
