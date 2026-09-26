"""v5: YALNIZ süre tahmincisini (proj_w + dp_feat_emb) yeniden eğit; kodlayıcı/çözücü DONDURULMUŞ (experiments.yaml v5-dp-linear).
  python -X utf8 -u -m dizgetts.scripts.train_dp --ckpt <v4-400 ep400.pt> --out D:/dizgetts/runs/v5_dplin

Neden yalıtılmış: dp girdisi kodlayıcı çıktısının detach()'lı kopyası; dp'nin kaybı kodlayıcıya gradyan göndermez (tests/test_dpfeat.py 3).
Hedefler = dondurulmuş modelin kendi MAS kareleri (gerçek mel'e). Böylece yalnız SÜRELER değişir; ses kalitesi, hizalama, telaffuz aynı kalır.
Kayıp: doğrusal kare üzerinde Huber (aritmetik ortalama; log-MSE geometrik ortalama öğretip her şeyi kısaltıyordu). Sentezde cfg.model.dp_round="cum".
Başlangıç: v4-400'ün dp ağırlıkları (sıcak başlangıç). Seçim: val Huber kaybı; erken durdurma.
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from matcha.utils.model import normalize as mel_norm
from matcha.utils.model import sequence_mask
from matcha.utils.utils import intersperse

from dizgetts import paths
from dizgetts.eval.prosody_acoustics import _load_model, mas_frames
from dizgetts.frontend.symbols import SYMBOL_TO_ID
from dizgetts.train.dpfeat import intersperse_feat, set_dp_feat, token_types

ROOT = paths.ANTALIA


@torch.no_grad()
def encoder_hidden(enc, x, x_lengths):
    """DPFeatTextEncoder.forward'un proj_w öncesi kısmı: (x_hidden, x_mask)."""
    import math
    h = enc.emb(x) * math.sqrt(enc.n_channels)
    h = torch.transpose(h, 1, -1)
    x_mask = torch.unsqueeze(sequence_mask(x_lengths, h.size(2)), 1).to(h.dtype)
    h = enc.prenet(h, x_mask)
    h = enc.encoder(h, x_mask)
    return h, x_mask


def load_split(m, stats, manifest, split, dev):
    rows = [json.loads(l) for l in open(f"{ROOT}/{split}{manifest}.jsonl", encoding="utf8")]
    out = []
    for r in rows:
        ids = intersperse([SYMBOL_TO_ID[t] for t in r["tokens"]], 0)
        set_dp_feat(m, None)
        fr, _ = mas_frames(m, ids, mel_norm(torch.load(f"{ROOT}/mels/{r['id']}.pt"), stats["mel_mean"], stats["mel_std"]))
        out.append(dict(ids=torch.tensor(ids), feat=torch.tensor(intersperse_feat(r["dp_feat"])), tgt=torch.tensor(fr, dtype=torch.float32),
                        ty=torch.tensor(token_types(r["tokens"]))))
    return out


def batches(data, bs, shuffle, rng):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        ch = [data[j] for j in idx[i:i + bs]]
        L = max(len(c["ids"]) for c in ch)
        pad = lambda k, v=0: torch.stack([F.pad(c[k], (0, L - len(c[k])), value=v) for c in ch])
        yield pad("ids"), torch.tensor([len(c["ids"]) for c in ch]), pad("feat"), pad("tgt"), pad("ty", -1)


def run_epoch(enc, dp, emb, data, bs, dev, rng, opt=None, beta=2.0, loss_kind="huber"):
    train = opt is not None
    dp.train(train); emb.train(train)
    tot, n, sums = 0.0, 0, {"pred": 0.0, "tgt": 0.0}
    by_ty = {}
    for x, xl, feat, tgt, ty in batches(data, bs, train, rng):
        x, xl, feat, tgt, ty = x.to(dev), xl.to(dev), feat.to(dev), tgt.to(dev), ty.to(dev)
        h, mask = encoder_hidden(enc, x, xl)
        with torch.set_grad_enabled(train):
            logw = dp(h + emb(feat).transpose(1, 2) * mask, mask)
            pred = torch.exp(logw[:, 0]) * mask[:, 0]
            m = mask[:, 0].bool()
            # huber (beta=2 kare): büyük hatada L1 -> ORTANCA öğretir (sağa çarpık sürede kısa); mse -> aritmetik ORTALAMA
            loss = F.mse_loss(pred[m], tgt[m]) if loss_kind == "mse" else F.smooth_l1_loss(pred[m], tgt[m], beta=beta)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss) * int(m.sum()); n += int(m.sum())
        sums["pred"] += float(pred[m].sum()); sums["tgt"] += float(tgt[m].sum())
        if not train:
            for t in range(6):
                s = m & (ty == t)
                if s.any():
                    d = by_ty.setdefault(t, [0.0, 0.0, 0]); d[0] += float(pred[s].sum()); d[1] += float(tgt[s].sum()); d[2] += int(s.sum())
    return tot / n, sums["pred"] / sums["tgt"], by_ty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--loss", choices=("huber", "mse"), default="huber")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); rng = random.Random(0)
    m, ck, stats = _load_model(a.ckpt)
    assert ck["cfg"]["model"].get("dp_feat"), "v4 (dp_feat) modeli bekleniyor"
    man = ck["cfg"]["manifest"]
    t0 = time.time()
    tr, va = load_split(m, stats, man, "train", "cpu"), load_split(m, stats, man, "val", "cpu")
    print(f"MAS hedefleri: train {len(tr)} val {len(va)} klip ({time.time() - t0:.0f} sn)", flush=True)
    m.to(dev).eval()
    for mod in m.modules():  # Matcha rotary önbelleği CPU'da (MAS aşamasında) kuruldu ve .to() ile taşınmaz -> sıfırla, GPU'da yeniden kurulsun
        if hasattr(mod, "cos_cached"):
            mod.cos_cached, mod.sin_cached = None, None
    enc = m.encoder
    for p in m.parameters():
        p.requires_grad = False
    dp, emb = copy.deepcopy(enc.proj_w), copy.deepcopy(enc.dp_feat_emb)  # sıcak başlangıç (v4-400)
    # DENETİM: yeniden kurulan yol (encoder_hidden + dp + emb) modelin kendi logw'siyle birebir aynı olmalı
    c = va[0]; x, xl, f = c["ids"][None].to(dev), torch.tensor([len(c["ids"])], device=dev), c["feat"][None].to(dev)
    dp.eval(); emb.eval()
    with torch.no_grad():
        set_dp_feat(m, f); _, lw_ref, _ = enc(x, xl); set_dp_feat(m, None)
        h, mk = encoder_hidden(enc, x, xl); lw = dp(h + emb(f).transpose(1, 2) * mk, mk)
    assert torch.allclose(lw, lw_ref, atol=1e-5), float((lw - lw_ref).abs().max())
    print("denetim: yeniden kurulan dp yolu = model logw (max fark %.1e)" % float((lw - lw_ref).abs().max()), flush=True)
    for p in list(dp.parameters()) + list(emb.parameters()):
        p.requires_grad = True
    opt = torch.optim.Adam(list(dp.parameters()) + list(emb.parameters()), lr=a.lr)
    names = ["blank", "ünlü", "ünsüz", "ayraç", "noktalama", "vurgu"]
    l0, r0, bt0 = run_epoch(enc, dp, emb, va, a.bs, dev, rng, loss_kind=a.loss)
    print(f"başlangıç (v4-400 dp, log-MSE ile eğitilmiş): val Huber {l0:.3f}, toplam tahmin/hedef {r0:.3f} | "
          + " ".join(f"{names[t]} {v[0] / v[1]:.2f}" for t, v in sorted(bt0.items())), flush=True)
    os.makedirs(a.out, exist_ok=True)
    best, bad, log = float("inf"), 0, []
    for ep in range(1, a.epochs + 1):
        lt, _, _ = run_epoch(enc, dp, emb, tr, a.bs, dev, rng, opt, loss_kind=a.loss)
        lv, rv, bt = run_epoch(enc, dp, emb, va, a.bs, dev, rng, loss_kind=a.loss)
        log.append(dict(epoch=ep, train=lt, val=lv, val_ratio=rv, by_type={names[t]: v[0] / v[1] for t, v in bt.items()}))
        print(f"ep{ep:3d} train {lt:.3f} val {lv:.3f} toplam oran {rv:.3f} | " + " ".join(f"{names[t]} {v[0] / v[1]:.2f}" for t, v in sorted(bt.items())), flush=True)
        if lv < best - 1e-4:
            best, bad = lv, 0
            best_dp, best_emb = copy.deepcopy(dp.state_dict()), copy.deepcopy(emb.state_dict())
        else:
            bad += 1
            if bad >= a.patience:
                print(f"erken durdurma (ep{ep}); en iyi val {best:.3f}", flush=True); break
    # yeni checkpoint: v4-400 aynen + yeni dp ağırlıkları + sentezde birikimli yuvarlama
    new = copy.deepcopy(ck)
    for k, v in best_dp.items():
        new["model"][f"encoder.proj_w.{k}"] = v.cpu()
    for k, v in best_emb.items():
        new["model"][f"encoder.dp_feat_emb.{k}"] = v.cpu()
    new["cfg"]["model"]["dp_round"] = "cum"
    new["cfg"]["name"] = "v5_dplin"
    new["dp_retrain"] = dict(base=a.ckpt, loss=a.loss + "_linear_frames", best_val=best, log=log)
    torch.save(new, os.path.join(a.out, f"ep400_dp_{a.loss}.pt"))
    json.dump(log, open(os.path.join(a.out, f"log_{a.loss}.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("->", os.path.join(a.out, f"ep400_dp_{a.loss}.pt"), flush=True)


if __name__ == "__main__":
    main()
