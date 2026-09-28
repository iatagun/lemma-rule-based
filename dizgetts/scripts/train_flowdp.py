"""v7: akış eşlemeli süre tahmincisini (train/flowdp.py) dondurulmuş v6-dp üzerinde eğit. Ses/telaffuz/deterministik dp DEĞİŞMEZ; yalnız FlowDP eğitilir.
  python -X utf8 -u -m dizgetts.scripts.train_flowdp --ckpt D:/dizgetts/runs/v6_dplin/ep400_dp_mse.pt --out D:/dizgetts/runs/v7_flowdp

Hedef: MAS kareleri (train_dp.py ile aynı düzenek). Seçim: val CFM kaybı (sabit tohum). Her 10 epoch'ta val'de ÖRNEKLENEN sürelerin token türüne göre
ortalama ve std oranı (tahmin/gerçek) raporlanır — amaç std oranını ~0,6'dan 1'e yaklaştırırken ortalamayı korumak.
"""
import argparse, copy, os, random, time

import numpy as np
import torch

from dizgetts.eval.prosody_acoustics import _load_model
from dizgetts.scripts.train_dp import batches, encoder_hidden, load_split
from dizgetts.train.dpfeat import cum_round_logw
from dizgetts.train.flowdp import FlowDP

NAMES = ["blank", "ünlü", "ünsüz", "ayraç", "noktalama", "vurgu"]


@torch.no_grad()
def cond_of(enc, x, xl, feat):
    h, mask = encoder_hidden(enc, x, xl)
    x_dp = h + enc.dp_feat_emb(feat).transpose(1, 2) * mask
    return x_dp, enc.proj_w(x_dp, mask), mask


@torch.no_grad()
def evaluate(enc, fd, data, dev, bs, seed=0):
    fd.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    tot, n = 0.0, 0
    per = {t: ([], []) for t in range(6)}
    torch.manual_seed(seed)
    for x, xl, feat, tgt, ty in batches(data, bs, False, None):
        x, xl, feat, tgt, ty = x.to(dev), xl.to(dev), feat.to(dev), tgt.to(dev), ty.to(dev)
        x_dp, lw, mask = cond_of(enc, x, xl, feat)
        tot += float(fd.loss(tgt, x_dp, lw, mask)) * int(mask.sum()); n += int(mask.sum())
        k = torch.ceil(torch.exp(cum_round_logw(fd.sample(x_dp, lw, mask), mask))) * mask
        m = mask[:, 0].bool()
        for t in range(6):
            s = m & (ty == t)
            per[t][0].append(k[:, 0][s].cpu()); per[t][1].append(tgt[s].cpu())
    stats = {}
    for t, (p, r) in per.items():
        if p:
            p, r = torch.cat(p).numpy(), torch.cat(r).numpy()
            if len(r):
                stats[NAMES[t]] = (p.mean() / r.mean(), p.std() / max(r.std(), 1e-6))
    return tot / n, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0); rng = random.Random(0)
    m, ck, stats = _load_model(a.ckpt)
    man = ck["cfg"]["manifest"]
    t0 = time.time()
    tr, va = load_split(m, stats, man, "train", "cpu"), load_split(m, stats, man, "val", "cpu")
    print(f"MAS hedefleri: train {len(tr)} val {len(va)} ({time.time() - t0:.0f} sn)", flush=True)
    m.to(dev).eval()
    for mod in m.modules():
        if hasattr(mod, "cos_cached"):
            mod.cos_cached, mod.sin_cached = None, None
    enc = m.encoder
    for p in m.parameters():
        p.requires_grad = False
    lk = torch.cat([torch.log(c["tgt"].clamp(min=1)) for c in tr])
    fd = FlowDP(cond_ch=enc.proj_w.in_channels).to(dev)
    fd.mu.fill_(float(lk.mean())); fd.sd.fill_(float(lk.std()))
    opt = torch.optim.Adam(fd.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(a.out, "tensorboard"))  # canlı izleme: tensorboard --logdir <out>/tensorboard --host 127.0.0.1
    best, best_sd, log = float("inf"), None, []
    for ep in range(1, a.epochs + 1):
        fd.train()
        tl, tn = 0.0, 0
        for x, xl, feat, tgt, _ in batches(tr, a.bs, True, rng):
            x, xl, feat, tgt = x.to(dev), xl.to(dev), feat.to(dev), tgt.to(dev)
            x_dp, lw, mask = cond_of(enc, x, xl, feat)
            loss = fd.loss(tgt, x_dp, lw, mask)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(fd.parameters(), 1.0); opt.step()
            tl += float(loss); tn += 1
        tb.add_scalar("kayıp/train", tl / tn, ep); tb.flush()
        if ep % 10 == 0 or ep == 1:
            vl, st = evaluate(enc, fd, va, dev, a.bs)
            tb.add_scalar("kayıp/val", vl, ep)
            for k, v in st.items():
                tb.add_scalar(f"ortalama_oranı/{k}", float(v[0]), ep); tb.add_scalar(f"std_oranı/{k}", float(v[1]), ep)
            tb.flush()
            log.append(dict(epoch=ep, train=tl / tn, val=vl, stats={k: [float(v[0]), float(v[1])] for k, v in st.items()}))
            print(f"ep{ep:4d} train {tl / tn:.4f} val {vl:.4f} | ort/std oranı " + " ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in st.items()), flush=True)
            if vl < best:
                best, best_sd = vl, copy.deepcopy(fd.state_dict())
    new = copy.deepcopy(ck)
    for k, v in best_sd.items():
        new["model"][f"encoder.flow_dp.{k}"] = v.cpu()
    new["cfg"]["model"]["flow_dp"] = dict(args=dict(cond_ch=enc.proj_w.in_channels), temperature=1.0)
    new["cfg"]["name"] = "v7_flowdp"
    new["flowdp_train"] = dict(base=a.ckpt, best_val=best, log=log)
    torch.save(new, os.path.join(a.out, "v7_flowdp.pt"))
    print("->", os.path.join(a.out, "v7_flowdp.pt"), f"(en iyi val {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
