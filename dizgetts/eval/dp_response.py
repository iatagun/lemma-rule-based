"""Süre tahmincisi sınır özniteliğine ne kadar TEPKİ veriyor? Aynı cümlelerde sınır kuralı kapalı/açık -> tahmin edilen süre farkı (seslendirme yok).
  python -X utf8 -m dizgetts.eval.dp_response CKPT [CKPT ...] --texts dosya.txt
Rapor: kuralın değiştirdiği sözcük sonlarında (ayraç token'ı + son hece) kare farkı; IP'ye yükseltilen / 0'a indirilen sınırlarda ayrı."""
import argparse, math

import numpy as np
import torch
from matcha.utils.utils import intersperse

from dizgetts.engine import Engine
from dizgetts.frontend.symbols import WORD_SEP
from dizgetts.train.dpfeat import cum_round_logw, intersperse_feat, set_dp_feat
from dizgetts.train.train import build_model

HOP_MS = 256 / 22050 * 1000


@torch.no_grad()
def frames(m, eng_ids, feat, dev):
    x = torch.tensor(eng_ids, device=dev)[None]
    set_dp_feat(m, torch.tensor(intersperse_feat(feat), device=dev)[None])
    _, logw, mask = m.encoder(x, torch.tensor([x.shape[1]], device=dev), None)
    return (torch.ceil(torch.exp(cum_round_logw(logw, mask))) * mask)[0, 0].cpu().numpy()  # cum_round_logw log(k-0,5) döndürür -> k kare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+"); ap.add_argument("--texts", required=True)
    a = ap.parse_args()
    dev = "cuda"
    texts = [l.strip() for l in open(a.texts, encoding="utf8") if l.strip() and not l.startswith("#")]
    for ck_path in a.ckpts:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        stats = dict(mel_mean=0.0, mel_std=1.0)
        m = build_model(ck["cfg"], len(ck["symbols"]), stats); m.load_state_dict(ck["model"]); m.to(dev).eval()
        ecfg = {k: v for k, v in ck["cfg"]["engine"].items() if k != "boundary_rules"}
        e0, e1 = Engine(**ecfg), Engine(**ecfg, boundary_rules=True)
        up, down = [], []
        for t in texts:
            u0, u1 = e0.frontend(t), e1.frontend(t)
            if u0.tokens != u1.tokens or u0.dp_feat == u1.dp_feat:
                continue
            ids = intersperse(e0.ids(u0), 0)
            f0, f1 = frames(m, ids, u0.dp_feat, dev), frames(m, ids, u1.dp_feat, dev)
            # sözcük sonu bölgesi: ayraç token'ı (interspersed 2j+1) ve komşu boşluklar
            for j, tok in enumerate(u0.tokens):
                if tok == WORD_SEP and u0.dp_feat[j] != u1.dp_feat[j]:
                    s = slice(2 * j, 2 * j + 3)
                    d = (f1[s].sum() - f0[s].sum()) * HOP_MS
                    (up if u1.dp_feat[j] > u0.dp_feat[j] else down).append(d)
        up, down = np.array(up), np.array(down)
        print(f"{ck_path}\n  sınır YÜKSELTİLDİ (n={len(up)}): duraklama değişimi medyan {np.median(up):+.0f} ms, ort {up.mean():+.0f}\n"
              f"  sınır İNDİRİLDİ  (n={len(down)}): medyan {np.median(down):+.0f} ms, ort {down.mean():+.0f}")


if __name__ == "__main__":
    main()
