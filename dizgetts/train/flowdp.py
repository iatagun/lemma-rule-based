"""Akış eşlemeli (OT-CFM) süre tahmincisi: deterministik dp'nin DARALTTIĞI süre dağılımını (tahmin std'si gerçeğin ~%60'ı, her token türünde;
reports/.. docs/prosody_research.md §8) örneklemeyle geri getirir. Matcha çözücüsünün mel için yaptığını süreler için yapar.

Girdi (koşul): kodlayıcı gizli durumu + sınır özniteliği gömmesi (x_dp, dondurulmuş) ve deterministik dp'nin log-süre tahmini (ortalama çapası).
Hedef: MAS karesi k >= 1 için y = (log k - mu) / sd. Sentez: gürültüden Euler ile y, sonra log k -> kodlayıcı logw'si yerine (birikimli yuvarlama sonra).
Kodlayıcıya bağlama: DPFeatTextEncoder.forward, `flow_dp` varsa ve eğitimde değilse logw'yi bununla değiştirir (train/dpfeat.py).
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _t_emb(t, dim=64):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * f[None] * 1000
    return torch.cat([a.sin(), a.cos()], -1)


class FlowDP(nn.Module):
    def __init__(self, cond_ch: int = 192, hid: int = 256, layers: int = 6, k: int = 3, p_drop: float = 0.1, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min
        self.t_mlp = nn.Sequential(nn.Linear(64, hid), nn.SiLU(), nn.Linear(hid, hid))
        self.inp = nn.Conv1d(1 + cond_ch + 1, hid, 1)
        self.convs = nn.ModuleList(nn.Conv1d(hid, hid, k, padding=k // 2) for _ in range(layers))
        self.films = nn.ModuleList(nn.Linear(hid, 2 * hid) for _ in range(layers))
        self.norms = nn.ModuleList(nn.GroupNorm(1, hid) for _ in range(layers))
        self.drop = nn.Dropout(p_drop)
        self.out = nn.Conv1d(hid, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer("mu", torch.tensor(0.0)); self.register_buffer("sd", torch.tensor(1.0))
        self.temperature, self.steps = 1.0, 10

    def velocity(self, y, t, cond, mask):
        h = self.inp(torch.cat([y, cond], 1)) * mask
        te = self.t_mlp(_t_emb(t))
        for conv, film, norm in zip(self.convs, self.films, self.norms):
            s, b = film(te).unsqueeze(-1).chunk(2, 1)
            h = h + self.drop(torch.nn.functional.gelu(norm(conv(h * mask)) * (1 + s) + b)) * mask
        return self.out(h) * mask

    def cond(self, x_dp, logw_det):
        return torch.cat([x_dp, (logw_det - self.mu) / self.sd], 1)

    def loss(self, frames, x_dp, logw_det, mask):
        y1 = (torch.log(frames.clamp(min=1.0)).unsqueeze(1) - self.mu) / self.sd * mask
        x0 = torch.randn_like(y1)
        t = torch.rand(y1.shape[0], device=y1.device)
        tt = t[:, None, None]
        yt = (1 - (1 - self.sigma_min) * tt) * x0 + tt * y1
        u = y1 - (1 - self.sigma_min) * x0
        v = self.velocity(yt, t, self.cond(x_dp, logw_det), mask)
        return (((v - u) ** 2) * mask).sum() / mask.sum()

    @torch.no_grad()
    def sample(self, x_dp, logw_det, mask):
        """-> log kare (kodlayıcı logw'si ile aynı ölçek)."""
        c = self.cond(x_dp, logw_det)
        y = torch.randn_like(logw_det) * self.temperature * mask
        dt = 1.0 / self.steps
        for i in range(self.steps):
            t = torch.full((y.shape[0],), i * dt, device=y.device)
            y = y + dt * self.velocity(y, t, c, mask)
        return (y * self.sd + self.mu) * mask


if __name__ == "__main__":  # öz-denetim: şekiller, sıfır başlatılmış çıkış -> hız 0, örnek = gürültü ölçeğinde
    f = FlowDP()
    B, T = 2, 7
    x, lw, m = torch.randn(B, 192, T), torch.randn(B, 1, T), torch.ones(B, 1, T)
    m[1, :, 5:] = 0
    assert f.velocity(torch.randn(B, 1, T), torch.rand(B), f.cond(x, lw), m).abs().max() == 0
    l = f.loss(torch.randint(1, 10, (B, T)).float(), x, lw, m)
    assert l.item() > 0 and f.sample(x, lw, m).shape == (B, 1, T) and f.sample(x, lw, m)[1, :, 5:].abs().max() == 0
    print("flowdp öz-denetim OK")
