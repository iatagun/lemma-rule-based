"""Süre tahmincisine sınır özniteliği (v4). Matcha TextEncoder'ın `x_dp` girdisine token başına bir gömme eklenir.

Neden yalnız süre tahmincisi: MAS hizalaması kodlayıcının `mu` çıktısını kullanır; `x_dp` hizalamaya girmez. v3'te sınır bilgisi TOKEN olarak
kodlayıcıya verilince hizalama oturmadı (experiments.yaml v3-g2ptts-tts); burada `mu` ve token dizisi v3a ile birebir aynı kalır.
Gömme SIFIRLA başlatılır: eğitimin başında model v3a ile aynı davranır (tests/test_dpfeat.py).
Öznitelik modele `set_dp_feat(model, feat)` ile verilir (Matcha forward/synthesise imzasına dokunmadan); feat = (B, T) interspersed uzunlukta.
"""
from __future__ import annotations

import math

import torch
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import sequence_mask

from dizgetts.engine import DP_FEAT

N_DP_FEAT = len(DP_FEAT)


class DPFeatTextEncoder(TextEncoder):
    """TextEncoder.forward ile aynı; tek fark `x_dp += dp_feat_emb(feat)` (matcha-tts 0.0.7.2 text_encoder.py:378-410)."""

    def forward(self, x, x_lengths, spks=None):
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.prenet(x, x_mask)
        if self.n_spks > 1:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(x, x_mask)
        mu = self.proj_m(x) * x_mask
        x_dp = torch.detach(x)
        feat = self._dp_feat
        if feat is not None:
            assert feat.shape == x_mask.shape[::2], (tuple(feat.shape), tuple(x_mask.shape))
            x_dp = x_dp + self.dp_feat_emb(feat).transpose(1, 2) * x_mask
        logw = self.proj_w(x_dp, x_mask)
        cal, grp = getattr(self, "_calib", None), getattr(self, "_calib_groups", None)
        if cal is not None and grp is not None:  # v4c: yalnız SENTEZDE (set_calib); eğitimde hiç ayarlanmaz
            logw = apply_calib(logw, x_mask, grp, cal)
        return mu, logw, x_mask


def enable(model) -> None:
    """MatchaTTS modelinin kodlayıcısını DPFeatTextEncoder'a çevirir ve sıfır başlatılmış gömmeyi ekler."""
    enc = model.encoder
    enc.__class__ = DPFeatTextEncoder
    enc.dp_feat_emb = torch.nn.Embedding(N_DP_FEAT, enc.proj_w.in_channels, padding_idx=DP_FEAT["pad"])
    torch.nn.init.zeros_(enc.dp_feat_emb.weight)
    enc._dp_feat = None


def set_dp_feat(model, feat) -> None:
    if isinstance(model.encoder, DPFeatTextEncoder):
        model.encoder._dp_feat = feat


def intersperse_feat(feat: list[int]) -> list[int]:
    """Token özniteliğini add_blank dizisine yay: [blank, t0, blank, t1, ..., blank]; bir token'dan sonraki boşluk o token'ın sınıfını alır."""
    out = [DP_FEAT["in"]]
    for f in feat:
        out += [f, f]
    return out


# ---------------------------------------------------------------- v4c: sentez anı süre kalibrasyonu (eğitim yok)
TOKEN_TYPES = ("blank", "vowel", "consonant", "sep", "punct", "stress")


def token_types(tokens: list[str]) -> list[int]:
    """add_blank dizisi için token türü: [blank, t0, blank, t1, ..., blank]."""
    from dizgetts.frontend.symbols import PAUSES, PHONES, STRESS, WORD_SEP

    def ty(t):
        if t == WORD_SEP:
            return 3
        if t in PAUSES:
            return 4
        if t == STRESS:
            return 5
        return 1 if PHONES.get(t, ("",))[0] == "ünlü" else 2
    out = [0]
    for t in tokens:
        out += [ty(t), 0]
    return out


def calib_groups(feat_inter: list[int], types_inter: list[int]) -> list[int]:
    return [f * len(TOKEN_TYPES) + t for f, t in zip(feat_inter, types_inter)]


def apply_calib(logw, x_mask, groups, table):
    """Grup başına log1p(süre) üzerinde moment eşleme: d' = my + (sy/sx)(d - mx). table: (N_grup, 4) = mx, sx, my, sy (özdeşlik: 0,1,0,1)."""
    d = torch.log1p(torch.exp(logw))
    p = table[groups]  # (B, T, 4)
    mx, sx, my, sy = (p[..., k].unsqueeze(1) for k in range(4))
    d2 = my + (sy / sx) * (d - mx)
    return torch.log(torch.expm1(d2).clamp(min=0) + 1e-8) * x_mask


def set_calib(model, table, groups) -> None:
    if isinstance(model.encoder, DPFeatTextEncoder):
        model.encoder._calib, model.encoder._calib_groups = table, groups


def identity_table():
    t = torch.zeros(N_DP_FEAT * len(TOKEN_TYPES), 4)
    t[:, 1] = 1.0; t[:, 3] = 1.0
    return t
