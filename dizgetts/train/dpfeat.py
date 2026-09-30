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
        ids = x
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
        logw = self.proj_w(x_dp, x_mask, ids) if getattr(self.proj_w, "wants_ids", False) else self.proj_w(x_dp, x_mask)  # v9: DurPredV2 açık öznitelikleri id'lerden hesaplar
        fd = getattr(self, "flow_dp", None)  # v7: akış eşlemeli süre örneklemesi (train/flowdp.py); yalnız SENTEZDE, MAS/eğitim deterministik logw'yi görür
        if fd is not None and not self.training and getattr(self, "_flow_on", True):
            logw = fd.sample(x_dp, logw, x_mask)
        ls = getattr(self, "long_scale", None)  # uzun ünlü (ː) süre düzeltmesi, yalnız SENTEZDE (set_long_scale; eval/long_vowel.py ölçer)
        if ls and not self.training:
            hit = torch.isin(ids, self._long_ids)
            hit = hit | torch.nn.functional.pad(hit[:, :-1], (1, 0))  # ünlü BİRİMİ: token + ardından gelen boşluk (ses ikisine yayılır)
            logw = logw + math.log(ls) * hit.unsqueeze(1).to(logw.dtype) * x_mask
        if getattr(self, "_round", None) == "cum":  # v5: yalnız SENTEZDE (set_round); eğitim/MAS logw'yi değiştirmez
            logw = cum_round_logw(logw, x_mask)
        return mu, logw, x_mask


def enable(model) -> None:
    """MatchaTTS modelinin kodlayıcısını DPFeatTextEncoder'a çevirir ve sıfır başlatılmış gömmeyi ekler."""
    enc = model.encoder
    enc.__class__ = DPFeatTextEncoder
    enc.dp_feat_emb = torch.nn.Embedding(N_DP_FEAT, enc.proj_w.in_channels, padding_idx=DP_FEAT["pad"])
    torch.nn.init.zeros_(enc.dp_feat_emb.weight)
    enc._dp_feat = None


LONG_SKIP = ("aː",)  # alıntı/Arapça uzun a zaten ~2 kat uzun üretiliyor (eval/long_vowel.py, v8): ölçeklenmez


def set_long_scale(model, scale, skip=LONG_SKIP) -> None:
    """Sentezde `ː` ile biten atomların (skip hariç) ünlü biriminin süresini `scale` ile çarp; scale None/1 = kapalı.
    2026-09-30: birim token + boşluk ve aː muaf (v7b'deki ilk sürüm yalnız token'ı ve tüm ː'leri ölçekliyordu; v7b reddedildi)."""
    from dizgetts.frontend.symbols import SYMBOLS
    enc = model.encoder
    enc.long_scale = float(scale) if scale and float(scale) != 1.0 else None
    ids = [i for i, s in enumerate(SYMBOLS) if s.endswith("ː") and s not in skip]
    enc.register_buffer("_long_ids", torch.tensor(ids, device=enc.emb.weight.device), persistent=False)


def set_dp_feat(model, feat) -> None:
    if isinstance(model.encoder, DPFeatTextEncoder):
        model.encoder._dp_feat = feat


def intersperse_feat(feat: list[int]) -> list[int]:
    """Token özniteliğini add_blank dizisine yay: [blank, t0, blank, t1, ..., blank]; bir token'dan sonraki boşluk o token'ın sınıfını alır."""
    out = [DP_FEAT["in"]]
    for f in feat:
        out += [f, f]
    return out


# ---------------------------------------------------------------- token türü (scripts/train_dp.py tür başına raporlama)
# (v4c sentez anı süre kalibrasyonu REDDEDİLDİ ve kod silindi: experiments.yaml v4c-calib, son sürüm git ad2b1a1'de)
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


# ---------------------------------------------------------------- v5: birikimli yuvarlama (sentez anı)
def cum_round_logw(logw, x_mask):
    """Matcha synthesise() süreleri ceil(exp(logw)) ile tamsayıya çevirir (sabit ~+0,5 kare yanlılığı). Burada hedef tamsayı k birikimli yuvarlamayla
    bulunur (toplam korunur, token hatası < 1 kare; MAS gibi en az 1 kare) ve logw = log(k - 0,5) döndürülür -> ceil(exp(logw)) = k tam olarak."""
    w = torch.exp(logw) * x_mask
    r = torch.round(torch.cumsum(w, dim=-1))
    k = torch.diff(r, dim=-1, prepend=torch.zeros_like(r[..., :1]))
    k = torch.clamp(k, min=1.0)
    return torch.log(k - 0.5) * x_mask


def set_round(model, mode) -> None:
    if isinstance(model.encoder, DPFeatTextEncoder):
        model.encoder._round = mode


# ---------------------------------------------------------------- v9: açık öznitelikli, geniş bağlamlı süre tahmincisi (docs/v9_plan.md)
# Öznitelikler token DİZİSİNDEN hesaplanır (model içinde, id'lerden): eğitim / sentez / ölçüm betikleri aynı yolu kullanır, manifest değişmez.
# Her özniteliğin sınıf sayısı (kategorik gömme). Boşluk token'ı kendinden önceki token'ın özniteliklerini alır (intersperse_feat gibi), türü "blank".
DUR_FEATS = {"type": 6, "stress": 3, "syl_from_start": 5, "syl_from_end": 5, "word_syls": 6, "closed": 3, "words_to_punct": 6, "words_from_punct": 5}


def dur_feats(tokens: list[str]) -> list[list[int]]:
    """Token listesi -> add_blank dizisi uzunluğunda öznitelik satırları [[type, stress, ...], ...]. Sözcük = WORD_SEP/noktalama arasındaki token'lar."""
    from dizgetts.frontend.symbols import BREAKS, PAUSES, PHONES, STRESS, WORD_SEP

    vowel = lambda t: PHONES.get(t, ("",))[0] == "ünlü"
    types = token_types(tokens)[1::2]  # token başına tür (boşluklar hariç)
    # sözcükleri bul: (başlangıç, bitiş) token indeksleri
    words, cur = [], []
    for i, t in enumerate(tokens):
        if t == WORD_SEP or t in PAUSES or t in BREAKS:
            if cur:
                words.append(cur); cur = []
        else:
            cur.append(i)
    if cur:
        words.append(cur)
    punct_after = [i for i, t in enumerate(tokens) if t in PAUSES]
    row = [[types[i], 0, 0, 0, 0, 0, 0, 0] for i in range(len(tokens))]
    for wi, w in enumerate(words):
        vidx = [i for i in w if vowel(tokens[i])]
        n = len(vidx)
        # noktalamaya sözcük uzaklığı (ileri / geri)
        end, start = w[-1], w[0]
        nxt = next((p for p in punct_after if p > end), None)
        prv = max((p for p in punct_after if p < start), default=None)
        to_p = sum(1 for w2 in words[wi + 1:] if nxt is None or w2[0] < nxt)
        from_p = sum(1 for w2 in words[:wi] if prv is None or w2[0] > prv)
        for i in w:
            before = sum(1 for v in vidx if v < i)
            if vowel(tokens[i]):
                syl = before
            else:  # ünsüz: hemen ardından (vurgu işareti atlanarak) ünlü geliyorsa sonraki hecenin başı (ya-rın), değilse önceki hecenin sonu (kuz-da)
                nxt_tok = next((tokens[j] for j in w if j > i and tokens[j] != STRESS), None)
                syl = before if nxt_tok is not None and vowel(nxt_tok) else before - 1
            syl = max(syl, 0)
            r = row[i]
            r[2] = min(syl, 4); r[3] = min(max(n - 1 - syl, 0), 4); r[4] = min(n, 5)
            r[6] = min(to_p, 5); r[7] = min(from_p, 4)
            if vowel(tokens[i]):
                r[1] = 2 if i and tokens[i - 1] == STRESS else 1
                nv = next((v for v in vidx if v > i), None)  # sonraki ünlüye kadar ünsüz sayısı (sözcük içi)
                cons = [j for j in w if i < j < (nv if nv is not None else w[-1] + 1) and not vowel(tokens[j]) and tokens[j] != STRESS]
                r[5] = 2 if (len(cons) >= 2 or (nv is None and cons)) else 1  # 2 kapalı, 1 açık
    out = [[0] * len(DUR_FEATS)]
    for r in row:
        out += [r, [0] + r[1:]]  # token, ardından onun özniteliklerini taşıyan boşluk (tür 0)
    return out


class DurPredV2(torch.nn.Module):
    """Geniş bağlamlı deterministik süre tahmincisi: x (detach'lı kodlayıcı + dp_feat) + açık öznitelik gömmeleri -> genişletilmiş evrişim yığını -> logw.
    Alıcı alan: kernel 3, genişleme (1,2,4,8) x `repeats` -> ~60 token (eski DurationPredictor: 5)."""
    wants_ids = True

    def __init__(self, in_channels: int, channels: int = 256, repeats: int = 2, dilations=(1, 2, 4, 8), feat_dim: int = 32, p_dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.feat_emb = torch.nn.ModuleList([torch.nn.Embedding(n, feat_dim) for n in DUR_FEATS.values()])
        self.inp = torch.nn.Conv1d(in_channels + feat_dim * len(DUR_FEATS), channels, 1)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(channels, channels, 3, padding=d, dilation=d) for _ in range(repeats) for d in dilations])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(channels) for _ in self.convs])
        self.drop = torch.nn.Dropout(p_dropout)
        self.proj = torch.nn.Conv1d(channels, 1, 1)
        self._cache = {}

    def feats(self, ids):
        """(B, T) id -> (B, T, n_feat) öznitelik; token dizisi id'lerden geri kurulur (boşluklar atlanır)."""
        from dizgetts.frontend.symbols import SYMBOLS
        out = []
        for row in ids.tolist():
            key = tuple(row)
            if key not in self._cache:
                toks = [SYMBOLS[i] for i in row[1::2]]
                f = dur_feats([t for t in toks if t != SYMBOLS[0]])  # sondaki dolgu (id 0) atılır
                f += [[0] * len(DUR_FEATS)] * (len(row) - len(f))
                if len(self._cache) > 4096:
                    self._cache.clear()
                self._cache[key] = f
            out.append(self._cache[key])
        return torch.tensor(out, device=ids.device)

    def forward(self, x, x_mask, ids):
        f = self.feats(ids)  # (B, T, F)
        e = torch.cat([emb(f[..., k]) for k, emb in enumerate(self.feat_emb)], dim=-1).transpose(1, 2)
        h = self.inp(torch.cat([x, e], dim=1) * x_mask)
        for conv, norm in zip(self.convs, self.norms):
            y = torch.relu(conv(h * x_mask))
            y = norm(y.transpose(1, 2)).transpose(1, 2)
            h = h + self.drop(y)
        return self.proj(h * x_mask) * x_mask


def enable_dp2(model, **kw) -> None:
    """Kodlayıcının süre tahmincisini DurPredV2 ile değiştirir (cfg.model.dp2 = kw)."""
    enc = model.encoder
    enc.proj_w = DurPredV2(enc.proj_w.in_channels, **kw)
