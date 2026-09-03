# -*- coding: utf-8 -*-
"""DizgeBERT-Joint — HF PreTrainedModel.

Tek ELECTRA'da UPOS + XPOS + FEATS + HEAD + DEPREL. Katman yapısı
`train_joint.JointModel` ile birebir aynı (aynı state_dict anahtarları).
`trust_remote_code=True` gerektirir; kendi içinde tam (Biaffine + Chu-Liu/Edmonds).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_dizgebert_joint import DizgeBertJointConfig


def _resize_token_type_embeddings(encoder, n: int) -> None:
    emb = encoder.embeddings.token_type_embeddings
    if emb.num_embeddings >= n:
        return
    new = nn.Embedding(n, emb.embedding_dim)
    with torch.no_grad():
        new.weight[: emb.num_embeddings] = emb.weight
        new.weight[emb.num_embeddings:] = emb.weight.mean(0, keepdim=True)
    encoder.embeddings.token_type_embeddings = new
    encoder.config.type_vocab_size = n


class Biaffine(nn.Module):
    def __init__(self, in1, in2, out, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x, self.bias_y = bias_x, bias_y
        self.U = nn.Parameter(torch.zeros(out, in1 + bias_x, in2 + bias_y))

    def forward(self, x1, x2):
        if self.bias_x:
            x1 = torch.cat([x1, x1.new_ones(*x1.shape[:-1], 1)], -1)
        if self.bias_y:
            x2 = torch.cat([x2, x2.new_ones(*x2.shape[:-1], 1)], -1)
        s = torch.einsum("bni,oij,bmj->bonm", x1, self.U, x2)
        return s.squeeze(1) if s.size(1) == 1 else s.permute(0, 2, 3, 1)


def mst(scores: np.ndarray) -> np.ndarray:
    """scores[d, h] = h→d. 0 = root. head[d] (d=1..n), head[0] = -1. Chu-Liu/Edmonds."""
    n = scores.shape[0]
    heads = scores.argmax(1)
    heads[0] = -1
    for _ in range(n):
        cyc = _find_cycle(heads, n)
        if cyc is None:
            return heads
        heads = _contract(scores, heads, cyc, n)
    return heads


def _find_cycle(heads, n):
    for s in range(1, n):
        seen, cur = set(), s
        while cur > 0 and cur not in seen:
            seen.add(cur)
            cur = heads[cur]
        if cur > 0:
            cyc, node = [], cur
            while True:
                cyc.append(node)
                node = heads[node]
                if node == cur:
                    break
            return set(cyc)
    return None


def _contract(scores, heads, cyc, n):
    cyc = sorted(cyc)
    in_s = {d: scores[d, heads[d]] for d in cyc}
    best, bd, bh = -1e18, None, None
    for d in cyc:
        for hh in range(n):
            if hh in cyc:
                continue
            g = scores[d, hh] - in_s[d]
            if g > best:
                best, bd, bh = g, d, hh
    heads = heads.copy()
    heads[bd] = bh
    return heads


class DizgeBertJointForParsing(PreTrainedModel):
    config_class = DizgeBertJointConfig

    def __init__(self, config: DizgeBertJointConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(config.encoder_name))
        _resize_token_type_embeddings(self.encoder, len(config.treebanks))
        h = self.encoder.config.hidden_size
        d = 2 * h + config.tb_emb_dim
        self.feat_names = list(config.feats_label_space.keys())
        self.tb_emb = nn.Embedding(len(config.treebanks), config.tb_emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.upos_head = nn.Linear(d, len(config.upos_labels))
        self.xpos_head = nn.Linear(d, len(config.xpos_labels))
        self.feat_heads = nn.ModuleDict(
            {n: nn.Linear(d, len(v)) for n, v in config.feats_label_space.items()})
        self.root = nn.Parameter(torch.zeros(d))
        mlp = lambda o: nn.Sequential(nn.Linear(d, o), nn.ReLU(), nn.Dropout(config.dropout))
        self.arc_h, self.arc_d = mlp(config.arc_dim), mlp(config.arc_dim)
        self.lab_h, self.lab_d = mlp(config.lab_dim), mlp(config.lab_dim)
        self.arc_biaf = Biaffine(config.arc_dim, config.arc_dim, 1, bias_x=True, bias_y=False)
        self.lab_biaf = Biaffine(config.lab_dim, config.lab_dim, len(config.deprels))
        self._tb = {tb: i for i, tb in enumerate(config.treebanks)}
        self.post_init()

    def forward(self, input_ids, attention_mask=None, treebank_id=None,
                first_pos=None, last_pos=None):
        B, L = input_ids.shape
        if treebank_id is None:
            treebank_id = input_ids.new_full((B,), self._tb[self.config.default_scheme])
        tti = treebank_id[:, None].expand(-1, L)
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=tti).last_hidden_state
        if first_pos is None:
            first_pos = last_pos = torch.arange(L, device=input_ids.device)[None].expand(B, -1)
        H = hs.size(-1)
        w = torch.cat([hs.gather(1, first_pos.unsqueeze(-1).expand(-1, -1, H)),
                       hs.gather(1, last_pos.unsqueeze(-1).expand(-1, -1, H))], -1)
        tb = self.tb_emb(treebank_id)[:, None, :].expand(-1, w.size(1), -1)
        z = self.dropout(torch.cat([w, tb], -1))
        ze = torch.cat([self.root.expand(B, 1, -1), z], 1)
        return ModelOutput(
            logits_upos=self.upos_head(z), logits_xpos=self.xpos_head(z),
            logits_feats={n: hd(z) for n, hd in self.feat_heads.items()},
            arc=self.arc_biaf(self.arc_d(ze), self.arc_h(ze)),
            lab=self.lab_biaf(self.lab_d(ze), self.lab_h(ze)),
        )

    @torch.no_grad()
    def predict(self, words: list[str], scheme: str | None = None, tokenizer=None):
        """Ön-token'lanmış cümle → [(form, upos, xpos, feats, head, deprel)]. head 0 = root."""
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config._name_or_path)
        scheme = scheme or self.config.default_scheme
        enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                        truncation=True, max_length=self.config.max_len).to(self.device)
        first, last = {}, {}
        for i, wid in enumerate(enc.word_ids()):
            if wid is None:
                continue
            first.setdefault(wid, i)
            last[wid] = i
        kept = sorted(first)
        fp = torch.tensor([[first[w] for w in kept]], device=self.device)
        lp = torch.tensor([[last[w] for w in kept]], device=self.device)
        tb = torch.tensor([self._tb[scheme]], device=self.device)
        o = self.forward(enc["input_ids"], enc["attention_mask"], tb, fp, lp)
        cfg = self.config
        up = o.logits_upos.argmax(-1)[0]
        xp = o.logits_xpos.argmax(-1)[0]
        fa = {n: o.logits_feats[n].argmax(-1)[0] for n in self.feat_names}
        sc = o.arc[0, :len(kept) + 1, :len(kept) + 1].float().cpu().numpy()
        heads = mst(sc)
        lab = o.lab[0].float().cpu()
        rows = []
        for k in range(len(kept)):
            pairs = sorted(((n, cfg.feats_label_space[n][int(fa[n][k])])
                            for n in self.feat_names if int(fa[n][k]) > 0),
                           key=lambda kv: kv[0].lower())
            feats = "|".join(f"{a}={b}" for a, b in pairs) or "_"
            hh = int(heads[k + 1])
            deprel = cfg.deprels[int(lab[k + 1, hh].argmax())]
            rows.append((words[k], cfg.upos_labels[up[k]], cfg.xpos_labels[xp[k]],
                         feats, hh, deprel))
        return rows
