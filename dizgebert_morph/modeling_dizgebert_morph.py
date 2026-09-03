# -*- coding: utf-8 -*-
"""DizgeBERT-Morph — HF PreTrainedModel sarmalayıcı.

Katman yapısı `train_morph_bert.MorphTagger` ile BİREBİR aynı (aynı state_dict
anahtarları) → eğitim checkpoint'i doğrudan yüklenir.

Kelime temsili = ilk subword ⊕ son subword. Treebank kimliği token_type_ids ile
encoder'a + tb_emb ile head'lere.

Kullanım:
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained("iatagun/DizgeBERT-Morph", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained("iatagun/DizgeBERT-Morph")
    print(m.predict(["Yarın", "İstanbul'a", "gideceğim", "."], scheme="boun", tokenizer=tok))
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_dizgebert_morph import DizgeBertMorphConfig


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


class DizgeBertMorphForMorphology(PreTrainedModel):
    config_class = DizgeBertMorphConfig

    def __init__(self, config: DizgeBertMorphConfig):
        super().__init__(config)
        # Kaydedilmiş modelde encoder ağırlıkları state_dict'te; iskeleti config'ten kur.
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(config.encoder_name))
        _resize_token_type_embeddings(self.encoder, len(config.treebanks))
        h = self.encoder.config.hidden_size
        self.tb_emb = nn.Embedding(len(config.treebanks), config.tb_emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        d = 2 * h + config.tb_emb_dim
        self.feat_names = list(config.feats_label_space.keys())
        self.upos_head = nn.Linear(d, len(config.upos_labels))
        self.xpos_head = nn.Linear(d, len(config.xpos_labels))
        self.feat_heads = nn.ModuleDict(
            {n: nn.Linear(d, len(v)) for n, v in config.feats_label_space.items()}
        )
        self._tb_to_id = {tb: i for i, tb in enumerate(config.treebanks)}
        self.post_init()

    def forward(self, input_ids, attention_mask=None, treebank_id=None,
                first_pos=None, last_pos=None):
        B, L = input_ids.shape
        if treebank_id is None:
            treebank_id = input_ids.new_full(
                (B,), self._tb_to_id[self.config.default_scheme]
            )
        tti = treebank_id[:, None].expand(-1, L)
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=tti).last_hidden_state
        if first_pos is None:  # her subword'ü kendi "kelimesi" say
            first_pos = last_pos = torch.arange(L, device=input_ids.device)[None].expand(B, -1)
        H = hs.size(-1)
        f = hs.gather(1, first_pos.unsqueeze(-1).expand(-1, -1, H))
        g = hs.gather(1, last_pos.unsqueeze(-1).expand(-1, -1, H))
        w = torch.cat([f, g], dim=-1)
        tb = self.tb_emb(treebank_id)[:, None, :].expand(-1, w.size(1), -1)
        z = self.dropout(torch.cat([w, tb], dim=-1))
        return ModelOutput(
            logits_upos=self.upos_head(z),
            logits_xpos=self.xpos_head(z),
            logits_feats={n: head(z) for n, head in self.feat_heads.items()},
        )

    # ── kolaylık: ön-token'lanmış kelime listesi → (upos, xpos, feats_str) ──
    @torch.no_grad()
    def predict(self, words: list[str], scheme: str | None = None, tokenizer=None):
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config._name_or_path)
        scheme = scheme or self.config.default_scheme
        enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                        truncation=True, max_length=self.config.max_len).to(self.device)
        word_ids = enc.word_ids()
        first, last = {}, {}
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            first.setdefault(wid, i)
            last[wid] = i
        kept = sorted(first)
        fp = torch.tensor([[first[w] for w in kept]], device=self.device)
        lp = torch.tensor([[last[w] for w in kept]], device=self.device)
        tb = torch.tensor([self._tb_to_id[scheme]], device=self.device)
        o = self.forward(enc["input_ids"], enc["attention_mask"], tb, fp, lp)
        cfg = self.config
        up = o.logits_upos.argmax(-1)[0]
        xp = o.logits_xpos.argmax(-1)[0]
        feats_arg = {n: o.logits_feats[n].argmax(-1)[0] for n in self.feat_names}
        results = []
        for k in range(len(kept)):
            pairs = [(n, cfg.feats_label_space[n][int(feats_arg[n][k])])
                     for n in self.feat_names if int(feats_arg[n][k]) > 0]
            pairs.sort(key=lambda kv: kv[0].lower())
            results.append((cfg.upos_labels[up[k]], cfg.xpos_labels[xp[k]],
                            "|".join(f"{a}={b}" for a, b in pairs) or "_"))
        return results
