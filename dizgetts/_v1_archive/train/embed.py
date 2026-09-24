"""Matcha TextEncoder için çarpanlı gömme: id = fonem + V*öznitelik. emb(fonem) + emb_feat(öznitelik). Öznitelik gömmesi sıfırdan başlar
(başlangıçta düz fonem modeliyle aynı davranır). Blank/PAD (id 0) -> fonem 0, öznitelik 0."""
import torch


class FactorizedEmbedding(torch.nn.Module):
    def __init__(self, n_vocab: int, n_feats: int, dim: int):
        super().__init__()
        self.n_vocab = n_vocab
        self.phone = torch.nn.Embedding(n_vocab, dim)
        self.feat = torch.nn.Embedding(n_feats, dim)
        torch.nn.init.normal_(self.phone.weight, 0.0, dim ** -0.5)  # Matcha TextEncoder ile aynı
        torch.nn.init.zeros_(self.feat.weight)

    def forward(self, ids):
        return self.phone(ids % self.n_vocab) + self.feat(ids // self.n_vocab)
