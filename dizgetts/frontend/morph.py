"""Morfolojik özellikler (UPOS + UD FEATS): iatagun/DizgeBERT-Morph. Vurgu kuralları (frontend/stress.py, M1b) bunlarla kapılanır:
Case=Ins (-(y)lA), Polarity=Neg (olumsuzluk), Person/Number (kişi eki), POS=ADV (belirteç) vb. Ek SINIRI bu modelden gelmez; sınır yüzey biçimiyle bulunur.

Model, cümleyi UD gibi önceden bölünmüş belirteçler (noktalama dahil) olarak alır. normalize() çıktısı zaten boşlukla ayrılmış belirteçlerdir.
"""
from __future__ import annotations

MODEL_ID = "iatagun/DizgeBERT-Morph"
SCHEME = "imst"  # UD literatürünün referans treebank şeması (model kartı)


def parse_feats(s: str) -> dict[str, str]:
    return {} if s in ("_", "", None) else dict(kv.split("=", 1) for kv in s.split("|"))


class MorphAnalyzer:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval().to(device)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self._torch = torch

    def analyze(self, tokens: list[str]) -> list[tuple[str, dict[str, str]]]:
        """tokens (noktalama dahil) -> her biri için (UPOS, FEATS sözlüğü)."""
        if not tokens:
            return []
        with self._torch.inference_mode():
            out = self.model.predict(tokens, scheme=SCHEME, tokenizer=self.tok)
        return [(u, parse_feats(f)) for u, _x, f in out]
