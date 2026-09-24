"""dizge-g2p-tts çıkarımı: cümle -> sözcük başına fonem (dizge, varyantlarıyla) + vurgu + sonraki sınır.
  python -X utf8 -m dizgetts.g2ptts.tagger "Şimdi gelme; belki yarın İstanbul'a gideriz."

Vurgu = SÖZLÜK ÖNCELİKLİ KARMA (g2ptts-v1 kararı, experiments.yaml): kural modülü belirleyici bir sonuç veriyorsa (kök sözlüğü + ağırlık kuralı,
clitic, ünlüsüz) o; varsayılan son heceye düşen sözcükte model (damıtılmış biçimbilim: koşaç, kişi eki, Ins, olumsuzluk...). DizgeBERT-Morph gerekmez.
Sınır = model (sesten ölçülen duraklamalarla eğitildi; eşik val'den) + noktalama; son sözcük hep "cümle".
Fonem: dizge birincil okuma; dizge birden çok okuma döndürdüyse `variants` (özenli / gündelik söyleyiş, bilinçli tasarım) — seçim kullanıcının.
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoTokenizer

from dizgetts.frontend.dep import MODEL_ID as DEP_ID, MODEL_REV as DEP_REV
from dizgetts.frontend.normalize import normalize, tr_lower
from dizgetts.frontend.phonemize import Phonemizer
from dizgetts.frontend.stress import StressRules, _n_vowels
from dizgetts.g2ptts.train import RUN, G2PTTS

PUNCT_BOUNDARY = {",": "ip", ";": "IP", ".": "cümle", "?": "cümle", "!": "cümle"}
RANK = ("0", "ip", "IP", "cümle")


class Tagger:
    def __init__(self, ckpt: str = f"{RUN}/best.pt", device: str = "cpu"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)  # kendi checkpoint'imiz
        self.model = G2PTTS(); self.model.load_state_dict(ck["state"]); self.model.eval().to(device)
        self.tau, self.device = float(ck["val"]["tau"]), device
        self.tok = AutoTokenizer.from_pretrained(DEP_ID, revision=DEP_REV)
        self.rules, self.ph = StressRules(), Phonemizer(bert_fallback=False)

    @torch.no_grad()
    def _model(self, toks: list[str]):
        enc = self.tok(toks, is_split_into_words=True, truncation=True, max_length=256, return_tensors="pt")
        first, seen = [0] * len(toks), set()
        for i, wid in enumerate(enc.word_ids(0)):
            if wid is not None and wid not in seen:
                seen.add(wid); first[wid] = i
        ls, lb = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device), torch.tensor([first], device=self.device))
        return ls[0].argmax(-1).tolist(), lb[0].softmax(-1).tolist()

    def stress(self, word: str, model_cls: int) -> tuple[int | None, str]:
        """(vurgulu seslem SONDAN sırası ya da None, kaynak)."""
        k, tag = self.rules.syllable(word)
        n = _n_vowels(tr_lower(word))
        if tag != "varsayılan_son":
            return (None if k is None else n - 1 - k), f"kural:{tag}"
        return (None if model_cls == 0 else min(model_cls - 1, n - 1)), "model"

    def __call__(self, text: str) -> dict:
        norm = normalize(text)
        toks = norm.split()
        scls, pb = self._model(toks) if toks else ([], [])
        words = []
        for i, t in enumerate(toks):
            if not t.isalpha():
                if words and t in PUNCT_BOUNDARY:
                    w = words[-1]
                    w["boundary"] = max(w["boundary"], PUNCT_BOUNDARY[t], key=RANK.index)
                continue
            r, src = self.stress(t, scls[i])
            p_break = pb[i][1] + pb[i][2]
            primary = self.ph.word(t)
            words.append(dict(word=t, phones=primary, variants=list(self.ph.variants.get(tr_lower(t), (primary,))),
                              stress_from_end=r, stress_src=src, p_break=round(p_break, 3),
                              boundary=("ip" if pb[i][1] >= pb[i][2] else "IP") if p_break > self.tau else "0"))
        if words:
            words[-1]["boundary"] = "cümle"
        return dict(text=text, norm=norm, words=words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    a = ap.parse_args()
    print(json.dumps(Tagger()(a.text), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
