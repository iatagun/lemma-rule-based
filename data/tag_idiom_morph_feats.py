#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deney Y — stage-2 için morfosentaktik kanonik-biçim sapması özelliği.

Deney I (UPOS enjeksiyonu, reddedildi) çok kaba kaldı: aynı isim+fiil ikilisi idyomatik ve
literal kullanımda AYNI POS dizisine sahip. Fazly, Cook & Stevenson (2009, "Unsupervised Type
and Token Identification of Idiomatic Expressions") literatüründeki bulgu: ayırt edici sinyal
POS değil, deyimin KENDİ KANONİK BİÇİMİNDEN SAPMA — hâl eki (case), çoğul/tekillik, çatı
(passivizasyon). DizgeBERT-Morph zaten bu FEATS'i (Case/Number/Definite/Voice) tahmin ediyor
(`morph_data/label_space.json`); bu modül `tag_idiom_upos.py::load_morph_upos_fn`'nin FEATS
ikizidir — aynı model, aynı hizalama yaklaşımı, ayrı bir çıktı.

`morph_deviation_vec()` TEK KAYNAK: hem `training/train_idiomaticity_clf.py` (eğitim, ClfDS)
hem `wrap_stage2` (çıkarım) bunu kullanır — kural biri değişince diğeri de değişsin.

Kanonik-biçim kuralı (per-deyim istatistik DEĞİL — çoğu deyimin örnek sayısı az, gürültülü
olurdu): dilbilimsel öncül — VID/LVC kanonik biçimi tipik olarak hâl-eksiz/belirsiz nesne +
etken çatı. 4 ikili özellik: [hâl-ekli mi, çoğul mu, belirli mi, çatı sapmış mı (edilgen/
ettirgen/dönüşlü/işteş)].
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MORPH_REPO = "iatagun/DizgeBERT-Morph"
SCHEME = "imst"

_DEVIANT_CASES = {"Acc", "Dat", "Abl", "Loc", "Gen", "Ins"}
_DEVIANT_VOICES = {"Pass", "Cau", "CauPass", "Rcp", "Rfl"}


def load_morph_feats_fn(device=None):
    """→ feats_for(words) -> list[dict] (words ile aynı uzunluk). Her sözlük UD FEATS'in
    yanında `_upos` alanı taşır (NOUN/VERB/... — kelimenin hangi bileşen olduğunu ayırt
    etmek için, bkz. `morph_deviation_vec`). Model `tag_idiom_upos.py` ile AYNI."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MORPH_REPO)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MORPH_REPO, trust_remote_code=True,
                                       torch_dtype=dtype).to(device).eval()
    tb = torch.tensor([{"kenet": 0, "boun": 1, "imst": 2}[SCHEME]], device=device)

    @torch.no_grad()
    def feats_for(words: list[str]) -> list[dict]:
        if not words:
            return []
        enc = tok(words, is_split_into_words=True, return_tensors="pt",
                  truncation=True, max_length=128).to(device)
        first, last = {}, {}
        for i, wid in enumerate(enc.word_ids()):
            if wid is None:
                continue
            first.setdefault(wid, i)
            last[wid] = i
        kept = sorted(first)
        fp = torch.tensor([[first[w] for w in kept]], device=device)
        lp = torch.tensor([[last[w] for w in kept]], device=device)
        out = model(enc["input_ids"], enc["attention_mask"], tb, fp, lp)
        cfg = model.config
        feat_names = list(cfg.feats_label_space.keys())
        feats_arg = {n: out.logits_feats[n].argmax(-1)[0] for n in feat_names}
        up = out.logits_upos.argmax(-1)[0]
        results = []
        for k in range(len(kept)):
            d = {n: cfg.feats_label_space[n][int(feats_arg[n][k])] for n in feat_names}
            d = {k2: v2 for k2, v2 in d.items() if v2 != "_"}
            d["_upos"] = cfg.upos_labels[int(up[k])]
            results.append(d)
        # truncation kelime düşürebilir — düşenlere boş FEATS ata
        results += [{"_upos": None}] * (len(words) - len(kept))
        return results[:len(words)]

    return feats_for


def morph_deviation_vec(feats: list[dict], s: int, e: int) -> list[float]:
    """Span [s,e) içindeki kelimelerden 4-boyutlu sapma vektörü (pure, model gerektirmez —
    hem eğitim hem çıkarımda AYNI çağrılır). Hâl/çoğul/belirlilik yalnız NESNE bileşeninde
    (NOUN/PROPN) bakılır (fiilin ÖZNE uyum eki her zaman bir Number taşır — alakasız gürültü
    olurdu); çatı sapması yalnız FİİL bileşeninde (VERB) bakılır."""
    span_feats = feats[s:e]
    nouns = [f for f in span_feats if f.get("_upos") in ("NOUN", "PROPN")]
    verbs = [f for f in span_feats if f.get("_upos") == "VERB"]
    has_case = any(f.get("Case") in _DEVIANT_CASES for f in nouns)
    is_plural = any(f.get("Number") == "Plur" for f in nouns)
    is_definite = any(f.get("Definite") == "Def" for f in nouns)
    is_deviant_voice = any(f.get("Voice") in _DEVIANT_VOICES for f in verbs)
    return [float(has_case), float(is_plural), float(is_definite), float(is_deviant_voice)]


FILES = [
    "corpus_examples_glu.json",  # yalnız bilgi amaçlı; stage-2 kendi ham havuzunu kullanıyor
]


def main() -> None:
    """Bağımsız smoke-test: birkaç örnek cümle üzerinde FEATS + sapma vektörünü yazdır."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    feats_for = load_morph_feats_fn(dev)
    examples = [
        (["Projede", "son", "aylarda", "önemli", "ölçüde", "yol", "aldık", "."], 5, 7),
        (["Otobüs", "kısa", "sürede", "çok", "yolu", "aldı", "."], 4, 6),
    ]
    for words, s, e in examples:
        feats = feats_for(words)
        vec = morph_deviation_vec(feats, s, e)
        print(" ".join(words), "→ span:", words[s:e], "feats:", feats[s:e], "vec:", vec)


if __name__ == "__main__":
    main()
