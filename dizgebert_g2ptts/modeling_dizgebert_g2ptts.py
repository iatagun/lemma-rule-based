# -*- coding: utf-8 -*-
"""DizgeBERT-G2PTTS — Türkçe metinden sözcük başına SESBİRİM + VURGU + SINIR (duraklama): tam TTS ön ucu.

HİBRİT hat (yalnız ağırlıklar yetmez):
  vurgu : SÖZLÜK ÖNCELİKLİ karma. Kural modülü (stress_rules.py + resources/*.tsv|txt: kök sözlüğü, clitic, ünlüsüz, sıfat sözlüğüne kapılı pekiştirme/-CIk)
          belirleyici bir sonuç verirse o; değilse (kural varsayılan son seslemi verdiyse) model başlığı (damıtılmış biçimbilim: koşaç, kişi eki, olumsuzluk...).
  sınır : model başlığı; P(ip)+P(IP) > tau -> sınır (ip < IP), noktalamayla birleştirilir; cümlenin son sözcüğü hep "cümle".
  sesbirim: kural tabanlı (phonemize.py: PyPI dizge==0.1.6, hata verirse iatagun/dizge-g2p yedeği) + söyleyiş sözlüğü (alıntılar, şapka) + uzun ünlü kuralları
          (pronounce.py, resources/pronunciation_exceptions.tsv); vurgu işareti ˈ vurgulu ünlünün önüne. Çıktı dizgetts.Engine(g2ptts=True) token'larıyla BİREBİR aynı.

Katman yapısı `dizgetts.g2ptts.train.G2PTTS` ile BİREBİR aynı (aynı state_dict anahtarları: bert.*, stress.*, boundary.*); mantık `dizgetts.g2ptts.tagger.Tagger` ile aynıdır
(round-trip testi: dizgetts/tests/test_hf_g2ptts.py birebir eşitliği doğrular).

Kullanım:
    from transformers import AutoModel
    m = AutoModel.from_pretrained("iatagun/DizgeBERT-G2PTTS", trust_remote_code=True)
    m.tag("Yarın İstanbul'a gideceğim, ama havanın nasıl olacağını bilmiyorum.")
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_dizgebert_g2ptts import DizgeBertG2ptttsConfig
from .normalize import normalize, tr_lower
from .stress import StressRules, _n_vowels, phone_stress_index
from .symbols import BREAK_MAJOR, BREAK_MID, STRESS, WORD_SEP, tokenize
# transformers uzak kodda yalnız modelleme dosyasının DOĞRUDAN göreli içe aktarmalarını önbelleğe kopyalar: sesbirim modülleri burada anılmalı
from .g2p import G2P  # noqa: F401,E402
from .phonemize import Phonemizer  # noqa: E402
from .pronounce import Exceptions  # noqa: F401,E402

PUNCT_BOUNDARY = {",": "ip", ";": "IP", ".": "cümle", "?": "cümle", "!": "cümle"}
RANK = ("0", "ip", "IP", "cümle")


class DizgeBertG2ptts(PreTrainedModel):
    config_class = DizgeBertG2ptttsConfig

    def __init__(self, config: DizgeBertG2ptttsConfig):
        super().__init__(config)
        # Kaydedilmiş modelde encoder ağırlıkları state_dict'te; iskeleti config'ten kur.
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained(config.encoder_name))
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(config.dropout)
        self.stress = nn.Linear(h, config.n_stress)
        self.boundary = nn.Linear(h, config.n_boundary)
        self._rules = None
        self._tok = None
        self._ph = None
        self.post_init()

    def forward(self, input_ids, attention_mask=None, first_pos=None):
        """first_pos: (B, W) her sözcüğün ilk alt-sözcük konumu (-1 = kesilmiş)."""
        hid = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        w = self.drop(torch.gather(hid, 1, first_pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, hid.size(-1))))
        return ModelOutput(logits_stress=self.stress(w), logits_boundary=self.boundary(w))

    # ------------------------------------------------------------------ kurallar / tokenizer
    def _resources_dir(self) -> str:
        name = self.config._name_or_path
        if os.path.isdir(name):
            return os.path.join(name, "resources")
        from huggingface_hub import snapshot_download

        return os.path.join(snapshot_download(name, allow_patterns=["resources/*"]), "resources")

    @property
    def rules(self) -> StressRules:
        if self._rules is None:
            self._rules = StressRules(res_dir=self._resources_dir())
        return self._rules

    @property
    def phonemizer(self):
        if self._ph is None:
            c = self.config
            self._ph = Phonemizer(bert_fallback=True, pron_exceptions=c.pron_exceptions, register=c.register, length_rules=c.length_rules, res_dir=self._resources_dir())
        return self._ph

    def _phones(self, word: str, stress_from_end):
        """(atomlar, vurgulu ünlü atomunun indeksi ya da None). Eşleme dizgetts Engine ile aynı: sözlük/kural değiştirdiyse ham atomlarda + köken."""
        ph = self.phonemizer
        atoms = tokenize(ph.word(word), strict=False)
        if stress_from_end is None or not atoms:
            return atoms, None
        tr = ph.trace(word)
        raw, origin = (tr.raw, tr.origin) if tr is not None and tr.a == atoms else (None, None)
        return atoms, phone_stress_index(word, atoms, _n_vowels(tr_lower(word)) - 1 - stress_from_end, raw, origin)[0]

    def _tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            self._tok = tokenizer
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.config._name_or_path)
        return self._tok

    # ------------------------------------------------------------------ çıkarım
    def _chunks(self, toks: list[str], tok) -> list[tuple[int, int]]:
        """Belirteç dizisini alt-sözcük sayısı <= max_subwords olan ardışık parçalara böler; kesim cümle sonunda (. ? !), yoksa herhangi bir noktalamada, o da yoksa sözcük sınırında."""
        n = [0] * len(toks)
        for w in tok(toks, is_split_into_words=True, add_special_tokens=False).word_ids():
            if w is not None:
                n[w] += 1
        out, start, used, last_sent, last_punct = [], 0, 0, None, None
        for i, c in enumerate(n):
            if used + c > self.config.max_subwords and i > start:
                cut = next((x + 1 for x in (last_sent, last_punct) if x is not None and x >= start), i)
                out.append((start, cut))
                start, used = cut, sum(n[cut:i])
                last_sent = last_punct = None
            used += c
            if toks[i] in PUNCT_BOUNDARY:
                last_punct = i
                if toks[i] in ".?!":
                    last_sent = i
        if toks:
            out.append((start, len(toks)))
        return out

    @torch.no_grad()
    def _model_chunk(self, toks: list[str], tok):
        enc = tok(toks, is_split_into_words=True, truncation=True, max_length=256, return_tensors="pt")
        first, seen = [0] * len(toks), set()
        for i, wid in enumerate(enc.word_ids(0)):
            if wid is not None and wid not in seen:
                seen.add(wid)
                first[wid] = i
        o = self.forward(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device), torch.tensor([first], device=self.device))
        return o.logits_stress[0].argmax(-1).tolist(), o.logits_boundary[0].softmax(-1).tolist()

    def _model(self, toks: list[str], tok):
        stress, prob = [], []
        for a, b in self._chunks(toks, tok):
            s, p = self._model_chunk(toks[a:b], tok)
            stress += s
            prob += p
        return stress, prob

    def _stress(self, word: str, model_cls: int):
        """(vurgulu seslem SONDAN sırası ya da None, kaynak)."""
        k, tag = self.rules.syllable(word, tiers=tuple(self.config.lexical_tiers))
        n = _n_vowels(tr_lower(word))
        if tag != "varsayılan_son":
            return (None if k is None else n - 1 - k), f"kural:{tag}"
        return (None if model_cls == 0 else min(model_cls - 1, n - 1)), "model"

    def tag(self, text: str, tokenizer=None) -> dict:
        """Ham metin -> {"text", "norm", "phonemes", "words": [{word, stress_from_end, stress_src, p_break, boundary, phones, stress_phone, punct}]}
        (normalize: sayı/kısaltma açılır, noktalama ayrılır). `phonemes`: TTS'e hazır dizi (sözcük arası boşluk, ˈ vurgu, noktalamasız ip/IP sınırına | / ‖)."""
        return self.tag_norm(normalize(text), text, tokenizer)

    def tag_norm(self, norm: str, text: str = "", tokenizer=None) -> dict:
        """normalize() çıktısı (boşlukla ayrılmış belirteçler, noktalama ayrı) üzerinde."""
        tok = self._tokenizer(tokenizer)
        toks = norm.split()
        scls, pb = self._model(toks, tok) if toks else ([], [])
        words = []
        for i, t in enumerate(toks):
            if not t.isalpha():
                if words and t in PUNCT_BOUNDARY:
                    w = words[-1]
                    w["boundary"] = max(w["boundary"], PUNCT_BOUNDARY[t], key=RANK.index)
                    w["punct"].append(t)
                continue
            r, src = self._stress(t, scls[i])
            p_break = pb[i][1] + pb[i][2]
            words.append(dict(word=t, stress_from_end=r, stress_src=src, p_break=round(p_break, 3),
                              boundary=("ip" if pb[i][1] >= pb[i][2] else "IP") if p_break > self.config.tau else "0", punct=[]))
        if words:
            words[-1]["boundary"] = "cümle"
        out = dict(text=text, norm=norm, words=words)
        if getattr(self.config, "phonemes", True):
            for w in words:
                w["phones"], w["stress_phone"] = self._phones(w["word"], w["stress_from_end"])
            out["phonemes"] = "".join(assemble(words))
        return out


def assemble(words: list[dict]) -> list[str]:
    """Sözcükler -> TTS token listesi (dizgetts.engine.AssembleStage(breaks=True) ile aynı sıra): vurgu ˈ ünlünün önünde, noktalamasız ip/IP'ye | / ‖, noktalama sonda."""
    toks: list[str] = []
    for i, w in enumerate(words):
        if i:
            toks.append(WORD_SEP)
        for j, p in enumerate(w["phones"]):
            if w["stress_phone"] == j:
                toks.append(STRESS)
            toks.append(p)
        if not w["punct"] and w["boundary"] in ("ip", "IP"):
            toks.append(BREAK_MID if w["boundary"] == "ip" else BREAK_MAJOR)
        toks.extend(w["punct"])
    return toks
