"""dizge-g2p sarmalayıcı: sözcük -> fonem dizgesi (toplu, tek başına çalışır).

Semantik `bert_turkish_finetune.DizgeBERT.g2p` ile aynıdır (D:/playground; kaynaktan doğrulandı):
sözcük karakterlerine bölünür (is_split_into_words), her karakterin İLK alt-token'ının etiketi alınır,
etiketler birleştirilir. Fark: toplu çalışır ve küçük harfe çevirme Türkçe'ye duyarlıdır
(wrapper `str.lower()` kullanır: 'İ'->'i̇' (2 kod noktası, hizalama bozuk), 'I'->'i' (yanlış, 'ı' olmalı)).
Sözcük düzeyindedir: sözcükler arası bağlam yok, vurgu yok.
"""
from __future__ import annotations

import glob
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_ID = "iatagun/dizge-g2p"
MAX_CHARS = 62  # wrapper: max_length(64) - [CLS] - [SEP]


from .normalize import tr_lower  # noqa: E402,F401  (torch'suz modülde tanımlı; burada yeniden dışa açılır)


def _local_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID)


class G2P:
    def __init__(self, model_dir: str | None = None, device: str | None = None):
        model_dir = model_dir or _local_dir()
        cfg = json.load(open(os.path.join(model_dir, "task_config.json"), encoding="utf8"))
        assert cfg["task"] == "g2p"
        self.label_list: list[str] = cfg["label_list"]
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    @torch.inference_mode()
    def g2p_batch(self, words: list[str], batch_size: int = 64) -> dict[str, str]:
        out: dict[str, str] = {}
        uniq = sorted(set(words), key=len)  # uzunluğa göre sırala: az padding
        for i in range(0, len(uniq), batch_size):
            chunk = uniq[i : i + batch_size]
            chars = [list(tr_lower(w))[:MAX_CHARS] for w in chunk]
            enc = self.tok(chars, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=64)
            pred = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.argmax(-1).cpu().tolist()
            for b, w in enumerate(chunk):
                labels, prev = [], None
                for j, wid in enumerate(enc.word_ids(b)):
                    if wid is not None and wid != prev:
                        labels.append(self.id2label[pred[b][j]])
                    prev = wid
                out[w] = "".join(labels[: len(chars[b])])
        return out

    def g2p(self, word: str) -> str:
        return self.g2p_batch([word])[word]
