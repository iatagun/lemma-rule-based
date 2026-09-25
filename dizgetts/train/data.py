"""Matcha-TTS eğitim verisi: *_phon.jsonl (Aşama 3) + önceden hesaplanmış mel'ler (D:/.../mels/*.pt, ham log-mel).

Mel normalizasyonu yükleme anında train istatistikleriyle (stats.json) yapılır; upstream text_mel_datamodule ile aynı mantık.
Token'lar Matcha'nın add_blank kuralıyla aralara 0 (PAD) eklenerek verilir.
"""
from __future__ import annotations

import json
import os
import random

import soundfile as sf
import torch
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import normalize
from matcha.utils.utils import intersperse
from torch.utils.data import Dataset

from dizgetts.frontend import symbols as dizge_syms
from dizgetts.frontend import symbols_espeak as espeak_syms


# frontend: "espeak" (karşılaştırma aracı) | "engine" (dizgetts.engine çıktısı: manifestteki `tokens`; vurgu/kırılma token'ları dahil). "dizge" = "engine" takma adı.
ENGINE_FRONTENDS = ("engine", "dizge")


def frontend_table(frontend: str):
    if frontend == "espeak":
        return espeak_syms.SYMBOLS, espeak_syms.SYMBOL_TO_ID
    if frontend in ENGINE_FRONTENDS:
        return dizge_syms.SYMBOLS, dizge_syms.SYMBOL_TO_ID
    raise ValueError(f"bilinmeyen frontend: {frontend}")


def row_tokens(row: dict, frontend: str, strip_stress: bool = False) -> list[str]:
    if frontend == "espeak":
        return espeak_syms.tokenize(row["espeak"], strip_stress=strip_stress)
    return row["tokens"]


def mel_file(root: str, clip_id: str) -> str:
    return os.path.join(root, "mels", clip_id + ".pt")


def ensure_mels(root: str, rows: list[dict], au: dict) -> int:
    """Eksik mel'leri hesaplayıp yazar; yazılan sayıyı döndürür."""
    os.makedirs(os.path.join(root, "mels"), exist_ok=True)
    n = 0
    for r in rows:
        p = mel_file(root, r["id"])
        if os.path.exists(p):
            continue
        y, sr = sf.read(os.path.join(root, r["wav"]), dtype="float32")
        assert sr == au["sample_rate"], (r["id"], sr)
        mel = mel_spectrogram(torch.from_numpy(y)[None], au["n_fft"], au["n_feats"], sr, au["hop_length"], au["win_length"],
                              au["f_min"], au["f_max"], center=False).squeeze(0)
        torch.save(mel.contiguous(), p)
        n += 1
    return n


class TTSDataset(Dataset):
    def __init__(self, root: str, split: str, frontend: str, stats: dict, strip_stress: bool = False, manifest: str = "_phon", dp_feat: bool = False):
        self.root = root
        self.rows = [json.loads(l) for l in open(os.path.join(root, f"{split}{manifest}.jsonl"), encoding="utf8")]
        _, self.s2i = frontend_table(frontend)
        self.mean, self.std = stats["mel_mean"], stats["mel_std"]
        self.ids = [intersperse([self.s2i[t] for t in row_tokens(r, frontend, strip_stress)], 0) for r in self.rows]
        self.dp = None
        if dp_feat:  # v4: süre tahmincisi özniteliği (manifestte `dp_feat`, token başına) -> add_blank dizisine yayılır
            from dizgetts.train.dpfeat import intersperse_feat
            self.dp = [intersperse_feat(r["dp_feat"]) for r in self.rows]
            bad = [r["id"] for r, a, b in zip(self.rows, self.ids, self.dp) if len(a) != len(b)]
            assert not bad, f"dp_feat uzunluğu token'la tutmuyor: {bad[:3]}"
        # mel uzunluğu: dosyadan okumadan tahmin için ilk çağrıda önbelleğe alınır
        self._len = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        mel = torch.load(mel_file(self.root, self.rows[i]["id"]))
        d = dict(x=torch.tensor(self.ids[i], dtype=torch.long), y=normalize(mel, self.mean, self.std), id=self.rows[i]["id"])
        if self.dp is not None:
            d["dp_feat"] = torch.tensor(self.dp[i], dtype=torch.long)
        return d

    def mel_lengths(self) -> list[int]:
        if self._len is None:
            self._len = [torch.load(mel_file(self.root, r["id"])).shape[-1] for r in self.rows]
        return self._len


def collate(batch):
    xl = torch.tensor([b["x"].shape[0] for b in batch])
    yl = torch.tensor([b["y"].shape[-1] for b in batch])
    x = torch.zeros(len(batch), int(xl.max()), dtype=torch.long)
    y = torch.zeros(len(batch), batch[0]["y"].shape[0], int(yl.max()))
    for i, b in enumerate(batch):
        x[i, : xl[i]] = b["x"]
        y[i, :, : yl[i]] = b["y"]
    out = dict(x=x, x_lengths=xl, y=y, y_lengths=yl, spks=None, durations=None, ids=[b["id"] for b in batch])
    if "dp_feat" in batch[0]:
        f = torch.zeros(len(batch), int(xl.max()), dtype=torch.long)  # 0 = pad
        for i, b in enumerate(batch):
            f[i, : xl[i]] = b["dp_feat"]
        out["dp_feat"] = f
    return out


class BucketBatches:
    """Uzunluğa göre gruplanmış batch'ler: bucket_size*batch_size'lık karışık dilimler kendi içinde sıralanır."""

    def __init__(self, lengths: list[int], batch_size: int, bucket_size: int, seed: int, shuffle: bool = True):
        self.lengths, self.bs, self.bucket, self.seed, self.shuffle = lengths, batch_size, bucket_size, seed, shuffle
        self.epoch = 0

    def __len__(self):
        return -(-len(self.lengths) // self.bs)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        chunk = self.bs * self.bucket
        batches = []
        for i in range(0, len(idx), chunk):
            c = sorted(idx[i : i + chunk], key=self.lengths.__getitem__)
            batches += [c[j : j + self.bs] for j in range(0, len(c), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)
