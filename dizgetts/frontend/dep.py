"""Bağımlılık çözümleme: iatagun/DizgeBERT-Dep (Biaffine + MST, UD). Öbek/sınır tahmini için (PhraseStage, eval/boundary_baseline.py).

Model sınıfı (BiaffineParser) HF model deposunda YOK ve repodaki training/train_dep_bert.py eski bir anlık görüntü; güncel sınıf
Space iatagun/dizge-demo'daki train_dep_simple.py'de. O dosya sabitlenmiş sürümüyle indirilip içe aktarılır.
Girdi normalize() çıktısı gibi önceden bölünmüş belirteçlerdir (noktalama dahil); uzun metin cümlelere bölünerek verilmeli (MAX_LEN=128 alt-sözcük).
"""
from __future__ import annotations

import importlib.util

MODEL_ID, MODEL_REV = "iatagun/DizgeBERT-Dep", "87110b15b04b19b58532179f07463070efa7c91e"
SPACE_ID, SPACE_REV = "iatagun/dizge-demo", "be39207e60bb465f7fce3c36ae200c3d707d079a"


class DepParser:
    def __init__(self, device: str = "cpu"):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        src = hf_hub_download(SPACE_ID, "train_dep_simple.py", repo_type="space", revision=SPACE_REV)
        spec = importlib.util.spec_from_file_location("_dizge_dep_model", src)
        T = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(T)
        self.T, self.torch, self.device = T, torch, device
        cfg = T.BiaffineConfig.from_pretrained(MODEL_ID, revision=MODEL_REV)
        self.model = T.BiaffineParser(cfg)
        missing, unexpected = self.model.load_state_dict(load_file(hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REV)), strict=False)
        if missing or unexpected:  # Space strict=False yüklüyor; biz sessiz eksik ağırlığa izin vermiyoruz
            raise RuntimeError(f"DizgeBERT-Dep ağırlık uyuşmazlığı: eksik={missing[:5]} fazla={unexpected[:5]}")
        self.model.eval().to(device)
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)

    def parse(self, tokens: list[str]) -> list[tuple[int, str]]:
        """tokens -> belirteç başına (baş, ilişki); baş 1-tabanlı, 0 = kök."""
        if not tokens:
            return []
        enc = self.tok(tokens, is_split_into_words=True, max_length=self.T.MAX_LEN, truncation=True, return_tensors="pt")
        n = len(tokens)
        if max(i for i in enc.word_ids(0) if i is not None) < n - 1:
            raise ValueError(f"cümle MAX_LEN={self.T.MAX_LEN} alt-sözcüğü aşıyor ({n} belirteç); daha kısa parçalar verin")
        with self.torch.inference_mode():
            out = self.model(input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device),
                             word_ids=enc.word_ids(0), num_words=n)
            d = self.model.mst_decode(out["arc_scores"], out["label_h"], out["label_d"], out.get("root_scores"))
        res = []
        for i, (h, l) in enumerate(zip(d["heads"], d["labels"])):
            h = h + 1
            res.append((0, "root") if h <= 0 or h == i + 1 else (h, self.T.ID_TO_REL.get(l, "dep")))  # mst_decode kökü -1 döndürür
        return res
