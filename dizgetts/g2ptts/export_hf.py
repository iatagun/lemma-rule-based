"""g2ptts checkpoint'ini HF paketine (trust_remote_code) dışa aktar. Kaynak dosyalar `dizgebert_g2ptts/` (config, modeling, kart) + `dizgetts/frontend/` (kural modülü) + `dizgetts/resources/`.
  python -X utf8 -m dizgetts.g2ptts.export_hf [--ckpt D:/dizgetts/runs/g2ptts_v3/best.pt] [--out dizgebert_g2ptts_hf]
Paket = model.safetensors (bert.*, stress.*, boundary.*: G2PTTS ile AYNI anahtarlar) + config.json (tau, kural özeti) + tokenizer (DizgeBERT-Dep, sabit revizyon) + modeling/config .py
        + stress_rules.py / normalize.py / symbols.py (frontend'den BİREBİR kopya) + resources/ (kök sözlüğü, clitic, sıfat sözlüğü) + README.md (kart).
Push ÖNCESİ zorunlu: python -X utf8 -m dizgetts.tests.test_hf_g2ptts (gidiş-dönüş: yerel Tagger ile birebir).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from dizgetts import paths
from dizgetts.frontend.dep import MODEL_ID as DEP_ID, MODEL_REV as DEP_REV
from dizgetts.frontend.stress import StressRules
from dizgetts.g2ptts.train import ENC, N_BOUND, N_STRESS

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "dizgebert_g2ptts"
FRONT = Path(__file__).resolve().parents[1] / "frontend"
RES = Path(__file__).resolve().parents[1] / "resources"
# (kaynak, paket içindeki ad): stress.py `from .normalize`/`from .symbols` ile göreli içe aktarır -> yanında normalize.py + symbols.py olmalı
RULE_FILES = (("stress.py", "stress_rules.py"), ("normalize.py", "normalize.py"), ("symbols.py", "symbols.py"))
RESOURCE_FILES = ("stress_roots.tsv", "clitics.tsv", "adj_lemmas.txt")


def export(ckpt: str, out: Path) -> Path:
    from dizgebert_g2ptts.configuration_dizgebert_g2ptts import DizgeBertG2ptttsConfig

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)  # kendi checkpoint'imiz
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha1(Path(ckpt).read_bytes()).hexdigest()[:12]
    a = ck["args"]
    cfg = DizgeBertG2ptttsConfig(
        # tau: np.arange gürültüsü (0.25000000000000006) atılır
        encoder_name=ENC, n_stress=N_STRESS, n_boundary=N_BOUND, tau=round(float(ck["val"]["tau"]), 6), rules_version=StressRules.version(),
        train_info=dict(ckpt_sha1=sha, epoch=int(ck["val"]["epoch"]), init=f"{DEP_ID}@{DEP_REV[:8]}", epochs=a["epochs"], lr=a["lr"], freeze_layers=a["freeze"],
                        antalia_repeat=a["antalia_repeat"], select=a["select"], val_boundary_f1=round(float(ck["val"]["sınır_F1"]), 4)))
    cfg.architectures = ["DizgeBertG2ptts"]
    cfg.auto_map = {"AutoConfig": "configuration_dizgebert_g2ptts.DizgeBertG2ptttsConfig", "AutoModel": "modeling_dizgebert_g2ptts.DizgeBertG2ptts"}
    cfg.save_pretrained(out)
    save_file({k: v.contiguous().clone() for k, v in ck["state"].items()}, out / "model.safetensors", metadata={"format": "pt"})
    AutoTokenizer.from_pretrained(DEP_ID, revision=DEP_REV).save_pretrained(out)
    for fn in ("configuration_dizgebert_g2ptts.py", "modeling_dizgebert_g2ptts.py"):
        shutil.copy(PKG / fn, out / fn)
    for src, dst in RULE_FILES:
        shutil.copy(FRONT / src, out / dst)
    (out / "resources").mkdir(exist_ok=True)
    for fn in RESOURCE_FILES:
        shutil.copy(RES / fn, out / "resources" / fn)
    shutil.copy(PKG / "MODEL_CARD.md", out / "README.md")
    print(f"HF paketi yazıldı: {out}  (ckpt sha1 {sha}, tau {cfg.tau:.2f})")
    print(f"  Doğrula: python -X utf8 -m dizgetts.tests.test_hf_g2ptts")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{paths.RUNS}/g2ptts_v3/best.pt")
    ap.add_argument("--out", default=str(REPO / "dizgebert_g2ptts_hf"))
    a = ap.parse_args()
    export(a.ckpt, Path(a.out))
