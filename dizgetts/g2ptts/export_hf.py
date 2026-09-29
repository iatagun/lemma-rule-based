"""g2ptts checkpoint'ini HF paketine (trust_remote_code) dışa aktar. Kaynak dosyalar `dizgebert_g2ptts/` (config, modeling, kart) + `dizgetts/frontend/` (kural modülü) + `dizgetts/resources/`.
  python -X utf8 -m dizgetts.g2ptts.export_hf [--ckpt D:/dizgetts/runs/g2ptts_v3/best.pt] [--out dizgebert_g2ptts_hf]
Paket = model.safetensors (bert.*, stress.*, boundary.*: G2PTTS ile AYNI anahtarlar) + config.json (tau, kural özeti) + tokenizer (DizgeBERT-Dep, sabit revizyon) + modeling/config .py
        + frontend'den BİREBİR kopya (stress/normalize/symbols: vurgu; phonemize/pronounce/g2p: sesbirim) + resources/ (kök sözlüğü, clitic, sıfat sözlüğü,
        söyleyiş sözlüğü) + README.md (kart). Sesbirim için kullanıcıda PyPI `dizge==0.1.6` kurulu olmalı.
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
from dizgetts.frontend.pronounce import Exceptions
from dizgetts.frontend.stress import StressRules
from dizgetts.g2ptts.train import ENC, N_BOUND, N_STRESS

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "dizgebert_g2ptts"
FRONT = Path(__file__).resolve().parents[1] / "frontend"
RES = Path(__file__).resolve().parents[1] / "resources"
# frontend'den adıyla kopyalanır: modüller birbirini göreli içe aktarır (pronounce -> .stress, phonemize -> .pronounce/.g2p)
RULE_FILES = ("stress.py", "normalize.py", "symbols.py", "phonemize.py", "pronounce.py", "g2p.py")
RESOURCE_FILES = ("stress_roots.tsv", "clitics.tsv", "adj_lemmas.txt", "pronunciation_exceptions.tsv", "loan_roots.tsv")


def export(ckpt: str, out: Path) -> Path:
    from dizgebert_g2ptts.configuration_dizgebert_g2ptts import DizgeBertG2ptttsConfig

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)  # kendi checkpoint'imiz
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha1(Path(ckpt).read_bytes()).hexdigest()[:12]
    a = ck["args"]
    cfg = DizgeBertG2ptttsConfig(
        # tau: np.arange gürültüsü (0.25000000000000006) atılır
        encoder_name=ENC, n_stress=N_STRESS, n_boundary=N_BOUND, tau=round(float(ck["val"]["tau"]), 6), rules_version=f"{StressRules.version()}+istisna={Exceptions().version().split('/')[0]}",
        train_info=dict(ckpt_sha1=sha, epoch=int(ck["val"]["epoch"]), init=f"{DEP_ID}@{DEP_REV[:8]}", epochs=a["epochs"], lr=a["lr"], freeze_layers=a["freeze"],
                        antalia_repeat=a["antalia_repeat"], select=a["select"], val_boundary_f1=round(float(ck["val"]["sınır_F1"]), 4)))
    cfg.architectures = ["DizgeBertG2ptts"]
    cfg.auto_map = {"AutoConfig": "configuration_dizgebert_g2ptts.DizgeBertG2ptttsConfig", "AutoModel": "modeling_dizgebert_g2ptts.DizgeBertG2ptts"}
    cfg.save_pretrained(out)
    save_file({k: v.contiguous().clone() for k, v in ck["state"].items()}, out / "model.safetensors", metadata={"format": "pt"})
    AutoTokenizer.from_pretrained(DEP_ID, revision=DEP_REV).save_pretrained(out)
    for fn in ("configuration_dizgebert_g2ptts.py", "modeling_dizgebert_g2ptts.py"):
        shutil.copy(PKG / fn, out / fn)
    for fn in RULE_FILES:
        shutil.copy(FRONT / fn, out / fn)
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
