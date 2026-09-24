"""Checkpoint'ten ses üret (Matcha + HiFi-GAN universal). Metin -> token: `dizgetts.engine.Engine` (frontend: engine) ya da espeak (karşılaştırma aracı).

  D:/dizgetts/venv/Scripts/python.exe -X utf8 -m dizgetts.eval.synth --ckpt D:/dizgetts/runs/<run>/ep150.pt [--device cpu] [--texts dosya.txt] [--out DIR]

Varsayılan metinler: dizgetts/eval/listen_sentences.txt (satır başına bir cümle; '#' ile başlayan satırlar yorum).
"""
import argparse, json, os, time

import soundfile as sf
import torch
import yaml
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.utils.utils import intersperse

from dizgetts.engine import Engine
from dizgetts.frontend import espeak, symbols_espeak
from dizgetts.frontend.normalize import normalize
from dizgetts.train.train import ROOT, build_model

VOCODER = "D:/dizgetts/pretrained/hifigan_univ_v1"


def load_vocoder(dev):
    g = HiFiGAN(AttrDict(v1)).to(dev)
    g.load_state_dict(torch.load(VOCODER, map_location=dev)["generator"])
    g.eval(); g.remove_weight_norm()
    return g, Denoiser(g, mode="zeros")


class Synth:
    def __init__(self, ckpt: str, device: str = "cpu", engine: Engine | None = None):
        self.dev = torch.device(device)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        dcfg = yaml.safe_load(open(os.path.join(ROOT, self.cfg["data_config"]), encoding="utf8"))
        stats = json.load(open(os.path.join(dcfg["out_root"], "stats.json"), encoding="utf8"))
        self.model = build_model(self.cfg, len(ck["symbols"]), stats)
        self.model.load_state_dict(ck["model"])
        self.model.to(self.dev).eval()
        self.epoch = ck["epoch"]
        self.vocoder, self.denoiser = load_vocoder(self.dev)
        self.fe = self.cfg["frontend"]
        self.engine = (engine or Engine(**self.cfg.get("engine", {}))) if self.fe in ("engine", "dizge") else None  # cfg["engine"]: morph/tiers (M1b)

    def ids(self, text: str):
        if self.fe == "espeak":
            norm = normalize(text)
            toks = symbols_espeak.tokenize(espeak.phonemize(norm), strip_stress=self.cfg["espeak_strip_stress"])
            return norm, toks, intersperse([symbols_espeak.SYMBOL_TO_ID[t] for t in toks], 0)
        u = self.engine.frontend(text)
        return u.norm, u.tokens, intersperse(self.engine.ids(u), 0)

    @torch.inference_mode()
    def __call__(self, text: str, steps: int = 10, temperature: float = 0.667, length_scale: float = 1.0):
        norm, toks, ids = self.ids(text)
        x = torch.tensor(ids, dtype=torch.long, device=self.dev)[None]
        xl = torch.tensor([x.shape[1]], device=self.dev)
        t = time.time()
        out = self.model.synthesise(x, xl, n_timesteps=steps, temperature=temperature, spks=None, length_scale=length_scale)
        wav = self.vocoder(out["mel"]).clamp(-1, 1)
        wav = self.denoiser(wav.squeeze(), strength=0.00025).cpu().squeeze()
        return wav.numpy(), norm, "".join(toks), time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--texts", default=os.path.join(os.path.dirname(__file__), "listen_sentences.txt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()
    s = Synth(a.ckpt, a.device)
    run = os.path.basename(os.path.dirname(a.ckpt))
    out = a.out or os.path.join("D:/dizgetts/samples", run, f"ep{s.epoch}")
    os.makedirs(out, exist_ok=True)
    lines = [l.strip() for l in open(a.texts, encoding="utf8") if l.strip() and not l.startswith("#")]
    log = []
    for i, t in enumerate(lines):
        wav, norm, ph, sec = s(t, steps=a.steps)
        sf.write(os.path.join(out, f"{i:02d}.wav"), wav, 22050, subtype="PCM_16")
        log.append(dict(i=i, text=t, norm=norm, phonemes=ph, sec_synth=round(sec, 2), dur=round(len(wav) / 22050, 2)))
        print(f"{i:02d} [{len(wav)/22050:.1f}s, {sec:.1f}s işlem] {t}", flush=True)
    json.dump(log, open(os.path.join(out, "index.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("->", out)
