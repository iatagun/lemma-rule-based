"""Checkpoint'ten ses üret (Matcha + HiFi-GAN universal). Frontend, checkpoint içindeki config'den gelir (espeak | dizge).

  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/eval/synth.py --ckpt D:/dizgetts/runs/<run>/ep25.pt [--device cpu] [--texts dosya.txt] [--out DIR]

Varsayılan metinler: dizgetts/eval/listen_sentences.txt (satır başına bir cümle; '#' ile başlayan satırlar yorum).
"""
import argparse, json, os, sys, time

import soundfile as sf
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "dizgetts"))
from matcha.hifigan.config import v1  # noqa: E402
from matcha.hifigan.denoiser import Denoiser  # noqa: E402
from matcha.hifigan.env import AttrDict  # noqa: E402
from matcha.hifigan.models import Generator as HiFiGAN  # noqa: E402
from matcha.utils.utils import intersperse  # noqa: E402

from frontend import espeak, symbols_espeak  # noqa: E402
from frontend.normalize import normalize  # noqa: E402
from frontend.phonemize import Phonemizer  # noqa: E402
from frontend.prosody import add_features  # noqa: E402
from frontend.stress import transfer_stress  # noqa: E402
from frontend.symbols import PAUSES, SYMBOL_TO_ID, SYMBOLS  # noqa: E402
from train.train import build_model  # noqa: E402

VOCODER = "D:/dizgetts/pretrained/hifigan_univ_v1"


def load_vocoder(dev):
    h = AttrDict(v1)
    g = HiFiGAN(h).to(dev)
    g.load_state_dict(torch.load(VOCODER, map_location=dev)["generator"])
    g.eval(); g.remove_weight_norm()
    return g, Denoiser(g, mode="zeros")


class Synth:
    def __init__(self, ckpt: str, device: str = "cpu"):
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
        self.space_id = (symbols_espeak.SYMBOL_TO_ID if self.fe == "espeak" else SYMBOL_TO_ID)[" "]  # nosep/breaks'te ayraç yok: boundary_scale etkisiz
        self.ph = Phonemizer() if self.fe.startswith("dizge") else None

    def ids(self, text: str):
        if self.fe == "espeak":
            norm = normalize(text)
            toks = symbols_espeak.tokenize(espeak.phonemize(norm), strip_stress=self.cfg["espeak_strip_stress"])
            s2i = symbols_espeak.SYMBOL_TO_ID
        else:
            norm, toks = self.ph(text)
            s2i = SYMBOL_TO_ID
            if self.fe == "dizge_stress":  # espeak vurgusunu dizge ünlülerine aktar (frontend/stress.py); çıkarımda espeak'e bağımlı (üst sınır deneyi)
                toks, _ = transfer_stress(toks, espeak.phonemize(norm))
            if self.fe == "dizge_nosep":
                toks = [t for t in toks if t != " "]
            elif self.fe == "dizge_breaks":  # ponytail: sınıf tahmincisi yok; yedek kural (noktalamadan sonra B3, aksi halde B1). Dep tabanlı tahminci sonra.
                toks = [("‖" if toks[i - 1] in PAUSES else None) if t == " " else t for i, t in enumerate(toks)]
                toks = [t for t in toks if t is not None]
        if self.fe in ("dizge_feat", "dizge_featm"):  # çıkarımda ölçülmüş sınıf yok: metin kuralı (B3 = noktalama/cümle sonu, aksi B1)
            toks, feats = add_features(toks)
            V = len(SYMBOLS)
            return norm, toks, intersperse([s2i[t] + V * f for t, f in zip(toks, feats)], 0)
        return norm, toks, intersperse([s2i[t] for t in toks], 0)

    @torch.inference_mode()
    def __call__(self, text: str, steps: int = 10, temperature: float = 0.667, length_scale: float = 1.0, boundary_scale: float = 1.0):
        """boundary_scale != 1: sözcük ayracı (boşluk) token'ının ve bitişik blank'lerin süresi bu çarpanla ölçeklenir (TEŞHİS amaçlı;
        sözcük arası boşluk sorunu incelemesi). 0 -> bu token'lar 0 kare."""
        norm, toks, ids = self.ids(text)
        x = torch.tensor(ids, dtype=torch.long, device=self.dev)[None]
        xl = torch.tensor([x.shape[1]], device=self.dev)
        t = time.time()
        enc = self.model.encoder
        orig = enc.forward
        if boundary_scale != 1.0:
            sp = (x[0] == self.space_id)
            sel = sp.clone(); sel[1:] |= sp[:-1]; sel[:-1] |= sp[1:]  # boşluk + bitişik blank'ler
            def patched(*a, **k):
                mu, logw, mask = orig(*a, **k)
                lw = logw.clone()
                lw[0, 0, sel] = lw[0, 0, sel] + (torch.log(torch.tensor(boundary_scale)) if boundary_scale > 0 else -1e4)
                return mu, lw, mask
            enc.forward = patched
        try:
            out = self.model.synthesise(x, xl, n_timesteps=steps, temperature=temperature, spks=None, length_scale=length_scale)
        finally:
            enc.forward = orig
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
        p = os.path.join(out, f"{i:02d}.wav")
        sf.write(p, wav, 22050, subtype="PCM_16")
        log.append(dict(i=i, text=t, norm=norm, phonemes=ph, sec_synth=round(sec, 2), dur=round(len(wav) / 22050, 2)))
        print(f"{i:02d} [{len(wav)/22050:.1f}s, {sec:.1f}s işlem] {t}", flush=True)
    json.dump(log, open(os.path.join(out, "index.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("->", out)
