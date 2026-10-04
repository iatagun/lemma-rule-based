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

from dizgetts import paths
from dizgetts.engine import Engine
from dizgetts.frontend import espeak, symbols_espeak
from dizgetts.frontend.normalize import normalize
from dizgetts.frontend.symbols import SYMBOLS
from dizgetts.train.data import probe_rows
from dizgetts.train.train import ROOT, build_model

VOCODER = paths.VOCODER


def load_vocoder(dev, path: str = VOCODER):
    g = HiFiGAN(AttrDict(v1)).to(dev)
    g.load_state_dict(torch.load(path, map_location=dev)["generator"])
    g.eval(); g.remove_weight_norm()
    return g, Denoiser(g, mode="zeros")


PROBE_N = 30          # checkpoint'in yanında taşınan örnek cümle sayısı (val manifestinin ilk N'i)
PROBE_MAX_DIFF = 0.2  # bu orandan fazla cümlede token farkı = SİSTEMATİK ön uç değişikliği -> hata (ör. v6 + bugünkü ön uç: ø/y, j, ɾ)


def frontend_probe(cfg: dict, n: int = PROBE_N) -> list[tuple]:
    """Checkpoint'in eğitim manifestinden (val) örnekler (probe_rows); manifest yoksa []."""
    path = os.path.join(paths.ANTALIA, f"val{cfg.get('manifest', '_phon')}.jsonl")
    if not os.path.exists(path):
        return []
    return probe_rows([json.loads(l) for l in open(path, encoding="utf8")], cfg, n)


def check_symbols(ck_symbols: list[str]) -> None:
    """Token -> id güncel tabloyla yapılır (Engine.ids); tablo `sorted(PHONES)` olduğundan yeni atom sonraki id'leri kaydırır ve eski checkpoint
    sessizce yanlış gömmeyi okurdu. Sona ekleme (BREAKS; 73 sembollük koşular) serbest: checkpoint tablosu güncel tablonun ÖNEKİ olmalı."""
    if list(ck_symbols) != SYMBOLS[:len(ck_symbols)]:
        d = next(((i, a, b) for i, (a, b) in enumerate(zip(ck_symbols, SYMBOLS)) if a != b), (len(SYMBOLS), "-", "-"))
        raise RuntimeError(f"sembol tablosu checkpoint'inkiyle uyuşmuyor (ilk fark id {d[0]}: checkpoint {d[1]!r}, güncel {d[2]!r}): id'ler kaymış, "
                           "bu checkpoint bu sembol tablosuyla kullanılamaz")


def check_frontend(engine: Engine, probe: list, allow: bool = False) -> dict:
    """Ön uç SÜRÜM TUZAĞI (docs/v9_plan.md §5): sentez her zaman güncel ön ucu kullanır; checkpoint başka token'larla eğitildiyse model görmediği girdi alır.
    Davranış denetimi: örnek cümleler güncel ön uçtan geçirilir, eğitim token'larıyla (ve örnek taşıyorsa dp_feat'iyle) karşılaştırılır. Farklı cümle oranı
    > PROBE_MAX_DIFF -> RuntimeError (allow=True ise uyarı); daha azı (ör. tek sözcüklük sözlük düzeltmesi) -> uyarı + farklı sözcükler. Örnek yoksa uyarı."""
    import warnings
    if not probe:
        warnings.warn("ön uç sürüm denetimi YAPILAMADI: checkpoint örnek taşımıyor, eğitim manifesti de yok; ön uç eğitimdekinden farklı olabilir", RuntimeWarning, stacklevel=3)
        return dict(checked=0)
    diffs = []
    for text, toks, *dp in probe:  # eski checkpoint'lerin örneği (metin, token): dp_feat denetlenmez
        u = engine.frontend(text)
        if u.tokens != toks:
            a, b = "".join(toks).split(" "), "".join(u.tokens).split(" ")
            diffs.append([(x, y) for x, y in zip(a, b) if x != y][:3] or [(len(a), len(b))])
        elif dp and dp[0] is not None and u.dp_feat != dp[0]:
            diffs.append([("dp_feat", sum(x != y for x, y in zip(dp[0], u.dp_feat)))])
    frac = len(diffs) / len(probe)
    msg = (f"ön uç, checkpoint'in eğitildiği girdiden farklı: {len(diffs)}/{len(probe)} örnek cümle ({frac:.0%}); ilk farklar: {diffs[:3]}")
    if frac > PROBE_MAX_DIFF and not allow:
        raise RuntimeError(msg + " -> SİSTEMATİK değişiklik: bu checkpoint'i bu ön uçla kullanma (eğitildiği ön uç sürümünü kullan ya da yeniden eğit); "
                           "bilinçli olarak geçmek için allow_frontend_mismatch=True")
    if diffs:
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return dict(checked=len(probe), differing=len(diffs))


class Synth:
    def __init__(self, ckpt: str, device: str = "cpu", engine: Engine | None = None, embed_alias: bool | None = None, vocoder: str = VOCODER,
                 long_vowel_scale: float | None = None, allow_frontend_mismatch: bool = False):
        self.dev = torch.device(device)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = ck["cfg"]
        stats = ck.get("stats")  # yayın paketi (scripts/export_tts_hf.py) mel istatistiğini taşır; eğitim checkpoint'i veri kökünden okur
        if stats is None:
            dcfg = yaml.safe_load(open(os.path.join(ROOT, self.cfg["data_config"]), encoding="utf8"))
            stats = json.load(open(os.path.join(dcfg["out_root"], "stats.json"), encoding="utf8"))
        self.model = build_model(self.cfg, len(ck["symbols"]), stats)
        self.model.load_state_dict(ck["model"])
        if (self.cfg["model"].get("embed_alias") is True) if embed_alias is None else embed_alias:  # "init": gömmeler eğitimde öğrenildi, bağlanmaz  # eğitimde görülmeyen a/aː gömmesi = ɑ/ɛ ortalaması (eski checkpoint'te de açılabilir)
            from dizgetts.train.embed_alias import tie
            tie(self.model, {s: i for i, s in enumerate(ck["symbols"])})
        if long_vowel_scale is not None:  # cfg'dekini ezer (checkpoint kopyalamadan dinleme denemesi)
            from dizgetts.train.dpfeat import set_long_scale
            set_long_scale(self.model, long_vowel_scale)
        self.model.to(self.dev).eval()
        self.epoch = ck["epoch"]
        self.vocoder, self.denoiser = load_vocoder(self.dev, vocoder)
        self.fe = self.cfg["frontend"]
        self.engine = (engine or Engine(**self.cfg.get("engine", {}))) if self.fe in ("engine", "dizge") else None  # cfg["engine"]: morph/tiers (M1b)
        if self.engine is not None:  # ön uç sürüm tuzağı: yanında taşınan örnek (yayın paketi / yeni ckpt) yoksa eğitim manifestinden
            check_symbols(ck["symbols"])
            self.frontend_check = check_frontend(self.engine, ck.get("frontend_probe") or frontend_probe(self.cfg), allow_frontend_mismatch)

    def ids(self, text: str):
        if self.fe == "espeak":
            norm = normalize(text)
            toks = symbols_espeak.tokenize(espeak.phonemize(norm), strip_stress=self.cfg["espeak_strip_stress"])
            return norm, toks, intersperse([symbols_espeak.SYMBOL_TO_ID[t] for t in toks], 0)
        u = self.engine.frontend(text)
        if not u.tokens:
            raise ValueError(f"konuşulacak fonem yok: {text!r} (norm: {u.norm!r})")
        self._dp = u.dp_feat
        return u.norm, u.tokens, intersperse(self.engine.ids(u), 0)

    @classmethod
    def from_hub(cls, repo: str = "iatagun/DizgeTTS-Antalia", device: str = "cpu", revision: str | None = None) -> "Synth":
        """HF paketinden yükle (yerel veri/koşu gerekmez): model.pt + vokoder + vurgu/sınır etiketleyici iatagun/DizgeBERT-G2PTTS."""
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel

        get = lambda f: hf_hub_download(repo, f, revision=revision)
        ck = get("model.pt")
        ecfg = dict(torch.load(ck, map_location="cpu", weights_only=False)["cfg"].get("engine", {}))
        if ecfg.pop("g2ptts", False):
            ecfg["g2ptts"], ecfg["g2ptts_tagger"] = True, AutoModel.from_pretrained(ecfg.pop("g2ptts_repo", "iatagun/DizgeBERT-G2PTTS"), trust_remote_code=True).eval()
        return cls(ck, device, engine=Engine(**ecfg), vocoder=get("hifigan_univ_v1"))

    @torch.inference_mode()
    def __call__(self, text: str, steps: int = 10, temperature: float = 0.667, length_scale: float | None = None):
        if length_scale is None:  # v5s: hız ayarı checkpoint config'inde (val'de seçildi); yoksa 1.0
            length_scale = float(self.cfg["model"].get("length_scale", 1.0))
        norm, toks, ids = self.ids(text)
        x = torch.tensor(ids, dtype=torch.long, device=self.dev)[None]
        xl = torch.tensor([x.shape[1]], device=self.dev)
        if self.cfg["model"].get("dp_feat"):  # v4: sınır özniteliği yalnız süre tahmincisine
            from dizgetts.train.dpfeat import intersperse_feat, set_dp_feat, set_round
            set_round(self.model, self.cfg["model"].get("dp_round"))  # v5: "cum" = birikimli yuvarlama (None = Matcha ceil)
            set_dp_feat(self.model, torch.tensor(intersperse_feat(self._dp), dtype=torch.long, device=self.dev)[None])
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
    out = a.out or os.path.join(paths.SAMPLES, run, f"ep{s.epoch}")
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
