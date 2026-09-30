"""Eğitim checkpoint'inden HF yayın paketi: model.pt (yalnız ağırlık + cfg + semboller + mel istatistiği; optimizer YOK) + HiFi-GAN vokoder + kart + örnek sesler.

  D:/dizgetts/venv/Scripts/python.exe -X utf8 -m dizgetts.scripts.export_tts_hf --ckpt D:/dizgetts/runs/<run>/ep250.pt [--out D:/dizgetts/hf/DizgeTTS-Antalia]
Sonra gidiş-dönüş: Synth(yerel ckpt) ile Synth(paket, HF etiketleyici) aynı token'ları ve (aynı tohumla) aynı mel'i üretmeli -> --check.
Push: python inference/push_g2ptts_hf.py --folder <out> --repo iatagun/DizgeTTS-Antalia [--public]
"""
import argparse, json, os, shutil

import soundfile as sf
import torch
import yaml

from dizgetts import paths
from dizgetts.train.train import ROOT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARD = os.path.join(HERE, "docs", "MODEL_CARD_tts.md")
SAMPLES = [  # kartta dinletilen cümleler (eğitim verisinde YOK)
    "Merhaba, ben DizgeTTS. Türkçe metinleri sesli okuyabilirim.",
    "Kağıt, kalem ve silgiyi masanın üstüne bıraktım; akşam yeniden bakacağım.",
    "Yarın sabah saat dokuzda İzmir'e gidiyoruz, değil mi?",
    "Bu sistem, sözcüklerin vurgusunu ve cümle içindeki duraklamaları kendisi tahmin ediyor.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=f"{paths.HOME}/hf/DizgeTTS-Antalia")
    ap.add_argument("--check", action="store_true", help="gidiş-dönüş denetimi (yerel Tagger vs HF etiketleyici; aynı tohumla mel farkı)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg.pop("init", None)
    cfg["engine"] = dict(cfg.get("engine", {}), g2ptts_repo="iatagun/DizgeBERT-G2PTTS")
    dcfg = yaml.safe_load(open(os.path.join(ROOT, cfg["data_config"]), encoding="utf8"))
    stats = json.load(open(os.path.join(dcfg["out_root"], "stats.json"), encoding="utf8"))
    out_ck = os.path.join(a.out, "model.pt")
    from dizgetts.eval.synth import frontend_probe
    probe = ck.get("frontend_probe") or frontend_probe(ck["cfg"])  # paket kullanıcısında manifest yok: ön uç denetimi için örnek pakette taşınır
    assert probe, "frontend_probe yok (manifest bulunamadı): paket ön uç sürüm denetimi olmadan yayınlanmamalı"
    torch.save(dict(model=ck["model"], cfg=cfg, symbols=ck["symbols"], epoch=ck["epoch"], stats=stats, source=os.path.basename(os.path.dirname(a.ckpt)),
                    frontend_probe=probe), out_ck)
    shutil.copy(paths.VOCODER, os.path.join(a.out, "hifigan_univ_v1"))
    shutil.copy(CARD, os.path.join(a.out, "README.md"))
    print("paket:", a.out, f"model.pt {os.path.getsize(out_ck)/1e6:.1f} MB")

    from dizgetts.eval.synth import Synth
    from dizgetts.engine import Engine
    from transformers import AutoModel

    ecfg = {k: v for k, v in cfg["engine"].items() if k != "g2ptts_repo"}
    hf = AutoModel.from_pretrained("iatagun/DizgeBERT-G2PTTS", trust_remote_code=True).eval()
    s = Synth(out_ck, "cpu", engine=Engine(**dict(ecfg, g2ptts_tagger=hf)), vocoder=os.path.join(a.out, "hifigan_univ_v1"))
    for i, t in enumerate(SAMPLES):
        torch.manual_seed(0)
        wav, norm, ph, _ = s(t)
        sf.write(os.path.join(a.out, "samples", f"{i:02d}.wav"), wav, 22050, subtype="PCM_16")
        print(f"{i:02d} {len(wav)/22050:.1f} sn  {ph}")
    if a.check:  # yerel (eğitimdeki) Tagger ile aynı token + aynı tohumla aynı mel
        loc = Synth(a.ckpt, "cpu")
        for t in SAMPLES:
            assert loc.ids(t)[1] == s.ids(t)[1], t
            outs = []
            for m in (loc, s):
                torch.manual_seed(0); outs.append(m(t)[0])
            d = float(abs(torch.tensor(outs[0]) - torch.tensor(outs[1])).max())
            assert d < 1e-4, (t, d)
        print("gidiş-dönüş OK: token birebir, dalga farkı < 1e-4")


if __name__ == "__main__":
    main()
