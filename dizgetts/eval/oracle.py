"""Kahin (oracle) deneyleri — analiz-yoluyla-sentez (IPO geleneği): sentezin açığı SÜREDE mi PERDEDE mi? Eğitim yok.

  D:/dizgetts/venv/Scripts/python.exe -X utf8 -m dizgetts.eval.oracle --ckpt D:/dizgetts/runs/<run>/ep150.pt [--label oracle_v6]
  python -X utf8 -m dizgetts.eval.compare oracle_v6_A oracle_v6_B oracle_v6_Bpsola oracle_v6_C oracle_v6_Rvoc oracle_v6_R

Koşullar (test klipleri, aynı tohum):
  A      normal sentez (süre tahmincisi)
  B      SÜRE KAHİNİ: token süreleri = gerçek kaydın MAS hizalaması (aynı modelin kodlayıcısıyla); çözücü aynı
  Bpsola B'nin kendi F0'ıyla Praat PSOLA yeniden sentezi (işlem kontrolü: C'deki farkın PSOLA'dan gelmediğini ayırmak için)
  C      SÜRE + F0 KAHİNİ: B'nin perdesi gerçek kaydın perde katmanıyla değiştirilir (B ile gerçek aynı zaman ekseninde; MAS tüm mel'i kaplar)
  Rvoc   gerçek mel -> HiFi-GAN (vokoder tavanı)
  R      gerçek kayıt
Süre etkisi = B - A; perde etkisi = C - Bpsola. Çıktı: <EVAL_OUT>/<label>_<koşul>/{NNN.wav, results.json} (compare.py biçimi).
"""
import argparse, json, math, os

import numpy as np
import parselmouth
import soundfile as sf
import torch
from parselmouth.praat import call

from dizgetts import paths
from dizgetts.eval.evaluate import canon, lev
from dizgetts.eval.synth import Synth
from dizgetts.eval.whisper_score import Scorer

ROOT = paths.ANTALIA
CONDS = ("A", "B", "Bpsola", "C", "Rvoc", "R")


@torch.no_grad()
def mas_frames(model, ids, mel_norm):
    """Interspersed token başına kare (Matcha MAS; prosody_acoustics.mas_frames ile aynı, cihazda)."""
    import matcha.utils.monotonic_align as MA
    from matcha.utils.model import sequence_mask

    dev = mel_norm.device
    x, xl = torch.tensor(ids, device=dev)[None], torch.tensor([len(ids)], device=dev)
    mu_x, _, x_mask = model.encoder(x, xl, None)
    y = mel_norm[None]
    y_mask = sequence_mask(torch.tensor([y.shape[-1]], device=dev), y.shape[-1]).unsqueeze(1).to(x_mask)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    const = -0.5 * math.log(2 * math.pi) * model.n_feats
    factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=dev)
    lp = (torch.matmul(factor.transpose(1, 2), y ** 2) - torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
          + torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1) + const)
    return MA.maximum_path(lp, attn_mask.squeeze(1))[0].sum(-1)


@torch.no_grad()
def synth_with_durations(model, ids, frames, steps=10, temperature=0.667):
    """Matcha synthesise ile aynı, tek fark: w_ceil = verilen kare sayıları (süre tahmincisi kullanılmaz)."""
    from matcha.utils.model import denormalize, fix_len_compatibility, generate_path, sequence_mask

    dev = frames.device
    x, xl = torch.tensor(ids, device=dev)[None], torch.tensor([len(ids)], device=dev)
    mu_x, _, x_mask = model.encoder(x, xl, None)
    w = frames.float()[None, None] * x_mask
    y_len = w.sum().long()
    y_max = fix_len_compatibility(int(y_len))
    y_mask = sequence_mask(y_len[None], y_max).unsqueeze(1).to(x_mask.dtype)
    attn = generate_path(w.squeeze(1), (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1)).unsqueeze(1)
    mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
    out = model.decoder(mu_y, y_mask, steps, temperature, None)[:, :, :int(y_len)]
    return denormalize(out, model.mel_mean, model.mel_std)


def psola(wav_path, out_path, floor, ceiling, pitch_from=None):
    """Praat Manipulation (PSOLA). pitch_from=None: kendi perdesi (kontrol); aksi halde o dosyanın perde katmanı."""
    s = parselmouth.Sound(wav_path)
    man = call(s, "To Manipulation", 0.01, floor, ceiling)
    if pitch_from is not None:
        src = call(parselmouth.Sound(pitch_from), "To Manipulation", 0.01, floor, ceiling)
        call([man, call(src, "Extract pitch tier")], "Replace pitch tier")
    call(call(man, "Get resynthesis (overlap-add)"), "Save as WAV file", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default="oracle_v6")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    sy = Synth(a.ckpt, a.device)
    m = sy.model
    from dizgetts.train.dpfeat import intersperse_feat, set_dp_feat
    from matcha.utils.model import normalize as mel_normalize

    rows = [json.loads(l) for l in open(f"{ROOT}/test{sy.cfg['manifest']}.jsonl", encoding="utf8")][:a.limit]
    # F0 aralığı gerçek kayıtlardan (Hirst iki geçiş, prosody_acoustics ile aynı kural)
    allf = np.concatenate([(lambda f: f[f > 0])(parselmouth.Sound(os.path.join(ROOT, r["wav"])).to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600)
                                                   .selected_array["frequency"]) for r in rows[::3]])
    q25, q75 = np.percentile(allf, [25, 75])
    floor, ceiling = round(0.75 * q25), round(1.5 * q75)
    print(f"F0 aralığı {floor}-{ceiling} Hz", flush=True)
    outs = {c: os.path.join(paths.EVAL_OUT, f"{a.label}_{c}") for c in CONDS}
    for d in outs.values():
        os.makedirs(d, exist_ok=True)
    for i, r in enumerate(rows):
        norm, toks, ids = sy.ids(r["text"])
        assert toks == r["tokens"], f"{r['id']}: sentez token'ı manifestten farklı"
        set_dp_feat(m, torch.tensor(intersperse_feat(sy._dp), dtype=torch.long, device=dev)[None])
        torch.manual_seed(i)
        sf.write(f"{outs['A']}/{i:03d}.wav", sy(r["text"])[0], 22050, subtype="PCM_16")
        mel = torch.load(os.path.join(ROOT, "mels", r["id"] + ".pt")).to(dev)
        frames = mas_frames(m, ids, mel_normalize(mel, m.mel_mean, m.mel_std))
        assert int(frames.sum()) == mel.shape[-1]
        torch.manual_seed(i)
        melB = synth_with_durations(m, ids, frames)
        with torch.inference_mode():
            wB = sy.denoiser(sy.vocoder(melB).clamp(-1, 1).squeeze(), strength=0.00025).cpu().squeeze().numpy()
            wR = sy.denoiser(sy.vocoder(mel[None]).clamp(-1, 1).squeeze(), strength=0.00025).cpu().squeeze().numpy()
        sf.write(f"{outs['B']}/{i:03d}.wav", wB, 22050, subtype="PCM_16")
        sf.write(f"{outs['Rvoc']}/{i:03d}.wav", wR, 22050, subtype="PCM_16")
        x, sr = sf.read(os.path.join(ROOT, r["wav"]), dtype="float32")
        sf.write(f"{outs['R']}/{i:03d}.wav", x, sr, subtype="PCM_16")
        psola(f"{outs['B']}/{i:03d}.wav", f"{outs['Bpsola']}/{i:03d}.wav", floor, ceiling)
        psola(f"{outs['B']}/{i:03d}.wav", f"{outs['C']}/{i:03d}.wav", floor, ceiling, pitch_from=f"{outs['R']}/{i:03d}.wav")
        if i % 10 == 0:
            print(f"sentez {i}/{len(rows)}  B {len(wB)/22050:.2f} sn, gerçek {len(x)/sr:.2f} sn", flush=True)
    del sy
    torch.cuda.empty_cache()
    sc = Scorer(device=a.device)
    utmos = torch.hub.load("tarepan/SpeechMOS:ed25eacbfa42b99156c36ebec67a733b5dbb9b79", "utmos22_strong", trust_repo=True).eval()
    for c, d in outs.items():
        res = []
        for i, r in enumerate(rows):
            p = f"{d}/{i:03d}.wav"
            x, sr = sf.read(p, dtype="float32")
            hyp = sc.transcribe(p)
            ref, h = canon(r["text"]), canon(hyp)
            with torch.no_grad():
                mos = float(utmos(torch.from_numpy(x)[None], sr))
            res.append(dict(i=i, id=r["id"], split="test", dur=round(len(x) / sr, 2), n_chars=len(ref), n_words=len(ref.split()),
                            cer_e=lev(ref, h), wer_e=lev(ref.split(), h.split()), utmos=round(mos, 3), asr=hyp, ref=ref))
        tot = lambda k, n: sum(x[k] for x in res) / sum(x[n] for x in res)
        summ = dict(label=f"{a.label}_{c}", ckpt=a.ckpt, n=len(res), cer=round(tot("cer_e", "n_chars"), 4), wer=round(tot("wer_e", "n_words"), 4),
                    utmos=round(float(np.mean([x["utmos"] for x in res])), 3))
        json.dump(dict(summary=summ, items=res), open(f"{d}/results.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
