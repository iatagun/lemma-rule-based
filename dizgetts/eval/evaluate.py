"""Aşama 6 değerlendirme çekirdeği: bir checkpoint'i test (+val) cümlelerinde sentezle, Whisper-small CER/WER ve UTMOS hesapla.
Cümleler eğitimde hiç görülmeyen test klipleri (val kayıp izlemede kullanıldı). Çıktı: D:/dizgetts/eval_out/<label>/{NNN.wav, results.json}.

  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/eval/evaluate.py --ckpt D:/dizgetts/runs/<run>/ep150.pt --label espeak_ep150 [--splits test val] [--device cuda]
"""
import argparse, json, os, sys, time

import numpy as np
import soundfile as sf
import torch
from dizgetts.eval.asr_floor import canon, lev  # noqa: E402
from dizgetts.eval.synth import Synth  # noqa: E402
from dizgetts.eval.whisper_score import Scorer  # noqa: E402

ROOT = "D:/dizgetts/data/processed/antalia"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = os.path.join("D:/dizgetts/eval_out", a.label)
    os.makedirs(out, exist_ok=True)
    rows = [dict(json.loads(l), split=sp) for sp in a.splits for l in open(f"{ROOT}/{sp}.jsonl", encoding="utf8")]
    torch.manual_seed(a.seed)
    sy = Synth(a.ckpt, a.device)
    sc = Scorer(device=a.device)
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    res = []
    t0 = time.time()
    for i, r in enumerate(rows):
        wav, norm, ph, sec = sy(r["text"])
        p = os.path.join(out, f"{i:03d}.wav")
        sf.write(p, wav, 22050, subtype="PCM_16")
        hyp = sc.transcribe(p)
        ref, h = canon(r["text"]), canon(hyp)
        with torch.no_grad():
            mos = float(utmos(torch.from_numpy(wav)[None].float(), 22050))
        res.append(dict(i=i, id=r["id"], split=r["split"], dur=round(len(wav) / 22050, 2), n_chars=len(ref), n_words=len(ref.split()),
                        cer_e=lev(ref, h), wer_e=lev(ref.split(), h.split()), utmos=round(mos, 3), asr=hyp, ref=ref))
        if i % 20 == 0:
            print(f"{a.label} {i}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    tot = lambda k, d: sum(x[k] for x in res) / sum(x[d] for x in res)
    summ = dict(label=a.label, ckpt=a.ckpt, n=len(res), cer=round(tot("cer_e", "n_chars"), 4), wer=round(tot("wer_e", "n_words"), 4),
                utmos=round(float(np.mean([x["utmos"] for x in res])), 3))
    json.dump(dict(summary=summ, items=res), open(os.path.join(out, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ), flush=True)
