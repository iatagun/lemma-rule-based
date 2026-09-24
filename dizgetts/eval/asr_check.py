"""Bir sentez klasörünü (synth.py çıktısı: NN.wav + index.json) Whisper-small ile yazıya çevirip CER/WER hesaplar.
Referans = cümlenin normalize edilmiş hali (canon). Aşama 6 değerlendirmesinin çekirdeği.

  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/eval/asr_check.py D:/dizgetts/samples/<run>/ep25 [--device cpu]
"""
import argparse, json, os, sys
from dizgetts.eval.asr_floor import canon, lev  # noqa: E402
from dizgetts.eval.whisper_score import Scorer  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    idx = json.load(open(os.path.join(a.dir, "index.json"), encoding="utf8"))
    sc = Scorer(device=a.device)
    ce = cl = we = wl = 0
    for r in idx:
        hyp = sc.transcribe(os.path.join(a.dir, f"{r['i']:02d}.wav"))
        ref, h = canon(r["text"]), canon(hyp)
        c, w = lev(ref, h), lev(ref.split(), h.split())
        r.update(asr=hyp, cer=round(c / max(1, len(ref)), 3), wer=round(w / max(1, len(ref.split())), 3))
        ce += c; cl += len(ref); we += w; wl += len(ref.split())
        print(f"{r['i']:02d} CER {r['cer']:.2f} | {r['text'][:60]}\n     -> {hyp[:80]}", flush=True)
    res = dict(cer=round(ce / cl, 4), wer=round(we / wl, 4), n=len(idx), model="openai/whisper-small", items=idx)
    json.dump(res, open(os.path.join(a.dir, "asr.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"TOPLAM CER {res['cer']:.3f} WER {res['wer']:.3f}  ({a.dir})")
