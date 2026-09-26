"""dizge-g2p-tts ön uç ABLASYONU (eğitim yok): aynı TTS modeli, aynı 143 test+val cümlesi; yalnız ön uç girdisi değişir.
  python -X utf8 -u -m dizgetts.eval.ablate_frontend --ckpt <v5 ckpt> --variant punct|oracle|m1a|default --label <etiket>
Çıktı evaluate.py ile aynı biçimde (D:/dizgetts/eval_out/<label>/NNN.wav + results.json) -> prosody_acoustics align/compare çalışır.

Varyantlar (Futamata vd. 2021 tasarımı: tahmin vs yalnız noktalama vs gerçek kayıttan KEHANET sınırı):
  default : g2ptts sınırı (model + noktalama) + g2ptts karma vurgu
  punct   : sınır YALNIZ noktalamadan (sınır modelinin katkısı = default - punct)
  oracle  : sınır gerçek kayıttaki ölçülen sessizlikten (breaks.jsonl: <60 ms 0, <250 ip, >=250 IP), noktalamayla birleşik (kalan pay = oracle - default)
  m1a     : vurgu M1a kuralı (kök sözlüğü + clitic + son seslem; ≈ son hece) — vurgu katmanının etkisi = default - m1a
Birleştirme kuralı g2ptts ile aynı: noktalamalı sınırda sınıf = max(noktalama sınıfı, kaynak sınıfı); son sözcük 'cümle'.
"""
import argparse
import json
import os



import soundfile as sf
import torch
from matcha.utils.utils import intersperse

from dizgetts import paths
from dizgetts.engine import AssembleStage
from dizgetts.eval.asr_floor import canon, lev
from dizgetts.eval.synth import Synth
from dizgetts.eval.whisper_score import Scorer

from dizgetts.frontend.stress import StressRules, to_phone_index

ROOT = paths.ANTALIA
RANK = ("0", "ip", "IP", "cümle")
PUNCT = {",": "ip", ";": "IP", ".": "cümle", "?": "cümle", "!": "cümle"}


def punct_class(w):
    return max((PUNCT[p] for p in w.punct), key=RANK.index, default="0")


def apply_variant(u, variant, oracle, R):
    ws = u.words
    if variant in ("punct", "oracle"):
        for k, w in enumerate(ws):
            src = "0"
            if variant == "oracle" and oracle is not None and k in oracle:
                s = oracle[k]
                src = "0" if s < 60 else "ip" if s < 250 else "IP"
            w.boundary = max(punct_class(w), src, key=RANK.index)
        ws[-1].boundary = "cümle"
    elif variant == "m1a":
        for w in ws:
            k, tag = R.syllable(w.text)  # tiers=() -> M1a
            w.stress_src = f"m1a:{tag}"
            w.stress = None if k is None else to_phone_index(w.text, w.phones, k)[0]
    return AssembleStage(breaks=False)(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", choices=("default", "punct", "oracle", "m1a"), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = os.path.join(paths.EVAL_OUT, a.label)
    os.makedirs(out, exist_ok=True)
    rows = [dict(json.loads(l), split=sp) for sp in ("test", "val") for l in open(f"{ROOT}/{sp}.jsonl", encoding="utf8")]
    br = {b["id"]: b for b in map(json.loads, open(f"{ROOT}/breaks.jsonl", encoding="utf8"))}
    torch.manual_seed(0)
    sy = Synth(a.ckpt, a.device)
    assert sy.cfg["model"].get("dp_feat"), "ablasyon dp_feat'li model (v4/v5) içindir"
    sc = Scorer(device=a.device)
    R = StressRules()
    res, n_oracle_ok, n_changed = [], 0, 0
    for i, r in enumerate(rows):
        u = sy.engine.frontend(r["text"])
        base = (list(u.tokens), list(u.dp_feat))
        oracle = None
        b = br.get(r["id"])
        if a.variant == "oracle" and b and b["n_words"] == b["n_words_dizge"] == len(u.words):
            oracle = {bd["k"]: bd["silence_ms"] for bd in b["boundaries"]}
            n_oracle_ok += 1
        u = apply_variant(u, a.variant, oracle, R)
        n_changed += (list(u.tokens), list(u.dp_feat)) != base
        sy.ids = lambda text, u=u: (u.norm, u.tokens, intersperse(sy.engine.ids(u), 0))  # sentez bu ön uç çıktısını kullanır
        sy._dp = u.dp_feat
        wav, *_ = sy(r["text"])
        p = os.path.join(out, f"{i:03d}.wav")
        sf.write(p, wav, 22050, subtype="PCM_16")
        hyp = sc.transcribe(p)
        ref, h = canon(r["text"]), canon(hyp)
        res.append(dict(i=i, id=r["id"], split=r["split"], dur=round(len(wav) / 22050, 2), n_chars=len(ref), n_words=len(ref.split()),
                        cer_e=lev(ref, h), wer_e=lev(ref.split(), h.split()), utmos=0.0, asr=hyp, ref=ref))
        if i % 40 == 0:
            print(f"{a.label} {i}/{len(rows)}", flush=True)
    tot = lambda k, d: sum(x[k] for x in res) / sum(x[d] for x in res)
    summ = dict(label=a.label, ckpt=a.ckpt, variant=a.variant, n=len(res), cer=round(tot("cer_e", "n_chars"), 4), wer=round(tot("wer_e", "n_words"), 4),
                changed_clips=n_changed, oracle_clips=n_oracle_ok if a.variant == "oracle" else None)
    json.dump(dict(summary=summ, items=res), open(os.path.join(out, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
