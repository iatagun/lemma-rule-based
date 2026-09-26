"""Gerçek kayıtların Whisper-small CER/WER'i = Aşama 6 için TAVAN/taban (TTS bunun altına inemez).
Karşılaştırma: iki taraf da normalize() + noktalamasız küçük harf.
  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/eval/asr_floor.py [split ...]
"""
import json, os, re, sys
from dizgetts import paths
from dizgetts.frontend.normalize import normalize, tr_lower  # noqa: E402
from dizgetts.eval.whisper_score import Scorer  # noqa: E402

ROOT = paths.ANTALIA


def canon(s):
    return " ".join(re.sub(r"[^\w\s]|_", " ", tr_lower(normalize(s))).split())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


if __name__ == "__main__":
    splits = sys.argv[1:] or ["val", "test"]
    sc = Scorer()
    ce = cl = we = wl = 0
    per, items = [], []
    for sp in splits:
        for l in open(f"{ROOT}/{sp}.jsonl", encoding="utf8"):
            r = json.loads(l)
            ref, hyp = canon(r["text"]), canon(sc.transcribe(f"{ROOT}/{r['wav']}"))
            c, w = lev(ref, hyp), lev(ref.split(), hyp.split())
            ce += c; cl += len(ref); we += w; wl += len(ref.split())
            items.append(dict(id=r["id"], split=sp, cer_e=c, n_chars=len(ref), wer_e=w, n_words=len(ref.split())))  # eval/compare.py bölüm bazında taban hesaplar
            per.append((w / max(1, len(ref.split())), r["id"], ref[:100], hyp[:100]))
    per.sort(reverse=True)
    res = dict(splits=splits, clips=len(per), cer=round(ce / cl, 4), wer=round(we / wl, 4), worst=per[:8], model="openai/whisper-small", items=items)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "reports", "asr_floor.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print({k: v for k, v in res.items() if k not in ("worst", "items")})
