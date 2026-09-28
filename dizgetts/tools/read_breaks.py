"""Kullanıcının Praat `duraklama` işaretlerini oku: sözcük başına (sözcük, sınır 0-3, sonraki sessizlik sn). Boşluk aralığına yazılan işaret önceki sözcüğe aittir."""
import glob, os

import parselmouth
from parselmouth.praat import call

DIR = "D:/dizgetts/user_prosody/praat"


def read(path):
    tg = parselmouth.read(path)
    rows = []
    for i in range(1, int(call(tg, "Get number of intervals", 1)) + 1):
        w = call(tg, "Get label of interval", 1, i).strip()
        s, e = call(tg, "Get start time of interval", 1, i), call(tg, "Get end time of interval", 1, i)
        b = call(tg, "Get label of interval", 2, int(call(tg, "Get interval at time", 2, (s + e) / 2))).strip()
        if w:
            rows.append(dict(word=w, s=s, e=e, brk=int(b) if b.isdigit() else 0, gap=0.0))
        elif rows:  # boşluk: önceki sözcüğün sonrası
            rows[-1]["gap"] += e - s
            if b.isdigit():
                rows[-1]["brk"] = max(rows[-1]["brk"], int(b))
    if rows:
        rows[-1]["gap"] = 0.0  # cümle sonu sessizliği sayılmaz
    return rows


def labeled():
    out = {}
    for p in sorted(glob.glob(f"{DIR}/*.TextGrid")):
        r = read(p)
        if any(x["brk"] for x in r):
            out[os.path.basename(p)[:2]] = r
    return out
