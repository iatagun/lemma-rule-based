"""Sınır kuralları (frontend/boundary_rules.py) açık/kapalı: (1) kullanıcı kulak etiketleri (40 cümle, GELİŞTİRME), (2) Antalia test kayıtlarında
ÖLÇÜLEN sessizlik (breaks.jsonl; kurallar yazılırken bakılmadı). Sınıf: <60 ms 0, 60-250 ip, >=250 IP (g2ptts eğitim eşikleri). Yalnız noktalamasız sınırlar."""
import collections, json, re

from dizgetts.engine import Engine
from dizgetts.tools.read_breaks import labeled

R = {"0": 0, "ip": 1, "IP": 2, "cümle": 3}
cls = lambda ms: 2 if ms >= 250 else (1 if ms >= 60 else 0)


def score(pairs, name):
    """pairs: (gerçek 0/1/2, tahmin 0/1/2)"""
    tp = sum(g > 0 and p > 0 for g, p in pairs); ng = sum(g > 0 for g, _ in pairs); np_ = sum(p > 0 for _, p in pairs)
    P, Rc = tp / max(np_, 1), tp / max(ng, 1)
    strong = [p for g, p in pairs if g == 2]
    print(f"  {name:10s} sınır P {P:.2f} R {Rc:.2f} F1 {2*P*Rc/max(P+Rc,1e-9):.2f} | güçlü (2) sınırlardan tahmin 2: {sum(p==2 for p in strong)}/{len(strong)}, "
          f"0: {sum(p==0 for p in strong)} | gerçek 0'da yanlış sınır: {sum(p>0 for g,p in pairs if g==0)}/{sum(g==0 for g,_ in pairs)}")


def main():
    eng = {k: Engine(g2ptts=True, g2ptts_breaks=False, boundary_rules=k) for k in (False, True)}
    key = lambda s: re.sub(r"[^\w]", "", s.lower().replace("i̇", "i"))
    print("(1) kullanıcı etiketleri, 40 cümle (GELİŞTİRME seti)")
    L = labeled()
    for k in (False, True):
        pairs = []
        for n, rows in L.items():
            ew, j, mb = eng[k].frontend(" ".join(r["word"] for r in rows)).words, 0, []
            for r in rows:
                acc, b = "", 0
                while j < len(ew) and key(acc) != key(r["word"]):
                    acc += ew[j].text; b = min(R.get(ew[j].boundary, 0), 2); j += 1
                mb.append(b)
            pairs += [(min(r["brk"], 2), mb[i]) for i, r in enumerate(rows[:-1]) if not re.search(r"[,;:]$", r["word"])]
        score(pairs, "kurallı" if k else "yalnız model")
    print("(2) Antalia TEST kayıtları, ölçülen sessizlik (bakılmamış)")
    man = {json.loads(l)["id"]: json.loads(l) for l in open("D:/dizgetts/data/processed/antalia/test_phon.jsonl", encoding="utf8")}
    B = [json.loads(l) for l in open("D:/dizgetts/data/processed/antalia/breaks.jsonl", encoding="utf8")]
    B = [b for b in B if b["split"] == "test"]
    for k in (False, True):
        pairs, skip = [], 0
        for b in B:
            ws = eng[k].frontend(man[b["id"]]["text"]).words
            if len(ws) != b["n_words"]:
                skip += 1; continue
            pairs += [(cls(x["silence_ms"]), min(R.get(ws[x["k"]].boundary, 0), 2)) for x in b["boundaries"] if not x["punct"] and x["k"] < len(ws) - 1]
        score(pairs, "kurallı" if k else "yalnız model")
    print(f"  (atlanan klip: {skip}; konum: {len(pairs)})")


if __name__ == "__main__":
    main()
