"""Sınır (duraklama) iç değerlendirmesi, eğitimsiz: metin kuralları vs Antalia kayıtlarından ÖLÇÜLEN sözcük-sınırı sessizliği.
  python -X utf8 -m dizgetts.eval.boundary_baseline

Etiket: <out_root>/breaks.jsonl (v1 _v1_archive/scripts/derive_breaks.py; baseline MAS hizalaması + enerji). Sınır k = sözcük k ile k+1 arası.
  kırılma = sessizlik >= 60 ms (B2+; symbols.py B1/B2/B3 eşiği). Sözcük sayısı engine ile tutmayan klipler atlanır.
Sistemler: noktalama (yalnız noktalamada kırılma) | dep_K (noktalama + DizgeBERT-Dep: sözcük k'de BİTEN en büyük öbek >= K sözcükse ve
  bir sonraki sözcüğe bağlanmıyorsa kırılma; K train'de seçilir). Raporlanan: noktalamasız sınırlarda P/R/F1 + tüm sınırlarda F1, bölüm bölüm.
Dep çözümlemeleri <out_root>/dep_parse.jsonl'e önbelleklenir (sınır modeli eğitimi de kullanır).
"""
from __future__ import annotations

import json
import os

import yaml

HERE = os.path.join(os.path.dirname(__file__), "..")
SENT_END = {".", "?", "!", ";"}
BREAK_MS = 60


def dep_cache(root: str) -> dict[str, dict]:
    path = os.path.join(root, "dep_parse.jsonl")
    if not os.path.exists(path):
        import torch

        from dizgetts.frontend.dep import DepParser

        p = DepParser("cuda" if torch.cuda.is_available() else "cpu")
        with open(path + ".tmp", "w", encoding="utf8") as out:
            for sp in ("train", "val", "test"):
                for l in open(os.path.join(root, f"{sp}_phon.jsonl"), encoding="utf8"):
                    r = json.loads(l)
                    toks, heads, rels, s = r["text_norm"].split(), [], [], 0
                    for i, t in enumerate(toks + ["."]):  # cümle cümle çözümle (MAX_LEN), başları genel indekse kaydır
                        if t in SENT_END or i == len(toks):
                            part = toks[s:i + 1]
                            if part:
                                for h, rel in p.parse(part):
                                    heads.append(h + s if h else 0)
                                    rels.append(rel)
                            s = i + 1
                    out.write(json.dumps(dict(id=r["id"], split=sp, tokens=toks, heads=heads, rels=rels), ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
    return {d["id"]: d for d in map(json.loads, open(path, encoding="utf8"))}


def closing_phrase(heads: list[int], t: int) -> tuple[int, int]:
    """Belirteç t'de (0-tabanlı) BİTEN en büyük öbek: (boyut, öbek başının başı [1-tabanlı, 0=kök]). Öbek = bir düğümün alt ağacı."""
    n = len(heads)
    kids = [[] for _ in range(n)]
    for i, h in enumerate(heads):
        if h:
            kids[h - 1].append(i)
    span = {}

    def sp(i):
        if i not in span:
            lo = hi = i
            for c in kids[i]:
                a, b = sp(c)
                lo, hi = min(lo, a), max(hi, b)
            span[i] = (lo, hi)
        return span[i]

    best, a = (1, heads[t]), t
    while heads[a] and sp(heads[a] - 1)[1] == t:  # ata da t'de bitiyorsa yukarı çık
        a = heads[a] - 1
    lo, hi = sp(a)
    if hi == t:
        best = (hi - lo + 1, heads[a])
    return best


def main():
    root = yaml.safe_load(open(os.path.join(HERE, "configs", "data.yaml"), encoding="utf8"))["out_root"]
    deps = dep_cache(root)
    rows, skipped = [], 0  # (split, noktalama_var, gerçek_kırılma, öbek_boyutu, öbek_sonraki_sözcüğe_mi)
    for l in open(os.path.join(root, "breaks.jsonl"), encoding="utf8"):
        b = json.loads(l)
        d = deps.get(b["id"])
        if d is None:
            skipped += 1
            continue
        widx = [i for i, t in enumerate(d["tokens"]) if t.isalpha()]
        if len(widx) != b["n_words"] or b["n_words"] != b["n_words_dizge"]:
            skipped += 1
            continue
        for bd in b["boundaries"]:
            k = bd["k"]
            if k + 1 >= len(widx):
                continue
            t, t_next = widx[k], widx[k + 1]
            size, head = closing_phrase(d["heads"], t)
            rows.append((d["split"], bool(bd["punct"]), bd["silence_ms"] >= BREAK_MS, size, head == t_next + 1))

    def score(pred, split, nopunct_only):
        sel = [(pred(r), r[2]) for r in rows if r[0] in split and (not nopunct_only or not r[1])]
        tp = sum(p and g for p, g in sel); fp = sum(p and not g for p, g in sel); fn = sum(g and not p for p, g in sel)
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        return P, R, 2 * P * R / (P + R) if P + R else 0.0, len(sel), sum(g for _, g in sel)

    dep = lambda K: (lambda r: r[1] or (r[3] >= K and not r[4]))
    K = max(range(1, 13), key=lambda K: score(dep(K), ("train",), True)[2])
    print(f"klip atlandı: {skipped}; sınır: {len(rows)}; K (train'de seçildi) = {K}")
    for name, split in (("train", ("train",)), ("val+test", ("val", "test"))):
        for sname, f in (("noktalama", lambda r: r[1]), (f"dep_K{K}", dep(K)), ("dep_K1", dep(1))):
            P, R, F, n, g = score(f, split, True)
            Fall = score(f, split, False)[2]
            print(f"  {name:8s} {sname:10s} noktalamasız: P {P:.3f} R {R:.3f} F1 {F:.3f} (n={n}, kırılma={g})  |  tüm sınırlar F1 {Fall:.3f}")


if __name__ == "__main__":
    main()
