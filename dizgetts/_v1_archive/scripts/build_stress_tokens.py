"""dizge token'larına espeak vurgusunu aktar (frontend/stress.py) -> *_phon.jsonl `tokens_stress` alanı (frontend: dizge_stress).

  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/scripts/build_stress_tokens.py

Rapor: reports/stress_transfer_stats.json (sayaçlar, vurgu-konum dağılımı, inceleme örnekleri).
"""
import collections, json, os, random, sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "dizgetts"))
from frontend.stress import ESPEAK_VOWELS, dizge_words, espeak_stress_index, espeak_words, transfer_stress  # noqa: E402
from frontend.symbols import PHONES, STRESS, SYMBOL_TO_ID  # noqa: E402


def from_end(tokens: list[str], w: list[int]) -> int | None:
    """Sözcükte vurgulu ünlünün sondan sırası (0 = son hece); vurgu yoksa None."""
    v = [i for i in w if tokens[i] in PHONES and PHONES[tokens[i]][0] == "ünlü"]
    for k, i in enumerate(reversed(v)):
        if i > 0 and tokens[i - 1] == STRESS:
            return k
    return None


def main():
    dcfg = yaml.safe_load(open(os.path.join(ROOT, "dizgetts", "configs", "data.yaml"), encoding="utf8"))
    root = dcfg["out_root"]
    rng = random.Random(0)
    total = collections.Counter()
    pos_dizge, pos_espeak = collections.Counter(), collections.Counter()
    examples = []
    for split in ("train", "val", "test"):
        p = os.path.join(root, f"{split}_phon.jsonl")
        rows = [json.loads(l) for l in open(p, encoding="utf8")]
        for r in rows:
            out, st = transfer_stress(r["tokens"], r["espeak"])
            assert all(t in SYMBOL_TO_ID for t in out)
            assert [t for t in out if t != STRESS] == r["tokens"], "vurgu dışında hiçbir token değişmemeli"
            r["tokens_stress"] = out
            total.update(st)
            if st["cümle_yedek_son_ünlü"]:
                continue  # sözcük eşlemesi güvenilmez; konum istatistiğine katma
            dw, ew = dizge_words(out), espeak_words(r["espeak"])
            for w, e in zip(dw, ew):
                k = from_end(out, w)
                pos_dizge["vurgusuz" if k is None else k] += 1
                s, n = espeak_stress_index(e)
                pos_espeak["vurgusuz" if s is None else n - 1 - s] += 1
                if rng.random() < 0.002:
                    examples.append(dict(espeak=e, dizge="".join(out[i] for i in w if out[i] != STRESS), dizge_vurgulu="".join(out[i] for i in w)))
        with open(p, "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(rows), flush=True)
    n_esp_marks = 0
    for split in ("train", "val", "test"):
        for l in open(os.path.join(root, f"{split}_phon.jsonl"), encoding="utf8"):
            r = json.loads(l)
            n_esp_marks += sum(1 for w in espeak_words(r["espeak"]) if espeak_stress_index(w)[0] is not None)
    rep = dict(sayaçlar=dict(total), espeak_vurgulu_sözcük=n_esp_marks, dizge_vurgu_konumu_sondan=dict(pos_dizge),
               espeak_vurgu_konumu_sondan=dict(pos_espeak), ornekler=examples[:40])
    json.dump(rep, open(os.path.join(ROOT, "dizgetts", "reports", "stress_transfer_stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "ornekler"}, ensure_ascii=False, indent=1))
    for e in examples[:30]:
        print(e["espeak"].ljust(22), e["dizge"].ljust(24), "->", e["dizge_vurgulu"])


if __name__ == "__main__":
    main()
