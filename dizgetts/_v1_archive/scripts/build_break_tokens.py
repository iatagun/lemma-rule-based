"""Aşama 5 ablasyon token'ları: dizge token'larından iki varyant üretir ve *_phon.jsonl'e ekler.

  tokens_nosep   : sözcük ayracı (' ') YOK; noktalama token'ları kalır.        (frontend: dizge_nosep)
  tokens_feat + feat_text/feat_meas : ayraçsız token + fonem başına öznitelik (frontend/prosody.py)  (frontend: dizge_feat / dizge_featm)
  tokens_breaks  : ayraç yerine sesten ölçülen kırılma sınıfı (breaks.jsonl):    (frontend: dizge_breaks)
                   B1 (<60 ms) token yok, B2 (60-250 ms) "|", B3 (>=250 ms) "‖"; noktalama token'ları kalır.
                   espeak/dizge sözcük sayısı uyuşmayan ya da ölçümü eksik sınırlarda yedek kural: noktalamadan sonra B3, aksi halde B1.

  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/scripts/build_break_tokens.py
"""
import collections, json, os, sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "dizgetts"))
from frontend.prosody import add_features  # noqa: E402
from frontend.symbols import BREAK_MAJOR, BREAK_MID, PAUSES, SYMBOL_TO_ID, WORD_SEP  # noqa: E402

MID_MS, MAJOR_MS = 60, 250  # eşikler: reports/breaks_stats.json (sözcük-içi sınırların %90'ı <=60 ms; noktalama sınırlarının %73'ü >=250 ms)


def klass(silence_ms: int) -> str:
    return BREAK_MAJOR if silence_ms >= MAJOR_MS else BREAK_MID if silence_ms >= MID_MS else ""


def rebuild(tokens: list[str], by_k: dict, usable: bool, mode: str, stats: collections.Counter) -> list[str]:
    out, k = [], 0
    for i, t in enumerate(tokens):
        if t != WORD_SEP:
            out.append(t)
            continue
        if mode == "nosep":
            k += 1
            continue
        if usable and k in by_k:
            c = klass(by_k[k])
            stats["ölçülen"] += 1
        else:
            c = BREAK_MAJOR if (i > 0 and tokens[i - 1] in PAUSES) else ""
            stats["yedek_kural"] += 1
        stats["B3" if c == BREAK_MAJOR else "B2" if c == BREAK_MID else "B1"] += 1
        if c:
            out.append(c)
        k += 1
    return out


def main():
    dcfg = yaml.safe_load(open(os.path.join(ROOT, "dizgetts", "configs", "data.yaml"), encoding="utf8"))
    root = dcfg["out_root"]
    br = {}
    for l in open(os.path.join(root, "breaks.jsonl"), encoding="utf8"):
        r = json.loads(l)
        br[r["id"]] = (r["n_words"] == r["n_words_dizge"], {b["k"]: b["silence_ms"] for b in r["boundaries"]})
    rep = {}
    for split in ("train", "val", "test"):
        p = os.path.join(root, f"{split}_phon.jsonl")
        rows = [json.loads(l) for l in open(p, encoding="utf8")]
        st = collections.Counter()
        for r in rows:
            usable, by_k = br[r["id"]]
            st["klip_yedek"] += not usable
            r["tokens_nosep"] = rebuild(r["tokens"], by_k, usable, "nosep", st)
            r["tokens_breaks"] = rebuild(r["tokens"], by_k, usable, "breaks", st)
            # öznitelik yolu: ayraçsız token + fonem başına (sözcük-başı, final sınıf); metin kuralı ve ölçülen sınıf sürümleri
            r["tokens_feat"], r["feat_text"] = add_features(r["tokens"])
            cls = {k: (3 if v >= MAJOR_MS else 2 if v >= MID_MS else 1) for k, v in by_k.items()} if usable else None
            _, r["feat_meas"] = add_features(r["tokens"], cls)
            assert len(r["feat_text"]) == len(r["tokens_feat"]) == len(r["feat_meas"])
            assert all(t in SYMBOL_TO_ID for t in r["tokens_breaks"] + r["tokens_nosep"])
        with open(p, "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rep[split] = dict(st)
        print(split, dict(st), flush=True)
    json.dump(dict(mid_ms=MID_MS, major_ms=MAJOR_MS, **rep), open(os.path.join(ROOT, "dizgetts", "reports", "break_tokens_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
