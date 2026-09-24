"""Eğitim manifestlerini Engine ile (yeniden) üret: `tokens` = Engine.frontend(text).tokens. v1 artık alanlarını atar.

  python -X utf8 -m dizgetts.scripts.build_manifest
Girdi/çıktı: <out_root>/{train,val,test}_phon.jsonl (yerinde; `text`, `espeak` alanları korunur). Rapor: reports/manifest_stats.json.
Engine sürümleri her satıra `engine_versions` olarak yazılır; train.py env.json'a alır.
"""
import collections, json, os

import yaml

from dizgetts.engine import Engine

HERE = os.path.join(os.path.dirname(__file__), "..")
LEGACY = ("tokens_nosep", "tokens_breaks", "tokens_feat", "feat_text", "feat_meas", "tokens_stress")


def main():
    root = yaml.safe_load(open(os.path.join(HERE, "configs", "data.yaml"), encoding="utf8"))["out_root"]
    e = Engine(bert_fallback=True)
    ver = e.versions()
    stats, per_split = collections.Counter(), {}
    for split in ("train", "val", "test"):
        p = os.path.join(root, f"{split}_phon.jsonl")
        rows = [json.loads(l) for l in open(p, encoding="utf8")]
        n_words = 0
        for r in rows:
            u = e.frontend(r["text"])
            r["text_norm"], r["tokens"], r["engine_versions"] = u.norm, u.tokens, ver
            for k in LEGACY:
                r.pop(k, None)
            for k, v in u.meta.get("stress", {}).items():
                stats[k] += v
            n_words += len(u.words)
            if u.meta.get("unknown_chars"):
                stats["unknown_chars"] += len(u.meta["unknown_chars"])
        with open(p, "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_split[split] = dict(clips=len(rows), words=n_words)
        print(split, per_split[split], flush=True)
    rep = dict(engine_versions=ver, splits=per_split, stress_counters=dict(stats))
    json.dump(rep, open(os.path.join(HERE, "reports", "manifest_stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
