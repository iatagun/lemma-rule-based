"""Eğitim manifestlerini Engine ile (yeniden) üret: `tokens` = Engine.frontend(text).tokens. v1 artık alanlarını atar.

  python -X utf8 -m dizgetts.scripts.build_manifest
Girdi/çıktı: <out_root>/{train,val,test}_phon.jsonl (yerinde; `text`, `espeak` alanları korunur). Rapor: reports/manifest_stats.json.
Engine sürümleri her satıra `engine_versions` olarak yazılır; train.py env.json'a alır.
"""
import argparse, collections, json, os

import yaml

from dizgetts.engine import Engine

HERE = os.path.join(os.path.dirname(__file__), "..")
LEGACY = ("tokens_nosep", "tokens_breaks", "tokens_feat", "feat_text", "feat_meas", "tokens_stress")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-suffix", default="_phon", help="çıktı: {split}<suffix>.jsonl (girdi her zaman *_phon.jsonl'in `text` alanı)")
    ap.add_argument("--tiers", default="", help="M1b vurgu katmanları, virgülle (yor,neg,ins,person); boş = M1a")
    ap.add_argument("--g2ptts", action="store_true", help="dizge-g2p-tts ön ucu (karma vurgu + model sınır token'ları); --tiers yok sayılır")
    ap.add_argument("--no-breaks", action="store_true", help="--g2ptts ile: sınır token'ı ekleme (yalnız karma vurgu; v3a)")
    a = ap.parse_args()
    tiers = tuple(t for t in a.tiers.split(",") if t)
    root = yaml.safe_load(open(os.path.join(HERE, "configs", "data.yaml"), encoding="utf8"))["out_root"]
    cache = None
    if tiers:  # morfolojik özellikler önbellekten (scripts/morph_corpus.py); yoksa canlı çözümlenir
        mp = os.path.join(root, "morph_feats.jsonl")
        if os.path.exists(mp):
            cache = {}
            for l in open(mp, encoding="utf8"):
                m = json.loads(l)
                cache[" ".join(m["tokens"])] = list(zip(m["upos"], m["feats"]))
    e = Engine(bert_fallback=True, g2ptts=True, g2ptts_breaks=not a.no_breaks) if a.g2ptts else Engine(bert_fallback=True, morph=bool(tiers), tiers=tiers, morph_cache=cache)
    ver = e.versions()
    stats, per_split = collections.Counter(), {}
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(os.path.join(root, f"{split}_phon.jsonl"), encoding="utf8")]
        p = os.path.join(root, f"{split}{a.out_suffix}.jsonl")
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
            if u.meta.get("g2ptts_hizalama_hatasi"):
                stats["g2ptts_hizalama_hatasi"] += 1
            stats["kırılma_ip"] += u.tokens.count("|")
            stats["kırılma_IP"] += u.tokens.count("‖")
            if u.meta.get("morph_hizalama_hatasi"):
                stats["morph_hizalama_hatasi"] += 1
        with open(p, "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_split[split] = dict(clips=len(rows), words=n_words)
        print(split, per_split[split], flush=True)
    rep = dict(engine_versions=ver, tiers=tiers, out_suffix=a.out_suffix, splits=per_split, stress_counters=dict(stats))
    json.dump(rep, open(os.path.join(HERE, "reports", f"manifest_stats{a.out_suffix}.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
