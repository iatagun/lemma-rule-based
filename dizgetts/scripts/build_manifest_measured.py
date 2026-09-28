"""v6m: süre tahmincisi özniteliği (dp_feat) ÖLÇÜLEN sınırlardan (Antalia sessizliği, breaks.jsonl) — token dizisi v6 manifestiyle birebir aynı.
Neden: v6-dp tahmin edilen (g2ptts) sınırlarla eğitildi; tahmin gerçek duraklamayla zayıf örtüşünce (F1 ~0,5) dp özniteliği yok saymayı öğrendi
(sınır kuralları sentezde süreyi medyanda 0 ms değiştirdi). Ölçülen sınırla eğitilen dp "IP = gerçek duraklama" ilişkisini öğrenir.
Sınıf: <60 ms 0, 60-250 ip, >=250 IP (g2ptts eşikleri); cümle sonu `cümle` kalır. Sözcük sayısı tutmayan klipte tahmin edilen sınır kalır (sayılır).

  python -X utf8 -m dizgetts.scripts.build_manifest_measured      # -> {split}_phon_v6m.jsonl
"""
import json

from dizgetts import paths
from dizgetts.engine import AssembleStage, Engine

ROOT = paths.ANTALIA
cls = lambda ms: "IP" if ms >= 250 else ("ip" if ms >= 60 else "0")


def main():
    e = Engine(g2ptts=True, g2ptts_breaks=False, pron_exceptions=True, register="özenli", length_rules=True)
    asm = AssembleStage(breaks=False)
    br = {b["id"]: b for b in map(json.loads, open(f"{ROOT}/breaks.jsonl", encoding="utf8"))}
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(f"{ROOT}/{split}_phon_v6.jsonl", encoding="utf8")]
        n_meas = n_fb = changed = 0
        for r in rows:
            u = e.frontend(r["text"])
            assert u.tokens == r["tokens"], f"{r['id']}: ön uç v6 manifestinden farklı (sürüm kayması)"
            b = br.get(r["id"])
            if b is None or b["n_words"] != len(u.words):
                n_fb += 1
                continue
            before = [w.boundary for w in u.words]
            for x in b["boundaries"]:
                if x["k"] < len(u.words) - 1 and u.words[x["k"]].boundary != "cümle":
                    u.words[x["k"]].boundary = cls(x["silence_ms"])
            u = asm(u)
            assert u.tokens == r["tokens"]
            changed += sum(a != w.boundary for a, w in zip(before, u.words))
            r["dp_feat"], r["dp_feat_src"] = u.dp_feat, "ölçülen"
            n_meas += 1
        with open(f"{ROOT}/{split}_phon_v6m.jsonl", "w", encoding="utf8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: ölçülen {n_meas}, tahminde kalan {n_fb}, değişen sınır {changed}", flush=True)


if __name__ == "__main__":
    main()
