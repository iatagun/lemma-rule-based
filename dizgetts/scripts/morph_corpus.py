"""Antalia manifestindeki her klibin normalize edilmiş cümlesini DizgeBERT-Morph ile çözümle -> <out_root>/morph_feats.jsonl (önbellek)
ve vurgu kuralı adaylarının sıklığını raporla (reports/morph_stats.json). CPU'da çalışır (GPU eğitimi bozmasın).

  python -X utf8 -m dizgetts.scripts.morph_corpus
"""
import collections, json, os, re

import yaml

from dizgetts.frontend.morph import MorphAnalyzer

HERE = os.path.join(os.path.dirname(__file__), "..")


def main():
    root = yaml.safe_load(open(os.path.join(HERE, "configs", "data.yaml"), encoding="utf8"))["out_root"]
    m = MorphAnalyzer("cpu")
    out = open(os.path.join(root, "morph_feats.jsonl"), "w", encoding="utf8")
    n_clips, pos, adv, neg, ins, yor, fin_person = 0, collections.Counter(), collections.Counter(), 0, 0, 0, 0
    words_total = 0
    for sp in ("train", "val", "test"):
        for l in open(os.path.join(root, f"{sp}_phon.jsonl"), encoding="utf8"):
            r = json.loads(l)
            toks = r["text_norm"].split()
            res = m.analyze(toks)
            out.write(json.dumps(dict(id=r["id"], tokens=toks, upos=[u for u, _ in res], feats=[f for _, f in res]), ensure_ascii=False) + "\n")
            n_clips += 1
            for t, (u, f) in zip(toks, res):
                if not t.isalpha():
                    continue
                words_total += 1
                pos[u] += 1
                lw = t.lower()
                if u == "ADV":
                    adv[lw] += 1
                neg += f.get("Polarity") == "Neg" and u in ("VERB", "AUX")
                ins += f.get("Case") == "Ins"
                yor += bool(re.search(r"yor", lw)) and u in ("VERB", "AUX")
                fin_person += u == "VERB" and "Person" in f and f.get("VerbForm") in (None, "Fin")
            if n_clips % 100 == 0:
                print(n_clips, flush=True)
    out.close()
    rep = dict(clips=n_clips, words=words_total, upos=dict(pos), negatif_fiil=neg, case_ins=ins, yor_fiil=yor, sonlu_fiil_kişi=fin_person,
               adv_top=adv.most_common(60), adv_tür=len(adv), adv_toplam=sum(adv.values()))
    json.dump(rep, open(os.path.join(HERE, "reports", "morph_stats.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "adv_top"}, ensure_ascii=False))
    print("ADV en sık:", rep["adv_top"][:40])


if __name__ == "__main__":
    main()
