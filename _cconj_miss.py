"""Why do explicit CCONJ coordinations miss?"""
import sys, pathlib
sys.path.insert(0, ".")
from morphology import create_default_analyzer
from morphology.sentence import SentenceAnalyzer
from morphology.dependency import DependencyParser
from benchmark.eval_dep import parse_conllu, _strip_punct_and_reindex
from collections import Counter

a = create_default_analyzer()
sa = SentenceAnalyzer(a)
dp = DependencyParser()
conllu = pathlib.Path("benchmark/test.conllu")

cconj_misses = []

for sent_id, text, gold_tokens_raw in parse_conllu(conllu):
    gold_tokens = _strip_punct_and_reindex(gold_tokens_raw)
    try:
        st_tokens = sa.analyze(text)
        pred_tokens = dp.parse(st_tokens, text=text)
    except Exception:
        continue
    n = min(len(gold_tokens), len(pred_tokens))
    for k in range(n):
        g = gold_tokens[k]
        p = pred_tokens[k]
        if g["deprel"] != "conj" or p.deprel == "conj":
            continue
        # Check for CCONJ between head and conj
        head_id = g["head"]
        zone = range(min(head_id, k+1), max(head_id, k+1))
        has_cconj = any(gold_tokens[z]["upos"] == "CCONJ" for z in zone if z < n)
        if not has_cconj:
            continue
        
        # Check if this token was assigned
        gold_head = next((t for t in gold_tokens if t["id"] == g["head"]), None)
        cconj_misses.append({
            "form": p.form,
            "upos": p.upos,
            "gold_upos": g["upos"],
            "pred_deprel": p.deprel,
            "pred_assigned": p.is_assigned,
            "gold_head_form": gold_head["form"] if gold_head else "?",
            "gold_head_deprel": gold_head["deprel"] if gold_head else "?",
            "distance": abs(k + 1 - head_id),
        })

print(f"CCONJ-based conj misses: {len(cconj_misses)}")

print("\nBy pred deprel:")
for dr, c in Counter(d["pred_deprel"] for d in cconj_misses).most_common():
    print(f"  {c:3d}  {dr}")

print("\nBy gold UPOS:")
for u, c in Counter(d["gold_upos"] for d in cconj_misses).most_common():
    print(f"  {c:3d}  {u}")

print("\nBy gold head deprel:")
for dr, c in Counter(d["gold_head_deprel"] for d in cconj_misses).most_common():
    print(f"  {c:3d}  {dr}")

print("\nBy distance (conj to head):")
dists = [d["distance"] for d in cconj_misses]
for d_val in sorted(set(dists)):
    c = dists.count(d_val)
    if c >= 3:
        print(f"  dist={d_val:2d}: {c:3d}")

print("\nExamples:")
for d in cconj_misses[:20]:
    print(f"  {d['form']:20s} pred={d['pred_deprel']:12s} gUPOS={d['gold_upos']:5s} head={d['gold_head_form']:15s} dist={d['distance']}")
