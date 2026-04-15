"""Why do comma coordinations miss?"""
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

comma_misses = []

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
        head_id = g["head"]
        zone = range(min(head_id, k+1), max(head_id, k+1))
        has_cconj = any(gold_tokens[z]["upos"] == "CCONJ" for z in zone if z < n)
        if has_cconj:
            continue
        # Check if preceding token has comma
        has_comma_before = False
        if k > 0 and k - 1 < len(pred_tokens):
            has_comma_before = pred_tokens[k - 1].has_comma_after
        if not has_comma_before:
            continue
        
        gold_head = next((t for t in gold_tokens if t["id"] == g["head"]), None)
        head_upos = gold_head["upos"] if gold_head else "?"
        comma_misses.append({
            "form": p.form,
            "upos": p.upos,
            "gold_upos": g["upos"],
            "pred_deprel": p.deprel,
            "pred_assigned": p.is_assigned,
            "gold_head_upos": head_upos,
            "gold_head_deprel": gold_head["deprel"] if gold_head else "?",
        })

print(f"Comma-based conj misses: {len(comma_misses)}")

print("\nBy pred deprel:")
for dr, c in Counter(d["pred_deprel"] for d in comma_misses).most_common():
    print(f"  {c:3d}  {dr}")

print("\nUPOS match vs mismatch:")
match = sum(1 for d in comma_misses if d["upos"] == d["gold_head_upos"])
mismatch = len(comma_misses) - match
print(f"  match={match}  mismatch={mismatch}")

print("\nBy (pred_upos, gold_head_upos) where mismatch:")
for pair, c in Counter((d["upos"], d["gold_head_upos"]) for d in comma_misses if d["upos"] != d["gold_head_upos"]).most_common():
    print(f"  {c:3d}  conj={pair[0]:6s} head={pair[1]:6s}")

print("\nBy is_assigned:")
print(f"  assigned={sum(1 for d in comma_misses if d['pred_assigned'])}")
print(f"  not_assigned={sum(1 for d in comma_misses if not d['pred_assigned'])}")
