"""Deeper coordination analysis: comma + CCONJ patterns."""
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

# For each gold conj token, check:
# 1. Is there a CCONJ between it and its head?
# 2. Is there a comma before it?
# 3. What's the distance from head?
conj_types = Counter()  # "cconj", "comma", "neither"
miss_types = Counter()

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
        if g["deprel"] != "conj":
            continue
        
        # Check for CCONJ between head and this token
        head_id = g["head"]
        if head_id > k + 1:  # head is to the RIGHT of conj
            zone = range(k + 1, min(head_id, n))
        else:  # head is to the LEFT of conj (typical)
            zone = range(head_id, k)
        
        has_cconj = any(gold_tokens[z]["upos"] == "CCONJ" for z in zone if z < n)
        
        # Check comma before this token (on the preceding token)
        has_comma_before = False
        if k > 0:
            prev_p = pred_tokens[k - 1] if k - 1 < len(pred_tokens) else None
            if prev_p and getattr(prev_p, "has_comma_after", False):
                has_comma_before = True
        
        ctype = "cconj" if has_cconj else ("comma" if has_comma_before else "neither")
        conj_types[ctype] += 1
        
        if p.deprel != "conj":
            miss_types[ctype] += 1

print("Gold conj by connection type:")
for ct in ["cconj", "comma", "neither"]:
    total = conj_types[ct]
    missed = miss_types.get(ct, 0)
    hit = total - missed
    print(f"  {ct:10s}: total={total:3d} hit={hit:3d} miss={missed:3d} recall={hit/total*100:.1f}%")

print(f"\nTotal: {sum(conj_types.values())}")
print(f"  Neither category = {conj_types['neither']} ({conj_types['neither']/sum(conj_types.values())*100:.1f}%)")
