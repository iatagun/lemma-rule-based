"""Detailed coordination analysis: what does gold conj look like?"""
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

# Analyze all gold conj tokens
conj_hit = []
conj_miss = []

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
        # Gold head info
        gold_head = next((t for t in gold_tokens if t["id"] == g["head"]), None)
        gold_head_upos = gold_head["upos"] if gold_head else "?"
        gold_head_deprel = gold_head["deprel"] if gold_head else "?"
        rec = {
            "form": p.form,
            "upos": p.upos,
            "gold_upos": g["upos"],
            "pred_deprel": p.deprel,
            "gold_head_upos": gold_head_upos,
            "gold_head_deprel": gold_head_deprel,
            "has_comma": getattr(p, "has_comma_after", False),
        }
        if p.deprel == "conj":
            conj_hit.append(rec)
        else:
            conj_miss.append(rec)

print(f"conj hit: {len(conj_hit)}, miss: {len(conj_miss)}")

print("\n=== MISSED conj ===")
print("By pred deprel:")
for dr, c in Counter(d["pred_deprel"] for d in conj_miss).most_common():
    print(f"  {c:3d}  {dr}")

print("\nBy gold UPOS of conjunct:")
for u, c in Counter(d["gold_upos"] for d in conj_miss).most_common():
    print(f"  {c:3d}  {u}")

print("\nBy gold head's deprel:")
for dr, c in Counter(d["gold_head_deprel"] for d in conj_miss).most_common():
    print(f"  {c:3d}  {dr}")

print("\nBy gold head's UPOS:")
for u, c in Counter(d["gold_head_upos"] for d in conj_miss).most_common():
    print(f"  {c:3d}  {u}")

# How many gold conj have left comma?
# Need to check: does the conjunct's LEFT sibling have comma_after?
print("\n=== HIT conj ===")
print("By gold head deprel (what do we correctly coordinate?):")
for dr, c in Counter(d["gold_head_deprel"] for d in conj_hit).most_common():
    print(f"  {c:3d}  {dr}")
