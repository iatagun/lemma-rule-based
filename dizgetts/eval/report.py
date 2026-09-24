"""experiments.yaml + eval_out/<etiket>/results.json -> reports/EXPERIMENTS.md   (python -m dizgetts.eval.report)"""
import json, os

import yaml

HERE = os.path.join(os.path.dirname(__file__), "..")
exps = yaml.safe_load(open(os.path.join(HERE, "experiments.yaml"), encoding="utf8"))["experiments"]
lines = ["# Deneyler (otomatik üretildi: `python -m dizgetts.eval.report`; kaynak `experiments.yaml`)", "",
         "| id | durum | hipotez | CER % | WER % | UTMOS | not |", "|---|---|---|---|---|---|---|"]
for e in exps:
    cer = wer = mos = ""
    lab = e.get("eval_label")
    p = f"D:/dizgetts/eval_out/{lab}/results.json" if lab and lab != "TBD" else None
    if p and os.path.exists(p):
        s = json.load(open(p, encoding="utf8"))["summary"]
        cer, wer, mos = f"{s['cer']*100:.1f}", f"{s['wer']*100:.1f}", f"{s['utmos']:.2f}"
    lines.append(f"| {e['id']} | {e['status']} | {e.get('hypothesis','')} | {cer} | {wer} | {mos} | {e.get('note', e.get('decision_rule',''))} |")
out = os.path.join(HERE, "reports", "EXPERIMENTS.md")
open(out, "w", encoding="utf8").write("\n".join(lines) + "\n")
print("\n".join(lines))
