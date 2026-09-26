"""Koşu durumu + deney kuyruğu.   python -X utf8 -m dizgetts.eval.status"""
import csv, glob, os, time

import yaml

from dizgetts import paths

HERE = os.path.join(os.path.dirname(__file__), "..")
print("== Eğitim koşuları (D:/dizgetts/runs, yalnız v2)")
for run in sorted(glob.glob(f"{paths.RUNS}/v2_*")):
    m = os.path.join(run, "metrics.csv")
    rows = list(csv.DictReader(open(m, encoding="utf8"))) if os.path.exists(m) else []
    if not rows:
        continue
    cfg = open(os.path.join(run, "config.resolved.yaml"), encoding="utf8").read()
    total = int(next(l.split(":")[1] for l in cfg.splitlines() if l.strip().startswith("epochs:")))
    last, vals = rows[-1], [r for r in rows if r["val_total"]]
    ep = int(last["epoch"])
    secs = sum(float(r["sec_epoch"]) for r in rows[-10:]) / len(rows[-10:]) + 1
    age = time.time() - os.path.getmtime(m)
    state = "ÇALIŞIYOR" if age < 240 else ("BİTTİ" if ep >= total else f"DURDU ({age/60:.0f} dk önce)")
    print(f"{os.path.basename(run)}: epoch {ep}/{total} [{state}] train {float(last['train_total']):.3f}"
          + (f" | val {float(vals[-1]['val_total']):.3f} (en iyi {min(float(r['val_total']) for r in vals):.3f})" if vals else "")
          + (f" | ~{(total - ep) * secs / 3600:.1f} sa kaldı" if state == "ÇALIŞIYOR" else ""))
print("\n== Deney kaydı (experiments.yaml, v2)")
for e in yaml.safe_load(open(os.path.join(HERE, "experiments.yaml"), encoding="utf8"))["experiments"]:
    if e["id"].startswith("v2"):
        print(f"  {e['id']:12s} {e['status']}" + (f"  (bekliyor: {e['blocked_on']})" if e.get("blocked_on") else "") + (f"  (sıra: {e['depends_on']} sonrası)" if e.get("depends_on") else ""))
