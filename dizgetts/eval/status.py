"""Koşu durumu: her run için son epoch, kayıplar, kalan süre tahmini.   python dizgetts/eval/status.py"""
import csv, glob, os, time

for run in sorted(glob.glob("D:/dizgetts/runs/*_e*_2*")):
    m = os.path.join(run, "metrics.csv")
    rows = list(csv.DictReader(open(m, encoding="utf8"))) if os.path.exists(m) else []
    if not rows:
        continue
    total = int(next((l.split(":")[1] for l in open(os.path.join(run, "config.resolved.yaml"), encoding="utf8") if l.strip().startswith("epochs:")), 0))
    last, vals = rows[-1], [r for r in rows if r["val_total"]]
    ep = int(last["epoch"])
    secs = sum(float(r["sec_epoch"]) for r in rows[-10:]) / len(rows[-10:]) + 1
    age = time.time() - os.path.getmtime(m)
    state = "ÇALIŞIYOR" if age < 180 else ("BİTTİ" if ep >= total else f"DURDU ({age/60:.0f} dk önce)")
    print(f"{os.path.basename(run)}: epoch {ep}/{total} [{state}] train {float(last['train_total']):.3f}"
          + (f" | val {float(vals[-1]['val_total']):.3f} (en iyi {min(float(r['val_total']) for r in vals):.3f})" if vals else "")
          + f" | ~{(total - ep) * secs / 3600:.1f} sa kaldı" + f" | ckpt: {', '.join(sorted(os.path.basename(p) for p in glob.glob(run + '/ep*.pt')))}")
