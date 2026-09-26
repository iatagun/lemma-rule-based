"""Matcha-TTS eğitimi (baseline_espeak.yaml / main_dizge.yaml). fp32, GTX 1650 4 GB.

  cd <repo>; D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/train/train.py --config dizgetts/configs/baseline_espeak.yaml [--pilot] [--resume RUN/last.pt]

--pilot: 60 adım + 1 doğrulama, süre/bellek ölçer, checkpoint yazmaz (epoch süresi tahmini için).
Her koşu <run_root>/<name>_<zaman>/ altında: config.resolved.yaml, env.json (git commit, sürümler, tohum), symbols.json, metrics.csv,
tensorboard/, last.pt, best.pt (val kaybı), ep{N}.pt (save_every). Val kaybı zayıf vekildir (rastgele-t CFM); nihai seçim Aşama 6 ASR metriğiyle.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 4 GB kartta parçalanma -> paylaşımlı belleğe taşma (ölçüldü: reserved 5,5 GiB)
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # repo kökü
from matcha.models.matcha_tts import MatchaTTS  # noqa: E402
from dizgetts.train.dpfeat import set_dp_feat  # noqa: E402
from dizgetts.train.data import BucketBatches, TTSDataset, collate, ensure_mels, frontend_table, row_tokens  # noqa: E402


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(open(path, encoding="utf8"))
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = deep_merge(base, cfg)
    return cfg


def git_info() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain", "--", "dizgetts")))


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_model(cfg: dict, n_vocab: int, stats: dict) -> MatchaTTS:
    m = cfg["model"]
    model = MatchaTTS(n_vocab=n_vocab, n_spks=m["n_spks"], spk_emb_dim=m["spk_emb_dim"], n_feats=m["n_feats"],
                      encoder=OmegaConf.create(m["encoder"]), decoder=OmegaConf.create(m["decoder"]), cfm=OmegaConf.create(m["cfm"]),
                      data_statistics=dict(mel_mean=stats["mel_mean"], mel_std=stats["mel_std"]), out_size=m["out_size"],
                      prior_loss=m["prior_loss"])
    if m.get("dp_feat"):  # v4: süre tahmincisine sınır özniteliği (train/dpfeat.py)
        from dizgetts.train import dpfeat
        dpfeat.enable(model)
    return model


def load_pretrained(model: MatchaTTS, init: dict) -> dict:
    sd = torch.load(init["ckpt"], map_location="cpu", weights_only=False)["state_dict"]
    own = model.state_dict()
    keep = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in init["skip_prefixes"]) and k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return dict(loaded=len(keep), skipped_prefix=sorted(k for k in sd if any(k.startswith(p) for p in init["skip_prefixes"])),
                not_loaded=sorted(k for k in own if k not in keep))


@torch.no_grad()
def evaluate(model, loader, repeats: int, seed: int, dev) -> dict:
    model.eval()
    tot = dict(dur_loss=0.0, prior_loss=0.0, diff_loss=0.0)
    n = 0
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        for rep in range(repeats):
            torch.manual_seed(seed + rep)
            for b in loader:
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                set_dp_feat(model, b.get("dp_feat"))
                l = model.get_losses(b)
                for k in tot:
                    tot[k] += float(l[k])
                n += 1
    model.train()
    out = {k: v / n for k, v in tot.items()}
    out["total"] = sum(out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dcfg = yaml.safe_load(open(os.path.join(ROOT, cfg["data_config"]), encoding="utf8"))
    root, au = dcfg["out_root"], dcfg["audio"]
    stats = json.load(open(os.path.join(root, "stats.json"), encoding="utf8"))
    tr = cfg["train"]
    seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    assert not tr["amp"], "amp bu donanımda KAPALI olmalı (fp16 cuDNN NaN, reports/stage1_audit.md)"

    symbols, _ = frontend_table(cfg["frontend"])
    ds = {s: TTSDataset(root, s, cfg["frontend"], stats, cfg["espeak_strip_stress"], cfg.get("manifest", "_phon"), bool(cfg["model"].get("dp_feat")))
          for s in ("train", "val")}
    for s in ds:
        wrote = ensure_mels(root, ds[s].rows, au)
        if wrote:
            print(f"[{s}] {wrote} mel hesaplandı", flush=True)
    sampler = BucketBatches(ds["train"].mel_lengths(), tr["batch_size"], tr["bucket_size"], cfg["seed"])
    train_dl = DataLoader(ds["train"], batch_sampler=sampler, collate_fn=collate, num_workers=tr["num_workers"], pin_memory=True)
    val_dl = DataLoader(ds["val"], batch_size=tr["batch_size"], shuffle=False, collate_fn=collate, num_workers=0)

    model = build_model(cfg, len(symbols), stats)
    init_report = load_pretrained(model, cfg["init"]) if cfg.get("init") and not a.resume else None
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    epoch0 = step0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); epoch0, step0 = ck["epoch"], ck.get("step", 0)  # adım sayacı kaldığı yerden (TensorBoard/max_steps)
    n_params = sum(p.numel() for p in model.parameters())

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = os.path.join(cfg["run_root"], f"{cfg['name']}{'_pilot' if a.pilot else ''}_{stamp}")
    os.makedirs(run, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(run, "config.resolved.yaml"), "w", encoding="utf8"), allow_unicode=True, sort_keys=False)
    json.dump(dict(git=git_info(), seed=cfg["seed"], torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                   python=platform.python_version(), n_params=n_params, n_vocab=len(symbols), train_clips=len(ds["train"]),
                   val_clips=len(ds["val"]), init=init_report, engine_versions=ds["train"].rows[0].get("engine_versions"), manifest=cfg.get("manifest", "_phon"), cmd=sys.argv, resumed_from=a.resume, epoch0=epoch0, step0=step0),
              open(os.path.join(run, "env.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
    json.dump(symbols, open(os.path.join(run, "symbols.json"), "w", encoding="utf8"), ensure_ascii=False)
    tb = SummaryWriter(os.path.join(run, "tensorboard"))
    csv_f = open(os.path.join(run, "metrics.csv"), "a", newline="", encoding="utf8")
    cw = csv.writer(csv_f)
    cw.writerow(["epoch", "step", "train_dur", "train_prior", "train_diff", "train_total", "val_dur", "val_prior", "val_diff", "val_total",
                 "sec_epoch", "peak_gib"])
    print(f"koşu: {run}\nparametre: {n_params/1e6:.2f} M | vocab {len(symbols)} | train {len(ds['train'])} val {len(ds['val'])} | init: "
          f"{init_report and init_report['loaded']} tensör yüklendi", flush=True)

    max_steps = 30 if a.pilot else (a.max_steps or tr["max_steps"])
    step, best = step0, float("inf")
    save = lambda name, ep: torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep, cfg=cfg, step=step, symbols=symbols), os.path.join(run, name))
    for epoch in range(epoch0 + 1, tr["epochs"] + 1):
        sampler.epoch = epoch
        torch.cuda.reset_peak_memory_stats()
        t0, acc, nb = time.time(), dict(dur_loss=0.0, prior_loss=0.0, diff_loss=0.0), 0
        opt.zero_grad(set_to_none=True)
        pend = 0
        for b in train_dl:
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            set_dp_feat(model, b.get("dp_feat"))
            l = model.get_losses(b)
            loss = sum(l.values())
            (loss / tr["grad_accum"]).backward()
            pend += 1
            nb += 1
            for k in acc:
                acc[k] += float(l[k])
            if pend < tr["grad_accum"]:
                continue
            pend = 0
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            if torch.isfinite(gn):
                opt.step()
                step += 1
            else:
                print(f"UYARI: sonlu olmayan grad (epoch {epoch}, adım {step}) -> güncelleme atlandı", flush=True)
            opt.zero_grad(set_to_none=True)
            if step % tr["log_every"] == 0:
                tb.add_scalar("train/total", float(loss), step); tb.add_scalar("train/grad_norm", float(gn), step)
            if max_steps and step >= max_steps:
                break
        if pend:  # epoch sonunda kalan yarım grup
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            if torch.isfinite(gn):
                opt.step(); step += 1
            opt.zero_grad(set_to_none=True)
        sec = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        trn = {k: v / max(1, nb) for k, v in acc.items()}
        row = [epoch, step, trn["dur_loss"], trn["prior_loss"], trn["diff_loss"], sum(trn.values())]
        val = None
        if a.pilot or epoch % tr["eval_every"] == 0 or epoch == tr["epochs"] or (max_steps and step >= max_steps):
            tv = time.time()
            val = evaluate(model, val_dl, tr["val_repeats"], cfg["seed"] + 1000, dev)
            for k, v in val.items():
                tb.add_scalar(f"val/{k}", v, epoch)
            sec_val = time.time() - tv
        row += [val and val["dur_loss"], val and val["prior_loss"], val and val["diff_loss"], val and val["total"], round(sec, 1), round(peak, 2)]
        cw.writerow(row); csv_f.flush()
        tb.add_scalar("train/epoch_total", row[5], epoch)
        print(f"ep {epoch:4d} step {step:6d} train {row[5]:.3f} (dur {trn['dur_loss']:.3f} prior {trn['prior_loss']:.3f} diff {trn['diff_loss']:.3f})"
              + (f" | val {val['total']:.3f} ({sec_val:.0f}s)" if val else "") + f" | {sec:.0f}s/epoch, {nb} adım, peak {peak:.2f} GiB", flush=True)
        if a.pilot:
            print(f"PİLOT: {sec/max(1,nb):.2f} s/mini-batch; tam epoch = {len(sampler)} mini-batch ≈ {len(sampler)*sec/max(1,nb):.0f} s (+val {sec_val:.0f} s / {tr['eval_every']} epoch)")
            break
        save("last.pt", epoch)
        if val and val["total"] < best:
            best = val["total"]; save("best.pt", epoch)
        if epoch % tr["save_every"] == 0:
            save(f"ep{epoch}.pt", epoch)
        if max_steps and step >= max_steps:
            break


if __name__ == "__main__":
    main()
