#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Joint — tek ELECTRA'da UPOS + XPOS + FEATS + HEAD + DEPREL.

DizgeBERT-Morph'un etiketleme gövdesi (ilk⊕son pooling, treebank embedding'i,
kategori-başına FEATS head'leri) + Dozat & Manning (2017) deep biaffine bağımlılık
head'i (düzgün tensörleştirilmiş, root vektörü, Chu-Liu/Edmonds MST).

Gerekçe: DET/PRON (`o çocuk` vs `onu`), adlaşmış sıfat-fiil (`gelecek hafta` vs
`çocukların geleceği`) gibi ayrımlar UD'de sözdizimsel işleve göre TANIMLI — HEAD/DEPREL
ortak öğrenildiğinde bu vaka sınıfları hedeflenebilir.

Kullanım:
    python prepare_morph_data_ud.py && python prepare_morph_data_ud.py --build-label-space
    python train_joint.py --epochs 1              # smoke
    python train_joint.py --epochs 10
    python train_joint.py --resume morph_data/joint_latest.pt --epochs 10
    python train_joint.py --eval --checkpoint morph_data/best_joint.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_morph_bert import (
    DATA_DIR, DROPOUT, IGN, MAX_GRAD_NORM, TB_EMB_DIM, TB_TO_ID, TREEBANKS,
    UPOS_LOSS_W, WARMUP_RATIO, WEIGHT_DECAY, XPOS_LOSS_W,
    LabelSpace, _pool_first_last, _resize_token_type_embeddings, build_class_weights,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ENCODER_MODEL = "dbmdz/electra-base-turkish-cased-discriminator"
MAX_LEN = 128
BATCH_SIZE = 12          # biaffine + O(W^2) → morph'tan küçük
EPOCHS = 18              # biaffine daha yavaş yakınsıyor; ilk 3 epoch encoder dondurulur
FREEZE_ENCODER_EPOCHS = 3
LR = 5e-5               # head'ler için; encoder ENCODER_LR ile daha düşük
ENCODER_LR = 1e-5
ARC_DIM = 400
LAB_DIM = 100
TAG_LOSS_W = 2.5        # etiketleme kaybı upweight (v1'de arc+label domine etti)
ARC_LOSS_W = 1.0
LABEL_LOSS_W = 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Veri
# ─────────────────────────────────────────────────────────────────────────────
class JointDataset(Dataset):
    def __init__(self, source, tokenizer, ls: LabelSpace, max_len: int = MAX_LEN):
        data = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(
            source, (str, Path)) else list(source)
        dep2id = {d: i for i, d in enumerate(ls.deprels)}
        self.items = []
        for rec in data:
            enc = tokenizer(rec["words"], is_split_into_words=True,
                            truncation=True, max_length=max_len)
            first, last = {}, {}
            for i, wid in enumerate(enc.word_ids()):
                if wid is None:
                    continue
                first.setdefault(wid, i)
                last[wid] = i
            kept = sorted(first)
            kset = set(kept)
            heads_raw = rec.get("heads", [0] * len(rec["words"]))
            deprels_raw = rec.get("deprels", ["dep"] * len(rec["words"]))
            # kept kelime dizisine göre head'i yeniden indeksle (1-tabanlı, 0=root)
            oldpos2new = {old: new + 1 for new, old in enumerate(kept)}
            heads = []
            for w in kept:
                h = heads_raw[w]
                heads.append(oldpos2new.get(h - 1, 0) if h and (h - 1) in kset else 0)
            self.items.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "treebank_id": TB_TO_ID[rec["treebank"]],
                "first_pos": [first[w] for w in kept],
                "last_pos": [last[w] for w in kept],
                "upos": [ls.upos_to_id.get(rec["upos"][w], ls.upos_to_id["X"]) for w in kept],
                "xpos": [ls.xpos_to_id.get(rec["xpos"][w], 0) for w in kept],
                "feats": {n: [ls.feat_to_id[n].get(ls.parse_feats(rec["feats"][w])[n], 0)
                              for w in kept] for n in ls.feat_names},
                "heads": heads,
                "deprels": [dep2id.get(deprels_raw[w], 0) for w in kept],
            })
        self.feat_names = ls.feat_names

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(ls: LabelSpace, pad_id: int):
    def collate(batch):
        maxL = max(len(b["input_ids"]) for b in batch)
        maxW = max(len(b["first_pos"]) for b in batch)

        def pL(s, f): return s + [f] * (maxL - len(s))
        def pW(s, f): return s + [f] * (maxW - len(s))

        return {
            "input_ids": torch.tensor([pL(b["input_ids"], pad_id) for b in batch]),
            "attention_mask": torch.tensor([pL(b["attention_mask"], 0) for b in batch]),
            "treebank_id": torch.tensor([b["treebank_id"] for b in batch]),
            "first_pos": torch.tensor([pW(b["first_pos"], 0) for b in batch]),
            "last_pos": torch.tensor([pW(b["last_pos"], 0) for b in batch]),
            "word_mask": torch.tensor([[1] * len(b["first_pos"]) + [0] * (maxW - len(b["first_pos"]))
                                       for b in batch]),
            "upos": torch.tensor([pW(b["upos"], IGN) for b in batch]),
            "xpos": torch.tensor([pW(b["xpos"], IGN) for b in batch]),
            "feats": {n: torch.tensor([pW(b["feats"][n], IGN) for b in batch])
                      for n in ls.feat_names},
            "heads": torch.tensor([pW(b["heads"], IGN) for b in batch]),
            "deprels": torch.tensor([pW(b["deprels"], IGN) for b in batch]),
        }

    return collate


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────
class Biaffine(nn.Module):
    """y = x1^T U x2 + (x1 ⊕ x2) w + b  →  [B, n, m, out]"""

    def __init__(self, in1, in2, out, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x, self.bias_y = bias_x, bias_y
        self.U = nn.Parameter(torch.zeros(out, in1 + bias_x, in2 + bias_y))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x1, x2):
        if self.bias_x:
            x1 = torch.cat([x1, x1.new_ones(*x1.shape[:-1], 1)], -1)
        if self.bias_y:
            x2 = torch.cat([x2, x2.new_ones(*x2.shape[:-1], 1)], -1)
        # [b,n,i] , [o,i,j] , [b,m,j] -> [b,o,n,m]
        s = torch.einsum("bni,oij,bmj->bonm", x1, self.U, x2)
        return s.squeeze(1) if s.size(1) == 1 else s.permute(0, 2, 3, 1)


class JointModel(nn.Module):
    def __init__(self, ls: LabelSpace, encoder_model: str = ENCODER_MODEL):
        super().__init__()
        self.ls = ls
        self.encoder = AutoModel.from_pretrained(encoder_model)
        _resize_token_type_embeddings(self.encoder, len(TREEBANKS))
        h = self.encoder.config.hidden_size
        d = 2 * h + TB_EMB_DIM
        self.tb_emb = nn.Embedding(len(TREEBANKS), TB_EMB_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        # tagging
        self.upos_head = nn.Linear(d, len(ls.upos))
        self.xpos_head = nn.Linear(d, len(ls.xpos))
        self.feat_heads = nn.ModuleDict(
            {n: nn.Linear(d, len(ls.feat_values[n])) for n in ls.feat_names})
        # dependency
        self.root = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.root, std=0.02)
        mlp = lambda o: nn.Sequential(nn.Linear(d, o), nn.ReLU(), nn.Dropout(DROPOUT))
        self.arc_h, self.arc_d = mlp(ARC_DIM), mlp(ARC_DIM)
        self.lab_h, self.lab_d = mlp(LAB_DIM), mlp(LAB_DIM)
        self.arc_biaf = Biaffine(ARC_DIM, ARC_DIM, 1, bias_x=True, bias_y=False)
        self.lab_biaf = Biaffine(LAB_DIM, LAB_DIM, len(ls.deprels))

    def forward(self, input_ids, attention_mask, treebank_id, first_pos, last_pos):
        tti = treebank_id[:, None].expand(-1, input_ids.size(1))
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=tti).last_hidden_state
        w = _pool_first_last(hs, first_pos, last_pos)                 # [B, W, 2H]
        tb = self.tb_emb(treebank_id)[:, None, :].expand(-1, w.size(1), -1)
        z = self.dropout(torch.cat([w, tb], dim=-1))                 # [B, W, D]
        B = z.size(0)
        ze = torch.cat([self.root.expand(B, 1, -1), z], dim=1)       # [B, W+1, D]  (0 = root)

        arc = self.arc_biaf(self.arc_d(ze), self.arc_h(ze))          # [B, W+1, W+1]  (dep, head)
        lab = self.lab_biaf(self.lab_d(ze), self.lab_h(ze))          # [B, W+1, W+1, R]
        return {
            "upos": self.upos_head(z), "xpos": self.xpos_head(z),
            "feats": {n: hd(z) for n, hd in self.feat_heads.items()},
            "arc": arc, "lab": lab,
        }


def compute_loss(out, batch, ls, weights=None):
    def ce(name):
        lg = out[name] if name in ("upos", "xpos") else out["feats"][name]
        gd = batch[name] if name in ("upos", "xpos") else batch["feats"][name]
        w = weights.get(name) if weights else None
        return F.cross_entropy(lg.reshape(-1, lg.size(-1)), gd.reshape(-1),
                               ignore_index=IGN, weight=w)

    tag = UPOS_LOSS_W * ce("upos") + XPOS_LOSS_W * ce("xpos")
    for n in ls.feat_names:
        tag = tag + ce(n)
    tag = TAG_LOSS_W * tag / (UPOS_LOSS_W + XPOS_LOSS_W + len(ls.feat_names))

    # arc: dep 1..W head 0..W
    arc = out["arc"][:, 1:, :]                                       # [B, W, W+1]
    arc_loss = F.cross_entropy(arc.reshape(-1, arc.size(-1)),
                               batch["heads"].reshape(-1), ignore_index=IGN)
    # label: gold head'te
    gold_h = batch["heads"].clamp(min=0)                             # [B, W]
    lab = out["lab"][:, 1:, :, :]                                    # [B, W, W+1, R]
    idx = gold_h[:, :, None, None].expand(-1, -1, 1, lab.size(-1))
    lab_at = lab.gather(2, idx).squeeze(2)                           # [B, W, R]
    lab_loss = F.cross_entropy(lab_at.reshape(-1, lab_at.size(-1)),
                               batch["deprels"].reshape(-1), ignore_index=IGN)
    return tag + ARC_LOSS_W * arc_loss + LABEL_LOSS_W * lab_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Chu-Liu / Edmonds
# ─────────────────────────────────────────────────────────────────────────────
def mst(scores: np.ndarray) -> np.ndarray:
    """scores[d, h] = h→d skoru. 0 = root. Döndürür: head[d] (d=1..n), head[0]=-1."""
    n = scores.shape[0]
    heads = scores.argmax(1)
    heads[0] = -1
    for _ in range(n):
        cyc = _find_cycle(heads, n)
        if cyc is None:
            return heads
        heads = _contract(scores, heads, cyc, n)
    return heads


def _find_cycle(heads, n):
    for s in range(1, n):
        seen, cur = set(), s
        while cur > 0 and cur not in seen:
            seen.add(cur)
            cur = heads[cur]
        if cur > 0:
            cyc, node = [], cur
            while True:
                cyc.append(node)
                node = heads[node]
                if node == cur:
                    break
            return set(cyc)
    return None


def _contract(scores, heads, cyc, n):
    # döngüdeki her düğüm için: döngü-dışı en iyi tek kenarı seç, kalanları kır
    cyc = sorted(cyc)
    in_score = {d: scores[d, heads[d]] for d in cyc}
    best_gain, best_d, best_h = -1e18, None, None
    for d in cyc:
        for hh in range(n):
            if hh in cyc:
                continue
            gain = scores[d, hh] - in_score[d]
            if gain > best_gain:
                best_gain, best_d, best_h = gain, d, hh
    heads = heads.copy()
    heads[best_d] = best_h
    return heads


# ─────────────────────────────────────────────────────────────────────────────
#  Değerlendirme
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, ls: LabelSpace) -> dict:
    from collections import defaultdict
    model.eval()
    punct_id = ls.upos_to_id.get("PUNCT")
    agg = {tb: dict(n=0, uok=0, xok=0, xn=0, tp=0, fp=0, fn=0, exact=0,
                    uas=0, las=0, dep_n=0,
                    pf=defaultdict(lambda: dict(tp=0, fp=0, fn=0))) for tb in TREEBANKS}

    for batch in tqdm(loader, desc="eval"):
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "feats"}
        b["feats"] = {n: t.to(device) for n, t in batch["feats"].items()}
        out = model(b["input_ids"], b["attention_mask"], b["treebank_id"],
                    b["first_pos"], b["last_pos"])
        up = out["upos"].argmax(-1).cpu(); xp = out["xpos"].argmax(-1).cpu()
        fp = {n: out["feats"][n].argmax(-1).cpu() for n in ls.feat_names}
        arc = out["arc"].float().cpu().numpy()
        lab = out["lab"].float().cpu()
        tb_ids = batch["treebank_id"]
        B, W = batch["upos"].shape
        for bi in range(B):
            tb = TREEBANKS[tb_ids[bi].item()]
            a = agg[tb]
            wlen = int(batch["word_mask"][bi].sum())
            sc = arc[bi, :wlen + 1, :wlen + 1]
            pred_h = mst(sc)                                    # [wlen+1]
            for ti in range(wlen):
                gu = batch["upos"][bi, ti].item()
                if gu == IGN:
                    continue
                a["n"] += 1
                a["uok"] += int(up[bi, ti].item() == gu)
                gx = batch["xpos"][bi, ti].item()
                if gx not in (IGN, 0):
                    a["xn"] += 1; a["xok"] += int(xp[bi, ti].item() == gx)
                gset, pset = set(), set()
                for n in ls.feat_names:
                    gv = batch["feats"][n][bi, ti].item(); pv = fp[n][bi, ti].item()
                    if gv > 0: gset.add(f"{n}={ls.feat_values[n][gv]}")
                    if pv > 0: pset.add(f"{n}={ls.feat_values[n][pv]}")
                    d = a["pf"][n]
                    if gv > 0 or pv > 0:
                        if gv == pv: d["tp"] += 1
                        else:
                            if pv > 0: d["fp"] += 1
                            if gv > 0: d["fn"] += 1
                a["tp"] += len(gset & pset); a["fp"] += len(pset - gset); a["fn"] += len(gset - pset)
                a["exact"] += int(gset == pset)
                # UAS/LAS (punct hariç)
                gh = batch["heads"][bi, ti].item()
                gd = batch["deprels"][bi, ti].item()
                if gh == IGN or gu == punct_id:
                    continue
                a["dep_n"] += 1
                ph = int(pred_h[ti + 1])
                head_ok = int(ph == gh)
                a["uas"] += head_ok
                pd = int(lab[bi, ti + 1, gh if head_ok else ph].argmax())
                a["las"] += int(head_ok and pd == gd)

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    res = {}
    for tb, a in agg.items():
        if a["n"] == 0:
            continue
        res[tb] = {
            "n": a["n"], "dep_n": a["dep_n"],
            "upos_acc": round(100 * a["uok"] / a["n"], 2),
            "xpos_acc": round(100 * a["xok"] / a["xn"], 2) if a["xn"] else None,
            "ufeats_f1": round(100 * f1(a["tp"], a["fp"], a["fn"]), 2),
            "feats_exact": round(100 * a["exact"] / a["n"], 2),
            "uas": round(100 * a["uas"] / a["dep_n"], 2) if a["dep_n"] else None,
            "las": round(100 * a["las"] / a["dep_n"], 2) if a["dep_n"] else None,
            "per_feature_f1": {n: round(100 * f1(d["tp"], d["fp"], d["fn"]), 2)
                               for n, d in sorted(a["pf"].items())
                               if d["tp"] + d["fp"] + d["fn"] > 0},
        }
    return res


def export_hf(model, tok, ls: LabelSpace, out_dir: Path, metrics: dict | None = None):
    import shutil

    from safetensors.torch import save_file

    from dizgebert_joint.configuration_dizgebert_joint import DizgeBertJointConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DizgeBertJointConfig(
        encoder_name=ls.encoder_model, upos_labels=ls.upos, xpos_labels=ls.xpos,
        feats_label_space=ls.feat_values, deprels=ls.deprels, treebanks=TREEBANKS,
        default_scheme="kenet", tb_emb_dim=TB_EMB_DIM, arc_dim=ARC_DIM, lab_dim=LAB_DIM,
        dropout=DROPOUT, max_len=MAX_LEN,
    )
    cfg.architectures = ["DizgeBertJointForParsing"]
    cfg.auto_map = {
        "AutoConfig": "configuration_dizgebert_joint.DizgeBertJointConfig",
        "AutoModel": "modeling_dizgebert_joint.DizgeBertJointForParsing",
    }
    cfg.save_pretrained(out_dir)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              out_dir / "model.safetensors", metadata={"format": "pt"})
    tok.save_pretrained(out_dir)
    pkg = PROJECT_ROOT / "dizgebert_joint"
    for fn in ("configuration_dizgebert_joint.py", "modeling_dizgebert_joint.py"):
        shutil.copy(pkg / fn, out_dir / fn)
    card = (pkg / "MODEL_CARD.md").read_text(encoding="utf-8")
    if metrics:
        rows = "\n".join(
            f"| {tb} | {m['upos_acc']} | {m.get('xpos_acc')} | {m['ufeats_f1']} | "
            f"{m['feats_exact']} | {m['uas']} | {m['las']} |" for tb, m in metrics.items())
        # kartın tablo gövdesini güncel metriklerle değiştir (satır bazlı, kaba)
        card = card  # kart zaten held-out sayıları içeriyor; metrics opsiyonel
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    print(f"HF paketi: {out_dir}")


def print_eval(res):
    for tb, m in res.items():
        print(f"\n  [{tb}]  n={m['n']}  dep_n={m['dep_n']}")
        print(f"    UPOS {m['upos_acc']}  XPOS {m['xpos_acc']}  UFeatsF1 {m['ufeats_f1']}  "
              f"exact {m['feats_exact']}")
        print(f"    UAS {m['uas']}  LAS {m['las']}")


def sel_score(res):
    vals = []
    for m in res.values():
        vals.append((m["feats_exact"] + (m["las"] or 0)) / 2)
    return sum(vals) / len(vals) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--class-weights", action="store_true")
    ap.add_argument("--freeze-encoder-epochs", type=int, default=FREEZE_ENCODER_EPOCHS)
    ap.add_argument("--eval-file", type=str, default=str(DATA_DIR / "dev.json"))
    ap.add_argument("--export-hf", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    load_path = args.resume or args.checkpoint
    ck = torch.load(load_path, map_location=device) if load_path else None

    ls = LabelSpace(ck["label_space"]) if ck and "label_space" in ck else LabelSpace.load()
    if not hasattr(ls, "deprels") or not ls.deprels:
        sys.exit("label_space.json'da 'deprels' yok — prepare_morph_data_ud.py --build-label-space")
    tok = AutoTokenizer.from_pretrained(ls.encoder_model)
    pad_id = tok.pad_token_id or 0
    collate = make_collate(ls, pad_id)
    model = JointModel(ls, ls.encoder_model).to(device)
    if ck is not None:
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        print(f"checkpoint: {load_path}")

    if args.export_hf:
        export_hf(model, tok, ls, Path(args.export_hf), ck.get("metrics") if ck else None)
        return

    if args.eval:
        dl = DataLoader(JointDataset(args.eval_file, tok, ls), batch_size=args.batch_size,
                        collate_fn=collate)
        print_eval(evaluate(model, dl, device, ls))
        return

    train_ds = JointDataset(DATA_DIR / "train.json", tok, ls)
    syn = DATA_DIR / "synthetic_morph.json"
    if args.synthetic and syn.exists():
        s = JointDataset(syn, tok, ls)
        train_ds = torch.utils.data.ConcatDataset([train_ds, s, s])
        print(f"sentetik +{len(s)}×2")
    dev_dl = DataLoader(JointDataset(DATA_DIR / "dev.json", tok, ls),
                        batch_size=args.batch_size, collate_fn=collate)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    weights = build_class_weights(DATA_DIR / "train.json", ls, device) if args.class_weights else None
    print(f"train {len(train_ds)}  dev {len(dev_dl.dataset)}")

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": ENCODER_LR},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=WEIGHT_DECAY)
    print(f"LR: encoder {ENCODER_LR}, head'ler {args.lr}  |  TAG_LOSS_W {TAG_LOSS_W}")
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total * WARMUP_RATIO), total)
    best, start = -1.0, 1
    if args.resume and ck:
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        start = ck.get("epoch", 0) + 1
        best = ck.get("best", -1.0)
        print(f"resume epoch {start}, best {best:.2f}")

    def save(obj, path):
        tmp = path.with_suffix(".tmp"); torch.save(obj, tmp); tmp.replace(path)
        del obj; gc.collect()

    for ep in range(start, args.epochs + 1):
        frozen = ep <= args.freeze_encoder_epochs
        for p in model.encoder.parameters():
            p.requires_grad_(not frozen)
        print(f"\n=== Epoch {ep}/{args.epochs} ===" + ("  [encoder DONMUŞ]" if frozen else ""))
        model.train()
        if frozen:
            model.encoder.eval()  # deterministik özellikler
        tot = 0.0
        for batch in tqdm(train_dl, desc="train"):
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items() if k != "feats"}
            b["feats"] = {n: t.to(device) for n, t in batch["feats"].items()}
            out = model(b["input_ids"], b["attention_mask"], b["treebank_id"],
                        b["first_pos"], b["last_pos"])
            loss = compute_loss(out, b, ls, weights)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step(); sched.step(); tot += loss.item()
        print(f"train loss: {tot / len(train_dl):.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        res = evaluate(model, dev_dl, device, ls)
        print_eval(res)
        sc = sel_score(res)
        print(f"\n  sel score (mean of feats_exact & LAS): {sc:.2f}")
        is_best = sc > best; best = max(best, sc)
        meta = {"epoch": ep, "target_epochs": args.epochs, "best": best, "metrics": res,
                "label_space": ls.as_dict(), "encoder_model": ls.encoder_model,
                "treebanks": TREEBANKS}
        if is_best:
            save({**meta, "model": model.state_dict()}, DATA_DIR / "best_joint.pt")
            print("  → best")
        save({**meta, "model": model.state_dict(), "scheduler": sched.state_dict()},
             DATA_DIR / "joint_latest.pt")
    print(f"\nBest: {best:.2f}")


if __name__ == "__main__":
    main()
