"""dizge-g2p-tts modeli: DizgeBERT-Dep kodlayıcısı + sözcük başına iki başlık (vurgu 5 sınıf, sınır 3 sınıf). Deney: experiments.yaml g2ptts-v1.
  python -X utf8 -u -m dizgetts.g2ptts.train            # eğit (seçim: val noktalamasız sınır F1)
  python -X utf8 -u -m dizgetts.g2ptts.train --test     # karar ölçümü: test + dep_K2 kıyası (klip bootstrap) + stress_gold.tsv
Veri: g2ptts/data.py çıktısı (D:/dizgetts/data/g2ptts). Sözcük temsili = ilk alt-sözcük (Dep ile aynı).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from dizgetts.frontend.dep import MODEL_ID as DEP_ID, MODEL_REV as DEP_REV

DATA = "D:/dizgetts/data/g2ptts"
RUN = "D:/dizgetts/runs/g2ptts_v1"
ENC = "dbmdz/electra-base-turkish-cased-discriminator"
PUNCT = {",", ".", "?", "!", ";"}
N_STRESS, N_BOUND = 5, 3


class G2PTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained(ENC))
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.stress = nn.Linear(h, N_STRESS)
        self.boundary = nn.Linear(h, N_BOUND)

    def forward(self, input_ids, attention_mask, first):
        hid = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        w = self.drop(torch.gather(hid, 1, first.clamp(min=0).unsqueeze(-1).expand(-1, -1, hid.size(-1))))
        return self.stress(w), self.boundary(w)


def init_from_dep(m: G2PTTS, freeze: int):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download(DEP_ID, "model.safetensors", revision=DEP_REV))
    m.bert.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("bert.")}, strict=True)
    for p in m.bert.embeddings.parameters():
        p.requires_grad = False
    for layer in m.bert.encoder.layer[:freeze]:
        for p in layer.parameters():
            p.requires_grad = False


def load(split):
    return [json.loads(l) for l in open(f"{DATA}/{split}.jsonl", encoding="utf8")]


def batchify(rows, tok, device):
    enc = tok([r["tokens"] for r in rows], is_split_into_words=True, truncation=True, max_length=256, padding=True, return_tensors="pt")
    W = max(len(r["tokens"]) for r in rows)
    first = torch.full((len(rows), W), -1, dtype=torch.long)
    ys, yb = torch.full((len(rows), W), -100), torch.full((len(rows), W), -100)
    for b, r in enumerate(rows):
        seen = set()
        for i, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid not in seen:
                seen.add(wid); first[b, wid] = i
        n = len(r["tokens"])
        ys[b, :n], yb[b, :n] = torch.tensor(r["stress"]), torch.tensor(r["boundary"])
        ys[b][first[b] < 0], yb[b][first[b] < 0] = -100, -100  # kesilen sözcükler
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), first.to(device), ys.to(device), yb.to(device)


@torch.no_grad()
def predict(m, rows, tok, device, bs=32):
    m.eval()
    out = []
    for i in range(0, len(rows), bs):
        ch = rows[i:i + bs]
        ids, am, first, _, _ = batchify(ch, tok, device)
        ls, lb = m(ids, am, first)
        ps, pb = ls.argmax(-1).cpu(), lb.softmax(-1).cpu()
        for b, r in enumerate(ch):
            n = len(r["tokens"])
            out.append((ps[b, :n].tolist(), (pb[b, :n, 1] + pb[b, :n, 2]).tolist()))
    return out


def boundary_items(rows, preds):
    """(klip_id, noktalamalı_mı, gerçek_kırılma, olasılık) — etiketli her sınır."""
    it = []
    for r, (_, pbreak) in zip(rows, preds):
        for i, y in enumerate(r["boundary"]):
            if y != -100:
                nxt = r["tokens"][i + 1] if i + 1 < len(r["tokens"]) else "."
                it.append((r["id"], nxt in PUNCT, y > 0, pbreak[i]))
    return it


def f1(items, pred):
    tp = sum(pred(x) and x[2] for x in items); fp = sum(pred(x) and not x[2] for x in items); fn = sum(x[2] and not pred(x) for x in items)
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0, P, R


def evaluate(m, rows, tok, device, tau=None):
    preds = predict(m, rows, tok, device)
    res = {}
    for src in ("ud", "antalia"):
        pairs = [(p, y) for r, (ps, _) in zip(rows, preds) if r["src"] == src for p, y in zip(ps, r["stress"]) if y != -100]
        nf = [(p, y) for p, y in pairs if y != 1]
        res[f"vurgu_uyum_{src}"] = sum(p == y for p, y in pairs) / len(pairs)
        res[f"vurgu_uyum_sondışı_{src}"] = sum(p == y for p, y in nf) / max(len(nf), 1)
    items = [x for x in boundary_items(rows, preds) if not x[1]]
    if tau is None:
        tau = max(np.arange(0.05, 0.95, 0.025), key=lambda t: f1(items, lambda x: x[3] > t)[0])
    res["tau"] = float(tau)
    res["sınır_F1"], res["sınır_P"], res["sınır_R"] = f1(items, lambda x: x[3] > tau)
    return res, preds


def train(a):
    device = "cuda"
    torch.manual_seed(0); random.seed(0)
    tok = AutoTokenizer.from_pretrained(DEP_ID, revision=DEP_REV)
    m = G2PTTS()
    init_from_dep(m, a.freeze)
    m.to(device)
    tr, va = load("train"), load("val")
    ant = [r for r in tr if r["src"] == "antalia"]
    params = [{"params": [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("bert.")], "lr": a.lr},
              {"params": [p for n, p in m.named_parameters() if not n.startswith("bert.")], "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    os.makedirs(RUN, exist_ok=True)
    log, best = [], -1
    for ep in range(1, a.epochs + 1):
        m.train()
        data = [r for r in tr if r["src"] == "ud"] + ant * a.antalia_repeat
        random.shuffle(data)
        t0, tot = time.time(), 0.0
        for i in range(0, len(data), a.bs):
            ids, am, first, ys, yb = batchify(data[i:i + a.bs], tok, device)
            ls, lb = m(ids, am, first)
            loss = F.cross_entropy(ls.reshape(-1, N_STRESS), ys.reshape(-1), ignore_index=-100)
            if (yb != -100).any():
                loss = loss + F.cross_entropy(lb.reshape(-1, N_BOUND), yb.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
            if (i // a.bs) % 200 == 0:
                print(f"ep{ep} adım {i // a.bs}/{len(data) // a.bs} kayıp {loss.item():.4f}", flush=True)
        res, _ = evaluate(m, va, tok, device)
        res.update(epoch=ep, train_loss=tot / (len(data) / a.bs), sec=round(time.time() - t0))
        log.append(res)
        print("VAL", json.dumps(res, ensure_ascii=False), flush=True)
        if res["sınır_F1"] > best:  # seçim = dağıtım metriği (noktalamasız sınır F1); vurgu karar kuralında ayrıca denetlenir
            best = res["sınır_F1"]
            torch.save(dict(state=m.state_dict(), val=res, args=vars(a)), f"{RUN}/best.pt")
        json.dump(log, open(f"{RUN}/log.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print("Bitti. En iyi val sınır F1:", best, flush=True)


def test(a):
    from dizgetts.eval.boundary_baseline import closing_phrase, dep_cache
    from dizgetts.eval.stress_intrinsic import GOLD
    from dizgetts.frontend.normalize import tr_lower
    from dizgetts.frontend.stress import _n_vowels
    import yaml

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(DEP_ID, revision=DEP_REV)
    ck = torch.load(f"{RUN}/best.pt", map_location="cpu")
    m = G2PTTS(); m.load_state_dict(ck["state"]); m.to(device)
    te = load("test")
    res, preds = evaluate(m, te, tok, device, tau=ck["val"]["tau"])
    # dep_K2 aynı sınırlarda
    root = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "data.yaml"), encoding="utf8"))["out_root"]
    deps = dep_cache(root)
    items = []  # (klip, gerçek, model, dep)
    for r, (_, pb) in zip(te, preds):
        if r["src"] != "antalia":
            continue
        d = deps[r["id"]]
        assert d["tokens"] == r["tokens"], r["id"]
        for i, y in enumerate(r["boundary"]):
            if y == -100 or (i + 1 < len(r["tokens"]) and r["tokens"][i + 1] in PUNCT):
                continue
            nxt = next(j for j in range(i + 1, len(r["tokens"])) if r["tokens"][j].isalpha())
            size, head = closing_phrase(d["heads"], i)
            items.append((r["id"], y > 0, pb[i] > res["tau"], size >= 2 and head != nxt + 1))

    def F(sub, k):  # k=0 model, k=1 dep_K2
        return f1([(None, False, x[1], x[2 + k]) for x in sub], lambda x: x[3])[0]

    clips = sorted({x[0] for x in items})
    by = {c: [x for x in items if x[0] == c] for c in clips}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(1000):
        s = [x for c in rng.choice(clips, len(clips)) for x in by[c]]
        diffs.append(F(s, 0) - F(s, 1))
    res.update(sınır_F1_model=F(items, 0), sınır_F1_dep_K2=F(items, 1), fark_CI95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
               n_sınır=len(items), n_klip=len(clips))
    # stress_gold.tsv (35 sözcük, tek başına)
    gold = [l.split("\t") for l in GOLD.read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]
    ok = 0
    for w, g, cat, *_ in gold:
        text = w[:1].replace("i", "İ").upper() + w[1:] if cat.startswith("yer adı") else w
        p = predict(m, [dict(tokens=[text], stress=[0], boundary=[-100])], tok, device)[0][0][0]
        ok += p == 1 + min(int(g), 3)
    res["gold_doğru"] = f"{ok}/{len(gold)}"
    print("TEST", json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(f"{RUN}/test.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--freeze", type=int, default=6)
    ap.add_argument("--antalia-repeat", type=int, default=8)
    a = ap.parse_args()
    test(a) if a.test else train(a)
