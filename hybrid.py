#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DizgeBERT-Hybrid — DizgeBERT-Morph (etiketleme) + DizgeBERT-Joint (ayrıştırma) birleşik çıkarım.

Motivasyon: DizgeBERT-Morph saf morfolojide en iyi ama UD'de *sözdizimsel olarak tanımlı*
UPOS ayrımlarında (DET/PRON `o çocuk` vs `onu`; genitive tamlayan → nominal) yapısal tavana
çarpıyor. Joint model bunları çözüyor ama saf morfolojide ~3p geride.

Hibrit: XPOS/FEATS → daima Morph. HEAD/DEPREL → Joint. UPOS → Morph; ancak Morph≠Joint
çeliştiğinde ve Joint'in deprel'i kendi UPOS'unun *kanonik işaretçisi* ise (ve Joint güveni
yüksekse) → Joint'in UPOS'u. Düzeltmeler `hybrid_eval` ile held-out'ta regresyonsuz doğrulandı.

Kullanım:
    python hybrid.py --in cumleler.txt --plain --out out.conllu --scheme kenet
    python hybrid.py --in x.conllu --out y.conllu
    from hybrid import HybridTagger
    HybridTagger().predict("Çocukların geleceği bizim elimizde .".split(), scheme="kenet")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent

import torch

MORPH_HF = "iatagun/DizgeBERT-Morph"
JOINT_HF = "iatagun/DizgeBERT-Joint"
MORPH_LOCAL = PROJECT_ROOT / "morph_data" / "best_morph_tagger.pt"   # DizgeBERT-Morph v3
JOINT_LOCAL = PROJECT_ROOT / "morph_data" / "best_joint_v2.pt"

CONF = 0.60
_DEM_ROLE = {"det": "DET", "nsubj": "PRON", "obj": "PRON", "iobj": "PRON", "obl": "PRON",
             "nmod": "PRON", "root": "PRON", "conj": "PRON", "nsubj:outer": "PRON"}
_POSS_DEPREL = {"nmod:poss"}
_NOMINALIZED = ("acağı", "eceği", "acağını", "eceğini", "dığı", "diği", "tığı",
                "dığını", "diğini", "ması", "mesi", "masını", "mesini", "ğı", "ği")


def _labels(cfg):
    """HF config ya da LabelSpace'ten (upos, xpos, feats_dict, deprels)."""
    if hasattr(cfg, "upos_labels"):
        return (cfg.upos_labels, cfg.xpos_labels, cfg.feats_label_space,
                getattr(cfg, "deprels", []))
    return cfg.upos, cfg.xpos, cfg.feat_values, getattr(cfg, "deprels", [])


class HybridTagger:
    def __init__(self, local: bool = False, device=None):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if local:
            from train_joint import JointModel
            from train_morph_bert import MorphTagger
            from train_morph_bert import LabelSpace
            cm = torch.load(MORPH_LOCAL, map_location=self.device)
            self.lsm = LabelSpace(cm["label_space"])
            self.morph = MorphTagger(self.lsm, self.lsm.encoder_model).to(self.device).eval()
            self.morph.load_state_dict(cm["model"])
            cj = torch.load(JOINT_LOCAL, map_location=self.device)
            self.lsj = LabelSpace(cj["label_space"])
            self.joint = JointModel(self.lsj, self.lsj.encoder_model).to(self.device).eval()
            self.joint.load_state_dict(cj["model"])
            self.tok = AutoTokenizer.from_pretrained(self.lsm.encoder_model)
            self._out_key = ("upos", "xpos", "feats", "arc", "lab")
        else:
            self.morph = AutoModel.from_pretrained(MORPH_HF, trust_remote_code=True).to(self.device).eval()
            self.joint = AutoModel.from_pretrained(JOINT_HF, trust_remote_code=True).to(self.device).eval()
            self.tok = AutoTokenizer.from_pretrained(MORPH_HF)
            self.lsm, self.lsj = self.morph.config, self.joint.config
            self._out_key = ("logits_upos", "logits_xpos", "logits_feats", "arc", "lab")

        (self.upos_m, self.xpos_m, self.feats_m, _) = _labels(self.lsm)
        (self.upos_j, _, _, self.deprels_j) = _labels(self.lsj)
        self.feat_names = list(self.feats_m.keys())
        self.tb_to_id = {tb: i for i, tb in enumerate(
            getattr(self.lsm, "treebanks", ["kenet", "boun", "imst"]))}

    def _run(self, model, enc, tb, fp, lp):
        o = model(enc["input_ids"], enc["attention_mask"], tb, fp, lp)
        return (lambda k: o[k] if isinstance(o, dict) else getattr(o, k))

    @torch.no_grad()
    def predict(self, words: list[str], scheme: str = "kenet"):
        enc = self.tok(words, is_split_into_words=True, return_tensors="pt",
                       truncation=True, max_length=128).to(self.device)
        first, last = {}, {}
        for i, wid in enumerate(enc.word_ids()):
            if wid is None:
                continue
            first.setdefault(wid, i)
            last[wid] = i
        kept = sorted(first)
        fp = torch.tensor([[first[w] for w in kept]], device=self.device)
        lp = torch.tensor([[last[w] for w in kept]], device=self.device)
        tb = torch.tensor([self.tb_to_id[scheme]], device=self.device)

        gm = self._run(self.morph, enc, tb, fp, lp)
        gj = self._run(self.joint, enc, tb, fp, lp)
        ku, kx, kf, ka, kl = self._out_key
        m_up = gm(ku).argmax(-1)[0]
        m_xp = gm(kx).argmax(-1)[0]
        m_fp = {n: gm(kf)[n].argmax(-1)[0] for n in self.feat_names}
        j_up = gj(ku).argmax(-1)[0]
        j_conf = gj(ku).softmax(-1).max(-1).values[0]
        arc = gj(ka)[0, :len(kept) + 1, :len(kept) + 1].float().cpu().numpy()
        lab = gj(kl)[0].float().cpu()

        from dizgebert_joint.modeling_dizgebert_joint import mst
        heads = mst(arc)

        nominal_ok = scheme != "imst"
        rows = []
        for k in range(len(kept)):
            m_upos = self.upos_m[int(m_up[k])]
            xpos = self.xpos_m[int(m_xp[k])]
            fpairs = sorted(((n, self.feats_m[n][int(m_fp[n][k])])
                             for n in self.feat_names if int(m_fp[n][k]) > 0),
                            key=lambda kv: kv[0].lower())
            feats = "|".join(f"{a}={b}" for a, b in fpairs) or "_"
            h = int(heads[k + 1])
            deprel = self.deprels_j[int(lab[k + 1, h].argmax())]
            j_upos = self.upos_j[int(j_up[k])]
            surf = words[k].lower().replace("i̇", "i")

            upos, corrected = m_upos, False
            if j_upos != m_upos and float(j_conf[k]) >= CONF:
                if "PronType=Dem" in feats and {m_upos, j_upos} <= {"DET", "PRON"}:
                    want = _DEM_ROLE.get(deprel)
                    if want and j_upos == want:
                        upos, corrected = want, True
                elif nominal_ok and deprel in _POSS_DEPREL and m_upos in ("ADJ", "ADV") \
                        and j_upos in ("NOUN", "PROPN", "PRON"):
                    upos, corrected = j_upos, True
                elif nominal_ok and m_upos == "ADJ" and j_upos == "NOUN" \
                        and deprel in ("nsubj", "obj", "root", "conj", "ccomp") \
                        and surf.endswith(_NOMINALIZED):
                    upos, corrected = "NOUN", True

            rows.append({"form": words[k], "upos": upos, "xpos": xpos, "feats": feats,
                         "head": h, "deprel": deprel, "corrected": corrected,
                         "morph_upos": m_upos, "joint_upos": j_upos})
        return rows


def _conllu(rows, meta=None):
    out = [f"# {k} = {v}" for k, v in (meta or {}).items()]
    for i, r in enumerate(rows, 1):
        out.append("\t".join([str(i), r["form"], "_", r["upos"], r["xpos"], r["feats"],
                              str(r["head"]), r["deprel"], "_",
                              "Corrected=Yes" if r["corrected"] else "_"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain", action="store_true", help="girdi düz metin (cümle/satır)")
    ap.add_argument("--scheme", choices=["kenet", "boun", "imst"], default="kenet")
    ap.add_argument("--local", action="store_true", help="HF yerine yerel .pt checkpoint'leri")
    args = ap.parse_args()

    ht = HybridTagger(local=args.local)
    blocks, n_corr = [], 0
    if args.plain:
        for ln, line in enumerate(Path(args.inp).read_text(encoding="utf-8").splitlines(), 1):
            ws = line.split()
            if not ws:
                continue
            rows = ht.predict(ws, args.scheme)
            n_corr += sum(r["corrected"] for r in rows)
            blocks.append(_conllu(rows, {"sent_id": ln, "text": line.strip()}))
    else:
        import conllu
        with open(args.inp, encoding="utf-8") as f:
            for sent in conllu.parse_incr(f):
                toks = [t for t in sent if isinstance(t["id"], int)]
                rows = ht.predict([t["form"] for t in toks], args.scheme)
                n_corr += sum(r["corrected"] for r in rows)
                blocks.append(_conllu(rows, dict(sent.metadata)))
    Path(args.out).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"yazıldı: {args.out}  ({n_corr} UPOS düzeltmesi)")


if __name__ == "__main__":
    main()
