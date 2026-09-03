#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Türkçe morfolojik belirsizlik — nokta atışı test seti.

Held-out treebank metrikleri (benchmark/eval_morph.py) genel doğruluğu ölçer ama
belirsizlik/eş-yazım vakalarını seyrek örnekler. Bu dosya, elle seçilmiş minimal
çiftleri ve sözdizimsel-tanımlı ayrımları hedefler — modeller arası kıyas için.

Kategoriler: isim↔fiil eş-yazımları, aorist↔türemiş isim, özel-isim↔sözcük,
-mA olumsuz-emir↔ad-fiil, -AcAk/-An/-DIk adlaşma, çatı, Gen↔iyelik, Acc↔iyelik,
DET↔PRON (sözdizimsel), edat, delillilik, üleştirme/sıra sayı.

Kullanım:
    python benchmark/eval_ambiguity.py --model hybrid --scheme kenet
    python benchmark/eval_ambiguity.py --model morph  --scheme imst
    python benchmark/eval_ambiguity.py --model joint   --scheme boun --local
    python benchmark/eval_ambiguity.py --compare        # morph vs joint vs hybrid
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))


# ═══════════════════════════════════════════════════════════════════════
#  Vaka seti — (kategori, hedef_yüzey, cümle, beklenen)
#  beklenen: "UPOS" veya "UPOS|feat=değer,feat=değer" (feat'ler substring aranır)
#  '~' önekli beklenen: treebank'ler tutarsız → bilgi amaçlı, skora katılmaz
# ═══════════════════════════════════════════════════════════════════════
CASES: list[tuple[str, str, str, str]] = [
    # ── isim ↔ fiil eş-yazımı ──
    ("isim/fiil", "dolar", "Cebinde yüz dolar vardı .", "NOUN"),
    ("isim/fiil", "dolar", "Bardağa su koyunca yavaşça dolar .", "VERB"),
    ("isim/fiil", "gülün", "Bahçedeki gülün kokusu çok güzeldi .", "NOUN|Case=Gen"),
    ("isim/fiil", "gülün", "Hep birlikte gülün ve eğlenin .", "VERB|Mood=Imp"),
    ("isim/fiil", "koyun", "Çayırda bir koyun otluyordu .", "NOUN"),
    ("isim/fiil", "koyun", "Kitapları rafa dikkatlice koyun .", "VERB|Mood=Imp"),
    ("isim/fiil", "at", "Çiftlikte üç at ve iki inek vardı .", "NOUN"),
    ("isim/fiil", "at", "Topu bana hızlıca at .", "VERB|Mood=Imp"),
    ("isim/fiil", "yüz", "Havuzda beş dakika yüz ve sonra çık .", "VERB|Mood=Imp"),
    ("isim/fiil", "yüz", "Salonda yüz kişilik bir masa vardı .", "NUM"),
    ("isim/fiil", "bin", "Bu araba tam iki bin lira .", "NUM"),
    ("isim/fiil", "bin", "Hemen atına bin ve köye git .", "VERB|Mood=Imp"),
    ("isim/fiil", "boya", "Duvara mavi boya sürdük .", "NOUN"),
    ("isim/fiil", "boya", "Şu duvarı hemen boya .", "VERB|Mood=Imp"),
    ("isim/fiil", "kaz", "Gölette bir kaz yüzüyordu .", "NOUN"),
    ("isim/fiil", "kaz", "Bahçeye derin bir çukur kaz .", "VERB|Mood=Imp"),

    # ── aorist fiil ↔ türemiş isim ──
    ("aor/isim", "geliri", "Ailenin aylık geliri oldukça düşüktü .", "NOUN"),
    ("aor/isim", "gelir", "O her akşam saat yedide eve gelir .", "VERB"),
    ("aor/isim", "çıkarını", "Herkes kendi çıkarını düşünüyordu .", "NOUN"),
    ("aor/isim", "çıkar", "Tren her sabah tam sekizde istasyondan çıkar .", "VERB"),
    ("aor/isim", "okuru", "Gazetenin sadık bir okuru olduğunu söyledi .", "NOUN"),
    ("aor/isim", "okur", "Dedem her sabah gazetesini dikkatle okur .", "VERB"),
    ("aor/isim", "yazar", "Bu romanı ünlü bir yazar yazdı .", "NOUN"),
    ("aor/isim", "yazar", "O her sabah günlüğüne bir şeyler yazar .", "VERB"),

    # ── özel isim ↔ sözcük ──
    ("özel-isim", "Güler", "Bebek annesini görünce tatlı tatlı güler .", "VERB"),
    ("özel-isim", "Güler", "Toplantıya Güler de katılacakmış .", "PROPN"),
    ("özel-isim", "Sever", "Annem taze çiçekleri çok sever .", "VERB"),

    # ── -mA olumsuz emir ↔ ad-fiil ──
    ("-mA", "gitme", "Sakın bu havada dışarı gitme .", "VERB|Polarity=Neg"),
    ("-mA", "gitmesi", "Onun aniden gitmesi herkesi üzdü .", "~VERB|VerbForm=Vnoun"),
    ("-mA", "yapma", "Sakın böyle bir şey yapma .", "VERB|Polarity=Neg"),
    ("-mA", "okuma", "Çocuk henüz okuma yazma bilmiyor .", "~VERB|VerbForm=Vnoun"),
    ("-mA", "okuma", "Sakın o mektubu okuma .", "VERB|Polarity=Neg"),

    # ── -AcAk gelecek ↔ sıfat ↔ isim ──
    ("-AcAk", "Gelecek", "Gelecek hafta sınavımız var .", "ADJ"),
    ("-AcAk", "geleceği", "Çocukların geleceği bizim elimizde .", "~NOUN"),
    ("-AcAk", "gelecek", "Yarın bize misafir gelecek .", "VERB"),

    # ── -An sıfat-fiil ↔ isim ──
    ("-An", "Bakanı", "Yeni Sağlık Bakanı bugün açıklama yaptı .", "NOUN"),
    ("-An", "bakan", "Pencereden dışarı bakan çocuk çok mutluydu .", "~VERB|VerbForm=Part"),
    ("-An", "yazan", "Tahtaya yazan öğrenci geç geldi .", "~VERB|VerbForm=Part"),

    # ── -DIk ──
    ("-DIk", "tanıdık", "Uzaktan bir tanıdık gördük .", "~NOUN"),
    ("-DIk", "tanıdık", "Onu kalabalığın içinde hemen tanıdık .", "VERB"),

    # ── çatı ──
    ("çatı", "yıkandı", "Bütün camlar sabahtan yıkandı .", "VERB|Voice=Pass"),

    # ── Gen ↔ iyelik-2 (aynı -In yüzeyi) ──
    ("Gen/psor2", "evin", "Senin evin gerçekten çok güzel .", "NOUN|Person[psor]=2"),
    ("Gen/psor2", "evin", "Evin kapısı sabahtan beri açıktı .", "NOUN|Case=Gen"),
    ("Gen/psor2", "kitabın", "Bu kitabın kapağı çok güzel .", "NOUN|Case=Gen"),
    ("Gen/psor2", "kitabın", "Senin kitabın masanın üstünde .", "NOUN|Person[psor]=2"),

    # ── Acc ↔ iyelik-3 ──
    ("Acc/psor3", "kalemi", "Masanın üstündeki kalemi bana uzat .", "NOUN|Case=Acc"),
    ("Acc/psor3", "kalemi", "Onun kalemi dün kırılmış .", "NOUN|Person[psor]=3"),
    ("Acc/psor3", "arabası", "Onun yeni arabası kırmızıymış .", "NOUN|Person[psor]=3"),

    # ── DET ↔ PRON (sözdizimsel-tanımlı; joint/hybrid'de çözülür) ──
    ("DET/PRON", "O", "O çocuk çok yaramaz .", "DET"),
    ("DET/PRON", "onu", "Dün akşam onu parkta gördüm .", "PRON"),
    ("DET/PRON", "Bu", "Bu senin defterin mi ?", "PRON"),
    ("DET/PRON", "Şu", "Şu kitabı bana ver .", "DET"),
    ("DET/PRON", "Bunu", "Bunu daha önce hiç duymamıştım .", "PRON"),

    # ── bir : DET / ADV ──
    ("bir", "bir", "Onu bir görsem çok sevinirim .", "ADV"),

    # ── ne : PRON / ADV ──
    ("ne", "ne", "Bu kutunun içinde ne var ?", "PRON"),
    ("ne", "ne", "Ne güzel bir gün !", "ADV"),

    # ── edat (ADP) ──
    ("edat", "göre", "Habere göre yarın hava yağmurluymuş .", "ADP"),
    ("edat", "kadar", "Akşama kadar çalıştık .", "ADP"),

    # ── delillilik (boun/imst) ──
    ("delil", "gelmiş", "Sen yokken kardeşin eve gelmiş .", "~VERB|Evident=Nfh"),

    # ── üleştirme / sıra sayı ──
    ("sayı", "beşer", "Öğrencilere beşer elma dağıtıldı .", "NUM|NumType=Dist"),
    ("sayı", "beşinci", "Sınavdan beşinci oldu .", "~NUM|NumType=Ord"),

    # ── -lI ──
    ("-lI", "evli", "Ablam iki yıldır evli .", "ADJ"),

    # ── nadir sözlüksel okuma ──
    ("nadir", "beni", "Yanağındaki beni herkes fark etti .", "~NOUN"),
]


def _low(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def _check(pred_upos: str, pred_feats: str, expected: str):
    """→ ('ok' | 'fail' | 'info', beklenen_temiz)"""
    soft = expected.startswith("~")
    exp = expected[1:] if soft else expected
    want_upos, _, want_feats = exp.partition("|")
    ok = pred_upos == want_upos
    if ok and want_feats:
        ok = all(f.strip() in pred_feats for f in want_feats.split(","))
    if soft:
        return ("info-ok" if ok else "info", exp)
    return ("ok" if ok else "fail", exp)


# ═══════════════════════════════════════════════════════════════════════
#  Model yükleyiciler
# ═══════════════════════════════════════════════════════════════════════
def make_predictor(model: str, scheme: str, local: bool):
    """→ fn(words) -> list[(upos, xpos, feats)]"""
    if model == "hybrid":
        from hybrid import HybridTagger
        ht = HybridTagger(local=local)
        return lambda ws: [(r["upos"], r["xpos"], r["feats"]) for r in ht.predict(ws, scheme)]

    if local:
        import torch
        from train_morph_bert import LabelSpace, MorphTagger, TB_TO_ID
        if model == "morph":
            ck = torch.load(_PROJECT / "morph_data/best_morph_tagger.pt", map_location="cpu")
            ls = LabelSpace(ck["label_space"])
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(ls.encoder_model)
            m = MorphTagger(ls, ls.encoder_model).eval()
            m.load_state_dict(ck["model"])
        else:
            from train_joint import JointModel
            ck = torch.load(_PROJECT / "morph_data/best_joint_v2.pt", map_location="cpu")
            ls = LabelSpace(ck["label_space"])
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(ls.encoder_model)
            m = JointModel(ls, ls.encoder_model).eval()
            m.load_state_dict(ck["model"])

        @torch.no_grad()
        def pred_local(ws):
            enc = tok(ws, is_split_into_words=True, return_tensors="pt",
                      truncation=True, max_length=128)
            first, last = {}, {}
            for i, w in enumerate(enc.word_ids()):
                if w is None:
                    continue
                first.setdefault(w, i)
                last[w] = i
            kept = sorted(first)
            fp = torch.tensor([[first[w] for w in kept]])
            lp = torch.tensor([[last[w] for w in kept]])
            tb = torch.tensor([TB_TO_ID[scheme]])
            o = m(enc["input_ids"], enc["attention_mask"], tb, fp, lp)
            up = o["upos"].argmax(-1)[0]
            xp = o["xpos"].argmax(-1)[0]
            fpr = {n: o["feats"][n].argmax(-1)[0] for n in ls.feat_names}
            return [(ls.upos[int(up[k])], ls.xpos[int(xp[k])],
                     ls.feats_to_string({n: ls.feat_values[n][int(fpr[n][k])]
                                         for n in ls.feat_names}))
                    for k in range(len(kept))]
        return pred_local

    # HF
    from transformers import AutoModel, AutoTokenizer
    repo = "iatagun/DizgeBERT-Morph" if model == "morph" else "iatagun/DizgeBERT-Joint"
    m = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(repo)
    if model == "morph":
        return lambda ws: list(m.predict(ws, scheme=scheme, tokenizer=tok))
    return lambda ws: [(u, x, f) for (_frm, u, x, f, _h, _d)
                       in m.predict(ws, scheme=scheme, tokenizer=tok)]


def _tokenize(sent):
    out = []
    for tok in sent.split():
        mm = re.match(r"^(.*?)([.,!?;:]+)$", tok)
        out += [mm.group(1), mm.group(2)] if mm and mm.group(1) else [tok]
    return out


def run(model, scheme, local, cases=CASES):
    predict = make_predictor(model, scheme, local)
    rows, n_ok, n_fail = [], 0, 0
    for cat, tgt, sent, exp in cases:
        ws = _tokenize(sent)
        idx = next((i for i, w in enumerate(ws) if _low(w) == _low(tgt)), None)
        if idx is None:
            rows.append((cat, tgt, "?", "hedef bulunamadı", exp)); continue
        u, x, f = predict(ws)[idx]
        status, expc = _check(u, f, exp)
        n_ok += status == "ok"
        n_fail += status == "fail"
        rows.append((cat, tgt, status, f"{u} {f}", expc))
    return rows, n_ok, n_fail


def _print(rows, n_ok, n_fail, title):
    print(f"\n=== {title} ===")
    cur = None
    for cat, tgt, st, got, exp in rows:
        if cat != cur:
            print(f"\n  ── {cat} ──"); cur = cat
        mark = {"ok": "✓", "fail": "✗", "info": "·", "info-ok": "·"}.get(st, "?")
        print(f"  {mark} {tgt:12s} {got[:52]:52s}  bkln: {exp}")
    scored = n_ok + n_fail
    print(f"\n  skor: {n_ok}/{scored}  (~ ile işaretli treebank-tutarsız vakalar hariç)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["morph", "joint", "hybrid"], default="hybrid")
    ap.add_argument("--scheme", choices=["kenet", "boun", "imst"], default="kenet")
    ap.add_argument("--local", action="store_true", help="HF yerine yerel .pt")
    ap.add_argument("--compare", action="store_true", help="morph + joint + hybrid yan yana")
    args = ap.parse_args()

    if args.compare:
        results = {}
        for mdl in ("morph", "joint", "hybrid"):
            rows, ok, fail = run(mdl, args.scheme, args.local)
            results[mdl] = {(c, t): s for c, t, s, _g, _e in rows}
            print(f"{mdl:7s} skor: {ok}/{ok+fail}")
        print(f"\n=== FARKLAR (scheme={args.scheme}) ===")
        keys = list(results["morph"])
        for k in keys:
            sts = [results[m].get(k, "?") for m in ("morph", "joint", "hybrid")]
            if len(set(s for s in sts if s in ("ok", "fail"))) > 1:
                print(f"  {k[1]:12s} ({k[0]})  morph={sts[0]}  joint={sts[1]}  hybrid={sts[2]}")
        return

    rows, ok, fail = run(args.model, args.scheme, args.local)
    _print(rows, ok, fail, f"{args.model}  scheme={args.scheme}  {'(yerel)' if args.local else '(HF)'}")


if __name__ == "__main__":
    main()
