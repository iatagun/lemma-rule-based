"""KÖR ÇOKLU KARŞILAŞTIRMA (2+ sürüm, aynı cümle): dinleyici her cümlede "en iyi"yi seçer. Kalibrasyon için (ör. uzun ünlü çarpanı 1,0 / 1,25 / 1,5).
  python -X utf8 -m dizgetts.scripts.make_abc_test --labels v8dp_lv10 v8dp_lv125 v8dp_lv15 --texts dizgetts/eval/long_vowel_sentences.txt --name lv_kalibrasyon \\
      --question "Hangisinde ğ'li uzun ünlüler doğru uzunlukta?"
  sonuç: ... --analyze <indirilen tsv> --name lv_kalibrasyon
Körlük ve yanlılık önlemleri make_ab_test ile aynı: sayfada sürüm adı yok, sıra cümle başına rastgele, ses düzeyi LUFS -23, hepsi dinlenmeden seçilemez.
Anahtar AB_KEYS/<name>.json (test klasörünün dışında).
"""
import argparse
import collections
import csv
import json
import os
import random

import soundfile as sf

from dizgetts import paths
from dizgetts.scripts.make_ab_test import loud

PAGE = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kör Karşılaştırma</title><style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a64;--line:#dedcd4;--acc:#1f5f8b;--ok:#2e7d4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#212120;--ink:#ecebe6;--mute:#a3a29b;--line:#3a3935;--acc:#6aa9d8;--ok:#6cc08f}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}main{max-width:760px;margin:0 auto;padding:20px 16px 48px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:14px 0}.t{font-size:18px;margin-bottom:10px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:6px 0}audio{height:36px;max-width:100%}
label{display:flex;gap:6px;align-items:center;cursor:pointer}.mute{color:var(--mute);font-size:14px}
button{font:inherit;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:8px;padding:9px 16px;cursor:pointer}
</style></head><body><main><h1 style="font-size:20px">__Q__</h1>
<p class="mute">Her cümlenin tüm seslerini dinleyin, sonra birini seçin (fark duymazsanız "fark yok"). İlerleme tarayıcıda saklanır. Bitince "Dışa aktar".</p>
<div id="L"></div><button id="exp">Dışa aktar (TSV)</button></main><script>
const ITEMS=__ITEMS__,KEY="abc___NAME__";let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const L=document.getElementById("L");
ITEMS.forEach(it=>{const c=document.createElement("div");c.className="card";c.innerHTML=`<div class="mute">${it.n} / ${ITEMS.length}</div><div class="t">${it.text}</div>`;
 const heard=new Set();const radios=[];
 it.files.forEach((f,k)=>{const r=document.createElement("div");r.className="row";const lab=String.fromCharCode(65+k);
  r.innerHTML=`<b>${lab}</b><audio controls preload="none" src="audio/${f}"></audio><label><input type="radio" name="p${it.n}" value="${lab}" disabled> ${lab} en iyi</label>`;
  const a=r.querySelector("audio");a.addEventListener("play",()=>{heard.add(k);if(heard.size===it.files.length)radios.forEach(x=>x.disabled=false)});
  radios.push(r.querySelector("input"));c.appendChild(r)});
 const n=document.createElement("div");n.className="row";n.innerHTML=`<label><input type="radio" name="p${it.n}" value="fark_yok" disabled> fark yok</label>`;radios.push(n.querySelector("input"));c.appendChild(n);
 radios.forEach(x=>{if(st[it.n]===x.value){x.checked=true;x.disabled=false};x.onchange=()=>{st[it.n]=x.value;save()}});
 if(st[it.n])radios.forEach(x=>x.disabled=false);L.appendChild(c)});
document.getElementById("exp").onclick=()=>{const rows=["çift\tseçim"].concat(ITEMS.map(it=>it.n+"\t"+(st[it.n]||"")));
 const b=new Blob([rows.join("\n")+"\n"],{type:"text/tab-separated-values"}),u=URL.createObjectURL(b),l=document.createElement("a");l.href=u;l.download="abc___NAME__.tsv";l.click();URL.revokeObjectURL(u)};
</script></body></html>"""


def build(a):
    rng = random.Random(a.seed)
    texts = [l.strip() for l in open(a.texts, encoding="utf8") if l.strip() and not l.startswith("#")]
    res = {lab: json.load(open(f"{paths.EVAL_OUT}/{lab}/results.json", encoding="utf8"))["items"] for lab in a.labels}
    n_items = len(res[a.labels[0]])
    assert all(len(v) == n_items == len(texts) for v in res.values()), "etiketlerin cümle sayısı ve metin dosyası uyuşmalı"
    out = f"{paths.AB}/{a.name}"
    os.makedirs(f"{out}/audio", exist_ok=True); os.makedirs(paths.AB_KEYS, exist_ok=True)
    items, key = [], []
    for n in range(1, n_items + 1):
        order = list(a.labels); rng.shuffle(order)
        files = []
        for k, lab in enumerate(order):
            x, sr = sf.read(f"{paths.EVAL_OUT}/{lab}/{res[lab][n - 1]['i']:03d}.wav", dtype="float64")
            fn = f"c{n:02d}_{chr(65 + k)}.wav"
            sf.write(f"{out}/audio/{fn}", loud(x, sr, a.lufs), sr, subtype="PCM_16")
            files.append(fn)
        items.append(dict(n=n, text=texts[n - 1], files=files))
        key.append(dict(n=n, order=order))
    page = PAGE.replace("__Q__", a.question).replace("__NAME__", a.name).replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
    open(f"{out}/index.html", "w", encoding="utf8").write(page)
    json.dump(dict(labels=a.labels, items=key), open(f"{paths.AB_KEYS}/{a.name}.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"{n_items} cümle x {len(a.labels)} sürüm -> {out}/index.html  (anahtar: {paths.AB_KEYS}/{a.name}.json)")


def analyze(a):
    key = {it["n"]: it["order"] for it in json.load(open(f"{paths.AB_KEYS}/{a.name}.json", encoding="utf8"))["items"]}
    c = collections.Counter()
    for r in csv.DictReader(open(a.analyze, encoding="utf8"), delimiter="\t"):
        s = r["seçim"]
        if s:
            c["fark_yok" if s == "fark_yok" else key[int(r["çift"])][ord(s) - 65]] += 1
    print(" | ".join(f"{k}: {v}" for k, v in c.most_common()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--texts")
    ap.add_argument("--name", required=True)
    ap.add_argument("--question", default="Hangisi en iyi?")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lufs", type=float, default=-23.0)
    ap.add_argument("--analyze", default=None)
    a = ap.parse_args()
    analyze(a) if a.analyze else build(a)
