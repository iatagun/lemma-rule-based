"""KÖR AB dinleme testi: iki değerlendirme etiketinin (evaluate.py çıktısı) aynı cümleleri -> tek dosyalık çevrimdışı sayfa.
  python -X utf8 -m dizgetts.scripts.make_ab_test --a v3a_g2ptts_nb_ep150_x543 --b v4_g2ptts_dp_ep150_x543 --name v3a_v4
  -> D:/dizgetts/ab/<name>/index.html (+ audio/), anahtar: D:/dizgetts/ab_keys/<name>.json (test klasörünün ve reponun DIŞINDA)
  sonuç: python -X utf8 -m dizgetts.scripts.make_ab_test --analyze <indirilen tsv> --name <name>

Körlük: sayfada model adı yok; dosyalar pNN_1/pNN_2; hangi tarafın hangi model olduğu çift başına rastgele ve yalnız anahtar dosyasında.
Yanlılık önlemleri: tüm sesler LUFS -23'e eşitlenir (yüksek ses 'daha iyi' algılanır); çift sırası rastgele; iki ses de dinlenmeden cevap verilemez.
"""
import argparse
import json
import os
import random

import numpy as np
import soundfile as sf

EVAL = "D:/dizgetts/eval_out"
AB, KEYS = "D:/dizgetts/ab", "D:/dizgetts/ab_keys"
ANTALIA = "D:/dizgetts/data/processed/antalia"
EXTRA = os.path.join(os.path.dirname(__file__), "..", "eval", "extra_sentences_ud.txt")


def texts():
    t = {}
    for sp in ("test", "val"):
        for l in open(f"{ANTALIA}/{sp}.jsonl", encoding="utf8"):
            r = json.loads(l); t[r["id"]] = r["text"]
    ex = [l.strip() for l in open(EXTRA, encoding="utf8") if l.strip() and not l.startswith("#")]
    for i, s in enumerate(ex):
        t[f"extra{i:03d}"] = s
    return t


def loud(x, sr, target=-23.0):
    import pyloudnorm as pyln
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # kırpma build()'de tepe denetimiyle yakalanır (sessiz np.clip YOK)
        return pyln.normalize.loudness(x, pyln.Meter(sr).integrated_loudness(x), target)


def build(a):
    rng = random.Random(a.seed)
    T = texts()
    items = {lab: {it["id"]: it for it in json.load(open(f"{EVAL}/{lab}/results.json", encoding="utf8"))["items"]} for lab in (a.a, a.b)}
    used = set()
    for kf in a.exclude_key or []:  # önceki testlerin cümleleri tekrar kullanılmaz (turlar bağımsız, birleştirilebilir)
        used |= {p["id"] for p in json.load(open(f"{KEYS}/{kf}.json", encoding="utf8"))["pairs"]}
    common = [k for k in items[a.a] if k in items[a.b] and k not in used]
    ant = [k for k in common if not k.startswith("extra")]
    ud = [k for k in common if k.startswith("extra")]
    pick = rng.sample(ant, a.n // 2) + rng.sample(ud, a.n - a.n // 2)
    rng.shuffle(pick)
    out = f"{AB}/{a.name}"
    os.makedirs(f"{out}/audio", exist_ok=True); os.makedirs(KEYS, exist_ok=True)
    key, data = [], []
    for n, cid in enumerate(pick, 1):
        swap = rng.random() < 0.5
        order = (a.b, a.a) if swap else (a.a, a.b)
        files = []
        for side, lab in zip((1, 2), order):
            x, sr = sf.read(f"{EVAL}/{lab}/{items[lab][cid]['i']:03d}.wav", dtype="float64")
            fn = f"p{n:02d}_{side}.wav"
            y = loud(x, sr, a.lufs)
            peak = float(np.abs(y).max())
            assert peak < 0.99, f"{lab}/{cid}: LUFS {a.lufs} ile tepe {peak:.2f} (kırpma) -> --lufs değerini düşürün"
            sf.write(f"{out}/audio/{fn}", y, sr, subtype="PCM_16")
            files.append(fn)
        key.append(dict(pair=n, id=cid, side1=order[0], side2=order[1]))
        data.append(dict(p=n, t=T[cid], f=files))
    json.dump(dict(a=a.a, b=a.b, seed=a.seed, pairs=key), open(f"{KEYS}/{a.name}.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__NAME__", a.name).replace("__ONEQ__", "true" if a.one_question else "false")
    open(f"{out}/index.html", "w", encoding="utf8").write(html)
    print(f"{len(data)} çift -> {out}/index.html  (anahtar: {KEYS}/{a.name}.json)")


def analyze(a):
    from scipy.stats import binomtest

    key = json.load(open(f"{KEYS}/{a.name}.json", encoding="utf8"))
    side = {p["pair"]: (p["side1"], p["side2"]) for p in key["pairs"]}
    rows = [l.rstrip("\n").split("\t") for l in open(a.analyze, encoding="utf8") if l.strip() and not l.startswith("#")]
    for qi, q in ((1, "doğallık/akıcılık"), (2, "telaffuz/vurgu")):
        wins = {key["a"]: 0, key["b"]: 0, "fark yok": 0}
        for r in rows:
            c = r[qi]
            if c == "0":
                wins["fark yok"] += 1
            elif c in ("1", "2"):
                wins[side[int(r[0])][int(c) - 1]] += 1
        na, nb = wins[key["a"]], wins[key["b"]]
        p = binomtest(nb, na + nb, 0.5).pvalue if na + nb else float("nan")
        print(f"{q}: {key['b']} {nb} | {key['a']} {na} | fark yok {wins['fark yok']}  (işaret testi, fark yok hariç: p = {p:.3f})")


HTML = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kör AB dinleme testi</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a64;--line:#dedcd4;--acc:#1f5f8b;--acc-ink:#fff;--ok:#2e7d4f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#212120;--ink:#ecebe6;--mute:#a3a29b;--line:#3a3935;--acc:#6aa9d8;--acc-ink:#0d1b26;--ok:#6cc08f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:16px}
h1{font-size:18px;margin:0}.mute{color:var(--mute);font-size:14px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:6px;width:220px}.bar i{display:block;height:100%;background:var(--ok)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 20px}
.text{font-size:19px;margin:6px 0 18px}
.players{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pl{border:1px solid var(--line);border-radius:10px;padding:10px}.pl b{display:block;margin-bottom:6px}
audio{width:100%}
.q{margin-top:20px}.q p{margin:0 0 8px;font-weight:600}
.opts{display:flex;gap:8px;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:8px 14px;cursor:pointer}
button:disabled{opacity:.4;cursor:not-allowed}button.on{background:var(--acc);color:var(--acc-ink);border-color:var(--acc)}
button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.primary{background:var(--acc);color:var(--acc-ink);border-color:var(--acc)}
input[type=text]{width:100%;margin-top:16px;font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
nav{display:flex;justify-content:space-between;margin-top:18px}
.hint{font-size:13px;margin-top:10px}
@media (max-width:560px){.players{grid-template-columns:1fr}}
</style></head><body><main>
<header><div><h1>Kör AB dinleme testi</h1><div class="mute"><span id="done">0</span>/<span id="total">0</span> çift</div><div class="bar"><i id="prog"></i></div></div>
<button class="primary" id="export">Dışa aktar (.tsv)</button></header>
<section class="card">
<div class="mute" id="pos"></div>
<div class="text" id="text"></div>
<div class="players">
 <div class="pl"><b>Ses 1</b><audio id="a1" controls preload="auto"></audio></div>
 <div class="pl"><b>Ses 2</b><audio id="a2" controls preload="auto"></audio></div>
</div>
<p class="hint mute" id="gate">Cevaplamak için iki sesi de dinleyin.</p>
<div class="q"><p>Hangisi daha doğal ve akıcı?</p><div class="opts" data-q="1">
 <button data-v="1">Ses 1</button><button data-v="0">Fark yok</button><button data-v="2">Ses 2</button></div></div>
<div class="q" id="q2box"><p>Hangisinin telaffuzu ve vurgusu daha doğru?</p><div class="opts" data-q="2">
 <button data-v="1">Ses 1</button><button data-v="0">Fark yok</button><button data-v="2">Ses 2</button></div></div>
<input type="text" id="note" placeholder="Not (isteğe bağlı): ör. Ses 2'de 'gidiyorum' yanlış vurgulu">
<nav><button id="prev">← Önceki</button><button id="next">Sonraki →</button></nav>
</section>
<p class="hint mute">Her çift için iki sesi dinleyip iki soruyu cevaplayın. Hangi sesin hangi sistemden geldiği bilinmez ve her çiftte rastgeledir;
sesler aynı yükseklikte. İlerleme bu tarayıcıda saklanır. Bitince <b>Dışa aktar</b> ile inen dosyayı bana verin.</p>
</main>
<script>
const DATA = __DATA__;
const ONEQ = __ONEQ__;  // tek soru: ikinci soru gizlenir, cevabı birinciyle aynı kaydedilir
if (ONEQ) { document.getElementById("q2box").style.display = "none"; document.querySelector('[data-q="1"]').previousElementSibling.textContent = "Hangisi genel olarak daha iyi (doğallık, akıcılık, telaffuz)?"; }
const KEY = "dizgetts-ab-__NAME__";
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const $ = id => document.getElementById(id);
let cur = 0;
const rec = i => st[DATA[i].p] || (st[DATA[i].p] = {});
const done = r => r.q1 != null && r.q2 != null;
function gate() {
  const r = rec(cur), ok = r.h1 && r.h2;
  document.querySelectorAll(".opts button").forEach(b => b.disabled = !ok);
  $("gate").style.visibility = ok ? "hidden" : "visible";
}
function render() {
  const d = DATA[cur], r = rec(cur);
  $("pos").textContent = `Çift ${cur + 1} / ${DATA.length}`;
  $("text").textContent = d.t;
  $("a1").src = "audio/" + d.f[0]; $("a2").src = "audio/" + d.f[1];
  document.querySelectorAll(".opts").forEach(o => o.querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", String(r["q" + o.dataset.q]) === b.dataset.v)));
  $("note").value = r.note || "";
  const n = DATA.filter((_, i) => done(st[DATA[i].p] || {})).length;
  $("done").textContent = n; $("total").textContent = DATA.length; $("prog").style.width = (100 * n / DATA.length) + "%";
  gate();
}
["a1", "a2"].forEach((id, k) => $(id).addEventListener("play", () => { rec(cur)["h" + (k + 1)] = true; save(); gate(); }));
document.querySelectorAll(".opts").forEach(o => o.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b || b.disabled) return;
  rec(cur)["q" + o.dataset.q] = +b.dataset.v; if (ONEQ) rec(cur).q2 = +b.dataset.v; save(); render();
  if (done(rec(cur)) && cur < DATA.length - 1) setTimeout(() => { cur++; render(); }, 250);
}));
$("note").oninput = e => { rec(cur).note = e.target.value.replace(/[\t\n]/g, " "); save(); };
$("prev").onclick = () => { if (cur > 0) { cur--; render(); } };
$("next").onclick = () => { if (cur < DATA.length - 1) { cur++; render(); } };
$("export").onclick = () => {
  const lines = ["# çift\tdoğallık(1/2/0)\ttelaffuz(1/2/0)\tnot", `# kör AB testi __NAME__, ${new Date().toISOString().slice(0, 10)}`];
  DATA.forEach(d => { const r = st[d.p] || {}; if (done(r)) lines.push(`${d.p}\t${r.q1}\t${r.q2}\t${r.note || ""}`); });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/tab-separated-values"}));
  a.download = "ab___NAME__.tsv"; a.click();
};
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lufs", type=float, default=-23.0, help="ortak ses yüksekliği; kırpma olursa düşürün")
    ap.add_argument("--name", required=True)
    ap.add_argument("--analyze", default=None, help="dışa aktarılan tsv")
    ap.add_argument("--exclude-key", nargs="*", default=None, help="bu testlerin (anahtar adı) cümlelerini kullanma")
    ap.add_argument("--one-question", action="store_true", help="yalnız genel tercih sorusu (v3a_v4 testinde iki soru 29/30 aynı cevaplandı)")
    a = ap.parse_args()
    analyze(a) if a.analyze else build(a)


if __name__ == "__main__":
    main()
