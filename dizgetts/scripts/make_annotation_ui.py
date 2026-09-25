"""Vurgu etiketleme arayüzü: reports/stress_annotation_sheet.tsv -> reports/stress_annotation.html (tek dosya, çevrimdışı, tarayıcıda aç).
  python -X utf8 -m dizgetts.scripts.make_annotation_ui

KÖR etiketleme: modelin önerisi GÖSTERİLMEZ (öneriye çapalanma bağımsız ölçümü bozar). Kullanıcı vurgulu seslemi tıklar / 1-9 tuşu.
İlerleme tarayıcıda (localStorage) saklanır. "Dışa aktar" -> stress_gold_random.tsv (stress_gold.tsv biçimi: sözcük, SONDAN sıra ya da "-" = vurgusuz,
kategori, not); dosyayı dizgetts/tests/ altına koyun ve:  python -X utf8 -m dizgetts.eval.stress_intrinsic --gold dizgetts/tests/stress_gold_random.tsv
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SRC, OUT = HERE / "reports" / "stress_annotation_sheet.tsv", HERE / "reports" / "stress_annotation.html"

HTML = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a64;--line:#dedcd4;--acc:#1f5f8b;--acc-ink:#fff;--ok:#2e7d4f;--warn:#a86b00;--skip:#8a8a8a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#212120;--ink:#ecebe6;--mute:#a3a29b;--line:#3a3935;--acc:#6aa9d8;--acc-ink:#0d1b26;--ok:#6cc08f;--warn:#e0a84a;--skip:#8f8f8f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:16px}
h1{font-size:18px;margin:0}.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:6px 0 0;width:220px}.bar i{display:block;height:100%;background:var(--ok)}
.mute{color:var(--mute);font-size:14px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:8px 14px;cursor:pointer}
button:hover{border-color:var(--acc)}button:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.primary{background:var(--acc);color:var(--acc-ink);border-color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px 20px}
.meta{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.syl{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:28px 0 8px}
.syl button{font-size:30px;padding:10px 18px;min-width:64px;position:relative}
.syl button small{position:absolute;top:2px;left:6px;font-size:11px;color:var(--mute)}
.syl button.on{background:var(--acc);color:var(--acc-ink);border-color:var(--acc);text-transform:uppercase}
.ex{text-align:center;color:var(--mute);margin:12px 0 20px}.ex b{color:var(--ink)}
.row{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.row button.on{border-color:var(--warn);color:var(--warn);font-weight:600}
input[type=text]{width:100%;margin-top:16px;font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
nav{display:flex;justify-content:space-between;margin-top:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(18px,1fr));gap:3px;margin-top:24px}
.grid span{height:18px;border-radius:3px;background:var(--line);cursor:pointer}
.grid span.s{background:var(--ok)}.grid span.v{background:var(--acc)}.grid span.u{background:var(--warn)}.grid span.cur{outline:2px solid var(--ink)}
.help{margin-top:18px;font-size:13px}kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:12px}
.tag{font-size:12px;border:1px solid var(--line);border-radius:10px;padding:1px 8px}
</style></head><body><main>
<header><div><h1>__TITLE__</h1><div class="mute"><span id="done">0</span>/<span id="total">0</span> etiketlendi</div><div class="bar"><i id="prog"></i></div></div>
<div class="row"><button id="next-empty">İlk boş</button><button class="primary" id="export">Dışa aktar (.tsv)</button></div></header>
<section class="card" aria-live="polite">
<div class="meta"><span class="mute" id="pos"></span><span class="mute" id="freq"></span></div>
<div class="syl" id="syl"></div>
<p class="ex" id="ex"></p>
<div class="row"><button id="b-none" title="Tuş: 0">Vurgusuz (clitic)</button><button id="b-unsure" title="Tuş: ?">Emin değilim</button></div>
<input type="text" id="note" placeholder="Not (isteğe bağlı): ör. özel ad olarak okudum, iki okuma var…">
<nav><button id="prev">← Önceki</button><button id="next">Sonraki →</button></nav>
</section>
<div class="grid" id="grid" aria-label="Tüm sözcükler"></div>
<p class="help mute">Vurgulu hecenin üstüne tıklayın ya da numarasını basın (<kbd>1</kbd>–<kbd>9</kbd>); seçim sonrası bir sonrakine geçilir.
<kbd>0</kbd> vurgusuz, <kbd>?</kbd> emin değilim, <kbd>←</kbd>/<kbd>→</kbd> gezin. İlerleme bu tarayıcıda saklanır; bitince <b>Dışa aktar</b>.
Renkler: <span class="tag" style="color:var(--ok)">etiketli</span> <span class="tag" style="color:var(--acc)">vurgusuz</span> <span class="tag" style="color:var(--warn)">emin değil</span></p>
</main>
<script>
const DATA = __DATA__;
const KEY = "__KEY__";
let st = {}; try { st = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const lo = s => s.toLocaleLowerCase("tr");
let cur = 0;
const $ = id => document.getElementById(id);
function rec(i) { return st[DATA[i].w] || (st[DATA[i].w] = {}); }
function status(r) { return r.k != null ? "s" : r.none ? "v" : r.unsure ? "u" : ""; }
function render() {
  const d = DATA[cur], r = st[d.w] || {};
  $("pos").textContent = `${cur + 1} / ${DATA.length}`;
  $("freq").textContent = `derlemde ${d.f} kez` + (d.cap ? " · cümle içinde büyük harfle de geçiyor" : "");
  const box = $("syl"); box.innerHTML = "";
  d.s.forEach((s, j) => { const b = document.createElement("button"); b.innerHTML = `<small>${j + 1}</small>${s}`;
    b.className = r.k === j ? "on" : ""; b.onclick = () => pick(j); box.appendChild(b); });
  const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  const m = new RegExp(`(^|[^\\p{L}])${d.w}(?=$|[^\\p{L}])`, "u").exec(lo(d.ex));  // tr küçük harf uzunluğu korur (İ -> i)
  const at = m ? m.index + m[1].length : -1;
  $("ex").innerHTML = at < 0 ? esc(d.ex) : esc(d.ex.slice(0, at)) + "<b>" + esc(d.ex.slice(at, at + d.w.length)) + "</b>" + esc(d.ex.slice(at + d.w.length));
  $("b-none").className = r.none ? "on" : ""; $("b-unsure").className = r.unsure ? "on" : "";
  $("note").value = r.note || "";
  const done = DATA.filter(x => status(st[x.w] || {})).length;
  $("done").textContent = done; $("total").textContent = DATA.length; $("prog").style.width = (100 * done / DATA.length) + "%";
  [...$("grid").children].forEach((c, i) => c.className = status(st[DATA[i].w] || {}) + (i === cur ? " cur" : ""));
}
function go(i) { cur = Math.max(0, Math.min(DATA.length - 1, i)); render(); }
function pick(j) { const r = rec(cur); r.k = j; delete r.none; delete r.unsure; save(); go(cur + 1); }
function flag(f) { const r = rec(cur); delete r.k; delete r.none; delete r.unsure; r[f] = true; save(); go(cur + 1); }
$("b-none").onclick = () => flag("none"); $("b-unsure").onclick = () => flag("unsure");
$("prev").onclick = () => go(cur - 1); $("next").onclick = () => go(cur + 1);
$("note").oninput = e => { rec(cur).note = e.target.value.replace(/[\t\n]/g, " "); save(); };
$("next-empty").onclick = () => { const i = DATA.findIndex(x => !status(st[x.w] || {})); go(i < 0 ? 0 : i); };
DATA.forEach((d, i) => { const s = document.createElement("span"); s.title = d.w; s.onclick = () => go(i); $("grid").appendChild(s); });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (/^[1-9]$/.test(e.key) && +e.key <= DATA[cur].s.length) pick(+e.key - 1);
  else if (e.key === "0") flag("none"); else if (e.key === "?") flag("unsure");
  else if (e.key === "ArrowRight") go(cur + 1); else if (e.key === "ArrowLeft") go(cur - 1);
});
$("export").onclick = () => {
  const lines = ["# sözcük\tvurgulu hecenin SONDAN sırası (0 = son; - = vurgusuz)\tkategori\tnot",
    `# Kaynak: kullanıcının KÖR etiketi (model önerisi gösterilmedi), ${new Date().toISOString().slice(0, 10)}; "emin değilim" ve boş satırlar dahil edilmedi.`];
  let n = 0;
  DATA.forEach(d => { const r = st[d.w] || {};
    if (r.k != null) { lines.push(`${d.w}\t${d.s.length - 1 - r.k}\trastgele${d.cap ? " (büyük harfle de geçiyor)" : ""}\t${r.note || ""}`); n++; }
    else if (r.none) { lines.push(`${d.w}\t-\trastgele\t${r.note || ""}`); n++; } });
  const unsure = DATA.filter(d => (st[d.w] || {}).unsure).map(d => d.w);
  if (unsure.length) lines.push(`# emin değil (${unsure.length}): ${unsure.join(", ")}`);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/tab-separated-values"}));
  a.download = "__EXPORT__"; a.click();
  alert(`${n} satır dışa aktarıldı. Dosyayı dizgetts/tests/ altına koyun (__EXPORT__).`);
};
render();
</script></body></html>"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--export-name", default="stress_gold_random.tsv", help="dışa aktarılan dosya (tarayıcı kaydı da bu ada göre ayrılır)")
    ap.add_argument("--title", default="Vurgu etiketleme")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.src, encoding="utf8"), delimiter="\t"))
    data = [dict(w=r["sözcük"], f=int(r["sıklık"]), cap=bool(r["özel_ad_olası"]), ex=r["örnek_cümle"],
                 s=[p.replace("İ", "i").replace("I", "ı").lower() for p in r["model_hece"].split("-")]) for r in rows]
    key = "dizgetts-vurgu-etiket-v1" if args.export_name == "stress_gold_random.tsv" else "dizgetts-vurgu-" + Path(args.export_name).stem
    html = (HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__KEY__", key)
            .replace("__EXPORT__", args.export_name).replace("__TITLE__", args.title))
    Path(args.out).write_text(html, encoding="utf8")
    print(len(data), "sözcük ->", args.out)


if __name__ == "__main__":
    main()
