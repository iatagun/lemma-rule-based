"""Sesbirim (g2ptts v1 / Engine özenli: dizge + söyleyiş sözlüğü + alıntı kökleri + şapka + uzun ünlü kuralları) için değerlendirme seti + etiketleme sayfası.
  python -X utf8 -m dizgetts.scripts.make_phoneme_eval        -> reports/phoneme_eval_sheet.tsv + reports/phoneme_eval.html (tarayıcıda aç, çevrimdışı)
  python -X utf8 -m dizgetts.scripts.make_phoneme_eval --score dizgetts/tests/phoneme_gold_dev.tsv   -> katman ve kural kaynağına göre doğruluk (set 1 = geliştirme)
  --tag 2 --seed N --exclude reports/phoneme_eval_sheet.tsv  -> YENİ kör set (önceki setin sözcükleri dışarıda; set 1 kuralları düzeltmek için kullanıldı = geliştirme seti)

Sözcük havuzu: UD Türkçe treebank'leri + Antalia metni (gerçek Türkçe; altyazı listesindeki ASCII gürültüsü dışarıda). Sabit tohum. Katmanlar:
  R  rastgele (geçiş sıklığıyla ağırlıklı)            -> genel doğruluk
  G  ğ içeren                                        -> uzun ünlü / geçiş kuralları
  A  alıntı adayı, kurallarımız DOKUNMADI (kök içi ünlü uyumu bozuk) -> kapsam dışında kalan alıntılar
  K  kurallarımızın DEĞİŞTİRDİĞİ biçim               -> sözlük/alıntı kurallarının görülmemiş biçimlere genellemesi
  B  sözcük başı ünsüz öbeği (Batı alıntısı)          -> türeme ünlüsü
Sayfa katmanı ve kural kaynağını GÖSTERMEZ (sayfa düzeni karışık); yargı sesbilgisel olarak anlamlı ayrımlar üzerinedir (açıklama sayfada).
Çapalama notu: çıktı gösterilerek yargılanır (kör transkripsiyon IPA'da çok emek ister); bu yüzden "yanlış" oranı alt sınırdır.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import random
import re
import warnings
from pathlib import Path

from dizgetts import paths
from dizgetts.frontend.normalize import tr_lower
from dizgetts.frontend.phonemize import Phonemizer
from dizgetts.frontend.symbols import PHONES, tokenize

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
QUOTA = {"R": 50, "G": 15, "A": 25, "K": 20, "B": 10}
V, BACK, FRONT = set("aeıioöuüâîû"), set("aıou"), set("eiöü")
WORD = re.compile(r"[a-zçğıöşüâîû]+")


def sheet_paths(tag: str = "") -> tuple[Path, Path]:
    sfx = f"_{tag}" if tag else ""
    return HERE / "reports" / f"phoneme_eval{sfx}_sheet.tsv", HERE / "reports" / f"phoneme_eval{sfx}.html"


def vocab() -> collections.Counter:
    c = collections.Counter()
    for f in glob.glob(str(REPO / "data" / "treebanks" / "**" / "*.conllu"), recursive=True):
        for line in open(f, encoding="utf8"):
            col = line.split("\t")
            if len(col) > 2 and col[0].isdigit() and col[3] != "PROPN":
                c[tr_lower(col[1])] += 1
    for sp in ("train", "val", "test"):
        p = Path(paths.ANTALIA) / f"{sp}.jsonl"
        if p.exists():
            for line in open(p, encoding="utf8"):
                c.update(WORD.findall(tr_lower(json.loads(line)["text"])))
    return collections.Counter({w: n for w, n in c.items() if WORD.fullmatch(w) and len(w) > 1})


def build(seed: int = 20260929, exclude: set[str] = frozenset()) -> list[dict]:
    warnings.filterwarnings("ignore")
    voc = vocab()
    bare = Phonemizer(bert_fallback=False)
    full = Phonemizer(bert_fallback=False, pron_exceptions=True, length_rules=True)
    rnd = random.Random(seed)
    words = sorted(voc)
    rnd.shuffle(words)

    def disharmonic(w):
        vs = [c for c in w if c in V]
        return len(vs) >= 2 and set(vs) & BACK and set(vs) & FRONT

    def stratum(w):
        changed_by_rules = full.word(w) != Phonemizer(bert_fallback=False, length_rules=True).word(w)
        if changed_by_rules:
            return "K"
        if w[0] not in V and len(w) > 3 and w[1] not in V and w[1] not in "ğy":
            return "B"
        if "ğ" in w:
            return "G"
        if disharmonic(w) and voc[w] >= 2:
            return "A"
        return None

    picked, seen = [], set(exclude)
    # R: geçiş sıklığıyla ağırlıklı çekiliş (tekrarsız)
    pool, weights = words, [voc[w] for w in words]
    while sum(r["stratum"] == "R" for r in picked) < QUOTA["R"]:
        w = rnd.choices(pool, weights)[0]
        if w not in seen and full.word(w):
            seen.add(w)
            picked.append(dict(word=w, stratum="R"))
    for s in ("G", "A", "K", "B"):
        n = 0
        for w in words:
            if n == QUOTA[s]:
                break
            if w in seen or voc[w] < 2 or not full.word(w):
                continue
            if stratum(w) == s:
                seen.add(w)
                picked.append(dict(word=w, stratum=s))
                n += 1
    rnd.shuffle(picked)
    exc_only = Phonemizer(bert_fallback=False, pron_exceptions=True)
    for i, r in enumerate(picked, 1):
        w = r["word"]
        r["id"] = f"s{i:03d}"
        r["phones"] = " ".join(tokenize(full.word(w), strict=False))
        layers = [name for name, a, b in (("sözlük/alıntı/şapka", bare.word(w), exc_only.word(w)),
                                          ("uzun_ünlü", exc_only.word(w), full.word(w))) if a != b]
        r["source"] = "+".join(layers) or "dizge"
        r["freq"] = voc[w]
    return picked


HTML = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sesbirim Değerlendirmesi __TAG__</title>
<style>
:root{--bg:#f7f6f2;--card:#fff;--ink:#1d1d1b;--mute:#6b6a64;--line:#dedcd4;--acc:#1f5f8b;--acc-ink:#fff;--ok:#2e7d4f;--bad:#b23b3b;--warn:#a86b00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#212120;--ink:#ecebe6;--mute:#a3a29b;--line:#3a3935;--acc:#6aa9d8;--acc-ink:#0d1b26;--ok:#6cc08f;--bad:#e07a7a;--warn:#e0a84a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 48px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;margin-bottom:16px}
h1{font-size:18px;margin:0}.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:6px;width:220px}.bar i{display:block;height:100%;background:var(--ok)}
.mute{color:var(--mute);font-size:14px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:8px 14px;cursor:pointer}
button:hover{border-color:var(--acc)}button:focus-visible,input:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px 20px}
.word{font-size:34px;font-weight:600;margin:6px 0 2px}
.ph{font:26px/1.6 "Charis SIL","Doulos SIL","Noto Serif","Times New Roman",serif;letter-spacing:.02em;display:flex;flex-wrap:wrap;gap:4px 6px;margin:10px 0 4px}
.ph span{border-bottom:2px solid var(--line);padding:0 2px;cursor:help}
.gloss{min-height:1.5em}
.acts{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}.acts button{flex:1 1 150px}
.sel-ok{background:var(--ok);color:#fff;border-color:var(--ok)}.sel-bad{background:var(--bad);color:#fff;border-color:var(--bad)}.sel-unk{background:var(--warn);color:#fff;border-color:var(--warn)}
input[type=text]{width:100%;font:inherit;padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);margin-top:12px}
.nav{display:flex;justify-content:space-between;gap:8px;margin-top:16px}
details{margin-top:18px}summary{cursor:pointer;color:var(--mute)}
table{border-collapse:collapse;font-size:14px;margin-top:8px}td{padding:2px 10px 2px 0;vertical-align:top}td:first-child{font:18px serif;white-space:nowrap}
</style></head><body><main>
<header><div><h1>Sesbirim değerlendirmesi</h1><div class="mute" id="prog"></div><div class="bar"><i id="barfill"></i></div></div>
<div><button id="exp">Dışa aktar (TSV)</button></div></header>
<div class="card">
 <div class="meta mute" id="meta"></div>
 <div class="word" id="w"></div>
 <div class="ph" id="ph" aria-label="sesbirimler"></div>
 <div class="gloss mute" id="gloss">Bir simgenin üstüne gelin (ya da dokunun): anlamı burada görünür.</div>
 <div class="acts">
  <button id="bok" title="1">Doğru (1)</button><button id="bbad" title="2">Yanlış (2)</button><button id="bunk" title="3">Emin değilim (3)</button>
 </div>
 <input type="text" id="note" placeholder="Yanlışsa: hangi ses, doğrusu ne? (ör. 'a kalın olmalı', 'l ince', 'uzun değil')">
 <div class="nav"><button id="prev">← Önceki</button><button id="next">Sonraki →</button></div>
</div>
<details><summary>Ne yargılanıyor?</summary><p class="mute">Yalnız sesbilgisel olarak anlamlı ayrımlar: ünlü niteliği (kalın ɑ / ön a / e, o / œ, u / ü benzeri Y), uzunluk (ː), ince/kalın ünsüz (k/c, g/ɟ, l/ł),
ğ'nin davranışı (uzatma, geçiş, y), türeme ünlüsü (sıpor), eksik/fazla ses. Özenli söyleyiş. Aspirasyon (ʰ), sözcük sonu r (ɣ), ön/arka h (x/ç), gevşek ünlü (I, U, Y) dizge'nin
alofon ayrıntılarıdır: yalnız açıkça yanlışsa işaretleyin.</p></details>
<details><summary>Simge tablosu</summary><table id="leg"></table></details>
</main>
<script>
const ITEMS=__ITEMS__, LEG=__LEGEND__, KEY="phoneme_eval_v1__TAG__";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0; try{i=+localStorage.getItem(KEY+"_i")||0}catch(e){}
const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(st));localStorage.setItem(KEY+"_i",i)}catch(e){}}
function render(){
 const it=ITEMS[i], a=st[it.id]||{};
 $("meta").textContent=`${i+1} / ${ITEMS.length}`;
 $("w").textContent=it.word;
 $("ph").innerHTML="";
 it.phones.split(" ").forEach(p=>{const s=document.createElement("span");s.textContent=p;s.tabIndex=0;
  const show=()=>{$("gloss").textContent=`${p}: ${LEG[p]||""}`};s.onmouseenter=show;s.onfocus=show;s.onclick=show;$("ph").appendChild(s)});
 $("bok").className=a.v==="doğru"?"sel-ok":"";$("bbad").className=a.v==="yanlış"?"sel-bad":"";$("bunk").className=a.v==="emin_değil"?"sel-unk":"";
 $("note").value=a.note||"";
 const done=ITEMS.filter(x=>st[x.id]&&st[x.id].v).length;
 $("prog").textContent=`${done} / ${ITEMS.length} işaretlendi`;$("barfill").style.width=(100*done/ITEMS.length)+"%";
}
function mark(v){const it=ITEMS[i];st[it.id]=Object.assign(st[it.id]||{},{v});save();if(v==="doğru"&&i<ITEMS.length-1){i++;save()}render();if(v!=="doğru")$("note").focus()}
$("bok").onclick=()=>mark("doğru");$("bbad").onclick=()=>mark("yanlış");$("bunk").onclick=()=>mark("emin_değil");
$("note").oninput=e=>{const it=ITEMS[i];st[it.id]=Object.assign(st[it.id]||{},{note:e.target.value});save()};
$("prev").onclick=()=>{if(i>0){i--;save();render()}};$("next").onclick=()=>{if(i<ITEMS.length-1){i++;save();render()}};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"){if(e.key==="Enter"){$("next").click()}return}
 if(e.key==="1")mark("doğru");else if(e.key==="2")mark("yanlış");else if(e.key==="3")mark("emin_değil");else if(e.key==="ArrowRight")$("next").click();else if(e.key==="ArrowLeft")$("prev").click()});
$("exp").onclick=()=>{const rows=["id\tsözcük\tsesbirim\tyargı\tnot"].concat(ITEMS.map(x=>{const a=st[x.id]||{};return [x.id,x.word,x.phones,a.v||"",(a.note||"").replace(/\t|\n/g," ")].join("\t")}));
 const b=new Blob([rows.join("\n")+"\n"],{type:"text/tab-separated-values"});const u=URL.createObjectURL(b);const l=document.createElement("a");l.href=u;l.download="phoneme_gold__TAG__.tsv";l.click();URL.revokeObjectURL(u)};
const used=new Set(ITEMS.flatMap(x=>x.phones.split(" ")));
$("leg").innerHTML=[...used].sort().map(p=>`<tr><td>${p}</td><td>${LEG[p]||""}</td></tr>`).join("");
render();
</script></body></html>"""


def write(rows: list[dict], tag: str) -> None:
    SHEET, OUT = sheet_paths(tag)
    with open(SHEET, "w", encoding="utf8", newline="") as f:
        wr = csv.DictWriter(f, ["id", "word", "stratum", "source", "freq", "phones"], delimiter="\t")
        wr.writeheader()
        wr.writerows(rows)
    legend = {a: note for a, (_, note) in PHONES.items()}
    legend.update({"a": "ön a (ae karışımı): kağıt, hal, normal", "aː": "uzun ön a: saat, zaten, hakim", "ø": "ö", "y": "ü", "j": "y (ünsüz)",
                   "œ": "yalnız alıntılarda 'oe' karışımı (rol, kontrol)", "Y": "yalnız alıntılarda 'ü benzeri u' (kabul, mahsul)", "ɑ": "a (kalın, normal)", "ɑː": "uzun kalın a (dağ)", "l": "ince l", "ł": "kalın l",
                   "c": "ince k", "cʰ": "ince k (aspire)", "ɟ": "ince g", "ː": "uzunluk"})
    items = [{k: r[k] for k in ("id", "word", "phones")} for r in rows]  # katman/kaynak sayfada YOK
    OUT.write_text(HTML.replace("__TAG__", tag).replace("__ITEMS__", json.dumps(items, ensure_ascii=False)).replace("__LEGEND__", json.dumps(legend, ensure_ascii=False)), encoding="utf8")
    print(f"{len(rows)} sözcük -> {SHEET.name}, {OUT}")
    print(collections.Counter(r["stratum"] for r in rows), collections.Counter(r["source"] for r in rows))


def score(gold: str, tag: str) -> None:
    meta = {r["id"]: r for r in csv.DictReader(open(sheet_paths(tag)[0], encoding="utf8"), delimiter="\t")}
    g = [r for r in csv.DictReader(open(gold, encoding="utf8"), delimiter="\t") if r["yargı"]]
    for key in ("stratum", "source"):
        by = collections.defaultdict(collections.Counter)
        for r in g:
            by[meta[r["id"]][key]][r["yargı"]] += 1
            by["TÜMÜ"][r["yargı"]] += 1 if key == "stratum" else 0
        print(f"== {key}")
        for k, c in sorted(by.items()):
            n = c["doğru"] + c["yanlış"]
            if n:
                print(f"  {k:24} doğru {c['doğru']}/{n} = {100 * c['doğru'] / n:.1f}%  (emin değil {c['emin_değil']})")
    print("== yanlışlar")
    for r in g:
        if r["yargı"] == "yanlış":
            m = meta[r["id"]]
            print(f"  [{m['stratum']}/{m['source']}] {r['sözcük']}: {r['sesbirim']}  ->  {r['not']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score")
    ap.add_argument("--tag", default="", help="set etiketi (dosya adı ve tarayıcı kaydı); boş = ilk set")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--exclude", nargs="*", default=[], help="bu sayfalardaki sözcükler sete girmez")
    a = ap.parse_args()
    excl = {r["word"] for f in a.exclude for r in csv.DictReader(open(f, encoding="utf8"), delimiter="	")}
    score(a.score, a.tag) if a.score else write(build(a.seed, excl), a.tag)
