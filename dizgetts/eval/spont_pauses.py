"""Doğal konuşmada (araştırma görüşmeleri) sözcük sınırı duraklamaları: Antalia (okuma) ile AYNI ölçümle kıyas (docs/prosody_research.md §9).
Soru: sınır kurallarının (frontend/boundary_rules.py K1-K5) işaret ettiği yerlerde konuşmacı gerçekten duruyor mu?

Aşamalar (her biri diske yazar; yeniden çalıştırınca var olanı atlar):
  asr    m4a -> 16 kHz wav; faster-whisper large-v3-turbo -> sözcük + zaman + güven          (HOME/data/processed/gorusme/<ad>/asr.jsonl)
  spk    WavLM x-vector, 1,5 sn pencere, k=2 -> pencere başına konuşmacı                      (.../spk.json)
  align  metin -> normalize -> MMS_FA zorlamalı hizalama (<= 25 sn parçalar, genel zaman ekseni) (.../words.jsonl)
  table  sınır tablosu: her sözcük arası konum için sessizlik (ms), konuşmacı, süzgeç bayrakları                 (.../boundaries.jsonl)
  antalia --variant gold | asr | deg:<kayıt adının başı>   Antalia kliplerini AYNI hizalama + AYNI sessizlik ölçümünden geçirir: altın metin (tüm klipler),
         Whisper metni (test+val), görüşme koşullarına bozulmuş ses (test+val; `table` sonrası)                  (HOME/.../gorusme/antalia_<variant>.jsonl)
  validate  ölçünün MAS tabanlı bağımsız ölçümle (breaks.jsonl) uyumu, temiz ve bozulmuş seste; eşik seçimi buradan  (.../validate.json)
  report [--key sil25] [--p-min 0.5] [--inner]   konum türüne göre duraklama oranı, küme bootstrap GA, konum - denetim farkı (.../report_*.json)

  python -X utf8 -m dizgetts.eval.spont_pauses <aşama> [--src "D:/dizgetts/data/raw/*.m4a"]        sıra: asr spk align table antalia(x4) validate report
Kişi adları yalnız yerel dosya adlarında kalır; rapora G1/G2/G3 olarak geçer. Kayıtlar eğitimde KULLANILMAZ, yerelden çıkmaz.
"""
import argparse, glob, json, os, re, subprocess

import numpy as np
import soundfile as sf

from dizgetts.paths import ANTALIA, HOME

OUT = f"{HOME}/data/processed/gorusme"
SR = 16000


def units(src):
    """-> [(ad, m4a yolu, çıktı klasörü)]"""
    out = []
    for p in sorted(glob.glob(src)):
        name = os.path.splitext(os.path.basename(p))[0]
        d = f"{OUT}/{name}"
        os.makedirs(d, exist_ok=True)
        out.append((name, p, d))
    return out


def wav16(src, dst):
    if not os.path.exists(dst):
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(SR), dst], check=True)
    return dst


def whisper():
    from faster_whisper import WhisperModel
    for ct in ("int8_float32", "float32", "int8"):  # GTX 1650: fp16 kullanma
        try:
            return WhisperModel("large-v3-turbo", device="cuda", compute_type=ct, download_root=f"{HOME}/cache/fw")
        except ValueError:
            continue
    raise RuntimeError("faster-whisper CUDA'da açılamadı")


def transcribe(model, x, vad=True):
    """-> sözcük listesi [{s, e, w (noktalama yapışık), p}]; bağlam taşınmaz (uydurma döngüsü olmasın).
    Görüşmelerde vad=False: Silero VAD mikrofona uzak (10-14 dB kısık) konuşmacının dakikalarca süren yanıtlarını konuşma saymıyor (bir kayıtta 57 dk'nın
    yalnız 16'sı yazıya dökülmüştü). Uydurma riski sözcük güveni + hizalama puanı + sessizlik ölçümüyle aşağıda süzülür."""
    segs, _ = model.transcribe(x, language="tr", beam_size=5, vad_filter=vad, word_timestamps=True, condition_on_previous_text=False)
    return [dict(s=round(w.start, 3), e=round(w.end, 3), w=w.word.strip(), p=round(w.probability, 3), seg=i, nsp=round(s.no_speech_prob, 3),
                 lp=round(s.avg_logprob, 3)) for i, s in enumerate(segs) for w in s.words or []]


def dump(path, rows):
    with open(path, "w", encoding="utf8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf8")]


def stage_asr(a):
    model = whisper()
    for name, src, d in units(a.src):
        if os.path.exists(f"{d}/asr.jsonl"):
            continue
        x, _ = sf.read(wav16(src, f"{d}/audio16k.wav"), dtype="float32")
        dump(f"{d}/asr.jsonl", transcribe(model, x, vad=False))
        print(f"asr {name}: {len(x) / SR / 60:.1f} dk", flush=True)
    out = f"{OUT}/antalia_whisper.jsonl"  # ASR metninin ölçüme etkisini sınamak için: Antalia test + val
    if not os.path.exists(out):
        import librosa
        rows = []
        for split in ("test", "val"):
            for r in load(f"{ANTALIA}/{split}.jsonl"):
                x, _ = librosa.load(f"{ANTALIA}/{r['wav']}", sr=SR)
                rows.append(dict(id=r["id"], split=split, words=transcribe(model, x)))
        dump(out, rows)
        print("asr antalia test+val", len(rows), flush=True)


# ---------------------------------------------------------------- konuşmacı
def stage_spk(a):
    import torch
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from transformers import AutoFeatureExtractor, WavLMForXVector
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv", cache_dir=f"{HOME}/cache/hf")
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv", cache_dir=f"{HOME}/cache/hf").to(dev).eval()
    for name, _, d in units(a.src):
        if os.path.exists(f"{d}/spk.json"):
            continue
        x, _ = sf.read(f"{d}/audio16k.wav", dtype="float32")
        reg = []  # konuşma bölgeleri: ASR sözcükleri, 0,3 sn'den kısa boşluklar kapalı
        for w in load(f"{d}/asr.jsonl"):
            if reg and w["s"] - reg[-1][1] < 0.3:
                reg[-1][1] = max(reg[-1][1], w["e"])
            else:
                reg.append([w["s"], w["e"]])
        wins = []
        for s, e in reg:  # 1,5 sn pencere, 0,5 sn adım; kısa bölge: ortalanmış tek pencere
            wins += [(t, t + 1.5) for t in np.arange(s, e - 1.5, 0.5)] + [(e - 1.5, e)] if e - s >= 1.5 else [(max(0, (s + e) / 2 - 0.75), (s + e) / 2 + 0.75)]
        E = []
        with torch.no_grad():
            for i in range(0, len(wins), 16):
                b = fe([x[int(p * SR):int(q * SR)] for p, q in wins[i:i + 16]], sampling_rate=SR, return_tensors="pt", padding=True).to(dev)
                E.append(torch.nn.functional.normalize(sv(**b).embeddings, dim=-1).cpu().numpy())
        E = np.concatenate(E)
        km = KMeans(2, n_init=10, random_state=0).fit(E)
        c = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        sim = E @ c.T
        sub = np.random.default_rng(0).choice(len(E), min(len(E), 3000), replace=False)
        sil = {k: round(float(silhouette_score(E[sub], KMeans(k, n_init=10, random_state=0).fit_predict(E[sub]), metric="cosine")), 3) for k in (2, 3)}
        json.dump(dict(wins=[(round(float(p), 2), round(float(q), 2)) for p, q in wins], margin=[round(float(v), 3) for v in sim[:, 1] - sim[:, 0]],
                       silhouette=sil), open(f"{d}/spk.json", "w"))
        print(f"spk {name}: {len(wins)} pencere, silhouette {sil}, küme-1 payı {float((sim[:, 1] > sim[:, 0]).mean()):.2f}", flush=True)


def word_speakers(d, words):
    """Her sözcük için (konuşmacı 0/1, |ortalama marj|): orta noktasını kapsayan pencerelerin işaretli marj ortalaması (>0 = küme 1)."""
    S = json.load(open(f"{d}/spk.json"))
    ws, m = np.array(S["wins"]), np.array(S["margin"])
    order = np.argsort(ws[:, 0], kind="stable")
    ws, m = ws[order], m[order]
    out = []
    for w in words:
        mid = (w["s"] + w["e"]) / 2
        i0, i1 = np.searchsorted(ws[:, 0], mid - 1.5), np.searchsorted(ws[:, 0], mid, side="right")
        sel = [j for j in range(i0, i1) if ws[j, 1] >= mid] or [int(np.argmin(np.abs((ws[:, 0] + ws[:, 1]) / 2 - mid)))]  # kapsayan yoksa en yakın pencere
        v = float(np.mean(m[sel]))
        out.append((int(v > 0), abs(v)))
    return out


# ---------------------------------------------------------------- hizalama
def text_words(text):
    """metin -> [[sözcük, sonrasındaki noktalama]] (engine ile aynı normalize + belirteçleme)"""
    from dizgetts.engine import _TOK
    from dizgetts.frontend.normalize import normalize
    out = []
    for t in _TOK.findall(normalize(text)):
        if t in ",.?!;":
            if out:
                out[-1][1] += t
        else:
            out.append([t, ""])
    return out


def asr_words(ws):
    """ASR sözcükleri -> [(sözcük, noktalama, ASR sözcüğü)]: sayı vb. açılımı sözcük sözcük (alt sözcükler ASR güvenini miras alır).
    normalize() her girdinin sonuna cümle noktası ekler; burada yalnız ASR sözcüğünün KENDİ noktalaması son alt sözcükte kalır."""
    out = []
    for w in ws:
        tw = text_words(w["w"])
        for i, (t, _) in enumerate(tw):
            out.append((t, "".join(c for c in w["w"] if c in ",.?!;:…").replace(":", ",").replace("…", ".") if i == len(tw) - 1 else "", w))
    return out


class FA:
    """torchaudio MMS_FA: (16 kHz dalga, sözcükler) -> sözcük başına (başlangıç sn, bitiş sn, ortalama simge puanı) ya da None (hizalanamayan)."""

    def __init__(self):
        import torch, torchaudio
        from dizgetts.tools.make_textgrid import words_of
        b = torchaudio.pipelines.MMS_FA
        self.torch, self.key = torch, lambda w: (words_of(w) or [("", "")])[0][1]
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok, self.align = b.get_model().to(self.dev).eval(), b.get_tokenizer(), b.get_aligner()

    def __call__(self, x, words):
        keys = [self.key(w) for w in words]
        keep = [i for i, k in enumerate(keys) if k]
        with self.torch.inference_mode():
            em, _ = self.model(self.torch.from_numpy(x)[None].to(self.dev))
            spans = self.align(em[0], self.tok([keys[i] for i in keep]))
        r = len(x) / em.shape[1] / SR
        out = [None] * len(words)
        for i, s in zip(keep, spans):
            out[i] = (s[0].start * r, s[-1].end * r, float(np.mean([t.score for t in s])))
        return out


def chunks(ws, soft=15.0, hard=25.0, gap=2.0):
    """ASR sözcüklerini hizalama parçalarına böl: >= 2 sn boşlukta; 15 sn'yi geçince ilk >= 0,2 sn boşlukta; en geç 25 sn'de."""
    cur = [ws[0]]
    for w in ws[1:]:
        g, dur = w["s"] - cur[-1]["e"], w["e"] - cur[0]["s"]
        if g > gap or dur > hard or (cur[-1]["e"] - cur[0]["s"] > soft and g >= 0.2):
            yield cur
            cur = []
        cur.append(w)
    yield cur


def stage_align(a):
    fa = FA()
    for name, _, d in units(a.src):
        if os.path.exists(f"{d}/words.jsonl"):
            continue
        x, _ = sf.read(f"{d}/audio16k.wav", dtype="float32")
        asr = [w for w in load(f"{d}/asr.jsonl") if w["e"] > w["s"]]
        rows, cs = [], list(chunks(asr))
        for ci, c in enumerate(cs):
            gp = c[0]["s"] - cs[ci - 1][-1]["e"] if ci else 1.0
            gn = cs[ci + 1][0]["s"] - c[-1]["e"] if ci + 1 < len(cs) else 1.0
            t0 = max(0.0, c[0]["s"] - min(0.5, max(0.1, gp / 2)))  # Whisper zamanları kaba: kenarda pay bırak, komşu parçaya en çok yarı boşluk kadar gir
            t1 = min(len(x) / SR, c[-1]["e"] + min(0.5, max(0.1, gn / 2)))
            ws = asr_words(c)
            if not ws:
                continue
            al = fa(x[int(t0 * SR):int(t1 * SR)], [t for t, _, _ in ws])
            for k, ((t, p, w), sp) in enumerate(zip(ws, al)):
                if sp:
                    rows.append(dict(w=t, punct=p, s=round(t0 + sp[0], 3), e=round(t0 + sp[1], 3), sc=round(sp[2], 3), p=w["p"], chunk=ci,
                                     edge=k == 0 or k == len(ws) - 1, asr_s=w["s"], asr_e=w["e"]))
        dump(f"{d}/words.jsonl", rows)
        print(f"align {name}: {len(rows)} sözcük, {len(cs)} parça", flush=True)


# ---------------------------------------------------------------- sessizlik ölçümü
def band_db(x, hop=0.010):
    """300-3000 Hz bant enerjisi (dB), 10 ms kare. Tam bant enerjisi görüşme kayıtlarında işe yaramaz (alçak frekans gürültüsü sessizliği örter)."""
    from scipy.signal import butter, sosfiltfilt
    h = int(SR * hop)
    y = sosfiltfilt(butter(4, [300, 3000], btype="band", fs=SR, output="sos"), x)
    n = len(y) // h
    return 10 * np.log10(np.maximum((y[: n * h].reshape(n, h) ** 2).mean(1), 1e-12))


THR = (15, 20, 25, 30)  # dB; başvuru düzeyi = konuşmacının SÖZCÜK İÇİ karelerinin %95'liği (duraklama payından bağımsız)


def ref_level(e, words):
    f = np.concatenate([e[int(w["s"] * 100):int(w["e"] * 100) + 1] for w in words]) if words else e
    return float(np.percentile(f if len(f) else e, 95))


def silences(e, a, b, ref):
    """a sözcüğünün sonu .. b sözcüğünün başı bölgesinde eşik altı karelerin toplamı (ms); `p` = +-40 ms paylı bölge (hizalayıcı sözcüğü sessizliğe taşırdıysa)."""
    i0, i1 = int(round(a["e"] * 100)), int(round(b["s"] * 100))
    seg, segp = e[i0:max(i0, i1)], e[max(0, i0 - 4):max(i0, i1) + 4]
    out = dict(region_ms=int(max(0, i1 - i0) * 10))
    for T in THR:
        out[f"sil{T}"], out[f"sil{T}p"] = int((seg < ref - T).sum() * 10), int((segp < ref - T).sum() * 10)
    return out


# K3 örüntüsünün yanlış yakaladıkları (Antalia'daki 258 tür + görüşmelerdeki 124 örnek tek tek okunarak): -An ortacı (gerek-en), ad + -DAn/-CA, özel ad.
# Antalia'da 12/678 (%2), görüşmelerde 20/124 (%16). boundary_rules.CONVERB_NOT'a eklenmeleri gerekir (kural açılırsa); burada yalnız ölçümden çıkarılır.
K3_NOT = {"gereken", "döken", "çöken", "mesken", "kümeden", "karınca", "gönlünce", "yeterince", "olabildiğince", "sakınca", "romadan", "narinca"}


def site_tags(ws):
    """ws: [(sözcük, noktalama)] -> her sözcükten SONRAKİ sınır için konum etiketleri (boundary_rules K1-K5 ile aynı yüzey örüntüleri)."""
    from dizgetts.frontend import boundary_rules as B
    from dizgetts.frontend.normalize import tr_lower
    t, n = [tr_lower(w) for w, _ in ws], len(ws)
    starts = [i == 0 or any(c in ".?!" for c in ws[i - 1][1]) for i in range(n)]
    tags = [set() for _ in range(n)]
    for i in range(n):
        if starts[i]:
            for seq in B.DISCOURSE_IP | B.DISCOURSE_ip:
                if tuple(t[i:i + len(seq)]) == seq and i + len(seq) < n:
                    tags[i + len(seq) - 1].add("K1")
    for i in range(1, n):
        if t[i] in B.CONJ and not starts[i] and (t[i] != "ya" or (i + 1 < n and t[i + 1] == "da")):
            if B.QPART.match(t[i - 1]) or B.FINITE.search(t[i - 1]) or ws[i - 1][1]:
                tags[i - 1].add("K2")
                last = i + 1 if t[i] == "ya" else i
                if last < n - 1:
                    tags[last].add("K2_sonra")
        # noktalamadan BAĞIMSIZ bağlaç konumları: Whisper duraklamayı nokta yapıp bağlacı cümle başına atar ("... gelmişti. Ama ..."), K2 bunları saymaz
        if t[i] in B.CONJ and (t[i] != "ya" or (i + 1 < n and t[i + 1] == "da")):
            tags[i - 1].add("bağ_önce")
            if i + (t[i] == "ya") < n - 1:
                tags[i + (t[i] == "ya")].add("bağ_sonra")
        if t[i] == "ve":
            tags[i - 1].add("ve_önce")
            if i < n - 1:
                tags[i].add("ve_sonra")
    since = 0
    for i in range(n - 1):
        since += 1
        if B.CONVERB.search(t[i]) and t[i] not in B.CONVERB_NOT and len(t[i]) > 4:
            tags[i].add("K3_dışlanan")  # örüntü tuttu; aşağıda K3 sayılmazsa yalnız bu etiket kalır (yanlış yakalananlar + "-mAdAn önce")
            if t[i] not in K3_NOT and t[i + 1] not in ("önce", "sonra"):  # "yemeden önce": öbek içi
                tags[i] |= {"K3", "K3_uzun" if since >= 3 else "K3_kısa", "K3_koşul" if re.search(r"s[ae](m|n|k|n[ıiuü]z)?$", t[i]) else "K3_zarf"}
        if t[i] == "ise" and since >= 3:
            tags[i].add("K4")
        if ws[i][1]:
            since = 0
        if t[i] in B.NUMS and t[i + 1] in B.NUMS:
            tags[i].add("K5_sayi")
        if t[i + 1] == "için" and re.search(r"m[ae]k$", t[i]):
            tags[i].add("K5_icin")
        if B.GEN.search(t[i]) and t[i] not in B.GEN_NOT and len(t[i]) > 4:
            tags[i].add("K5_tamlayan")
    return tags


def stage_table(a):
    for name, _, d in units(a.src):
        x, _ = sf.read(f"{d}/audio16k.wav", dtype="float32")
        e, words = band_db(x), load(f"{d}/words.jsonl")
        spk = word_speakers(d, words)
        if json.load(open(f"{d}/spk.json"))["silhouette"]["2"] < 0.3:  # iki ses ayrışmıyor (tek ses baskın ya da çok benzer): hepsi tek konuşmacı sayılır, raporda işaretlenir
            spk = [(1, 1.0)] * len(words)
        main = int(np.mean([s for s, _ in spk]) > 0.5)  # görüşülen kişi (K) = çok konuşan küme; S = soran
        for w, (s, m) in zip(words, spk):
            w["spk"], w["spk_m"] = ("K" if s == main else "S"), round(m, 3)
        refs = {s: ref_level(e, [w for w in words if w["spk"] == s and w["spk_m"] >= 0.1]) for s in "KS"}
        tags, rows = site_tags([(w["w"], w["punct"]) for w in words]), []
        ch = np.flatnonzero([p["spk"] != q["spk"] for p, q in zip(words, words[1:])])  # konuşmacı değişen sınırlar
        for k in range(len(words) - 1):
            p, q = words[k], words[k + 1]
            rows.append(dict(k=k, w=p["w"], nxt=q["w"], punct=p["punct"], tags=sorted(tags[k]), **silences(e, p, q, refs[p["spk"]]), sc=min(p["sc"], q["sc"]),
                             p=min(p["p"], q["p"]), spk=p["spk"], same=p["spk"] == q["spk"], spk_m=min(p["spk_m"], q["spk_m"]), chunk=p["chunk"],
                             cross=p["chunk"] != q["chunk"], edge=p["edge"] or q["edge"], t=p["e"],
                             turn_dist=int(np.min(np.abs(ch - k))) if len(ch) else 99))
        dump(f"{d}/boundaries.jsonl", rows)
        dump(f"{d}/words_spk.jsonl", words)
        print(f"table {name}: {len(rows)} sınır; başvuru düzeyi K {refs['K']:.1f} S {refs['S']:.1f} dB; K payı {np.mean([w['spk'] == 'K' for w in words]):.2f}", flush=True)


# ---------------------------------------------------------------- Antalia (aynı ölçüm)
def degrader(d, ant_rows, seed=0):
    """Antalia sesini görüşme kaydı koşullarına getirir: uzun dönem spektrumu eşle (FIR) + düzeyi eşle + kaydın KENDİ duraklama gürültüsünü ekle."""
    import librosa
    from scipy.signal import firwin2, lfilter, welch
    x, _ = sf.read(f"{d}/audio16k.wav", dtype="float32")
    words = [w for w in load(f"{d}/words_spk.jsonl") if w["spk"] == "K" and w["spk_m"] >= 0.1]
    e = band_db(x)
    sp = np.zeros(len(e), bool)
    for w in load(f"{d}/words_spk.jsonl"):
        sp[int(w["s"] * 100):int(w["e"] * 100) + 1] = True
    quiet = (~sp) & (e < np.percentile(e, 15))  # gürültü: sözcük dışı, bant enerjisi alt %15'te, >= 200 ms kesintisiz
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]])))
    fade, parts = np.hanning(320)[:160], []
    for i, j in zip(edges[::2], edges[1::2]):
        if j - i >= 20:
            seg = x[i * 160:j * 160].copy()
            seg[:160] *= fade
            seg[-160:] *= fade[::-1]
            parts.append(seg)
    noise = np.concatenate(parts)
    spx = np.concatenate([x[int(w["s"] * SR):int(w["e"] * SR)] for w in words[:6000]])
    ant = np.concatenate([librosa.load(f"{ANTALIA}/{r['wav']}", sr=SR)[0] for r in ant_rows[:60]])
    f, pi = welch(spx, SR, nperseg=512)
    _, pa = welch(ant, SR, nperseg=512)
    g = np.convolve(np.sqrt(pi / pa), np.ones(5) / 5, mode="same")
    g = np.clip(g / np.median(g[(f > 300) & (f < 3000)]), 10 ** (-40 / 20), 10 ** (20 / 20))
    fir = firwin2(255, f / (SR / 2), g)
    lvl = np.percentile(np.abs(spx), 95)  # görüşülen kişinin sözcük içi genlik düzeyi
    rng = np.random.default_rng(seed)

    def apply(y):
        y = lfilter(fir, 1.0, np.concatenate([y, np.zeros(127, np.float32)]))[127:]
        y = y * lvl / max(np.percentile(np.abs(y[band_db(y).repeat(160)[:len(y)].argsort()[len(y) // 2:]]), 95), 1e-6)  # yüksek enerjili yarının %95'i ~ konuşma düzeyi
        o = rng.integers(0, len(noise) - len(y) - 1)
        return (y + noise[o:o + len(y)]).astype(np.float32)
    return apply, dict(noise_s=round(len(noise) / SR, 1), n_runs=len(parts))


def stage_antalia(a):
    """--variant gold (tüm klipler) | asr (test+val, Whisper metni) | deg:<görüşme adının başı> (test+val, bozulmuş ses, altın metin)"""
    import librosa
    out = f"{OUT}/antalia_{a.variant.replace(':', '_')}.jsonl"
    if os.path.exists(out):
        return
    splits = ("train", "val", "test") if a.variant == "gold" else ("val", "test")
    rows = [dict(r, split=s) for s in splits for r in load(f"{ANTALIA}/{s}.jsonl")]
    asr = {r["id"]: r["words"] for r in load(f"{OUT}/antalia_whisper.jsonl")} if a.variant == "asr" else None
    deg = None
    if a.variant.startswith("deg:"):
        deg, info = degrader([d for n, _, d in units(a.src) if n.startswith(a.variant[4:])][0], rows)
        print("bozucu:", info, flush=True)
    fa, res = FA(), []
    for n, r in enumerate(rows):
        x, _ = librosa.load(f"{ANTALIA}/{r['wav']}", sr=SR)
        if deg:
            x = deg(x)
            if n < 3:
                sf.write(f"{OUT}/_ornek_{a.variant[4:8]}_{n}.wav", x, SR)
        tw = [(t, p, w["p"]) for t, p, w in asr_words(asr[r["id"]])] if asr else [(t, p, 1.0) for t, p in text_words(r["text"])]
        if len(tw) < 2:
            continue
        words = [dict(w=t, punct=p, s=sp[0], e=sp[1], sc=round(sp[2], 3), p=pr) for (t, p, pr), sp in zip(tw, fa(x, [t for t, _, _ in tw])) if sp]
        e = band_db(x)
        ref, tags = ref_level(e, words), site_tags([(w["w"], w["punct"]) for w in words])
        b = [dict(k=k, w=p["w"], nxt=q["w"], punct=p["punct"], tags=sorted(tags[k]), **silences(e, p, q, ref), sc=min(p["sc"], q["sc"]), p=min(p["p"], q["p"]))
             for k, (p, q) in enumerate(zip(words, words[1:]))]
        res.append(dict(id=r["id"], split=r["split"], parent=r.get("parent"), n_words=len(words), n_text_words=len(tw), boundaries=b))
        if n % 200 == 0:
            print(a.variant, n, len(rows), flush=True)
    dump(out, res)


# ---------------------------------------------------------------- doğrulama ve rapor
def antalia_rows(variant):
    return [dict(b, unit=r["id"], split=r["split"], n_words=r["n_words"]) for r in load(f"{OUT}/antalia_{variant}.jsonl") for b in retag(r["boundaries"])]


def prf(truth, pred):
    tp = int((truth & pred).sum())
    P, R = tp / max(int(pred.sum()), 1), tp / max(int(truth.sum()), 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def stage_validate(a):
    """Sessizlik ölçümünün geçerliliği. Başvuru: Antalia MAS hizalamalı sessizlik (breaks.jsonl; tam bant enerji, p95-30 dB) — bağımsız hizalayıcı, temiz ses.
    (1) eşik/pay seçimi TRAIN'de, (2) seçilen ölçünün test+val uyumu, (3) görüşme koşullarına bozulmuş seste aynı ölçü."""
    mas = {(r["id"], b["k"]): b["silence_ms"] for r in load(f"{ANTALIA}/breaks.jsonl") for b in r["boundaries"]}
    nw = {r["id"]: r["n_words"] for r in load(f"{ANTALIA}/breaks.jsonl")}
    res = {}
    for variant in [os.path.basename(p)[8:-6] for p in sorted(glob.glob(f"{OUT}/antalia_*.jsonl")) if "asr" not in p and "whisper" not in p]:
        rows = [r for r in antalia_rows(variant) if nw.get(r["unit"]) == r["n_words"] and (r["unit"], r["k"]) in mas]
        for part in (("train",), ("val", "test")):
            rs = [r for r in rows if r["split"] in part]
            if not rs:
                continue
            t = np.array([mas[(r["unit"], r["k"])] for r in rs])
            for key in [f"sil{T}{s}" for T in THR for s in ("", "p")]:
                v = np.array([r[key] for r in rs])
                res[f"{variant}|{'+'.join(part)}|{key}"] = dict(
                    n=len(rs), f1_250=round(prf(t >= 250, v >= 250)[2], 3), f1_100=round(prf(t >= 100, v >= 100)[2], 3), r=round(float(np.corrcoef(t, v)[0, 1]), 3),
                    mad=int(np.median(np.abs(t - v))), bias_paused=int(np.median((v - t)[t >= 250])) if (t >= 250).any() else None,
                    rate250=round(float((v >= 250).mean()), 4), rate250_mas=round(float((t >= 250).mean()), 4))
    json.dump(res, open(f"{OUT}/validate.json", "w", encoding="utf8"), indent=1)
    for k, v in res.items():
        print(k, v)


CATS = ["denetim", "virgül", "cümle", "K3", "K3_zarf", "K3_koşul", "K3_uzun", "K3_kısa", "K3_dışlanan", "K2", "K2_sonra", "bağ_önce", "bağ_sonra", "ve_önce", "ve_sonra",
        "K1", "K4", "K5_sayi", "K5_icin", "K5_tamlayan"]


def retag(bs):
    """Ardışık sınır satırlarının konum etiketlerini saklanan sözcük + noktalamadan yeniden üret (etiket tanımı değişince hizalamayı yeniden koşmamak için)."""
    tags = site_tags([(b["w"], b["punct"]) for b in bs] + [(bs[-1]["nxt"], ".")])
    for b, t in zip(bs, tags):
        b["tags"] = sorted(t)
    return bs


def cats(r):
    """Sınırın sınıfları: konum etiketleri (noktalamadan bağımsız) + noktalama sınıfı; hiçbiri yoksa denetim."""
    c = set(r["tags"]) - {"K3_dışlanan"} if "K3" in r["tags"] else set(r["tags"])
    if any(x in r["punct"] for x in ".?!"):
        c.add("cümle")
    elif r["punct"]:
        c.add("virgül")
    return c or {"denetim"}


def corpora(a, p_min=0.5, sc_min=None, inner=False):
    """-> {ad: sınır satırları (unit = bootstrap kümesi)}. Görüşmeler: sıra değişimine 2 sözcükten uzak (sıra arası boşluk duraklama sayılmasın), konuşmacı ataması net, ASR güveni ve hizalama puanı yeterli."""
    out = {"Antalia (okuma)": antalia_rows("gold")}
    sc_min = sc_min if sc_min is not None else float(np.percentile([r["sc"] for r in out["Antalia (okuma)"]], 5))
    drop = {}
    for i, (name, _, d) in enumerate(units(a.src), 1):
        rows = retag(load(f"{d}/boundaries.jsonl"))
        mixed = all(r["spk"] == "K" for r in rows)  # konuşmacı ayrılamadı: soranın (okunan) soruları ve sıra arası boşluklar İÇİNDE
        for s, label in (("K", f"G{i} ayrılmamış" if mixed else f"G{i} katılımcı"), ("S", f"G{i} soran")):
            rs = [r for r in rows if r["spk"] == s]
            if not rs:
                continue
            keep = [dict(r, unit=f"G{i}-{r['chunk']}") for r in rs if r["turn_dist"] > 2 and r["spk_m"] >= 0.1 and r["p"] >= p_min and r["sc"] >= sc_min
                    and not (inner and (r["cross"] or r["edge"]))]
            drop[label] = dict(n=len(rs), kalan=len(keep), sira_degisimine_yakin=sum(r["turn_dist"] <= 2 for r in rs), belirsiz_konusmaci=sum(r["spk_m"] < 0.1 for r in rs),
                               dusuk_asr=sum(r["p"] < p_min for r in rs), dusuk_hizalama=sum(r["sc"] < sc_min for r in rs))
            out[label] = keep
    return out, drop, sc_min


def boot(rows, f, n=2000, seed=0):
    """Küme (klip / hizalama parçası) düzeyinde bootstrap %95 GA."""
    by = {}
    for r in rows:
        by.setdefault(r["unit"], []).append(r)
    us, rng = list(by.values()), np.random.default_rng(seed)
    v = [f([r for j in rng.integers(0, len(us), len(us)) for r in us[j]]) for _ in range(n)]
    v = [x for x in v if x is not None]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (None, None)


def stage_report(a):
    C, drop, sc_min = corpora(a, a.p_min, inner=a.inner)
    key = a.key
    res = dict(key=key, sc_min=round(sc_min, 3), suzgec=drop, tablo={})
    for name, rows in C.items():
        for r in rows:
            r["c"] = cats(r)
        rate = lambda rs, c, thr=250: (np.mean([r[key] >= thr for r in rs if c in r["c"]]) if any(c in r["c"] for r in rs) else None)
        base = rate(rows, "denetim")
        for c in CATS:
            sel = [r for r in rows if c in r["c"]]
            if len(sel) < 5:
                continue
            for sub, ss in (("", sel), ("|noktalamasız", [r for r in sel if not r["punct"]])):
                if c in ("denetim", "virgül", "cümle") and sub or len(ss) < 5:
                    continue
                v = np.array([r[key] for r in ss])
                ids = set(map(id, ss))
                lo, hi = boot(ss, lambda rs: float(np.mean([r[key] >= 250 for r in rs])), 1000)

                def diff(rs):  # konum - denetim (aynı konuşmacı içinde): konuşma tarzının genel duraklama eğilimini çıkarır
                    s_, d_ = [r[key] >= 250 for r in rs if id(r) in ids], [r[key] >= 250 for r in rs if "denetim" in r["c"]]
                    return float(np.mean(s_) - np.mean(d_)) if s_ and d_ else None
                dlo, dhi = boot(rows, diff, 1000) if c != "denetim" else (0.0, 0.0)
                res["tablo"][f"{name}|{c}{sub}"] = dict(n=len(ss), p250=round(float((v >= 250).mean()), 3), ga=[round(lo, 3), round(hi, 3)],
                                                        p400=round(float((v >= 400).mean()), 3), p100=round(float((v >= 100).mean()), 3), medyan=int(np.median(v)),
                                                        medyan_duran=int(np.median(v[v >= 100])) if (v >= 100).any() else None,
                                                        fark=round(float((v >= 250).mean() - base), 3), fark_ga=[round(dlo, 3), round(dhi, 3)])
    json.dump(res, open(f"{OUT}/report_{key}_p{a.p_min}{'_inner' if a.inner else ''}.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"ölçü {key}, ASR güveni >= {a.p_min}, hizalama puanı >= {sc_min:.2f}{', parça kenarı/arası hariç' if a.inner else ''}")
    for k, v in drop.items():
        print("  süzgeç", k, v)
    names = list(C)
    print(f"\n{'sınıf':24s}" + "".join(f"{n[:16]:>30s}" for n in names))
    for c in [c + s for c in CATS for s in ("", "|noktalamasız")]:
        cells = [res["tablo"].get(f"{n}|{c}") for n in names]
        if any(cells):
            print(f"{c:24s}" + "".join(f"{f'%{100*x['p250']:.0f} [{100*x['ga'][0]:.0f}-{100*x['ga'][1]:.0f}] n={x['n']} md={x['medyan_duran']}':>30s}" if x else f"{'-':>30s}" for x in cells))
    print("\nkonum - denetim farkı (yüzde puan, >=250 ms) [%95 GA] | >=400 ms oranı")
    for c in [c + s for c in CATS[1:] for s in ("", "|noktalamasız")]:
        cells = [res["tablo"].get(f"{n}|{c}") for n in names]
        if any(cells):
            print(f"{c:24s}" + "".join(f"{f'{100*x['fark']:+.0f} [{100*x['fark_ga'][0]:+.0f},{100*x['fark_ga'][1]:+.0f}] | %{100*x['p400']:.0f}':>30s}" if x else f"{'-':>30s}" for x in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="sil25")  # doğrulamada seçildi: gürültüye dayanıklı en sıkı eşik (30 dB gürültü tabanına çarpıyor)
    ap.add_argument("stage", choices=["asr", "spk", "align", "table", "antalia", "validate", "report"])
    ap.add_argument("--src", default=f"{HOME}/data/raw/*.m4a")
    ap.add_argument("--variant", default="gold")
    ap.add_argument("--p-min", type=float, default=0.5)  # sınırın iki yanındaki sözcüklerin en düşük ASR güveni
    ap.add_argument("--inner", action="store_true")  # duyarlılık: hizalama parçası kenarındaki / parçalar arası sınırları at
    a = ap.parse_args()
    globals()[f"stage_{a.stage}"](a)
