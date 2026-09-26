"""Ezgi / akustik karşılaştırma: GERÇEK Antalia kaydı vs SENTEZ (aynı 143 test+val cümlesi). Eğitim yok, ölçüm.
  python -X utf8 -m dizgetts.eval.prosody_acoustics --stage align     # A: hizalama + Praat ölçümleri -> önbellek + doğrulama raporu
  python -X utf8 -m dizgetts.eval.prosody_acoustics --stage report    # B: metrikler, grafikler, reports/prosody_acoustic_<label>.md

Yöntem (iki kaynağa AYNI işlem, yöntem yanlılığı olmasın):
  hizalama : sentezleyen Matcha modelinin kendi MAS'ı (Antalia sesini tanır), fonem düzeyi; token'lar = manifest (sentezde kullanılanla eşitliği denetlenir)
  ses      : sentez LUFS -23'e getirilir (gerçek kayıtlar ön işlemede öyle; mel log-ölçekli, kazanç hizalamayı bozar). Enerji yalnız GÖRELİ (cümle ortalamasına göre).
  F0       : Praat AC, konuşmacıya uyarlı aralık (Hirst: taban = 0,75*q25, tavan = 1,5*q75, gerçek kayıtların ilk geçişinden), yarım ton (gerçek konuşmacı medyanına göre)
  eğim     : ünlü başına 0-1 kHz / 1-4 kHz bant enerjisi farkı (alfa oranı, dB)
  formant  : Burg, ünlü ortası F1/F2
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from dizgetts import paths

EVAL_ROOT = paths.EVAL_OUT
ROOT = paths.ANTALIA
HOP_S = 256 / 22050
PUNCT = {",", ".", "?", "!", ";"}


def _load_model(ckpt):
    import torch
    from dizgetts.train.train import build_model

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    stats = json.load(open(os.path.join(ROOT, "stats.json"), encoding="utf8"))
    m = build_model(ck["cfg"], len(ck["symbols"]), stats)
    m.load_state_dict(ck["model"])
    return m.eval(), ck, stats


def mas_frames(m, ids, mel_norm):
    """Matcha MAS: interspersed token başına mel karesi sayısı (derive_breaks.py ile aynı)."""
    import torch
    import matcha.utils.monotonic_align as MA
    from matcha.utils.model import sequence_mask

    with torch.no_grad():
        x = torch.tensor(ids)[None]
        xl = torch.tensor([len(ids)])
        mu_x, _, x_mask = m.encoder(x, xl, None)
        y = mel_norm[None]
        y_mask = sequence_mask(torch.tensor([y.shape[-1]]), y.shape[-1]).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        const = -0.5 * math.log(2 * math.pi) * m.n_feats
        factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype)
        lp = (torch.matmul(factor.transpose(1, 2), y ** 2) - torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
              + torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1) + const)
        attn = MA.maximum_path(lp, attn_mask.squeeze(1))
        return attn[0].sum(-1).numpy().astype(int), float((lp * attn).sum() / y.shape[-1] / m.n_feats)


def segments(tokens, frames):
    """Fonem ve sözcük aralıkları (sn). tokens = interspersesiz manifest token'ları; frames = interspersed kare sayıları."""
    from dizgetts.frontend.symbols import PHONES, STRESS, WORD_SEP

    c = np.concatenate([[0], np.cumsum(frames)])
    words, cur, stress_next = [], None, False
    for j, t in enumerate(tokens):
        # token j = interspersed 2j+1. add_blank boşlukları (2j, 2j+2) kareleri iki komşuya YARI YARIYA dağıtılır: yalnız token kareleri
        # ünlüyü 1-2 kareye indiriyordu (MAS kareleri boşluklara da veriyor); böylece aralıklar zaman eksenini boşluksuz kaplar.
        s = (c[2 * j + 1] - (0 if j == 0 else frames[2 * j] / 2)) * HOP_S
        e = (c[2 * j + 2] + (frames[2 * j + 2] if j == len(tokens) - 1 else frames[2 * j + 2] / 2)) * HOP_S
        if t == WORD_SEP or t in PUNCT:
            if cur:
                words.append(cur); cur = None
            if t in PUNCT and words:
                words[-1]["punct"] += t
            continue
        if t == STRESS:
            stress_next = True
            continue
        if t not in PHONES:  # | ‖ (v3a'da yok) vb.
            continue
        if cur is None:
            cur = dict(phones=[], punct="")
        vowel = PHONES[t][0] == "ünlü"
        cur["phones"].append(dict(p=t, s=round(s, 4), e=round(e, 4), v=vowel, stress=bool(vowel and stress_next)))
        if vowel:
            stress_next = False
    if cur:
        words.append(cur)
    for w in words:
        w["s"], w["e"] = w["phones"][0]["s"], w["phones"][-1]["e"]
    return words


def praat_measures(wav_path, floor, ceiling, gain_to_lufs=None):
    import parselmouth
    import soundfile as sf

    x, sr = sf.read(wav_path, dtype="float64")
    if gain_to_lufs is not None:
        import pyloudnorm as pyln

        x = pyln.normalize.loudness(x, pyln.Meter(sr).integrated_loudness(x), gain_to_lufs)
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array["frequency"]
    inten = snd.to_intensity(minimum_pitch=floor, time_step=0.01)
    return x.astype(np.float32), sr, snd, dict(t=pitch.xs().round(4).tolist(), f0=f0.round(2).tolist(),
                                               it=inten.xs().round(4).tolist(), db=inten.values[0].round(2).tolist())


def vowel_measures(snd, words, max_formant):
    """Ünlü başına alfa oranı (dB) ve orta noktada F1/F2."""
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=max_formant)
    for w in words:
        for ph in w["phones"]:
            if not ph["v"] or ph["e"] - ph["s"] < 0.03:
                continue
            mid = (ph["s"] + ph["e"]) / 2
            ph["F1"], ph["F2"] = round(fm.get_value_at_time(1, mid), 1), round(fm.get_value_at_time(2, mid), 1)
            spec = snd.extract_part(ph["s"], ph["e"], preserve_times=False).to_spectrum()
            lo, hi = spec.get_band_energy(0, 1000), spec.get_band_energy(1000, 4000)
            ph["alpha_db"] = round(10 * math.log10(hi / lo), 2) if lo > 0 and hi > 0 else None


def stage_align(a):
    import torch
    import soundfile as sf
    from matcha.utils.audio import mel_spectrogram
    from matcha.utils.model import normalize as mel_normalize
    from matcha.utils.utils import intersperse
    from dizgetts.engine import Engine
    from dizgetts.frontend.symbols import SYMBOL_TO_ID

    m, ck, stats = _load_model(a.ckpt)
    res = json.load(open(f"{EVAL_ROOT}/{a.label}/results.json", encoding="utf8"))["items"]
    items = [it for it in res if it["split"] in ("test", "val")]
    man = {}
    for sp in ("test", "val"):
        for l in open(f"{ROOT}/{sp}{ck['cfg'].get('manifest', '_phon')}.jsonl", encoding="utf8"):
            r = json.loads(l); man[r["id"]] = r
    eng = Engine(**ck["cfg"].get("engine", {}))
    # --- DOĞRULAMA 1: hizalama SENTEZDE KULLANILAN token'larla yapılır (iki kaynakta aynı); manifestten farkı bilgi olarak raporlanır
    #     (ör. v2-m1a manifesti donduruldu, sentez güncel kural sürümüyle yapıldı)
    used = {it["id"]: eng.frontend(man[it["id"]]["text"]).tokens for it in items}
    bad = [i for i in used if used[i] != man[i]["tokens"]]
    print(f"[D1] sentez token'ı == manifest: {len(items) - len(bad)}/{len(items)}" + (f"  (farklı: {len(bad)}; hizalama sentez token'larıyla)" if bad else ""), flush=True)
    # --- F0 aralığı: gerçek kayıtlardan iki geçiş (Hirst)
    import parselmouth
    allf = []
    for it in items[::3]:
        s = parselmouth.Sound(os.path.join(ROOT, man[it["id"]]["wav"]))
        f = s.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600).selected_array["frequency"]
        allf.extend(f[f > 0])
    q25, q75 = np.percentile(allf, [25, 75])
    floor, ceiling = round(0.75 * q25), round(1.5 * q75)
    max_formant = 5500 if np.median(allf) > 165 else 5000
    print(f"[F0] ilk geçiş medyan {np.median(allf):.0f} Hz, q25 {q25:.0f} q75 {q75:.0f} -> taban {floor} tavan {ceiling}; maks formant {max_formant}", flush=True)
    au = dict(n_fft=1024, n_feats=80, hop=256, win=1024, fmin=0, fmax=8000)
    out = []
    for n, it in enumerate(items):
        row = man[it["id"]]
        toks = used[it["id"]]
        ids = intersperse([SYMBOL_TO_ID[t] for t in toks], 0)
        rec = dict(id=it["id"], i=it["i"], split=it["split"], text=row["text"], tokens=toks)
        for src, path, lufs in (("real", os.path.join(ROOT, row["wav"]), None), ("synth", f"{EVAL_ROOT}/{a.label}/{it['i']:03d}.wav", -23.0)):
            x, sr, snd, tracks = praat_measures(path, floor, ceiling, lufs)
            assert sr == 22050, (path, sr)
            if src == "real":
                mel = torch.load(os.path.join(ROOT, "mels", it["id"] + ".pt"))  # eğitimde kullanılan mel'in kendisi
            else:
                mel = mel_spectrogram(torch.from_numpy(x)[None], au["n_fft"], au["n_feats"], sr, au["hop"], au["win"], au["fmin"], au["fmax"],
                                      center=False).squeeze(0)
            frames, ll = mas_frames(m, ids, mel_normalize(mel, stats["mel_mean"], stats["mel_std"]))
            words = segments(toks, frames)
            vowel_measures(snd, words, max_formant)
            rec[src] = dict(dur=round(len(x) / sr, 3), mel_frames=int(mel.shape[-1]), frames_sum=int(frames.sum()), mas_ll=round(ll, 4),
                            words=words, tracks=tracks)
        out.append(rec)
        if n % 20 == 0:
            print(f"  {n}/{len(items)}", flush=True)
    os.makedirs(f"{EVAL_ROOT}/{a.label}/prosody", exist_ok=True)
    json.dump(dict(floor=floor, ceiling=ceiling, max_formant=max_formant, f0_ref_hz=float(np.median(allf)), ckpt=a.ckpt, clips=out),
              open(f"{EVAL_ROOT}/{a.label}/prosody/align.json", "w", encoding="utf8"), ensure_ascii=False)
    validate(out)


def validate(out):
    # --- DOĞRULAMA 2: MAS kare toplamı == mel uzunluğu; sözcük sayısı iki kaynakta aynı
    d2 = sum(r[s]["frames_sum"] == r[s]["mel_frames"] for r in out for s in ("real", "synth"))
    d2w = sum(len(r["real"]["words"]) == len(r["synth"]["words"]) for r in out)
    print(f"[D2] kare toplamı = mel uzunluğu: {d2}/{2 * len(out)}; sözcük sayısı eşit: {d2w}/{len(out)}")
    # --- DOĞRULAMA 3: hizalama olabilirliği (sentez kendi modelinin çıktısı -> daha yüksek olmalı) ve gerçek kayıtta noktalamalı sınırda boşluk
    ll = {s: np.mean([r[s]["mas_ll"] for r in out]) for s in ("real", "synth")}
    print(f"[D3] ort. MAS log-olabilirlik/kare/boyut: gerçek {ll['real']:.3f}  sentez {ll['synth']:.3f}")
    for s in ("real", "synth"):
        gp = [w2["s"] - w1["e"] for r in out for w1, w2 in zip(r[s]["words"], r[s]["words"][1:]) if w1["punct"]]
        gn = [w2["s"] - w1["e"] for r in out for w1, w2 in zip(r[s]["words"], r[s]["words"][1:]) if not w1["punct"]]
        print(f"[D3] {s}: sözcük arası boşluk (hizalamadan) noktalamalı medyan {1000 * np.median(gp):.0f} ms (>=250 ms %{100 * np.mean(np.array(gp) >= .25):.0f}), "
              f"noktalamasız medyan {1000 * np.median(gn):.0f} ms (>=60 ms %{100 * np.mean(np.array(gn) >= .06):.0f})")
    # --- DOĞRULAMA 4: F0 izleme sağlığı: seslilik oranı ve oktav sıçraması (ardışık seslı karelerde > 7 yarım ton)
    for s in ("real", "synth"):
        vf, jumps, pairs = [], 0, 0
        for r in out:
            f = np.array(r[s]["tracks"]["f0"])
            vf.append((f > 0).mean())
            st = 12 * np.log2(f[f > 0]) if (f > 0).any() else np.array([])
            both = (f[1:] > 0) & (f[:-1] > 0)
            d = np.abs(12 * np.log2(np.where(both, f[1:], 1) / np.where(both, f[:-1], 1)))[both]
            jumps += int((d > 7).sum()); pairs += int(both.sum())
        print(f"[D4] {s}: seslilik oranı {np.mean(vf):.2f}; oktav-benzeri sıçrama {jumps}/{pairs} = %{100 * jumps / max(pairs, 1):.2f}")


# ---------------------------------------------------------------- B: metrikler
# Tanımlar (hepsi yarım ton [yt] gerçek konuşmacı medyanına göre; yalnız SESLİ kareler; ünlü değeri = ünlü aralığındaki seslı karelerin medyanı):
#  sözcük_aralığı   : sözcük içi F0 p95 - p5 (>= 5 seslı kare)
#  vurguya_yükseliş : vurgulu ünlü - sözcüğün ilk ünlüsü (vurgu ilk ünlüde değilse; L...H* yükselişi)
#  son_hece_değişim : son ünlü - sondan ikinci ünlü, sınır türüne göre (yok / virgül-; / cümle sonu)
#  tepe_vurguda     : sözcüğün en yüksek ünlüsü işaretli vurgulu ünlü mü (>= 2 ünlülü sözcük)
#  alçalma          : ezgisel öbek (. ? ! ile biten dilim) içinde sözcük medyan F0'ının sıraya göre eğimi (yt/sözcük)
#  sınır_sıçraması  : sonraki sözcüğün ilk ünlüsü - bu sözcüğün son ünlüsü (noktalamasız sınır)
#  duraklama        : bu sözcüğün SON ÜNLÜSÜ sonu -> sonraki sözcüğün İLK ÜNLÜSÜ başı arasında sessiz süre (Praat yoğunluk < klip p95 - 30 dB;
#                     v1 derive_breaks eşiği). Ünlüden ünlüye bölge: MAS'ın sessizliği hangi foneme yüklediğinden bağımsız.
#  vurgu_farkı      : vurgulu ünlü - aynı sözcükteki diğer ünlülerin ortalaması: F0 (yt), yoğunluk (dB), süre (ms), alfa oranı (dB)
#  ünlü_dağılımı    : klip içi ünlülerin F1/F2 (Bark) ağırlık merkezine ort. uzaklığı (eklemleme netliği)
#  konuşma_hızı     : ünlü sayısı / (süre - sessiz süre)  [hece/sn]
#  son_uzama        : sözcüğün son ünlüsü süresi / aynı sözcüğün diğer ünlülerinin ort. süresi, sınır türüne göre (öbek sonu uzaması her sözcükte mi?)
#  yoğunluk_eğimi   : son ünlü yoğunluğu - ilk ünlü yoğunluğu (dB), noktalamasız sözcükler (sözcük başı güçlü, sonu sönük mü?)
#  nPVI             : ardışık ünlü sürelerinin normalize ikili değişkenlik indeksi (klip içi, sessizlik aşan çiftler hariç; ritim)


def _st(hz, ref):
    return 12 * np.log2(hz / ref)


def _bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def clip_metrics(r, src, ref):
    tr = r[src]["tracks"]
    t, f = np.array(tr["t"]), np.array(tr["f0"], float)
    it, db = np.array(tr["it"]), np.array(tr["db"], float)
    thr = np.nanpercentile(db, 95) - 30
    silent_t = it[db < thr]
    step = float(np.median(np.diff(it))) if len(it) > 1 else 0.01

    def f0_in(s, e):
        v = f[(t >= s) & (t < e) & (f > 0)]
        return _st(v, ref) if len(v) else np.array([])

    def med(s, e):
        v = f0_in(s, e)
        return float(np.median(v)) if len(v) else np.nan

    def db_in(s, e):
        v = db[(it >= s) & (it < e)]
        return float(np.mean(v)) if len(v) else np.nan

    m = {k: [] for k in ("word_range", "rise_to_stress", "stress_peak", "reset", "pause_np", "pause_p",
                         "last_change_none", "last_change_comma", "last_change_end",
                         "d_f0", "d_db", "d_dur", "d_alpha", "len_none", "len_comma", "len_end", "int_slope")}
    words = r[src]["words"]
    for k, w in enumerate(words):
        vw = [p for p in w["phones"] if p["v"]]
        v = f0_in(w["s"], w["e"])
        if len(v) >= 5:
            m["word_range"].append(float(np.percentile(v, 95) - np.percentile(v, 5)))
        vf = [med(p["s"], p["e"]) for p in vw]
        si = next((j for j, p in enumerate(vw) if p["stress"]), None)
        if len(vw) >= 2:
            if si is not None and si > 0 and not np.isnan(vf[si]) and not np.isnan(vf[0]):
                m["rise_to_stress"].append(vf[si] - vf[0])
            ok = [(j, x) for j, x in enumerate(vf) if not np.isnan(x)]
            if si is not None and len(ok) >= 2 and not np.isnan(vf[si]):
                m["stress_peak"].append(float(max(ok, key=lambda z: z[1])[0] == si))
            if not np.isnan(vf[-1]) and not np.isnan(vf[-2]):
                key = "last_change_end" if any(c in w["punct"] for c in ".?!") else "last_change_comma" if w["punct"] else "last_change_none"
                m[key].append(vf[-1] - vf[-2])
            key = "len_end" if any(c in w["punct"] for c in ".?!") else "len_comma" if w["punct"] else "len_none"
            m[key].append((vw[-1]["e"] - vw[-1]["s"]) / float(np.mean([p["e"] - p["s"] for p in vw[:-1]])))
            if not w["punct"]:
                a_, b_ = db_in(vw[0]["s"], vw[0]["e"]), db_in(vw[-1]["s"], vw[-1]["e"])
                if not np.isnan(a_) and not np.isnan(b_):
                    m["int_slope"].append(b_ - a_)
            if si is not None:
                others = [j for j in range(len(vw)) if j != si]
                sf0 = vf[si]; of0 = [vf[j] for j in others if not np.isnan(vf[j])]
                if not np.isnan(sf0) and of0:
                    m["d_f0"].append(sf0 - float(np.mean(of0)))
                sp = vw[si]
                sdb = db_in(sp["s"], sp["e"]); odb = [db_in(vw[j]["s"], vw[j]["e"]) for j in others]
                odb = [x for x in odb if not np.isnan(x)]
                if not np.isnan(sdb) and odb:
                    m["d_db"].append(sdb - float(np.mean(odb)))
                m["d_dur"].append(1000 * ((sp["e"] - sp["s"]) - float(np.mean([vw[j]["e"] - vw[j]["s"] for j in others]))))
                oa = [vw[j].get("alpha_db") for j in others if vw[j].get("alpha_db") is not None]
                if sp.get("alpha_db") is not None and oa:
                    m["d_alpha"].append(sp["alpha_db"] - float(np.mean(oa)))
        if k + 1 < len(words):
            nx = words[k + 1]
            lv = vw[-1] if vw else None
            fv = next((p for p in nx["phones"] if p["v"]), None)
            if lv and fv:
                s, e = lv["e"], fv["s"]
                pause = step * float(((silent_t >= s) & (silent_t < e)).sum())
                m["pause_p" if w["punct"] else "pause_np"].append(1000 * pause)
                if not w["punct"]:
                    a, b = med(lv["s"], lv["e"]), med(fv["s"], fv["e"])
                    if not np.isnan(a) and not np.isnan(b):
                        m["reset"].append(b - a)
    # alçalma: . ? ! ile biten dilimler
    slopes, seg = [], []
    for w in words:
        mv = f0_in(w["s"], w["e"])
        seg.append(float(np.median(mv)) if len(mv) else np.nan)
        if any(c in w["punct"] for c in ".?!"):
            y = np.array(seg); x = np.arange(len(y)); ok = ~np.isnan(y)
            if ok.sum() >= 3:
                slopes.append(float(np.polyfit(x[ok], y[ok], 1)[0]))
            seg = []
    m["declination"] = slopes
    # ünlü dağılımı + konuşma hızı
    F = np.array([(p["F1"], p["F2"]) for w in words for p in w["phones"] if p["v"] and p.get("F1") and p.get("F2")
                  and 200 < p["F1"] < 1000 and 500 < p["F2"] < 3000], float)
    m["vowel_disp"] = [float(np.mean(np.linalg.norm(_bark(F) - _bark(F).mean(0), axis=1)))] if len(F) >= 10 else []
    nv = sum(p["v"] for w in words for p in w["phones"])
    speech = r[src]["dur"] - step * len(silent_t)
    m["artic_rate"] = [nv / speech] if speech > 0 else []
    vd = [(p["s"], p["e"]) for w in words for p in w["phones"] if p["v"]]
    pv = []
    for (s1, e1), (s2, e2) in zip(vd, vd[1:]):
        d1, d2 = e1 - s1, e2 - s2
        if d1 > 0 and d2 > 0 and step * float(((silent_t >= e1) & (silent_t < s2)).sum()) < 0.03:  # araya >= 30 ms sessizlik giren çift hariç
            pv.append(abs(d1 - d2) / ((d1 + d2) / 2))
    m["npvi"] = [100 * float(np.mean(pv))] if len(pv) >= 5 else []
    v = f[f > 0]
    m["clip_range"] = [float(np.percentile(_st(v, ref), 95) - np.percentile(_st(v, ref), 5))] if len(v) >= 20 else []
    return m


LABELS = [  # (anahtar, açıklama, birim, özet: "mean" ya da eşik tuple)
    ("clip_range", "cümle F0 aralığı (p95-p5)", "yt", "mean"),
    ("declination", "öbek içi alçalma eğimi", "yt/sözcük", "mean"),
    ("word_range", "sözcük içi F0 aralığı", "yt", "mean"),
    ("rise_to_stress", "ilk ünlüden vurgulu ünlüye yükseliş", "yt", "mean"),
    ("stress_peak", "sözcük tepesi işaretli vurgulu ünlüde (oran)", "", "mean"),
    ("last_change_none", "son hece değişimi — noktalamasız", "yt", "mean"),
    ("last_change_comma", "son hece değişimi — virgül/;", "yt", "mean"),
    ("last_change_end", "son hece değişimi — cümle sonu", "yt", "mean"),
    ("reset", "sözcük sınırında F0 sıçraması (noktalamasız)", "yt", "mean"),
    ("pause_np", "duraklama — noktalamasız sınır", "ms", "mean"),
    ("pause_p", "duraklama — noktalamalı sınır", "ms", "mean"),
    ("d_f0", "vurgulu - vurgusuz ünlü F0", "yt", "mean"),
    ("d_db", "vurgulu - vurgusuz ünlü yoğunluk", "dB", "mean"),
    ("d_dur", "vurgulu - vurgusuz ünlü süre", "ms", "mean"),
    ("d_alpha", "vurgulu - vurgusuz ünlü spektral eğim (alfa)", "dB", "mean"),
    ("len_none", "son ünlü uzaması — noktalamasız (oran)", "x", "mean"),
    ("len_comma", "son ünlü uzaması — virgül/; (oran)", "x", "mean"),
    ("len_end", "son ünlü uzaması — cümle sonu (oran)", "x", "mean"),
    ("int_slope", "sözcük içi yoğunluk eğimi (son - ilk ünlü)", "dB", "mean"),
    ("npvi", "ünlü süresi nPVI (ritim)", "", "mean"),
    ("vowel_disp", "ünlü dağılımı (F1/F2 merkezine uzaklık)", "Bark", "mean"),
    ("artic_rate", "eklemleme hızı", "hece/sn", "mean"),
]


def _clips(clips, splits):
    """--splits: yalnız bu bölümlerin klipleri (test | val); karar için val kullanma (dp/epoch/tau/length_scale seçimi val'de yapıldı)."""
    return [r for r in clips if not splits or r["split"] in splits]


def stage_report(a):
    D = json.load(open(f"{EVAL_ROOT}/{a.label}/prosody/align.json", encoding="utf8"))
    ref, clips = D["f0_ref_hz"], _clips(D["clips"], a.splits)
    per = [(clip_metrics(r, "real", ref), clip_metrics(r, "synth", ref)) for r in clips]
    rng = np.random.default_rng(0)
    B = rng.integers(0, len(per), size=(2000, len(per)))
    rows, extra = [], {}
    for key, desc, unit, _ in LABELS:
        # klip düzeyi ortalama (her iki kaynakta değeri olan klipler), eşleşmiş klip bootstrap
        pairs = [(np.mean(r[key]), np.mean(s[key])) for r, s in per if r[key] and s[key]]
        if len(pairs) < 10:
            rows.append((desc, unit, len(pairs), None, None, None, None)); continue
        P = np.array(pairs)
        d = P[:, 1] - P[:, 0]
        idx = rng.integers(0, len(P), size=(2000, len(P)))
        ci = np.percentile(d[idx].mean(1), [2.5, 97.5])
        rows.append((desc, unit, len(P), P[:, 0].mean(), P[:, 1].mean(), d.mean(), ci))
    # duraklama eşik oranları (sınır düzeyi)
    for s in ("real", "synth"):
        pn = np.array([x for p in per for x in p[0 if s == "real" else 1]["pause_np"]])
        extra[s] = {f">={t} ms": float((pn >= t).mean()) for t in (30, 60, 100, 250)} | {"n": int(len(pn))}
        rs = np.array([x for p in per for x in p[0 if s == "real" else 1]["rise_to_stress"]])
        extra[s]["yükseliş>=3yt"] = float((rs >= 3).mean())
        lc = np.array([x for p in per for x in p[0 if s == "real" else 1]["last_change_none"]])
        extra[s]["son_hece_yükselişi>=3yt(noktalamasız)"] = float((lc >= 3).mean())
    json.dump(dict(rows=[list(map(lambda z: z.tolist() if isinstance(z, np.ndarray) else z, r)) for r in rows], extra=extra),
              open(f"{EVAL_ROOT}/{a.label}/prosody/metrics.json", "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(f"{'ölçü':52s} {'n':>4s} {'GERÇEK':>9s} {'SENTEZ':>9s} {'fark':>8s}  %95 GA")
    for desc, unit, n, rv, sv, dv, ci in rows:
        if rv is None:
            print(f"{desc:52s} {n:4d}  (yetersiz)"); continue
        sig = "*" if ci[0] > 0 or ci[1] < 0 else " "
        print(f"{desc + ' [' + unit + ']':52s} {n:4d} {rv:9.2f} {sv:9.2f} {dv:+8.2f}  [{ci[0]:+.2f}, {ci[1]:+.2f}] {sig}")
    for s in ("real", "synth"):
        print(s, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in extra[s].items()}, ensure_ascii=False))


def stage_compare(a):
    """İki SENTEZİ (a.label vs a.against) aynı klipler üzerinde eşleşmiş karşılaştırır; gerçek kayıt referans olarak yanında."""
    L = {k: json.load(open(f"{EVAL_ROOT}/{k}/prosody/align.json", encoding="utf8")) for k in (a.label, a.against)}
    C = {k: _clips(L[k]["clips"], a.splits) for k in L}
    assert [r["id"] for r in C[a.label]] == [r["id"] for r in C[a.against]], "klip sırası farklı"
    print(f"klip: {len(C[a.label])} (bölümler: {a.splits or 'hepsi'})")
    ref = L[a.against]["f0_ref_hz"]
    new = [clip_metrics(r, "synth", ref) for r in C[a.label]]
    old = [clip_metrics(r, "synth", ref) for r in C[a.against]]
    real = [clip_metrics(r, "real", ref) for r in C[a.against]]
    rng = np.random.default_rng(0)
    out = []
    print(f"{'ölçü':52s} {'GERÇEK':>8s} {a.against[:10]:>10s} {a.label[:10]:>10s} {'fark':>8s}  %95 GA (yeni - eski)")
    for key, desc, unit, _ in LABELS:
        tri = [(np.mean(r[key]), np.mean(o[key]), np.mean(n[key])) for r, o, n in zip(real, old, new) if r[key] and o[key] and n[key]]
        if len(tri) < 10:
            continue
        T = np.array(tri); d = T[:, 2] - T[:, 1]
        ci = np.percentile(d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1), [2.5, 97.5])
        sig = "*" if ci[0] > 0 or ci[1] < 0 else " "
        out.append(dict(key=key, desc=desc, unit=unit, n=len(T), real=T[:, 0].mean(), old=T[:, 1].mean(), new=T[:, 2].mean(), diff=d.mean(), ci=ci.tolist()))
        print(f"{desc + ' [' + unit + ']':52s} {T[:, 0].mean():8.2f} {T[:, 1].mean():10.2f} {T[:, 2].mean():10.2f} {d.mean():+8.2f}  [{ci[0]:+.2f}, {ci[1]:+.2f}] {sig}")
    json.dump(out, open(f"{EVAL_ROOT}/{a.label}/prosody/compare_vs_{a.against}.json", "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("align", "validate", "report", "compare"), required=True)
    ap.add_argument("--against", default="v3a_g2ptts_nb_ep150_x543", help="compare: karşılaştırılan eski sentez etiketi")
    ap.add_argument("--label", default="v3a_g2ptts_nb_ep150_x543")
    ap.add_argument("--splits", nargs="*", default=None, help="report/compare: yalnız bu bölümler (test|val); karar için val kullanma")
    ap.add_argument("--ckpt", default=f"{paths.RUNS}/v3a_g2ptts_nb_e150_20260925-213638/ep150.pt")
    a = ap.parse_args()
    if a.stage == "align":
        stage_align(a)
    elif a.stage == "report":
        stage_report(a)
    elif a.stage == "compare":
        stage_compare(a)
    else:
        validate(json.load(open(f"{EVAL_ROOT}/{a.label}/prosody/align.json", encoding="utf8"))["clips"])


if __name__ == "__main__":
    main()
