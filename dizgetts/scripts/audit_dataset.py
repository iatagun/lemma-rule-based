"""Aşama 1 veri denetimi: sinyal (WAV'dan yeniden hesap, metadata'ya güvenmeden), metin, transkript-ses uyumu,
G2P kapsama. Yalnız stdlib+numpy (+ G2P bölümü için torch/transformers/dizge).

  python -X utf8 dizgetts/scripts/audit_dataset.py --root D:/dizgetts/data/raw/antalia --out dizgetts/reports/antalia_audit.json
"""
import argparse, collections, json, os, re, sys, wave

import numpy as np

from dizgetts import paths

FR = 0.010  # 10 ms çerçeve


def read_wav(p):
    with wave.open(p) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    assert sw == 2, f"{p}: {sw*8}-bit"
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(1), sr, ch


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-12))


def signal_stats(x, sr):
    hop = int(sr * FR)
    n = len(x) // hop
    e = (x[: n * hop].reshape(n, hop) ** 2).mean(1)  # çerçeve enerjisi
    edb = db(e)
    speech_thr = np.percentile(edb, 95) - 30  # tepe konuşma seviyesinin 30 dB altı: konuşma/sessizlik eşiği
    act = edb > speech_thr
    idx = np.flatnonzero(act)
    lead = idx[0] * FR if len(idx) else float("nan")
    trail = (n - 1 - idx[-1]) * FR if len(idx) else float("nan")
    noise = np.percentile(edb, 10)  # ponytail: yüzdelik SNR tahmini; kayıt sonrası gürültü kapısı varsa iyimser çıkar
    snr = np.percentile(edb, 90) - noise
    return dict(
        dur=len(x) / sr,
        peak=float(np.abs(x).max()),
        rms_dbfs=float(db((x**2).mean())),
        clip=float((np.abs(x) >= 0.999).mean()),
        lead_sil=float(lead), trail_sil=float(trail),
        snr_pct=float(snr), noise_floor_db=float(noise), active_ratio=float(act.mean()),
    )


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_for_cer(s):
    s = s.replace("İ", "i").replace("I", "ı").lower()
    return re.sub(r"[^\wçğıöşü]+", " ", s).strip()


def q(a, ps=(0, 5, 25, 50, 75, 95, 100)):
    a = np.asarray(a, float)
    return {f"p{p}": round(float(np.nanpercentile(a, p)), 3) for p in ps}


def hist(a, edges):
    h, _ = np.histogram(a, bins=edges)
    return {f"{edges[i]}-{edges[i+1]}": int(h[i]) for i in range(len(h))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{paths.HOME}/data/raw/antalia")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "antalia_audit.json"))
    ap.add_argument("--skip-g2p", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(a.root, "metadata.jsonl"), encoding="utf8")]
    recs = []
    for r in rows:
        x, sr, ch = read_wav(os.path.join(a.root, r["file_name"]))
        s = signal_stats(x, sr)
        t, asr = norm_for_cer(r["transcript"]), norm_for_cer(r["asr_text"])
        s.update(
            id=r["clip_id"], cat=r["campaign_category"], sr=sr, ch=ch, meta_dur=r["duration_seconds"],
            meta_snr=r["estimated_snr_db"], chars=len(r["transcript"]), words=len(r["transcript"].split()),
            cer_asr=lev(t, asr) / max(1, len(t)), same_norm=(r["transcript"] == r["normalized_transcript"]),
            has_digit=bool(re.search(r"\d", r["transcript"])), parent=r["clip_id"].split("-script-seg-")[0],
        )
        s["cps"] = s["chars"] / s["dur"]
        recs.append(s)
    g = lambda k: [r[k] for r in recs]

    total_h = sum(g("dur")) / 3600
    out = dict(
        n_clips=len(recs), total_hours=round(total_h, 3), sample_rates=dict(collections.Counter(g("sr"))),
        channels=dict(collections.Counter(g("ch"))),
        dur_quantiles=q(g("dur")), dur_hist=hist(g("dur"), [0, 1, 3, 5, 8, 10, 12, 15, 20, 25]),
        hours_in_1_10s=round(sum(d for d in g("dur") if 1 <= d <= 10) / 3600, 3),
        clips_in_1_10s=sum(1 for d in g("dur") if 1 <= d <= 10),
        clips_over_10s=sum(1 for d in g("dur") if d > 10),
        max_abs_dur_diff_vs_metadata=round(float(np.max(np.abs(np.array(g("dur")) - np.array(g("meta_dur"))))), 4),
        snr_pct_quantiles=q(g("snr_pct")), snr_meta_quantiles=q(g("meta_snr")),
        corr_snr_mine_vs_meta=round(float(np.corrcoef(g("snr_pct"), g("meta_snr"))[0, 1]), 3),
        noise_floor_db_quantiles=q(g("noise_floor_db")), rms_dbfs_quantiles=q(g("rms_dbfs")),
        peak_max=round(max(g("peak")), 4), clipping_clips=sum(1 for c in g("clip") if c > 0),
        lead_sil_quantiles=q(g("lead_sil")), trail_sil_quantiles=q(g("trail_sil")),
        chars_per_sec_quantiles=q(g("cps")), cer_transcript_vs_scribe_quantiles=q(g("cer_asr")),
        n_cer_gt_0_10=sum(1 for c in g("cer_asr") if c > 0.10),
        n_transcript_ne_normalized=sum(1 for s in g("same_norm") if not s), n_with_digits=sum(g("has_digit")),
        by_category={
            c: dict(n=len(v), hours=round(sum(r["dur"] for r in v) / 3600, 3), mean_cps=round(float(np.mean([r["cps"] for r in v])), 2),
                    n_over_10s=sum(1 for r in v if r["dur"] > 10))
            for c, v in ((c, [r for r in recs if r["cat"] == c]) for c in sorted(set(g("cat"))))
        },
        n_parent_recordings=len(set(g("parent"))),
    )
    # şüpheli: CER yüksek ya da hız aykırı (medyan±3 MAD)
    cps = np.array(g("cps")); med = np.median(cps); mad = np.median(np.abs(cps - med)) * 1.4826
    sus = [r for r in recs if r["cer_asr"] > 0.10 or abs(r["cps"] - med) > 3 * mad]
    byid = {r["clip_id"]: r for r in rows}
    out["suspects"] = [
        dict(id=r["id"], dur=round(r["dur"], 2), cps=round(r["cps"], 1), cer=round(r["cer_asr"], 3),
             text=byid[r["id"]]["transcript"][:120], asr=byid[r["id"]]["asr_text"][:120])
        for r in sorted(sus, key=lambda r: -r["cer_asr"])[:40]
    ]
    out["n_suspects"] = len(sus)
    dup = collections.Counter(byid[r["id"]]["transcript"] for r in recs)
    out["duplicate_transcripts"] = sum(1 for v in dup.values() if v > 1)

    # metin: karakter kümesi + sözcük dağarcığı
    text = [byid[r["id"]]["transcript"] for r in recs]
    chars = collections.Counter("".join(text))
    out["non_turkish_chars"] = {c: n for c, n in chars.items() if not re.match(r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ0-9 ]", c)}
    words = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ]+", " ".join(text))
    out["n_word_tokens"], out["n_word_types"] = len(words), len(set(map(norm_for_cer, words)))

    if not a.skip_g2p:
        import dizge
        from dizgetts.frontend.g2p import G2P, tr_lower
        from dizgetts.frontend.symbols import PHONES, tokenize

        vocab = sorted(set(tr_lower(w) for w in words))
        ref = set(w.strip() for w in open("D:/playground/turkish_words.txt", encoding="utf8"))
        oov = [w for w in vocab if w not in ref]
        pred = G2P().g2p_batch(vocab)
        tea = {}
        for w in vocab:
            try:
                r = dizge.g2p(w)
                tea[w] = r if isinstance(r, str) else r[0]
            except Exception as e:  # dizge bazı yabancı/kısaltmalarda patlayabilir
                tea[w] = None
        ok = [w for w in vocab if tea[w] is not None]
        dis = [w for w in ok if pred[w] != tea[w]]
        dis_oov = [w for w in dis if w in set(oov)]
        unk = collections.Counter()
        for w in vocab:
            for ch in pred[w]:
                if not any(ch in p for p in PHONES):
                    unk[ch] += 1
        out["g2p"] = dict(
            vocab=len(vocab), oov_vs_training_list=len(oov), dizge_failed=len(vocab) - len(ok),
            model_eq_teacher_all=round(1 - len(dis) / max(1, len(ok)), 4),
            model_eq_teacher_oov=round(1 - len(dis_oov) / max(1, len([w for w in oov if tea[w] is not None])), 4),
            model_unknown_chars=dict(unk), disagree_sample=[(w, tea[w], pred[w]) for w in dis_oov[:40]],
            oov_sample=oov[:60],
        )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("suspects", "g2p", "by_category")}, ensure_ascii=False, indent=1))
    if "g2p" in out:
        print(json.dumps({k: v for k, v in out["g2p"].items() if "sample" not in k}, ensure_ascii=False))


if __name__ == "__main__":
    main()
