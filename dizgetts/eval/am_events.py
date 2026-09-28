# AM olayları: gerçek vs sentez (v4-e400 hizalaması, 143 test+val cümlesi). Ünlü F0 = aralıktaki sesli karelerin medyanı (yt, konuşmacı medyanına göre).
import json, math, sys
import numpy as np
d = json.load(open("D:/dizgetts/eval_out/v4_e400_x543/prosody/align.json", encoding="utf8"))
ref = d["f0_ref_hz"]
QW = ("mi", "mı", "mu", "mü")

def vf0(tr, s, e):
    t, f = np.array(tr["t"]), np.array(tr["f0"])
    m = (t >= s) & (t <= e) & (f > 0)
    return 12 * math.log2(np.median(f[m]) / ref) if m.sum() >= 2 else None

def words_f0(src):
    out = []
    for w in src["words"]:
        vs = [(p, vf0(src["tracks"], p["s"], p["e"])) for p in w["phones"] if p["v"]]
        out.append(dict(punct=w["punct"], vs=vs))
    return out

R = {k: {"real": [], "synth": []} for k in ("hstar", "ip_LH", "q_final", "dot_final", "premi_peak", "mi_fall")}
for c in d["clips"]:
    txt = c["text"]
    for src in ("real", "synth"):
        ws = words_f0(c[src])
        allv = [f for w in ws for _, f in w["vs"] if f is not None]
        if not allv:
            continue
        mean = np.mean(allv)
        for i, w in enumerate(ws):
            vs = [(p, f) for p, f in w["vs"] if f is not None]
            # H*: çok heceli, vurgulu ünlüsü ölçülü sözcükte en yüksek F0 vurgulu ünlüde mi
            st = [f for p, f in vs if p["stress"]]
            if len(vs) >= 2 and st:
                R["hstar"][src].append(float(max(vs, key=lambda x: x[1])[0]["stress"]))
            if len(vs) >= 2:
                mv = vs[-1][1] - vs[-2][1]  # son ünlü - sondan ikinci (yt)
                if "," in w["punct"]:
                    R["ip_LH"][src].append(mv)
                if "?" in w["punct"]:
                    R["q_final"][src].append(mv)
                if "." in w["punct"]:
                    R["dot_final"][src].append(mv)
        # mi-soru: mi'den önceki sözcüğün vurgulu tepesi (cümle ortalamasına göre) ve mi'nin kendisi (önceki sözcüğün tepesine göre)
        toks = txt.replace("?", " ?").split()
        for i, w in enumerate(ws):
            if i and i < len(ws) and "?" in w["punct"]:
                vs_prev = [f for _, f in ws[i - 1]["vs"] if f is not None]
                vs_mi = [f for _, f in w["vs"] if f is not None]
                # kaba ölçüt: son sözcük kısa (<=2 ünlü) ise mi-benzeri kabul
                if vs_prev and vs_mi and len(w["vs"]) <= 2:
                    R["premi_peak"][src].append(max(vs_prev) - mean)
                    R["mi_fall"][src].append(vs_mi[-1] - max(vs_prev))

def fmt(x):
    x = np.array(x)
    return f"n={len(x):4d} ort {x.mean():+.2f}  medyan {np.median(x):+.2f}"
names = dict(hstar="H*: sözcük tepesi vurgulu ünlüde (oran)", ip_LH="virgül öncesi son ünlü - önceki (yt; LH>0)",
             q_final="soru sonu sözcük: son ünlü - önceki (yt)", dot_final="nokta sonu sözcük: son ünlü - önceki (yt)",
             premi_peak="mi'den önceki sözcük tepesi - cümle ort. (yt)", mi_fall="mi ünlüsü - önceki sözcük tepesi (yt)")
for k in R:
    print(names[k]); print("   gerçek ", fmt(R[k]["real"])); print("   sentez ", fmt(R[k]["synth"]))
print("--- virgül öncesi dağılım (yt): >+1 yükselen / -1..+1 düz / <-1 düşen; std")
for src in ("real", "synth"):
    x = np.array(R["ip_LH"][src]); print(f"   {src:6s} yükselen %{100*(x>1).mean():.0f}  düz %{100*((x>=-1)&(x<=1)).mean():.0f}  düşen %{100*(x<-1).mean():.0f}  std {x.std():.2f}")
print("--- H* sözcük tepesi: vurgulu / vurgudan sonraki ünlü / diğer")
for src in ("real","synth"):
    a=b=o=0
    for c in d["clips"]:
        for w in words_f0(c[src]):
            vs=[(j,p,f) for j,(p,f) in enumerate(w["vs"]) if f is not None]
            si=[j for j,p,f in vs if p["stress"]]
            if len(vs)<2 or not si: continue
            j=max(vs,key=lambda x:x[2])[0]
            if j==si[0]: a+=1
            elif j==si[0]+1: b+=1
            else: o+=1
    n=a+b+o; print(f"   {src:6s} vurgulu %{100*a/n:.0f}  sonraki %{100*b/n:.0f}  diğer %{100*o/n:.0f}")
