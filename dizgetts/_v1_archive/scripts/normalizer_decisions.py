"""Normalizer kararları için Whisper hipotez puanlaması (scripts/whisper_score.py). Sonuç: reports/normalizer_decisions.json

Her karar için: ham metinde ilgili kalıbı içeren klipler; adaylar = kalıbın farklı okunuşlarıyla YENİDEN normalize edilmiş metin.
Kazanan = klip başına en düşük toplam NLL. (Aday token sayıları farklıysa toplam NLL kısa adayı kayırır; bu yüzden
ayrıca token başına NLL de verilir ve karar ikisi de aynı yöndeyse alınır.)
"""
import json, os, re, sys, collections

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from frontend.normalize import cardinal, digits, normalize  # noqa: E402
from whisper_score import Scorer  # noqa: E402

ROOT = "D:/dizgetts/data/processed/antalia"
rows = [json.loads(l) for sp in ("train", "val", "test") for l in open(f"{ROOT}/{sp}.jsonl", encoding="utf8")]
sc = Scorer()


def alts_currency(raw):
    m = re.search(r"(\d+),(\d{2}) TL", raw)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    return {"lira_kuruş(mine)": raw,
            "virgül_türk_lirası(antalia)": re.sub(r"(\d+),(\d{2}) TL", lambda x: f"{cardinal(int(x.group(1)))} virgül {cardinal(int(x.group(2)))} türk lirası", raw)}


def alts_sms(raw):
    return {"es_em_es(mine)": raw, "se_me_se(antalia)": raw.replace("SMS", "se me se")} if "SMS" in raw else None


def alts_www(raw):
    if "www" not in raw:
        return None
    return {k: raw.replace("www", v) for k, v in {"ve_ve_ve(mine)": "ve ve ve", "vevevev": "vevevev", "çift_ve×3": "çift ve çift ve çift ve",
                                                  "dabılyu×3": "dabılyu dabılyu dabılyu", "vi_vi_vi": "vi vi vi"}.items()}


def alts_longid(raw):
    m = re.search(r"(?<![\d.,-])(\d{7,})(?![\d.,-])", raw)
    if not m:
        return None
    s = m.group(1)
    return {"cardinal(mine)": raw, "rakam_rakam(antalia)": raw.replace(s, digits(s)),
            "2li_gruplar": raw.replace(s, " ".join(cardinal(int(s[i:i + 2])) if s[i:i+2][0] != "0" else digits(s[i:i+2]) for i in range(0, len(s), 2)))}


def alts_suffix(raw):
    m = re.search(r"(\d+)['’]([iıuü])\b", raw)
    if not m:
        return None
    d = {"kural(mine)": raw}
    d.update({f"'{suf}": raw[: m.start(2)] + suf + raw[m.end(2):] for suf in ("i", "ı", "u", "ü", "si", "sı", "su", "sü", "yi", "yı", "yu", "yü")})
    return d


def alts_wifi(raw):
    return {k: re.sub(r"Wi-?Fi", v, raw) for k, v in {"vayfay(mine)": "vayfay", "vay_fay": "vay fay", "vi_fi": "vi fi", "vayfai": "vayfai", "vifi": "vifi"}.items()} if re.search(r"Wi-?Fi", raw) else None


def alts_pin(raw):
    return {"pin(mine)": raw, "pe_i_ne": re.sub(r"\bPIN\b", "pe i ne", raw), "pi_in": re.sub(r"\bPIN\b", "pi in", raw)} if re.search(r"\bPIN\b", raw) else None


def alts_dashcode(raw):
    m = re.search(r"\b(\d{4})-(\d{4})\b", raw)
    if not m:
        return None
    a, b = m.groups()
    sp = lambda s: " ".join(cardinal(int(s[i:i + 2])) if s[i] != "0" else digits(s[i:i + 2]) for i in range(0, 4, 2))
    return {"kardinal_gruplar(mine)": raw, "rakam_rakam": raw.replace(m.group(), digits(a) + " " + digits(b)), "ikişer": raw.replace(m.group(), sp(a) + " " + sp(b))}


def alts_code4(raw):
    m = re.search(r"\b[A-ZÇĞİÖŞÜ]{2,4}-(\d{4})-[A-ZÇĞİÖŞÜ]\b", raw)
    if not m:
        return None
    a = m.group(1)
    pair = " ".join(cardinal(int(a[i:i + 2])) if a[i] != "0" else digits(a[i:i + 2]) for i in range(0, 4, 2))
    return {"kardinal(mine)": raw, "rakam_rakam": raw.replace(a, digits(a)), "ikişer": raw.replace(a, pair)}


DECISIONS = dict(currency=alts_currency, sms=alts_sms, www=alts_www, long_id=alts_longid, suffix_i=alts_suffix, wifi=alts_wifi, pin=alts_pin, dash_code=alts_dashcode, code4=alts_code4)


def main():
    out, per_decision = {}, {}
    only = sys.argv[1:]
    for name, fn in DECISIONS.items():
        if only and name not in only:
            continue
        wins, wins_tok, n, detail = collections.Counter(), collections.Counter(), 0, []
        for r in rows:
            alts = fn(r["text"])
            if not alts:
                continue
            labels = list(alts)
            texts = [normalize(alts[k]).replace(" ,", ",").replace(" .", ".") for k in labels]
            res = sc.nll(f"{ROOT}/{r['wav']}", texts)
            tot = [x[0] for x in res]
            per = [x[0] / x[1] for x in res]
            wi = int(min(range(len(tot)), key=tot.__getitem__))
            wins[labels[wi]] += 1
            mi = next((i for i, k in enumerate(labels) if "mine" in k), None)
            if mi is not None:
                wins["_kazanan_metin==mine_metni"] += texts[wi] == texts[mi]
            wins_tok[labels[int(min(range(len(per)), key=per.__getitem__))]] += 1
            n += 1
            detail.append(dict(id=r["id"], nll={k: round(t, 2) for k, t in zip(labels, tot)}))
        out[name] = dict(n_clips=n, wins_total_nll=dict(wins), wins_nll_per_token=dict(wins_tok), sample=detail[:5])
        print(name, n, "toplam:", dict(wins), "| token-başı:", dict(wins_tok), flush=True)
    pth = os.path.join(os.path.dirname(__file__), "..", "reports", "normalizer_decisions.json")
    old = json.load(open(pth, encoding="utf8")) if os.path.exists(pth) else {}
    old.update(out)
    json.dump(old, open(pth, "w", encoding="utf8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
