"""Aşama 3: split manifestlerini normalize + fonemleştir (dizge & espeak-ng), bilinmeyen sembol/hata raporu.

  D:/dizgetts/venv/Scripts/python.exe -X utf8 dizgetts/scripts/build_phonemes.py

Çıktı: <out_root>/{split}_phon.jsonl (text_norm, tokens[dizge], espeak) + dizgetts/reports/frontend_report.json
"""
import argparse, collections, json, os, re, subprocess, sys

import yaml
from dizgetts.frontend import espeak  # noqa: E402
from dizgetts.frontend.normalize import tr_lower  # noqa: E402
from dizgetts.frontend.phonemize import Phonemizer  # noqa: E402
from dizgetts.frontend.symbols import PHONES, SYMBOL_TO_ID  # noqa: E402


def canon(s):  # normalizer karşılaştırması için: küçük harf, noktalama/tire yok
    return " ".join(re.sub(r"[^\w\s]|_", " ", tr_lower(s)).replace("-", " ").split())


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "data.yaml"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "frontend_report.json"))
    a = ap.parse_args()
    c = yaml.safe_load(open(a.config, encoding="utf8"))
    root = c["out_root"]
    ph = Phonemizer()
    esp_chars = collections.Counter()
    mism, n_cmp, cer_sum, cer_len, exact = [], 0, 0, 0, 0
    n_clips, n_no_tokens = collections.Counter(), 0
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in open(os.path.join(root, f"{split}.jsonl"), encoding="utf8")]
        with open(os.path.join(root, f"{split}_phon.jsonl"), "w", encoding="utf8") as f:
            for r in rows:
                norm, toks = ph(r["text"])
                if not toks:
                    n_no_tokens += 1
                e = espeak.phonemize(norm)
                esp_chars.update(e)
                ref, mine = canon(r["text_norm_antalia"]), canon(norm)
                d = lev(ref, mine)
                cer_sum += d; cer_len += len(ref); exact += d == 0; n_cmp += 1
                if d:
                    mism.append((d / max(1, len(ref)), r["id"], ref[:160], mine[:160]))
                assert all(t in SYMBOL_TO_ID for t in toks)
                f.write(json.dumps({**r, "text_norm": norm, "tokens": toks, "espeak": e}, ensure_ascii=False) + "\n")
                n_clips[split] += 1
    mism.sort(reverse=True)
    rep = dict(
        clips=dict(n_clips), clips_without_tokens=n_no_tokens, words=ph.stats["words"], unique_words=ph.word.cache_info().currsize,
        dizge_calls=ph.stats["dizge"], dizge_failures=len(ph.failed), dizge_failed_words=ph.failed,
        multi_variant_words={k: list(v) for k, v in ph.variants.items()}, unknown_dizge_chars=dict(ph.unknown),
        espeak_symbols=sorted(esp_chars), espeak_symbol_counts=esp_chars.most_common(),
        espeak_version=espeak.espeak_version(),
        normalizer_vs_antalia=dict(n=n_cmp, exact_canon_match=exact, exact_rate=round(exact / n_cmp, 4), cer=round(cer_sum / cer_len, 4),
                                   worst=mism[:25]),
        unused_dizge_atoms=sorted(set(PHONES)),  # doldurma aşağıda
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
    )
    used = set()
    for split in ("train", "val", "test"):
        for l in open(os.path.join(root, f"{split}_phon.jsonl"), encoding="utf8"):
            used.update(json.loads(l)["tokens"])
    rep["unused_dizge_atoms"] = sorted(set(PHONES) - used)
    rep["dizge_atoms_used"] = len(used & set(PHONES))
    json.dump(rep, open(a.out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("espeak_symbol_counts", "multi_variant_words", "dizge_failed_words")}, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
