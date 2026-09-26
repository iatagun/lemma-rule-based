"""Paketleme (2026-09-26 code review regresyonu): wheel'de g2ptts alt paketi ve resources/*.tsv|txt olmalı, tests olmamalı; wheel'den kurulan paket
kaynak dizini olmadan içe aktarılıp çalışmalı. Editable kurulumda bu hata GÖRÜNMEZDİ (kaynak dizinden okunuyor).
  python -X utf8 -m dizgetts.tests.test_packaging
`pip wheel` için setuptools indirilebilmeli (CI'da olur); geçici dizinde çalışır, kaynak dizinde `build/` bırakmaz."""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IGNORE = shutil.ignore_patterns("build", "__pycache__", "*.egg-info", "reports", "_v1_archive", "docs", "*.pt", ".git")

with tempfile.TemporaryDirectory() as tmp:
    src, out, site = (os.path.join(tmp, d) for d in ("src", "whl", "site"))
    shutil.copytree(HERE, src, ignore=IGNORE)  # kaynak dizini kirletme
    subprocess.run([sys.executable, "-m", "pip", "wheel", src, "--no-deps", "-w", out, "-q"], check=True)
    whl = glob.glob(os.path.join(out, "*.whl"))[0]
    names = zipfile.ZipFile(whl).namelist()
    need = ["dizgetts/g2ptts/tagger.py", "dizgetts/g2ptts/train.py", "dizgetts/engine.py", "dizgetts/paths.py",
            "dizgetts/resources/stress_roots.tsv", "dizgetts/resources/clitics.tsv", "dizgetts/resources/adj_lemmas.txt", "dizgetts/configs/data.yaml"]
    missing = [n for n in need if n not in names]
    assert not missing, f"wheel'de yok: {missing}"
    assert not [n for n in names if "/tests/" in n], "tests wheel'e sızmış"
    # wheel'i kaynaktan bağımsız kur ve çalıştır (dizge ortamda kurulu olmalı; bağımlılıklar --no-deps ile atlanır)
    subprocess.run([sys.executable, "-m", "pip", "install", whl, "--no-deps", "--target", site, "-q"], check=True)
    code = ("import dizgetts.engine as e, os; assert os.path.abspath(e.__file__).startswith(os.environ['SITE']), e.__file__\n"
            "from dizgetts.frontend.stress import StressRules\n"
            "StressRules()  # resources/*.tsv|txt yüklenir\n"
            "u = e.Engine(bert_fallback=False).frontend('Merhaba, dünya.')\n"
            "assert u.tokens and all(w.stress_src for w in u.words)\n"
            "import dizgetts.g2ptts  # alt paket kurulu\n"
            "print('paket kurulu ortamda çalıştı')")
    env = dict(os.environ, PYTHONPATH=site, SITE=os.path.abspath(site), PYTHONUTF8="1")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    print(r.stdout.strip())
print("OK")
