"""Ön uç sürüm tuzağı denetimi (eval/synth.check_frontend; docs/v9_plan.md §5).
  python -X utf8 -m dizgetts.tests.test_frontend_probe
Checkpoint eğitim token'larıyla güncel ön uç SİSTEMATİK olarak farklıysa (v6 + bugünkü ön uç) Synth hata vermeli; küçük sözlük düzeltmesi uyarı."""
import types
import warnings

from dizgetts.eval.synth import PROBE_MAX_DIFF, check_frontend, check_symbols
from dizgetts.frontend.symbols import SYMBOLS

probe = [(f"c{i}", ["a", " ", "b"]) for i in range(10)]
eng = lambda changed: types.SimpleNamespace(frontend=lambda t: types.SimpleNamespace(tokens=["a", " ", "x"] if int(t[1:]) < changed else ["a", " ", "b"]))

assert check_frontend(eng(0), probe) == dict(checked=10, differing=0)
with warnings.catch_warnings(record=True) as W:  # küçük fark: uyarı, hata değil
    warnings.simplefilter("always")
    assert check_frontend(eng(1), probe)["differing"] == 1
assert any("ön uç" in str(w.message) for w in W)
try:  # sistematik fark: hata
    check_frontend(eng(int(10 * PROBE_MAX_DIFF) + 1), probe)
    raise AssertionError("hata bekleniyordu")
except RuntimeError as e:
    assert "SİSTEMATİK" in str(e)
with warnings.catch_warnings():  # bilinçli geçiş
    warnings.simplefilter("ignore")
    assert check_frontend(eng(10), probe, allow=True)["differing"] == 10
with warnings.catch_warnings(record=True) as W:  # regresyon (2026-10-04 review): örnek yoksa denetim yok ama SESSİZ de değil
    warnings.simplefilter("always")
    assert check_frontend(eng(10), []) == dict(checked=0)
assert any("YAPILAMADI" in str(w.message) for w in W)

# regresyon (2026-10-04 review): v4+ sınır modele yalnız dp_feat ile gider; token'lar aynı, dp_feat farklıysa (etiketleyici/eşik/sınır kuralı değişti) yakalanmalı
dp_probe = [(f"c{i}", ["a", " ", "b"], [2, 2, 5]) for i in range(10)]
dp_eng = lambda changed: types.SimpleNamespace(frontend=lambda t: types.SimpleNamespace(tokens=["a", " ", "b"], dp_feat=[3, 3, 5] if int(t[1:]) < changed else [2, 2, 5]))
assert check_frontend(dp_eng(0), dp_probe) == dict(checked=10, differing=0)
try:
    check_frontend(dp_eng(10), dp_probe)
    raise AssertionError("hata bekleniyordu")
except RuntimeError as e:
    assert "dp_feat" in str(e)
assert check_frontend(dp_eng(10), [(t, k, None) for t, k, _ in dp_probe]) == dict(checked=10, differing=0)  # model dp_feat okumuyor: denetlenmez

# regresyon (2026-10-04 review): sembol tablosu checkpoint'inkinin devamı olmalı (araya atom girerse id'ler kayar)
check_symbols(SYMBOLS); check_symbols(SYMBOLS[:-2])
try:
    check_symbols(SYMBOLS[:10] + ["yeni"] + SYMBOLS[10:])
    raise AssertionError("hata bekleniyordu")
except RuntimeError as e:
    assert "id 10" in str(e)
print("OK")
