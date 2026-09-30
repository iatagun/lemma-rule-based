"""Ön uç sürüm tuzağı denetimi (eval/synth.check_frontend; docs/v9_plan.md §5).
  python -X utf8 -m dizgetts.tests.test_frontend_probe
Checkpoint eğitim token'larıyla güncel ön uç SİSTEMATİK olarak farklıysa (v6 + bugünkü ön uç) Synth hata vermeli; küçük sözlük düzeltmesi uyarı."""
import types
import warnings

from dizgetts.eval.synth import PROBE_MAX_DIFF, check_frontend

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
assert check_frontend(eng(10), []) == dict(checked=0)  # örnek yoksa denetim yok
print("OK")
