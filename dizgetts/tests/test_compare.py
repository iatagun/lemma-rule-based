"""eval/compare.py: öğeler `id` ile eşleşir ve bölüm süzülür (2026-09-26 code review regresyonu). Ağır bağımlılık yok.
  python -X utf8 -m dizgetts.tests.test_compare
Hata: `R[l][:n]` konumla eşleştiriyordu; sırası/uzunluğu farklı iki etiket sessizce yanlış klip çiftleriyle karşılaştırılıyordu."""
import json
import os
import tempfile

from dizgetts.eval.compare import load_items


def item(i, split, cer):
    return dict(id=i, split=split, cer_e=cer, n_chars=10, wer_e=0, n_words=2, utmos=3.0)


with tempfile.TemporaryDirectory() as root:
    for lab, items in (("a", [item("x", "test", 1), item("y", "val", 2), item("z", "extra", 3)]),
                       ("b", [item("z", "extra", 30), item("x", "test", 10), item("w", "test", 99)])):  # sıra farklı, "y" yok, "w" fazladan
        os.makedirs(f"{root}/{lab}")
        json.dump(dict(items=items), open(f"{root}/{lab}/results.json", "w", encoding="utf8"))
    R, dropped = load_items(["a", "b"], eval_root=root)
    assert [x["id"] for x in R["a"]] == [x["id"] for x in R["b"]] == ["x", "z"], R  # ilk etiketin sırası, yalnız ortak id'ler
    assert [x["cer_e"] for x in R["a"]] == [1, 3] and [x["cer_e"] for x in R["b"]] == [10, 30]  # konumla eşleşseydi (1,30) / (2,10) karışırdı
    assert dropped == {"a": 1, "b": 1}, dropped
    R, _ = load_items(["a", "b"], splits=["test"], eval_root=root)
    assert [x["id"] for x in R["a"]] == ["x"]
    R, _ = load_items(["a", "b"], splits=["val"], eval_root=root)
    assert R["a"] == [] and R["b"] == []  # ortak yok -> boş (val yalnız a'da)
print("OK")
