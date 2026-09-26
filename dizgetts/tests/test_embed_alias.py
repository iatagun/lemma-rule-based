"""Gömme bağı (train/embed_alias.py): eğitimde görülmeyen `a`/`aː` atomlarının gömmesi = ɑ/ɛ (ɑː/ɛː) ortalaması ("ae karışımı").
  python -X utf8 -m dizgetts.tests.test_embed_alias
Neden: Antalia'da `a` 0, `aː` 1 kez geçiyor; gradyan almayan gömme rastgele kalır ve o atomlu sözcük (kağıt, hakim) bozuk çıkar."""
from types import SimpleNamespace

import torch

from dizgetts.frontend.symbols import SYMBOL_TO_ID
from dizgetts.train.embed_alias import ALIASES, tie

torch.manual_seed(0)
emb = torch.nn.Embedding(len(SYMBOL_TO_ID), 8)
model = SimpleNamespace(encoder=SimpleNamespace(emb=emb))
W = lambda s: emb.weight[SYMBOL_TO_ID[s]].detach().clone()

before = {s: W(s) for s in SYMBOL_TO_ID}
tie(model, SYMBOL_TO_ID)
assert torch.allclose(W("a"), (W("ɑ") + W("ɛ")) / 2) and torch.allclose(W("aː"), (W("ɑː") + W("ɛː")) / 2)
changed = {s for s in SYMBOL_TO_ID if not torch.equal(before[s], W(s))}
assert changed == set(ALIASES), changed                      # yalnız takma atomlar değişir, kaynaklar ve diğerleri dokunulmaz

# eğitim adımı kaynakları değiştirir -> yeniden bağla ortalamayı izler (başlangıçta bir kez bağlamak YETMEZ)
with torch.no_grad():
    emb.weight[SYMBOL_TO_ID["ɑ"]] += 1.0
assert not torch.allclose(W("a"), (W("ɑ") + W("ɛ")) / 2)
tie(model, SYMBOL_TO_ID)
assert torch.allclose(W("a"), (W("ɑ") + W("ɛ")) / 2)

# tablo eşleşmiyorsa (ör. espeak sembolleri) sessiz geçmez
try:
    tie(model, {"x": 0})
    raise SystemExit("KeyError bekleniyordu")
except KeyError:
    pass
print("OK")
