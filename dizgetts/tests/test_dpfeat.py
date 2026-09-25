"""v4 süre tahmincisi özniteliği (train/dpfeat.py) güvenlik testleri.
  python -X utf8 -m dizgetts.tests.test_dpfeat
1) sıfır gömme -> v3a ile BİREBİR aynı mu/logw   2) öznitelik mu'yu (hizalamayı) DEĞİŞTİRMEZ, yalnız logw   3) gradyan gömmeye ulaşır
4) engine: dp_feat uzunluğu = token uzunluğu; add_blank yayılımı = ids uzunluğu"""
import copy
import os

import torch
from matcha.utils.utils import intersperse

from dizgetts.engine import DP_FEAT, Engine
from dizgetts.frontend.symbols import SYMBOL_TO_ID
from dizgetts.train import dpfeat
from dizgetts.train.train import build_model

CK = "D:/dizgetts/runs/v3a_g2ptts_nb_e150_20260925-213638/ep150.pt"

# 4) engine + yayılım (model gerektirmez)
e = Engine(bert_fallback=False)
for t in ("Merhaba, 3'te buluşalım!", "Evet; peki ama neden? Bilmiyorum.", "Tek"):
    u = e.frontend(t)
    assert len(u.dp_feat) == len(u.tokens), t
    assert len(dpfeat.intersperse_feat(u.dp_feat)) == len(intersperse(e.ids(u), 0)), t
    assert all(f in DP_FEAT.values() and f != DP_FEAT["pad"] for f in u.dp_feat), t
u = e.frontend("Merhaba, 3'te buluşalım!")
i = u.tokens.index(",")
assert u.dp_feat[i] == DP_FEAT["ip"] and u.dp_feat[i + 1] == DP_FEAT["ip"], (u.tokens, u.dp_feat)  # virgül + sonraki ayraç
assert u.dp_feat[-1] == DP_FEAT["cümle"] and u.dp_feat[0] == DP_FEAT["in"]
print("OK (engine dp_feat)")

if not os.path.exists(CK):
    print("checkpoint yok, model testleri atlandı"); raise SystemExit
ck = torch.load(CK, map_location="cpu", weights_only=False)
stats = {"mel_mean": -5.266, "mel_std": 1.957}
base = build_model(ck["cfg"], len(ck["symbols"]), stats); base.load_state_dict(ck["model"]); base.eval()
cfg = copy.deepcopy(ck["cfg"]); cfg["model"]["dp_feat"] = True
dp = build_model(cfg, len(ck["symbols"]), stats)
missing, unexpected = dp.load_state_dict(ck["model"], strict=False)
assert missing == ["encoder.dp_feat_emb.weight"] and not unexpected, (missing, unexpected)
dp.eval()

ids = torch.tensor([intersperse([SYMBOL_TO_ID[t] for t in u.tokens], 0)])
xl = torch.tensor([ids.shape[1]])
f = torch.tensor([dpfeat.intersperse_feat(u.dp_feat)])
with torch.no_grad():
    mu0, lw0, _ = base.encoder(ids, xl)
    dpfeat.set_dp_feat(dp, f)
    mu1, lw1, _ = dp.encoder(ids, xl)
assert torch.equal(mu0, mu1) and torch.allclose(lw0, lw1, atol=0), "sıfır gömme v3a'dan farklı çıktı"
print("OK (1: sıfır gömme = v3a)")

torch.nn.init.normal_(dp.encoder.dp_feat_emb.weight, std=0.5)
with torch.no_grad():
    mu2, lw2, _ = dp.encoder(ids, xl)
assert torch.equal(mu0, mu2), "öznitelik mu'yu (hizalama) değiştirdi!"
assert not torch.allclose(lw0, lw2), "öznitelik logw'yi değiştirmedi"
print("OK (2: mu sabit, logw değişiyor)")

dp.train()
torch.nn.init.zeros_(dp.encoder.dp_feat_emb.weight)
y = torch.randn(1, 80, 400)
dpfeat.set_dp_feat(dp, f)
dur, prior, diff, *_ = dp(x=ids, x_lengths=xl, y=y, y_lengths=torch.tensor([400]), out_size=172)
dur.backward()
g = dp.encoder.dp_feat_emb.weight.grad
assert g is not None and g[DP_FEAT["pad"]].abs().sum() == 0 and g[1:].abs().sum() > 0, g
assert all(p.grad is None or p.grad.abs().sum() == 0 for n, p in dp.named_parameters() if n.startswith("encoder.encoder.")), \
    "süre kaybı kodlayıcı gövdesine gradyan sızdırdı (x_dp detach olmalı)"
print("OK (3: gradyan yalnız süre tarafına)")
