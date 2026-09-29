"""Eğitimde görülmeyen atomlar için gömme bağı: `a` = (ɑ + ɛ) / 2, `aː` = (ɑː + ɛː) / 2 ("ae karışımı", kullanıcı tarifi 2026-09-26).

Neden: özenli söyleyiş sözlüğü (frontend/pronounce.py) kağıt/hakim için ön `a`/`aː` üretir, ama Antalia'da `a` 0, `aː` 1 kez geçiyor (`ɑ` 17.514, `ɛ` 10.322);
gradyan almayan gömme rastgele kalır. Bağ HER optimizer adımından sonra yenilenir (kaynak gömmeler eğitimde değişir; yalnız başlangıçta bağlamak yetmez) ve
sentezde uygulanır. cfg: `model.embed_alias: true`.
`model.embed_alias: init` (2026-09-29): alıntı kurallarıyla a/aː eğitimde yeterince geçiyor -> bağ YALNIZ başlangıçta (ortalamadan başlat), sonra serbest öğrenilir; sentezde bağlanmaz.
"""
import torch

ALIASES = {"a": ("ɑ", "ɛ"), "aː": ("ɑː", "ɛː")}


@torch.no_grad()
def tie(model, symbol_to_id: dict, aliases: dict = ALIASES) -> None:
    """model.encoder.emb (Matcha TextEncoder) satırlarını kaynakların ortalamasına eşitler. Tabloda olmayan sembol KeyError (sessiz geçmez)."""
    w = model.encoder.emb.weight
    for new, srcs in aliases.items():
        w[symbol_to_id[new]] = torch.stack([w[symbol_to_id[s]] for s in srcs]).mean(0)
