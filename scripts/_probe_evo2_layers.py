"""Probe which Evo-2 intermediate layer names yield non-zero embeddings."""

import numpy as np
import torch
from evo2 import Evo2

m = Evo2("evo2_7b")
seq = "ACGT" * 2048  # 8192 bp
ids = torch.tensor(m.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to("cuda:0")
layers = [f"blocks.{i}.mlp.l3" for i in range(20, 32)]
with torch.no_grad():
    _, emb = m(ids, return_embeddings=True, layer_names=layers)
for ln in layers:
    a = emb[ln].squeeze(0).float().cpu().numpy()
    print(
        f"{ln:24s} shape={a.shape} absmean={np.abs(a).mean():.5f} "
        f"allzero={bool(np.abs(a).sum() == 0)}"
    )
