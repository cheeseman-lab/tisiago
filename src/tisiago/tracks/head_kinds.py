"""Track-head kind names shared by the trainers and evaluators (no JAX import)."""

from __future__ import annotations

RNACOND_FACT_KINDS = (
    "rnacond_f",
    "rnacond_fo",
    "rnacond_foc",
    "rnacond_fod",
    "rnacond_foa",
    "rnacond_focd",
    "rnacond_foca",
    "rnacond_fc2",
    "rnacond_fc2d",
)
HEAD_CHOICES = (
    "mlp",
    "conv",
    "seqconv",
    "seqconv_tx",
    "resid_tx",
    "resid_chx",
    "rnacond",
    *RNACOND_FACT_KINDS,
)
TRANSCRIPT_KINDS = frozenset({"seqconv_tx", "resid_tx", "resid_chx"})
MOD_KINDS = frozenset({"resid_tx", "resid_chx"})
CHX_KINDS = frozenset({"resid_chx"})
RNA_KINDS = frozenset({"rnacond", *RNACOND_FACT_KINDS})
