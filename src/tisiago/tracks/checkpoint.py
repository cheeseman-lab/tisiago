"""Leaf-by-leaf checks for restoring a saved agft checkpoint into a freshly built model.

Pure Python (no JAX): the Orbax ``_METADATA`` file is read as JSON, and trees are nested
mappings / sequences whose leaves expose ``.shape``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

LeafShapes = dict[tuple[str, ...], tuple[int, ...]]


def tree_leaf_shapes(tree, prefix: tuple[str, ...] = ()) -> LeafShapes:
    """``{path: shape}`` of a nested mapping/sequence tree; sequence indices become ``"0"``..."""
    if isinstance(tree, Mapping):
        items = ((str(k), v) for k, v in tree.items())
    elif isinstance(tree, Sequence) and not isinstance(tree, str):
        items = ((str(i), v) for i, v in enumerate(tree))
    else:
        return {prefix: tuple(int(d) for d in tree.shape)}
    out: LeafShapes = {}
    for key, value in items:
        out.update(tree_leaf_shapes(value, prefix + (key,)))
    return out


def saved_leaf_shapes(checkpoint: str | Path) -> LeafShapes:
    """``{path: shape}`` recorded in an Orbax checkpoint directory's ``_METADATA``.

    Entries without a ``write_shape`` (empty containers such as an empty state dict) are skipped.
    """
    meta = json.loads((Path(checkpoint) / "_METADATA").read_text())["tree_metadata"]
    out: LeafShapes = {}
    for entry in meta.values():
        if "write_shape" not in entry["value_metadata"]:
            continue
        path = tuple(str(k["key"]) for k in entry["key_metadata"])
        out[path] = tuple(int(d) for d in entry["value_metadata"]["write_shape"])
    return out


def check_restorable(template: LeafShapes, saved: LeafShapes) -> int:
    """Raise ``ValueError`` unless ``saved`` has exactly ``template``'s leaves and shapes.

    Returns:
        The number of matched leaves.
    """
    missing = sorted(set(template) - set(saved))
    unexpected = sorted(set(saved) - set(template))
    mismatched = sorted(p for p in set(template) & set(saved) if template[p] != saved[p])
    if missing or unexpected or mismatched:
        lines = [f"checkpoint does not match the model ({len(template)} model leaves):"]
        for label, paths in (("missing", missing), ("unexpected", unexpected)):
            if paths:
                lines.append(f"  {label} ({len(paths)}): {['/'.join(p) for p in paths[:5]]}")
        for p in mismatched[:5]:
            lines.append(f"  shape {'/'.join(p)}: model {template[p]} vs saved {saved[p]}")
        raise ValueError("\n".join(lines))
    return len(template)


def is_head_leaf(path: tuple[str, ...], head: str) -> bool:
    """Whether an Orbax leaf path belongs to custom head ``head`` (agft's head-path rule)."""
    name = "/".join(path[1:])
    return f"/head/{head}/" in name or name.startswith(f"head/{head}/")


def init_leaves(
    template: LeafShapes, saved: LeafShapes, head: str, *, reuse_head: bool | str
) -> list:
    """Leaves of ``template`` to copy from ``saved`` for a two-stage init.

    Every leaf outside ``head`` (backbone parameters and state) is copied. ``head``'s leaves:
    ``reuse_head`` ``"all"`` (or True) copies all, ``"none"`` (or False) keeps them fresh,
    ``"matching"`` copies those present in ``saved`` with the same shape (a partly shared head).
    Raises ``ValueError`` when a leaf to copy is missing from ``saved`` or has another shape.
    """
    mode = {True: "all", False: "none"}.get(reuse_head, reuse_head)
    if mode not in ("all", "none", "matching"):
        raise ValueError(f"reuse_head must be all, matching or none, got {reuse_head!r}")

    def wanted(p):
        if not is_head_leaf(p, head):
            return True
        return mode == "all" or (mode == "matching" and saved.get(p) == template[p])

    load = sorted(p for p in template if wanted(p))
    missing = [p for p in load if p not in saved]
    mismatched = [p for p in load if p in saved and saved[p] != template[p]]
    if missing or mismatched:
        lines = [f"init checkpoint does not match the model ({len(load)} leaves to load):"]
        if missing:
            lines.append(f"  missing ({len(missing)}): {['/'.join(p) for p in missing[:5]]}")
        for p in mismatched[:5]:
            lines.append(f"  shape {'/'.join(p)}: model {template[p]} vs saved {saved[p]}")
        raise ValueError("\n".join(lines))
    return load


def finite_or_none(value: float) -> float | None:
    """``value`` if finite, else ``None`` (serialised as JSON ``null``)."""
    return value if math.isfinite(value) else None


def write_json_atomic(payload: dict, out: Path) -> None:
    """Write ``payload`` to ``out`` via a temp file in the same directory and ``os.replace``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
