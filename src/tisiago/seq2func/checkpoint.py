"""Leaf-by-leaf checks for restoring a saved agft checkpoint into a freshly built model.

Pure Python (no JAX): the Orbax ``_METADATA`` file is read as JSON, and trees are nested
mappings / sequences whose leaves expose ``.shape``.
"""

from __future__ import annotations

import json
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
    """``{path: shape}`` recorded in an Orbax checkpoint directory's ``_METADATA``."""
    meta = json.loads((Path(checkpoint) / "_METADATA").read_text())["tree_metadata"]
    out: LeafShapes = {}
    for entry in meta.values():
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
