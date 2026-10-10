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


def init_leaves(template: LeafShapes, saved: LeafShapes, head: str, *, reuse_head: bool) -> list:
    """Leaves of ``template`` to copy from ``saved`` for a two-stage init.

    Every leaf outside ``head`` (backbone parameters and state) is copied; ``head``'s leaves only
    when ``reuse_head``, otherwise they keep their fresh initialisation. Raises ``ValueError``
    when a leaf to copy is missing from ``saved`` or has another shape.
    """
    load = sorted(p for p in template if reuse_head or not is_head_leaf(p, head))
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
