import json

import numpy as np
import pytest

from tisiago.seq2func.checkpoint import check_restorable, saved_leaf_shapes, tree_leaf_shapes


def fake_model_trees():
    params = {
        "alphagenome/embed": {"embeddings": np.zeros((2, 8))},
        "head/p6_hela/~predict/hidden": {"w": np.zeros((8, 4)), "b": np.zeros(4)},
        "head/p6_hela/~predict/output": {"w": np.zeros((4, 6)), "b": np.zeros(6)},
    }
    state = {"alphagenome/norm": {"var_ema": np.zeros(())}}
    return params, state


def write_orbax_metadata(directory, shapes):
    tree = {}
    for path, shape in shapes.items():
        keys = [{"key": k, "key_type": 1 if k.isdigit() else 2} for k in path]
        tree[str(path)] = {
            "key_metadata": keys,
            "value_metadata": {"value_type": "jax.Array", "write_shape": list(shape)},
        }
    (directory / "_METADATA").write_text(json.dumps({"tree_metadata": tree}))


def test_tree_leaf_shapes_paths_follow_orbax_keys():
    shapes = tree_leaf_shapes(fake_model_trees())
    assert shapes[("0", "head/p6_hela/~predict/output", "w")] == (4, 6)
    assert shapes[("1", "alphagenome/norm", "var_ema")] == ()
    assert len(shapes) == 6


def test_saved_leaf_shapes_round_trip_matches_model(tmp_path):
    shapes = tree_leaf_shapes(fake_model_trees())
    write_orbax_metadata(tmp_path, shapes)
    assert saved_leaf_shapes(tmp_path) == shapes
    assert check_restorable(shapes, saved_leaf_shapes(tmp_path)) == 6


def test_missing_head_leaf_is_refused():
    model = tree_leaf_shapes(fake_model_trees())
    saved = {p: s for p, s in model.items() if "output" not in p[1]}
    with pytest.raises(ValueError, match=r"missing \(2\).*p6_hela/~predict/output"):
        check_restorable(model, saved)


def test_unexpected_leaf_is_refused():
    model = tree_leaf_shapes(fake_model_trees())
    saved = {**model, ("0", "head/p6_shared/~predict/output", "w"): (4, 36)}
    with pytest.raises(ValueError, match="unexpected"):
        check_restorable(model, saved)


def test_shape_mismatch_is_refused():
    model = tree_leaf_shapes(fake_model_trees())
    saved = {**model, ("0", "head/p6_hela/~predict/output", "w"): (4, 36)}
    with pytest.raises(ValueError, match=r"model \(4, 6\) vs saved \(4, 36\)"):
        check_restorable(model, saved)
