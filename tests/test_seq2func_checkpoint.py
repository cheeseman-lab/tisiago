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


def test_init_leaves_fresh_head_copies_everything_else():
    from tisiago.seq2func.checkpoint import init_leaves, is_head_leaf

    model = tree_leaf_shapes(fake_model_trees())
    saved = {p: s for p, s in model.items() if not is_head_leaf(p, "p6_hela")}
    saved[("0", "head/p6_hela/~predict/hidden", "w")] = (31, 8, 4)  # another head kind
    load = init_leaves(model, saved, "p6_hela", reuse_head=False)
    assert load == sorted(p for p in model if "p6_hela" not in p[1])
    assert len(load) == 2


def test_init_leaves_reuse_head_requires_matching_head():
    from tisiago.seq2func.checkpoint import init_leaves

    model = tree_leaf_shapes(fake_model_trees())
    assert init_leaves(model, model, "p6_hela", reuse_head=True) == sorted(model)
    saved = {**model, ("0", "head/p6_hela/~predict/hidden", "w"): (31, 8, 4)}
    with pytest.raises(ValueError, match=r"model \(8, 4\) vs saved \(31, 8, 4\)"):
        init_leaves(model, saved, "p6_hela", reuse_head=True)


def test_init_leaves_matching_copies_shared_head_leaves_only():
    from tisiago.seq2func.checkpoint import init_leaves, is_head_leaf

    model = tree_leaf_shapes(fake_model_trees())
    model[("0", "head/p6_hela/~predict/mod_output", "w")] = (1, 64, 4)
    saved = {p: s for p, s in model.items() if "mod_output" not in p[1]}
    load = init_leaves(model, saved, "p6_hela", reuse_head="matching")
    assert load == sorted(p for p in model if "mod_output" not in p[1])
    assert any(is_head_leaf(p, "p6_hela") for p in load)
    saved[("0", "head/p6_hela/~predict/hidden", "w")] = (31, 8, 4)  # shape differs: not copied
    skipped = init_leaves(model, saved, "p6_hela", reuse_head="matching")
    assert ("0", "head/p6_hela/~predict/hidden", "w") not in skipped
    assert init_leaves(model, saved, "p6_hela", reuse_head="none") == sorted(
        p for p in model if not is_head_leaf(p, "p6_hela")
    )
    with pytest.raises(ValueError, match="all, matching or none"):
        init_leaves(model, saved, "p6_hela", reuse_head="some")


def test_init_leaves_refuses_missing_backbone_leaf():
    from tisiago.seq2func.checkpoint import init_leaves

    model = tree_leaf_shapes(fake_model_trees())
    saved = {p: s for p, s in model.items() if p[1] != "alphagenome/embed"}
    with pytest.raises(ValueError, match=r"missing \(1\).*alphagenome/embed"):
        init_leaves(model, saved, "p6_hela", reuse_head=False)


def test_is_head_leaf_matches_agft_rule():
    from tisiago.seq2func.checkpoint import is_head_leaf

    assert is_head_leaf(("0", "head/h/~predict/hidden", "w"), "h")
    assert not is_head_leaf(("0", "head/hh/~predict/hidden", "w"), "h")
    assert not is_head_leaf(("0", "alphagenome/head/rna_seq/output", "w"), "h")
    assert not is_head_leaf(("1", "alphagenome/norm", "var_ema"), "h")


def test_saved_leaf_shapes_skips_empty_containers(tmp_path):
    shapes = tree_leaf_shapes(fake_model_trees())
    write_orbax_metadata(tmp_path, shapes)
    meta = json.loads((tmp_path / "_METADATA").read_text())
    meta["tree_metadata"]["('1',)"] = {
        "key_metadata": [{"key": "1", "key_type": 1}],
        "value_metadata": {"value_type": "Dict", "skip_deserialize": True},
    }
    (tmp_path / "_METADATA").write_text(json.dumps(meta))
    assert saved_leaf_shapes(tmp_path) == shapes
