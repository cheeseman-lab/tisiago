import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tisiago.seq2func import train
from tisiago.seq2func.samples import CONDITIONS
from tisiago.seq2func.train import (
    mean_window_kl,
    parse_arm,
    save_dir_atomic,
    train_loop,
    window_profile_kl,
)


def fake_step(params, state, opt_state, batch):
    return params - 1, opt_state + 1, {"loss": float(params)}


def scripted_eval(values):
    calls = []

    def eval_fn(params):
        calls.append(params)
        return {"val_profile_kl": values[len(calls) - 1]}

    return eval_fn, calls


def batches():
    while True:
        yield {"sequences": None}


def test_train_loop_selects_best_and_logs(tmp_path):
    eval_fn, eval_calls = scripted_eval([3.0, 2.0, 2.5])
    improved, checkpoints = [], []
    log = tmp_path / "log.jsonl"
    best = train_loop(
        fake_step,
        100,
        None,
        0,
        batches(),
        steps=30,
        eval_every=10,
        eval_fn=eval_fn,
        on_improve=lambda p, s: improved.append((p, s)),
        on_checkpoint=lambda p, o, s: checkpoints.append((p, o, s)),
        log_path=log,
    )
    assert eval_calls == [90, 80, 70]
    assert improved == [(90, 10), (80, 20)]
    assert checkpoints == [(90, 10, 10), (80, 20, 20), (70, 30, 30)]
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["step"] for r in lines] == [10, 20, 30]
    assert [r["val_profile_kl"] for r in lines] == [3.0, 2.0, 2.5]
    assert [r["improved"] for r in lines] == [True, True, False]
    assert lines[0]["train_loss"] == pytest.approx(np.mean(range(91, 101)))
    assert best["step"] == 20 and best["val_profile_kl"] == 2.0


def test_train_loop_logs_loss_components(tmp_path):
    def step(params, state, opt_state, batch):
        aux = {"loss": 10.0, "poisson": 6.0, "multinomial": 0.5, "frame": float(params)}
        return params - 1, opt_state, aux

    eval_fn, _ = scripted_eval([1.0, 0.5])
    log = tmp_path / "log.jsonl"
    train_loop(
        step,
        4,
        None,
        0,
        batches(),
        steps=4,
        eval_every=2,
        eval_fn=eval_fn,
        on_improve=lambda p, s: None,
        on_checkpoint=lambda p, o, s: None,
        log_path=log,
    )
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["train_frame"] for r in lines] == [3.5, 1.5]
    assert lines[0]["train_poisson"] == 6.0 and lines[0]["train_multinomial"] == 0.5


def test_window_frame_kl_matches_loss_definition():
    obs = np.zeros((60, 2))
    obs[0::3, 1] = 4.0
    obs[1::3, 1] = 1.0
    mask = np.ones_like(obs, bool)
    num, den = train.window_frame_kl(2 * obs, obs, mask, (1,), 10)
    assert num == pytest.approx(0.0, abs=1e-9) and den == obs.sum()
    shifted = np.roll(obs, 1, axis=0)
    assert train.window_frame_kl(shifted, obs, mask, (1,), 10)[0] > 0
    assert train.window_frame_kl(shifted, obs, mask, (0,), 10) == (0.0, 0.0)
    mask[5, 1] = False  # kills the first block only
    num_masked, den_masked = train.window_frame_kl(shifted, obs, mask, (1,), 10)
    assert den_masked == pytest.approx(den / 2)


def test_train_loop_evaluates_final_partial_interval(tmp_path):
    eval_fn, eval_calls = scripted_eval([1.0, 0.5])
    best = train_loop(
        fake_step,
        10,
        None,
        0,
        batches(),
        steps=7,
        eval_every=5,
        eval_fn=eval_fn,
        on_improve=lambda p, s: None,
        on_checkpoint=lambda p, o, s: None,
        log_path=tmp_path / "log.jsonl",
    )
    assert eval_calls == [5, 3]
    assert best["step"] == 7


def test_train_loop_non_finite_kl_never_improves(tmp_path):
    eval_fn, _ = scripted_eval([math.nan, 2.0, math.inf])
    improved = []
    log = tmp_path / "log.jsonl"
    best = train_loop(
        fake_step,
        100,
        None,
        0,
        batches(),
        steps=3,
        eval_every=1,
        eval_fn=eval_fn,
        on_improve=lambda p, s: improved.append(s),
        on_checkpoint=lambda p, o, s: None,
        log_path=log,
    )
    assert improved == [2]
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines[0]["val_profile_kl"] is None and lines[2]["val_profile_kl"] is None
    assert best["step"] == 2


def test_train_loop_exception_keeps_completed_log_and_reraises(tmp_path):
    def step(params, state, opt_state, batch):
        if params == 85:
            raise RuntimeError("boom")
        return fake_step(params, state, opt_state, batch)

    eval_fn, _ = scripted_eval([3.0, 2.0, 2.5])
    log = tmp_path / "log.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        train_loop(
            step,
            100,
            None,
            0,
            batches(),
            steps=30,
            eval_every=10,
            eval_fn=eval_fn,
            on_improve=lambda p, s: None,
            on_checkpoint=lambda p, o, s: None,
            log_path=log,
        )
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["step"] for r in lines] == [10]


def test_train_loop_rejects_bad_intervals(tmp_path):
    kwargs = dict(
        eval_fn=lambda p: {"val_profile_kl": 1.0},
        on_improve=lambda p, s: None,
        on_checkpoint=lambda p, o, s: None,
        log_path=tmp_path / "log.jsonl",
    )
    with pytest.raises(ValueError):
        train_loop(fake_step, 1, None, 0, batches(), steps=0, eval_every=1, **kwargs)
    with pytest.raises(ValueError):
        train_loop(fake_step, 1, None, 0, batches(), steps=5, eval_every=0, **kwargs)


def test_parse_arm():
    assert parse_arm("shared") == ("p6_shared", tuple(CONDITIONS))
    assert parse_arm("condition:HeLa") == ("p6_hela", ("HeLa",))
    assert parse_arm("condition:RPE1_Que") == ("p6_rpe1_que", ("RPE1_Que",))
    for bad in ("condition:Nope", "condition:", "hela", "shared:HeLa"):
        with pytest.raises(ValueError):
            parse_arm(bad)


def test_window_profile_kl_averages_channels_and_windows():
    target = np.array([[1.0, 0.0], [1.0, 0.0]])
    pred = np.array([[1.0, 1.0], [1.0, 1.0]])
    mask = np.ones_like(target, bool)
    assert window_profile_kl(pred, target, mask) == pytest.approx(0.0)
    assert math.isnan(window_profile_kl(pred, np.zeros_like(target), mask))
    assert mean_window_kl([1.0, math.nan, 3.0]) == pytest.approx(2.0)
    assert math.isnan(mean_window_kl([math.nan]))
    assert math.isnan(mean_window_kl([]))


def test_save_dir_atomic_replaces_existing(tmp_path):
    dest = tmp_path / "best"

    def writer(text):
        def write(path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "x.txt").write_text(text)

        return write

    save_dir_atomic(writer("one"), dest)
    assert (dest / "x.txt").read_text() == "one"
    save_dir_atomic(writer("two"), dest)
    assert (dest / "x.txt").read_text() == "two"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best"]


def test_save_dir_atomic_failure_keeps_previous(tmp_path):
    dest = tmp_path / "last"
    dest.mkdir()
    (dest / "x.txt").write_text("old")

    def broken(path):
        path.mkdir(parents=True, exist_ok=True)
        raise OSError("disk")

    with pytest.raises(OSError):
        save_dir_atomic(broken, dest)
    assert (dest / "x.txt").read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last"]


def test_train_module_never_reads_test_split():
    assert "test.parquet" not in inspect.getsource(train)


def test_cli_defaults_are_primary_config():
    args = train.build_parser().parse_args(["--arm", "shared"])
    assert args.train_backbone is True
    assert args.learning_rate == 1e-3
    assert args.backbone_learning_rate == 1e-4
    assert args.warmup_steps == 300
    assert args.clip_norm == 1.0
    assert args.head == "mlp"
    assert args.workers == 4
    assert str(args.tracks_dir) == "data/p6/tracks"
    assert args.frame_weight == 0.0
    assert args.frame_block == 10
    frozen = train.build_parser().parse_args(["--arm", "shared", "--heads-only"])
    assert frozen.train_backbone is False


def test_save_dir_atomic_rolls_back_when_swap_fails(tmp_path, monkeypatch):
    dest = tmp_path / "last"
    dest.mkdir()
    (dest / "x.txt").write_text("old")
    real_replace = train.os.replace

    def flaky_replace(src, dst):
        if str(src).endswith(".tmp"):
            raise OSError("swap failed")
        return real_replace(src, dst)

    monkeypatch.setattr(train.os, "replace", flaky_replace)

    def write(path):
        (path / "x.txt").write_text("new")

    with pytest.raises(OSError, match="swap failed"):
        save_dir_atomic(write, dest)
    assert (dest / "x.txt").read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last"]


@pytest.mark.parametrize("record", ["log.jsonl", "run.json", "best", "last"])
def test_existing_run_records_detected(tmp_path, record):
    assert train.existing_run_records(tmp_path) == []
    target = tmp_path / record
    target.mkdir() if "." not in record else target.write_text("{}")
    assert train.existing_run_records(tmp_path) == [record]
    with pytest.raises(ValueError, match="--overwrite"):
        train.check_out_dir(tmp_path, overwrite=False)
    train.check_out_dir(tmp_path, overwrite=True)
    assert target.exists()


def test_cli_refuses_existing_run_and_deletes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    (out / "log.jsonl").write_text('{"step": 1}\n')
    argv = ["train", "--arm", "condition:HeLa", "--checkpoint", "x", "--genome", "y"]
    monkeypatch.setattr("sys.argv", [*argv, "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code != 0
    assert (out / "log.jsonl").read_text() == '{"step": 1}\n'
    assert sorted(p.name for p in out.iterdir()) == ["log.jsonl"]


def test_cli_frame_options_parse():
    args = train.build_parser().parse_args(
        ["--arm", "shared", "--frame-weight", "0.5", "--frame-block", "4"]
    )
    assert (args.frame_weight, args.frame_block) == (0.5, 4)


def test_cli_rejects_bad_frame_options(tmp_path, monkeypatch):
    for bad in (["--frame-weight", "-1"], ["--frame-block", "0"]):
        argv = ["train", "--arm", "condition:HeLa", "--checkpoint", "x", "--genome", "y"]
        monkeypatch.setattr("sys.argv", [*argv, "--out", str(tmp_path / "r"), *bad])
        with pytest.raises(SystemExit) as exc:
            train.main()
        assert exc.value.code != 0
    assert not (tmp_path / "r").exists()


def test_default_out_separates_frozen_baseline():
    assert train.default_out("condition:HeLa", train_backbone=True) == Path(
        "data/p6/runs/condition_hela"
    )
    assert train.default_out("condition:HeLa", train_backbone=False) == Path(
        "data/p6/runs/condition_hela_frozen"
    )
    assert train.default_out("shared", True) != train.default_out("shared", False)


def _fake_run(tmp_path, head_name="p6_k562", kind=None):
    best = tmp_path / "src" / "best"
    (best / "checkpoint").mkdir(parents=True)
    args = {"arm": "condition:K562"} | ({"head": kind} if kind else {})
    run = {"args": args, "head_name": head_name, "best": {"step": 2000}}
    (best.parent / "run.json").write_text(json.dumps(run))
    return best


@pytest.mark.parametrize(
    ("source_kind", "kind", "source_head", "reuse"),
    [
        ("mlp", "mlp", "p6_k562", "all"),
        (None, "mlp", "p6_k562", "all"),  # runs before --head recorded mlp implicitly
        ("mlp", "seqconv", "p6_k562", "none"),
        ("mlp", "mlp", "p6_hela", "none"),
        ("mlp", "resid_tx", "p6_k562", "matching"),
        (None, "resid_tx", "p6_k562", "matching"),
        ("mlp", "resid_tx", "p6_hela", "none"),
        ("seqconv_tx", "resid_tx", "p6_k562", "none"),
        ("mlp", "resid_chx", "p6_k562", "matching"),
        (None, "resid_chx", "p6_k562", "matching"),
        ("mlp", "resid_chx", "p6_hela", "none"),
        ("resid_tx", "resid_chx", "p6_k562", "none"),
        ("resid_chx", "resid_chx", "p6_k562", "all"),
    ],
)
def test_init_source_reuses_head_only_for_same_arm_and_kind(
    tmp_path, source_kind, kind, source_head, reuse
):
    best = _fake_run(tmp_path, source_head, source_kind)
    init = train.init_source(best, "p6_k562", kind)
    assert init["reuse_head"] == reuse
    assert init["path"] == str(best)
    assert init["source_kind"] == (source_kind or "mlp")


def test_init_source_requires_checkpoint_dir(tmp_path):
    with pytest.raises(ValueError, match="no checkpoint/"):
        train.init_source(tmp_path, "p6_k562", "mlp")


def test_cli_two_stage_defaults_are_off():
    args = train.build_parser().parse_args(["--arm", "shared"])
    assert args.init_from is None
    assert args.freeze_backbone_steps == 0


@pytest.mark.parametrize(
    "bad",
    [
        ["--freeze-backbone-steps", "-1"],
        ["--freeze-backbone-steps", "10", "--steps", "10"],
        ["--freeze-backbone-steps", "5", "--heads-only"],
        ["--init-from", "does/not/exist"],
    ],
)
def test_cli_rejects_bad_two_stage_options(tmp_path, monkeypatch, bad):
    argv = ["train", "--arm", "condition:HeLa", "--checkpoint", "x", "--genome", "y"]
    monkeypatch.setattr("sys.argv", [*argv, "--out", str(tmp_path / "r"), *bad])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code != 0
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("best", [None, "missing"])
def test_init_source_refuses_unfinished_run(tmp_path, best):
    src = _fake_run(tmp_path)
    run = json.loads((src.parent / "run.json").read_text())
    if best == "missing":
        del run["best"]
    else:
        run["best"] = None
    (src.parent / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="not a finished run"):
        train.init_source(src, "p6_k562", "mlp")


def test_init_source_refuses_missing_run_json(tmp_path):
    src = _fake_run(tmp_path)
    (src.parent / "run.json").unlink()
    with pytest.raises(ValueError, match="no run.json"):
        train.init_source(src, "p6_k562", "mlp")


def test_cli_seqconv_tx_needs_gtf(tmp_path, monkeypatch):
    monkeypatch.delenv("TISIAGO_GTF", raising=False)
    args = train.build_parser().parse_args(["--arm", "shared", "--gtf", "a.gtf"])
    assert args.gtf == "a.gtf"
    argv = ["train", "--arm", "condition:HeLa", "--checkpoint", "x", "--genome", "y"]
    monkeypatch.setattr("sys.argv", [*argv, "--out", str(tmp_path / "r"), "--head", "seqconv_tx"])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code != 0
    assert not (tmp_path / "r").exists()


def test_cli_mod_learning_rate_parses_and_defaults_to_none():
    args = train.build_parser().parse_args(["--arm", "shared"])
    assert args.mod_learning_rate is None
    args = train.build_parser().parse_args(
        ["--arm", "shared", "--head", "resid_chx", "--mod-learning-rate", "1e-3"]
    )
    assert args.mod_learning_rate == 1e-3


@pytest.mark.parametrize(
    "bad",
    [
        ["--head", "mlp", "--mod-learning-rate", "1e-3"],
        ["--head", "seqconv_tx", "--mod-learning-rate", "1e-3"],
        ["--head", "resid_chx", "--mod-learning-rate", "-1e-3"],
    ],
)
def test_cli_rejects_mod_learning_rate_without_mod_head(tmp_path, monkeypatch, bad):
    argv = ["train", "--arm", "condition:HeLa", "--checkpoint", "x", "--genome", "y"]
    monkeypatch.setattr("sys.argv", [*argv, "--gtf", "a.gtf", "--out", str(tmp_path / "r"), *bad])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code != 0
    assert not (tmp_path / "r").exists()


def test_cli_mod_libraries_and_scale_parse_with_none_defaults():
    args = train.build_parser().parse_args(["--arm", "shared"])
    assert args.mod_libraries is None and args.mod_scale_tis is None
    args = train.build_parser().parse_args(
        ["--arm", "shared", "--mod-libraries", "chx,tis", "--mod-scale-tis", "3"]
    )
    assert args.mod_libraries == "chx,tis" and args.mod_scale_tis == 3.0


@pytest.mark.parametrize(
    "bad",
    [
        ["--head", "mlp", "--mod-libraries", "chx"],
        ["--head", "resid_tx", "--mod-libraries", "chx,tis"],
        ["--head", "resid_tx", "--mod-scale-tis", "3"],
        ["--head", "resid_chx", "--mod-libraries", "chx,harr"],
        ["--head", "resid_chx", "--mod-libraries", "chx,chx"],
        ["--head", "resid_chx", "--mod-libraries", ""],
        ["--head", "resid_chx", "--mod-libraries", "chx,tis", "--mod-scale-tis", "0"],
        ["--head", "resid_chx", "--mod-scale-tis", "3"],
    ],
)
def test_cli_rejects_bad_mod_libraries(tmp_path, monkeypatch, bad):
    argv = ["train", "--arm", "condition:HeLa", "--checkpoint", "x", "--genome", "y"]
    monkeypatch.setattr("sys.argv", [*argv, "--gtf", "a.gtf", "--out", str(tmp_path / "r"), *bad])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code != 0
    assert not (tmp_path / "r").exists()


def test_mod_libraries_parse():
    assert train.mod_libraries_of(None) == ("chx",)
    assert train.mod_libraries_of("chx,tis") == ("chx", "tis")
    assert train.mod_libraries_of(" tis , chx ") == ("tis", "chx")
    for bad in ("", "chx,", "chx,chx", "chx,harr"):
        with pytest.raises(ValueError, match="mod-libraries"):
            train.mod_libraries_of(bad)


def test_channel_config_frame_stays_chx_and_records_modulation():
    pytest.importorskip("alphagenome_ft")
    from tisiago.seq2func.model import MOD_SCALE
    from tisiago.seq2func.targets import channel_layout, library_channels

    layout = channel_layout(["HeLa", "K562"])
    chx = library_channels(layout, "chx")
    cfg = train.channel_config("resid_chx", layout, ("chx", "tis"), 3.0)
    assert cfg["frame_channels"] == list(chx)
    assert cfg["mod_libraries"] == ["chx", "tis"]
    assert cfg["mod_channels"] == [0, 1, 2, 3, 6, 7, 8, 9]
    assert cfg["mod_scales"] == [
        3.0 if layout[c][1] == "tis" else MOD_SCALE for c in cfg["mod_channels"]
    ]
    assert cfg["mod_scale_tis"] == 3.0
    old = train.channel_config("resid_chx", layout, ("chx",), MOD_SCALE)
    assert old["frame_channels"] == old["mod_channels"] == list(chx)
    assert old["mod_scales"] == [MOD_SCALE] * 4
    plain = train.channel_config("mlp", layout, ("chx",), MOD_SCALE)
    assert plain == {"frame_channels": list(chx)}


def test_cli_resid_chx_needs_gtf(tmp_path, monkeypatch):
    monkeypatch.delenv("TISIAGO_GTF", raising=False)
    argv = ["train", "--arm", "condition:HeLa", "--checkpoint", "x", "--genome", "y"]
    monkeypatch.setattr("sys.argv", [*argv, "--out", str(tmp_path / "r"), "--head", "resid_chx"])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code != 0
