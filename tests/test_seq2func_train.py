import inspect
import json
import math

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
    frozen = train.build_parser().parse_args(["--arm", "shared", "--heads-only"])
    assert frozen.train_backbone is False
