"""scripts/run_p6_trial.sh against fake sbatch / squeue / scancel (no Slurm)."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_p6_trial.sh"
LANE0 = f"{os.environ.get('USER', 'u')}_tisiago_p6ar_lane0"

FAKE_SBATCH = """#!/bin/bash
n=$(( $(cat "$FAKE_DIR/n" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$FAKE_DIR/n"
printf '%s\\n' "$*" >> "$FAKE_DIR/sbatch.log"
[[ "$n" == "${FAIL_AT:-0}" ]] && exit 1
echo "$((900 + n))"
"""
FAKE_SQUEUE = """#!/bin/bash
[[ -n "${SQUEUE_FAIL:-}" ]] && exit 1
cat "$FAKE_DIR/squeue.txt"
"""
FAKE_SCANCEL = """#!/bin/bash
printf '%s\\n' "$*" >> "$FAKE_DIR/scancel.log"
"""


@pytest.fixture
def fake(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE), ("scancel", FAKE_SCANCEL)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    refs = tmp_path / "refs"
    refs.mkdir()
    for name in ("ceiling_val.tsv", "uniform_val.tsv", "ref.tsv"):
        (refs / name).write_text("x\n")
    (tmp_path / "squeue.txt").write_text("")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_DIR": str(tmp_path),
        "P6_TRIAL_CEILING": str(refs / "ceiling_val.tsv"),
        "P6_TRIAL_REF_PERIODICITY": str(refs / "ref.tsv"),
        "P6_TRIAL_BASE": str(tmp_path / "no_base.json"),
        "P6_TRIAL_RESULTS": str(tmp_path / "results.tsv"),
        "P6_TRIAL_RUNS_DIR": str(tmp_path / "runs"),
        "TISIAGO_AGFT_PARTITION": "nvidia-A6000-20",
        "P6_TRIAL_LANES": "nvidia-A6000-20 nvidia-A6000-20",
    }

    def run(*args, squeue="", reset=True, **extra):
        (tmp_path / "squeue.txt").write_text(squeue)
        proc = subprocess.run(
            ["bash", str(SCRIPT), *args],
            env={**env, **extra},
            capture_output=True,
            text=True,
        )
        log = tmp_path / "sbatch.log"
        calls = log.read_text().splitlines() if log.exists() else []
        log.unlink(missing_ok=True)
        if reset:
            (tmp_path / "n").unlink(missing_ok=True)
        cancel = tmp_path / "scancel.log"
        cancelled = cancel.read_text().split() if cancel.exists() else []
        return proc, calls, cancelled

    return run


def dependency(call):
    return next(a.split("=", 1)[1] for a in call.split() if a.startswith("--dependency="))


def test_failed_squeue_aborts_without_submitting(fake):
    proc, calls, _ = fake("t1", "--steps", "10", SQUEUE_FAIL="1")
    assert proc.returncode != 0 and "squeue failed" in proc.stderr
    assert calls == []


def test_default_counts_held_and_dependency_jobs(fake):
    squeue = "500|smoke|nvidia-A6000-20|gres/gpu:1\n600|base|nvidia-A6000-20|gres/gpu:1\n"
    proc, calls, _ = fake("t1", "--steps", "10", squeue=squeue)
    assert proc.returncode != 0 and "GPU busy (2 GPU jobs" in proc.stderr
    assert calls == []
    proc, calls, _ = fake("--ignore-jobs", "500", "t1", "--steps", "10", squeue=squeue)
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 4
    assert dependency(calls[0]) == "singleton"


def test_cpu_jobs_are_not_counted(fake):
    squeue = "700|evaluate|20|N/A\n701|period|20|N/A\n"
    proc, calls, _ = fake("t1", "--steps", "10", squeue=squeue)
    assert proc.returncode == 0, proc.stderr


def test_queue_waits_on_outside_jobs_by_id_and_after(fake):
    squeue = "300|c|nvidia-A6000-20|gres/gpu:1\n100|a|nvidia-A6000-20|gres/gpu:1\n"
    squeue += "200|b|nvidia-A6000-20|gres/gpu:1\n"
    proc, calls, _ = fake("--queue", "--after", "100", "t1", "--steps", "10", squeue=squeue)
    assert proc.returncode == 0, proc.stderr
    assert dependency(calls[0]) == "singleton,afterany:100,afterany:300,afterok:100"
    assert f"--job-name={LANE0}" in calls[0]
    assert dependency(calls[1]) == "afterok:901,singleton"
    assert dependency(calls[2]) == "afterok:901"
    assert dependency(calls[3]) == "afterok:902:903"


def test_queue_second_lane_gets_second_outside_job(fake):
    squeue = "100|a|nvidia-A6000-20|gres/gpu:1\n200|b|nvidia-A6000-20|gres/gpu:1\n"
    squeue += f"800|{LANE0}|nvidia-A6000-20|gres/gpu:1\n"
    proc, calls, _ = fake("--queue", "t2", "--steps", "10", squeue=squeue)
    assert proc.returncode == 0, proc.stderr
    assert dependency(calls[0]) == "singleton,afterany:200"
    assert "_tisiago_p6ar_lane1" in calls[0]


def test_queue_busy_lane_waits_on_all_existing_lane_jobs(fake):
    squeue = "100|a|nvidia-A6000-20|gres/gpu:1\n"
    squeue += f"810|{LANE0}|nvidia-A6000-20|gres/gpu:1\n"
    squeue += f"800|{LANE0}|nvidia-A6000-20|gres/gpu:1\n"
    squeue += f"801|{LANE0[:-1]}1|nvidia-A6000-20|gres/gpu:1\n"
    squeue += f"811|{LANE0[:-1]}1|nvidia-A6000-20|gres/gpu:1\n"
    proc, calls, _ = fake("--queue", "t3", "--steps", "10", squeue=squeue)
    assert proc.returncode == 0, proc.stderr
    assert f"--job-name={LANE0}" in calls[0]  # tie -> lane0
    assert dependency(calls[0]) == "singleton,afterany:800,afterany:810,afterany:100"


def _deps(call):
    return {
        part.split(":", 1)[1].split(":")[i]
        for part in dependency(call).split(",")
        if part.startswith("after")
        for i in range(len(part.split(":")) - 1)
    }


def test_three_queued_trials_never_exceed_two_gpus(fake):
    outside = ["100", "200"]
    lines = [f"{j}|base{j}|nvidia-A6000-20|gres/gpu:1" for j in outside]
    gpu_deps: dict[str, set[str]] = {}
    n = 0
    for k in range(3):
        squeue = "\n".join(lines) + "\n"
        proc, calls, _ = fake("--queue", f"q{k}", "--steps", "10", squeue=squeue, reset=False)
        assert proc.returncode == 0, proc.stderr
        for call in calls:
            n += 1
            job = str(900 + n)
            lane = next((a.split("=", 1)[1] for a in call.split() if "_p6ar_lane" in a), None)
            if lane:
                gpu_deps[job] = _deps(call)
                lines.append(f"{job}|{lane}|nvidia-A6000-20|gres/gpu:1")
    assert len(gpu_deps) == 6
    # Event simulation: start every job whose dependencies are all done (singleton ignored).
    running, done, pending = list(outside), set(), dict(gpu_deps)
    started_before_outside_end = [j for j, d in pending.items() if not d]
    assert started_before_outside_end == []
    while running:
        done.add(running.pop(0))
        for job, deps in sorted(pending.items()):
            if deps <= done:
                running.append(job)
                del pending[job]
        assert len(running) <= 2, running
    assert not pending


def test_failed_sbatch_cancels_already_submitted(fake):
    proc, calls, cancelled = fake("t1", "--steps", "10", FAIL_AT="3")
    assert proc.returncode != 0
    assert "already submitted: 901 902" in proc.stderr
    assert cancelled == ["901", "902"]
    assert len(calls) == 3


def test_eval_only_unfinished_run_needs_after(fake, tmp_path):
    run = tmp_path / "runs" / "base"
    (run / "best").mkdir(parents=True)
    (run / "run.json").write_text('{"args": {"arm": "condition:K562"}}')
    proc, calls, _ = fake("--eval-only", str(run))
    assert proc.returncode != 0 and "not a finished run" in proc.stderr
    proc, calls, _ = fake("--eval-only", str(run), "--after", "555")
    assert proc.returncode == 0, proc.stderr
    assert dependency(calls[0]) == "singleton,afterok:555"
    assert dependency(calls[1]) == "afterok:555"


A6 = "nvidia-A6000-20"
L40 = "nvidia-L40S-24"
LANES3 = f"{A6} {A6} {L40}"


def lane_name(i):
    return f"{LANE0[:-1]}{i}"


def partition(call):
    return next(a.split("=", 1)[1] for a in call.split() if a.startswith("--partition=nvidia"))


def test_three_lanes_pick_fewest_jobs_then_lowest_index(fake):
    squeue = f"800|{lane_name(0)}|{A6}|gres/gpu:1\n801|{lane_name(1)}|{A6}|gres/gpu:1\n"
    proc, calls, _ = fake("t1", "--steps", "10", squeue=squeue, P6_TRIAL_LANES=LANES3)
    assert proc.returncode == 0, proc.stderr
    assert f"--job-name={lane_name(2)}" in calls[0] and partition(calls[0]) == L40
    assert f"--job-name={lane_name(2)}" in calls[1] and partition(calls[1]) == L40
    proc, calls, _ = fake("--queue", "t1", "--steps", "10", squeue="", P6_TRIAL_LANES=LANES3)
    assert f"--job-name={lane_name(0)}" in calls[0] and partition(calls[0]) == A6
    squeue = f"800|{lane_name(0)}|{A6}|gres/gpu:1\n"
    proc, calls, _ = fake("--queue", "t1", "--steps", "10", squeue=squeue, P6_TRIAL_LANES=LANES3)
    assert f"--job-name={lane_name(1)}" in calls[0] and partition(calls[0]) == A6


def test_outside_job_in_other_partition_does_not_gate(fake):
    squeue = f"500|x|{L40}|gres/gpu:1\n501|y|{L40}|gres/gpu:1\n"
    proc, calls, _ = fake("t1", "--steps", "10", squeue=squeue, P6_TRIAL_LANES=f"{A6} {A6}")
    assert proc.returncode == 0, proc.stderr
    assert partition(calls[0]) == A6 and dependency(calls[0]) == "singleton"
    proc, calls, _ = fake("--queue", "t1", "--steps", "10", squeue=squeue,
                          P6_TRIAL_LANES=f"{A6} {A6}")
    assert dependency(calls[0]) == "singleton"


def test_outside_job_in_same_partition_gates_right_lane(fake):
    squeue = f"500|x|{L40}|gres/gpu:1\n"
    proc, calls, _ = fake("t1", "--steps", "10", squeue=squeue, P6_TRIAL_LANES=LANES3)
    assert proc.returncode == 0, proc.stderr  # lane0 on A6000 is free
    proc, calls, _ = fake("t1", "--steps", "10", squeue=squeue, P6_TRIAL_LANES=f"{L40} {A6}")
    assert proc.returncode != 0 and f"GPU busy (1 GPU jobs in {L40}" in proc.stderr
    # queue: lane 2 (only L40 lane) waits on the L40 outside job; lanes 0/1 busy
    squeue += f"800|{lane_name(0)}|{A6}|gres/gpu:1\n801|{lane_name(1)}|{A6}|gres/gpu:1\n"
    squeue += f"802|{lane_name(0)}|{A6}|gres/gpu:1\n803|{lane_name(1)}|{A6}|gres/gpu:1\n"
    proc, calls, _ = fake("--queue", "t1", "--steps", "10", squeue=squeue, P6_TRIAL_LANES=LANES3)
    assert f"--job-name={lane_name(2)}" in calls[0]
    assert dependency(calls[0]) == "singleton,afterany:500"


def test_queue_distributes_outside_jobs_over_lanes_of_partition(fake):
    squeue = f"100|a|{A6}|gres/gpu:1\n200|b|{A6}|gres/gpu:1\n300|c|{A6}|gres/gpu:1\n"
    squeue += f"400|d|{L40}|gres/gpu:1\n"
    squeue += f"800|{lane_name(0)}|{A6}|gres/gpu:1\n"
    proc, calls, _ = fake("--queue", "t1", "--steps", "10", squeue=squeue, P6_TRIAL_LANES=LANES3)
    assert f"--job-name={lane_name(1)}" in calls[0]  # lane1 is 2nd A6000 lane: 200 only
    assert dependency(calls[0]) == "singleton,afterany:200"


def test_all_gpu_calls_use_lane_partition(fake):
    proc, calls, _ = fake("--queue", "t1", "--steps", "10", squeue=f"800|{lane_name(0)}|{A6}|g\n"
                          f"801|{lane_name(1)}|{A6}|g\n", P6_TRIAL_LANES=LANES3)
    assert partition(calls[0]) == L40 and partition(calls[1]) == L40


def test_rnacond_trial_is_train_then_evaluate_on_every_condition(fake, tmp_path):
    proc, calls, _ = fake("rnacond_hela", "--arm", "rnacond:HeLa", "--head", "rnacond")
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 2  # no periodicity / trial_score: they take per-condition arms only
    train, evaluate = calls
    assert "run_p6_train.sh" in train and "--arm rnacond:HeLa --head rnacond" in train
    assert "run_p6_evaluate.sh" in evaluate and "--arm rnacond:HeLa" in evaluate
    assert "--condition" not in evaluate  # evaluate defaults to all six conditions
    assert dependency(evaluate) == "afterok:901,singleton"
    assert "periodicity and trial_score skipped" in proc.stderr
