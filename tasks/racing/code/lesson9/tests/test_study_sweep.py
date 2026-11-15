"""Checks for `study_sweep.py`'s task construction and `launcher.py`'s CLI plumbing (main venv, in the
container; pure logic, no real flights, runs in under a second):

    python tasks/racing/code/lesson9/tests/test_study_sweep.py        # no pytest needed

Covers the fix found 2026-09-30: `driver.py`'s lam=0 seed-count shortcut has no notion of per-member
escalation, so `study_sweep.py`'s own escalated (mppi_l1) sub-task must explicitly opt back in via
`full_seeds_at_zero`, and `launcher.py` must actually thread that through to the CLI -- both are pure,
cheap-to-test logic, unlike `run_task` itself (real flights, covered instead in `test_driver.py`).

Also covers `--mpc-set cap07` (2026-11-15): capped members only, their own folder and logs, the original cell's
lambdas, midpoints from capped jumps only, and charts reading each family from its own file. Reads the stored
original sweep cells; writes only to temp folders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # the lesson9 modules
import launcher  # noqa: E402
import study_sweep as ss  # noqa: E402


def test_escalated_deterministic_subtask_sets_full_seeds_at_zero():
    batch1, batch2 = ss.cell_tasks(747, "wind_const", ss.COARSE_LAMS, {"fa0": "robust_full:x.zip"},
                                   escalated=["mppi_l1"])
    assert len(batch1) == 1 and len(batch2) == 1
    assert batch1[0].get("full_seeds_at_zero") is None or batch1[0]["full_seeds_at_zero"] is False
    assert batch2[0]["full_seeds_at_zero"] is True
    assert batch2[0]["only"] == "mppi_l1"
    assert batch2[0]["seeds"] == 3


def test_no_escalation_means_no_full_seeds_at_zero_anywhere():
    batch1, batch2 = ss.cell_tasks(747, "wind_const", ss.COARSE_LAMS, {"fa0": "robust_full:x.zip"}, escalated=[])
    assert len(batch1) == 1 and batch2 == []
    assert not batch1[0].get("full_seeds_at_zero")


def test_stochastic_conditions_never_set_full_seeds_at_zero():
    """Stochastic conditions already fly the full seed count uniformly (S4's own 25-seed rule) -- the
    lam=0 shortcut and its escalation override are both a deterministic-condition-only concept."""
    batch1, batch2 = ss.cell_tasks(747, "wind_gust", ss.COARSE_LAMS, {"fa0": "robust_full:x.zip"},
                                   escalated=["mppi_l1"])
    assert len(batch1) == 1 and batch2 == []
    assert not batch1[0].get("full_seeds_at_zero")


def test_task_cmd_adds_the_flag_only_when_the_task_sets_it():
    with_flag = launcher.task_cmd({"role": "study", "track": 747, "cond": "wind_const", "lams": [0.0],
                                    "seeds": 3, "out": "/tmp/x.csv", "full_seeds_at_zero": True})
    without_flag = launcher.task_cmd({"role": "study", "track": 747, "cond": "wind_const", "lams": [0.0],
                                       "seeds": 1, "out": "/tmp/x.csv"})
    assert "--full-seeds-at-zero" in with_flag
    assert "--full-seeds-at-zero" not in without_flag


# ---- --mpc-set cap07 (the matched 0.7 rad extension) ------------------------------------------------------------------
# Pure logic: `launch` is replaced by a recorder, and every test uses a throwaway copy of the cap07 set rooted in a
# temp folder, so nothing is written under results/. The original sweep's stored cells are only READ.
import csv  # noqa: E402
import tempfile  # noqa: E402

RL = {"s0": "robust:a.zip", "s1": "robust:b.zip", "s2": "robust:c.zip"}
RL_LABELS = ("s0", "s1", "s2", "fa0", "fa1", "fa2")


def _tmp_cap07() -> ss.SweepSet:
    root = Path(tempfile.mkdtemp()) / "study_sweep_cap07"
    return ss.SweepSet("cap07", root, ss.CAP07_LABELS, fly_rl=False, log_dir=root.parent / "logs")


def _record_launch():
    calls = []
    ss.launch = lambda tasks, workers=8, log_dir=None: calls.append(list(tasks))
    return calls


def _cell_with_missing_midpoint() -> tuple[int, str, list[float]]:
    """A stored cell whose lambdas lack at least one 0.05 midpoint, and the midpoints it lacks."""
    allmids = [round(a + 0.05, 2) for a in ss.COARSE_LAMS[:-1]]
    for track in ss.untouched_study_tracks():
        for cond in ss.STOCHASTIC:
            have = set(ss.original_cell_lams(track, cond))
            missing = [m for m in allmids if m not in have]
            if missing and len(missing) < len(allmids):
                return track, cond, missing
    raise AssertionError("no stored cell lacks a midpoint -- pick the test cell differently")


def test_cap07_set_is_separate_from_the_original():
    assert ss.CAP07.root.name == "study_sweep_cap07" and ss.CAP07.root != ss.OUT_ROOT
    assert ss.CAP07.root.parent == ss.OUT_ROOT.parent
    assert ss.CAP07_LABELS == ("M1@0.7", "M1+ESO@0.7", "M1+L1@0.7", "M1+mass@0.7", "mppi_l1@0.7")
    assert not ss.CAP07.fly_rl and ss.ORIGINAL.fly_rl
    assert ss.ORIGINAL.root == ss.OUT_ROOT and ss.ORIGINAL.mpc_labels == ss.MPC_LABELS and ss.ORIGINAL.log_dir is None


def test_cap07_cell_tasks_fly_capped_members_only_into_their_own_folder():
    s = _tmp_cap07()
    b1, b2 = ss.cell_tasks(747, "wind_gust", ss.COARSE_LAMS, RL, escalated=[], sset=s)
    assert b2 == [] and len(b1) == 1
    t = b1[0]
    assert t["only"].split(",") == list(ss.CAP07_LABELS) and t["members"] == {}
    assert Path(t["out"]) == s.root / "track_747" / "wind_gust" / "laps.csv"
    assert Path(t["log"]) == s.log_dir / "track_747_wind_gust.log"
    b1, b2 = ss.cell_tasks(747, "wind_const", ss.COARSE_LAMS, RL, escalated=["mppi_l1@0.7"], sset=s)
    assert b1[0]["only"].split(",") == list(ss.CAP07_LABELS[:4]) and b1[0]["seeds"] == 1
    assert b2[0]["only"] == "mppi_l1@0.7" and b2[0]["seeds"] == 3 and b2[0]["full_seeds_at_zero"] is True
    assert b1[0]["log"] == b2[0]["log"]          # one log per cell; the two batches run one after the other


def test_original_cell_tasks_are_unchanged():
    b1, _ = ss.cell_tasks(747, "wind_gust", ss.COARSE_LAMS, RL, escalated=[])
    t = b1[0]
    assert "log" not in t and t["members"] == RL
    assert t["only"].split(",") == list(ss.MPC_LABELS) + list(RL)
    assert Path(t["out"]) == ss.OUT_ROOT / "track_747" / "wind_gust" / "laps.csv"


def test_cap07_coarse_flies_every_original_lambda_and_never_an_rl_member():
    s, calls = _tmp_cap07(), _record_launch()
    track, cond, _ = _cell_with_missing_midpoint()
    ss.run_coarse([track], {**RL}, workers=1, conditions=[cond], sset=s)
    tasks = [t for batch in calls for t in batch]
    assert len(tasks) == 1
    assert tasks[0]["lams"] == sorted(set(ss.COARSE_LAMS) | set(ss.original_cell_lams(track, cond)))
    for t in tasks:
        assert t["members"] == {} and not set(t["only"].split(",")) & set(RL_LABELS)
        assert ss.OUT_ROOT not in Path(t["out"]).parents


def test_cap07_midpoints_come_from_capped_jumps_and_skip_the_original_midpoints():
    s, calls = _tmp_cap07(), _record_launch()
    track, cond, missing = _cell_with_missing_midpoint()
    out = ss.cell_out(track, cond, s)
    with open(out, "w", newline="") as fh:       # M1@0.7 alternates 1/0 on the coarse grid: every interval jumps
        w = csv.DictWriter(fh, fieldnames=["member", "lam", "completed"])
        w.writeheader()
        for i, lam in enumerate(ss.COARSE_LAMS):
            w.writerow({"member": "M1@0.7", "lam": lam, "completed": (i + 1) % 2})
    ss.run_midpoints([track], {**RL}, workers=1, conditions=[cond], sset=s)
    tasks = [t for batch in calls for t in batch]
    assert len(tasks) == 1 and tasks[0]["lams"] == sorted(missing)
    assert tasks[0]["members"] == {} and tasks[0]["only"].split(",") == list(ss.CAP07_LABELS)


def test_cap07_determinism_writes_into_its_own_folder():
    s, calls = _tmp_cap07(), _record_launch()
    try:
        ss.run_determinism([747], {**RL}, workers=1, conditions=["wind_const"], sset=s)
    except FileNotFoundError:
        pass                                      # the recorder flew nothing, so there is no CSV to read back
    t = calls[0][0]
    assert Path(t["out"]) == s.root / "determinism_check_wind_const.csv"
    assert Path(t["log"]) == s.log_dir / "determinism_check_wind_const.log"
    assert t["only"].split(",") == list(ss.CAP07_LABELS) and t["members"] == {}
    assert ss.determinism_path(s) == s.root / "determinism.json" != ss.determinism_path()


def test_matched07_chart_reads_mpc_and_rl_from_their_own_files():
    tmp = Path(tempfile.mkdtemp())
    cap_csv, rl_csv = tmp / "cap.csv", tmp / "rl.csv"
    for path, labels in ((cap_csv, ss.CAP07_LABELS), (rl_csv, ("s0", "s1", "s2"))):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["member", "lam", "completed", "rmse_3d", "race_time"])
            w.writeheader()
            for lab in labels:
                for lam in (0.0, 0.5, 1.0):
                    w.writerow({"member": lab, "lam": lam, "completed": 1, "rmse_3d": 0.1, "race_time": 4.0})
    seen = []
    orig = ss._mean_series
    ss._mean_series = lambda src, lab, stat: (seen.append((Path(src), lab)), orig(src, lab, stat))[1]
    try:
        ss._render("t", tmp / "c.png", cap_csv, ss.STATS["completion"], ss.CAP07_LABELS, ("s0", "s1", "s2"),
                   rl_csv=rl_csv)
    finally:
        ss._mean_series = orig
    assert (tmp / "c.png").exists()
    assert {lab for src, lab in seen if src == cap_csv} == set(ss.CAP07_LABELS)
    assert {lab for src, lab in seen if src == rl_csv} == {"s0", "s1", "s2"}


def test_task_log_key_does_not_reach_the_driver_cli():
    base = {"role": "study", "track": 747, "cond": "wind_gust", "lams": [0.0], "seeds": 1, "out": "/tmp/x.csv"}
    assert launcher.task_cmd(base) == launcher.task_cmd({**base, "log": "/tmp/x.log"})


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
