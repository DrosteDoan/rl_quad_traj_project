"""Checks for `study_sweep.py`'s task construction and `launcher.py`'s CLI plumbing (main venv, in the
container; pure logic, no real flights, runs in under a second):

    python tasks/racing/code/lesson9/test_study_sweep.py        # no pytest needed

Covers the fix found 2026-09-30: `driver.py`'s lam=0 seed-count shortcut has no notion of per-member
escalation, so `study_sweep.py`'s own escalated (mppi_l1) sub-task must explicitly opt back in via
`full_seeds_at_zero`, and `launcher.py` must actually thread that through to the CLI -- both are pure,
cheap-to-test logic, unlike `run_task` itself (real flights, covered instead in `test_driver.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
