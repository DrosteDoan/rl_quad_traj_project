"""Checks for the Lesson 9 track library (main venv, in the container):

    python tasks/racing/code/lesson9/test_track_lib.py        # no pytest needed
    python -m pytest tasks/racing/code/lesson9/test_track_lib.py -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track_lib as tl  # noqa: E402

UPSTREAM_TUBE = tl.ROOT / "tasks" / "racing" / "crazy_track" / "configs" / "togt" / "lsy_level2_tube.yaml"


def test_tube_yaml_reproduces_upstream_level2_tube():
    """Our writer, fed the level-2 gates, gives upstream's hand-made tube (positions to 1 mm, angles to 0.01 deg)."""
    mine = yaml.safe_load(tl.tube_yaml(tl.level2_gates()))
    theirs = yaml.safe_load(UPSTREAM_TUBE.read_text())
    assert mine["orders"] == theirs["orders"]
    for key in theirs["orders"]:
        np.testing.assert_allclose(mine[key]["position"], theirs[key]["position"], atol=1e-3, err_msg=key)
        np.testing.assert_allclose(mine[key]["rpy"], theirs[key]["rpy"], atol=1e-2, err_msg=key)
        for f in ("width", "height", "marginW", "marginH", "length", "midpoints"):
            assert mine[key][f] == theirs[key][f], (key, f)
    np.testing.assert_allclose(mine["endState"]["pos"], theirs["endState"]["pos"], atol=1e-3)
    np.testing.assert_allclose(mine["initState"]["pos"][:2], theirs["initState"]["pos"][:2], atol=1e-3)
    assert mine["initState"]["pos"][2] == tl.GROUND_START[2]          # the ONE difference: a ground start


def test_generator_is_deterministic_and_respects_lsy_constraints():
    a, b = tl.generate_layout(7), tl.generate_layout(7)
    assert a == b
    assert tl.generate_layout(8) != a
    for seed in range(20):
        gates = tl.generate_layout(seed)["gates"]
        np.testing.assert_allclose([g["pos"][2] for g in gates], tl.GATES_Z, atol=1e-6)   # float32 heights
        xy = np.array([g["pos"][:2] for g in gates])
        assert (np.abs(xy[:, 0]) <= 2.0 + 1e-4).all() and (np.abs(xy[:, 1]) <= 1.0 + 1e-4).all()  # border margin 0.5
        assert (np.linalg.norm(xy - np.asarray(tl.GROUND_START[:2]), axis=1) >= 1.0 - 1e-3).all()   # drone_excl_r


def test_pipeline_on_level2_gates_finds_the_known_gate3_frame_crossing():
    """The pipeline end to end on the known track. Upstream's tube (no pole vias) has ONE structural problem,
    Lesson 6 section 2: a second, unintended crossing of gate 3's plane 0.41 m from its centre, inside the
    frame zone (0.13-0.43 m). The filter must find that (the exact box test, contact.py, sees the same graze as
    a reference contact) and nothing else. Skipped without the driver."""
    if not tl.BIN.exists():
        print("  (skipped: TOGT driver not built)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        gates = tl.level2_gates()
        ok, msg = tl.run_togt(tl.tube_yaml(gates), "l2", Path(tmp) / "l2.csv")
        assert ok, msg
        d = tl.diagnose_plan(Path(tmp) / "l2.csv", gates)
    # the 3.0 cm margin rule flags it too: this plan's frame margin is 1.9 cm (Lesson 8's ground plan, with the
    # pole vias that steer it off gate 3, has 3.5 cm)
    assert set(d["reasons"]) <= {"frame_crossing", "ref_contact", "frame_margin"} and "frame_crossing" in d["reasons"], d
    assert d["frame_margin_m"] < tl.MIN_FRAME_MARGIN, d["frame_margin_m"]
    (fc,) = d["frame_crossings"]
    assert fc["gate"] == 3 and not fc["intended"] and abs(fc["offset"] - 0.41) < 0.02, fc
    assert 3.5 < d["plan_lap_s"] < 4.6                                  # Lesson 8: 4.02 s with vias, similar without
    assert d["thrust_peak"] < 0.9 * d["thrust_limit"]
    assert d["stretch"] >= 1.0 and d["v_peak_stretched"] <= tl.V_CAP + 0.05


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
