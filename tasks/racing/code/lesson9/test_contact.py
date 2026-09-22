"""Checks for the gate-frame contact test (main venv, in the container):

    python tasks/racing/code/lesson9/test_contact.py        # no pytest needed

Three independent checks: (1) Lesson 8 section 1's thresholds fall out of the geometry, (2) the whole
test agrees with MuJoCo's own collision on lsy's real gate XML for thousands of random poses, (3) a fast
step cannot tunnel through a 2 cm frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact as ct  # noqa: E402
import track_lib as tl  # noqa: E402

from crazy_track.trajectories.freestyle import RaceGate  # noqa: E402

GATE_XML = tl.ROOT / "repos" / "lsy_drone_racing" / "lsy_drone_racing" / "envs" / "assets" / "gate.xml"
LEVEL = np.array([0.0, 0.0, 0.0, 1.0])


def _threshold(gate: RaceGate, axis: str) -> float:
    """Smallest lateral ('y', in-plane horizontal) or vertical ('z') offset at which a LEVEL drone flying
    straight through the gate along its normal touches the frame, found by bisection to 0.1 mm."""
    e_y = np.array([-np.sin(gate.yaw), np.cos(gate.yaw), 0.0])
    e_dir = e_y if axis == "y" else np.array([0.0, 0.0, 1.0])
    s = np.linspace(-0.5, 0.5, 201)
    n = gate.normal

    def touches(off: float) -> bool:
        pos = gate.center + s[:, None] * n + off * e_dir
        return bool(ct.pose_contacts(pos, np.tile(LEVEL, (len(s), 1)), [gate])[0].any())

    lo, hi = 0.0, 0.3
    assert not touches(lo) and touches(hi)
    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if touches(mid) else (mid, hi)
    return hi


def test_lesson8_thresholds_fall_out_of_the_geometry():
    aligned = RaceGate((0.0, 0.0, 1.0), yaw=0.0)                     # G4-like: normal along world x
    assert abs(_threshold(aligned, "y") - 0.13) < 5e-4               # 0.20 - 0.07
    assert abs(_threshold(RaceGate((0.0, 0.0, 1.0), yaw=3.14), "y") - 0.13) < 2e-3    # G3
    for yaw in (-0.78, 2.35):                                        # G1, G2: ~45 degrees
        th = _threshold(RaceGate((0.0, 0.0, 1.0), yaw=yaw), "y")
        assert abs(th - 0.101) < 2e-3, (yaw, th)                     # 0.20 - 0.07 (|cos| + |sin|)
    assert abs(_threshold(aligned, "z") - 0.18) < 5e-4               # 0.20 - 0.02


def _mujoco_model():
    import mujoco

    spec = mujoco.MjSpec.from_file(str(GATE_XML))
    body = spec.worldbody.add_body(name="drone")
    body.add_freejoint()
    body.add_geom(name="drone_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(ct.DRONE_HALF),
                  contype=1, conaffinity=1)
    model = spec.compile()
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
           for n in ("drone_box", "col_top", "col_bottom", "col_left", "col_right")}
    return model, ids


def test_agrees_with_mujoco_signed_distance_on_random_poses():
    """4000 random poses (half arbitrary rotations, half nearly level), the gate at a random world pose. The
    truth is MuJoCo's own signed distance between the drone box and the four col_* boxes on lsy's real gate
    XML (touching = distance < 0, lsy's rule). A disagreement is allowed only when the pose is tangent.
    (MuJoCo's classic CONTACT GENERATOR also drops some deep, >= 6 cm, thin-box overlaps that its own distance
    reports; that regime is irrelevant to first contact, which is shallow: 1 cm of travel per 500 Hz step.)"""
    import mujoco

    model, ids = _mujoco_model()
    data = mujoco.MjData(model)
    cols = [ids[n] for n in ("col_top", "col_bottom", "col_left", "col_right")]
    rng = np.random.default_rng(0)
    n, mismatches, positives, fromto = 4000, [], 0, np.zeros(6)
    for k in range(n):
        gate = RaceGate((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.5, 1.5)),
                        yaw=rng.uniform(-np.pi, np.pi))
        loc_pos = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
        loc_rot = (R.from_euler("xyz", rng.uniform(-0.4, 0.4, 3)) if k % 2
                   else R.random(random_state=int(rng.integers(1 << 30))))
        qx, qy, qz, qw = loc_rot.as_quat()
        data.qpos[:3] = loc_pos                                       # MuJoCo has the gate at the origin
        data.qpos[3:7] = [qw, qx, qy, qz]                             # and is w-first
        mujoco.mj_forward(model, data)
        dist = min(mujoco.mj_geomDistance(model, data, ids["drone_box"], i, 1.0, fromto) for i in cols)
        truth = dist < 0
        Rg = np.asarray(gate.rotation)                                # mine: the same pose in the world
        mine = bool(ct.pose_contacts(gate.center + Rg @ loc_pos,
                                     R.from_matrix(Rg @ loc_rot.as_matrix()).as_quat(), [gate])[0][0])
        positives += truth
        if truth != mine and abs(dist) > 2e-4:
            mismatches.append((k, bool(truth), mine, round(float(dist), 5)))
    print(f"  {n} poses, {positives} in contact ({100 * positives / n:.0f} %), non-tangent mismatches: {len(mismatches)}")
    assert 0.15 < positives / n < 0.85, "the test set should contain both outcomes in quantity"
    assert not mismatches, mismatches[:5]


def test_agrees_with_mjx_the_backend_lsy_uses():
    """lsy's race_core reads `mjx_data._impl.contact.dist < 0` from MJX. Same gate XML, 600 nearly level poses
    (the flying regime): zero disagreements expected."""
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    model, ids = _mujoco_model()
    cols = [ids[n] for n in ("col_top", "col_bottom", "col_left", "col_right")]
    mx = mjx.put_model(model)

    @jax.jit
    def contacts(qpos):
        d = mjx.forward(mx, mjx.make_data(mx).replace(qpos=qpos))
        return d._impl.contact.dist, d._impl.contact.geom

    gate = RaceGate((0.0, 0.0, 0.0), yaw=0.0)
    rng = np.random.default_rng(1)
    n, hits, bad = 600, 0, []
    for k in range(n):
        loc_pos = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
        loc_rot = R.from_euler("xyz", rng.uniform(-0.5, 0.5, 3))
        qx, qy, qz, qw = loc_rot.as_quat()
        dist, geom = contacts(jnp.asarray([*loc_pos, qw, qx, qy, qz], dtype=jnp.float32))
        dist, geom = np.asarray(dist), np.asarray(geom)
        involved = np.isin(geom[:, 0], cols + [ids["drone_box"]]) & np.isin(geom[:, 1], cols + [ids["drone_box"]])
        truth = bool(((dist < 0) & involved).any())
        mine = bool(ct.pose_contacts(loc_pos, loc_rot.as_quat(), [gate])[0][0])
        hits += truth
        if truth != mine:
            bad.append((k, truth, mine))
    print(f"  {n} poses, {hits} in contact, MJX disagreements: {len(bad)}")
    assert hits > 50 and not bad, bad[:5]


def test_a_fast_step_cannot_tunnel_through_the_frame():
    gate = RaceGate((0.0, 0.0, 1.0), yaw=0.0)
    # fly along the normal at 0.25 m lateral offset (through the left post) at 5 m/s sampled at 100 Hz
    t = np.arange(0.0, 0.4, 0.01)
    pos = np.stack([-1.0 + 5.0 * t, np.full_like(t, 0.28), np.full_like(t, 1.0)], axis=1)
    quat = np.tile(LEVEL, (len(t), 1))
    hit = ct.first_contact(t, pos, quat, [gate])
    assert hit is not None and hit["gate"] == 1 and hit["box"] in ("left", "right", "top", "bottom"), hit
    assert 0.1 < hit["t"] < 0.3
    # and a clean pass through the middle is not a contact, with a margin of ~0.13 m to spare
    pos_ok = pos.copy()
    pos_ok[:, 1] = 0.0
    assert ct.first_contact(t, pos_ok, quat, [gate]) is None
    assert 0.12 < ct.frame_margin(t, pos_ok, quat, [gate]) < 0.14


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
