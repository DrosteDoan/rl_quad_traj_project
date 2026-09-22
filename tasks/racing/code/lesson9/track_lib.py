"""Track library for Lesson 9: random gate layouts -> TOGT ground-start tube plans -> a flyability verdict.

    layout  = generate_layout(seed)          # lsy's level-3 generator, gates only (poles are ignored)
    yaml    = tube_yaml(layout["gates"])     # the Lesson 6/7 tube, one entry + one exit corridor per gate
    ok, why = run_togt(yaml, "name", csv)    # TOGT driver from Lesson 6 (needs `bash togt/build.sh` once)
    diag    = diagnose_plan(csv, gates)      # what a tracker will have to fly, and whether it can

The verdict is about the PLAN, never about a controller (METHODOLOGY.md section 3): a track is rejected
for structural reasons only (TOGT fails, a gate is missed, the path goes through a frame, the crossing is
too oblique, the path leaves the arena). The speed stretch s = max(1, v_peak / V_CAP) is computed here and
applied at run time through SampledRaceTrajectory(time_scale=s).

Nothing in this file edits an existing file; it imports `togt_plan` and `race_refs` (Lessons 6-8) and the
vendored `crazy_track`. Runs in the MAIN venv, in the container.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]                     # tasks/racing/code
sys.path.insert(0, str(CODE))
import contact as ct  # noqa: E402  (next to this file: the exact gate-frame contact test)
from race_refs import crossing_angles, plane_crossings  # noqa: E402
from togt_plan import BIN, ROOT, patched_params  # noqa: E402

from crazy_track.trajectories.freestyle import (  # noqa: E402
    GRAVITY,
    RaceGate,
    feasibility_report,
    lsy_level2_race,
)
from crazy_track.trajectories.sampled import SampledRaceTrajectory  # noqa: E402

# ---- the fixed part of every track (METHODOLOGY.md section 3) ------------------------------------
GROUND_START = (-1.5, 0.75, 0.05)      # where the race puts the drone; Lesson 8 section 3
GATES_Z = (0.7, 1.2, 0.7, 1.2)         # lsy: tall and short gates alternate; height is never randomised
OBSTACLES_Z = (1.55, 1.55, 1.55, 1.55)  # the generator needs them; we ignore the poles it places
ARENA_LOW = np.array([-2.5, -1.5])      # lsy level3.toml safety_limits, xy
ARENA_HIGH = np.array([2.5, 1.5])
ARENA_Z_MAX = 2.0

# ---- planner and filter constants -----------------------------------------------------------------
THRUST_FRAC = 0.85                     # user: the usual 0.85 x TWR
TUBE_D = 0.6                           # entry / exit corridor distance along the gate normal (upstream tube)
TUBE_MARGIN = 0.30                     # marginW/H of a tube corridor: window = 0.5 * (0.4 - 0.30) = +-0.05 m
END_PAST = 1.0                         # the plan ends at rest this far past the last gate, on its axis
V_CAP = 4.4                            # m/s; stretch s = max(1, v_peak / V_CAP)
ANGLE_CAP_DEG = 25.0                   # Lesson 7's tube plans cross at 2-18 deg; above this a lag is a miss
ARENA_MARGIN = 0.10                    # the plan must stay this far inside lsy's safety limits
MIN_FRAME_MARGIN = 0.03                # m; user decision 2026-09-22. The reference's frame margin (contact.frame_margin) must
#                                        be at least this; Lesson 8's ground plan, which the corrected MPC flies at 18/20,
#                                        has 0.035 m. 0 would mean only "the reference does not touch a frame".


# ---- 1. layouts -----------------------------------------------------------------------------------
_SAMPLER = None


def _sampler():
    global _SAMPLER
    if _SAMPLER is None:
        import jax
        from lsy_drone_racing.envs.randomize import build_random_track_fn

        fn = build_random_track_fn(np.asarray(GATES_Z), np.asarray(OBSTACLES_Z), ARENA_LOW, ARENA_HIGH)
        _SAMPLER = jax.jit(fn)
    return _SAMPLER


def generate_layout(seed: int) -> dict:
    """One random layout from lsy's level-3 generator, built around the ground start.

    Returns {"seed", "gates": [{"pos": [x, y, z], "yaw": rad}, ...]}. Gate x, y and yaw vary; z and the
    order are lsy's. The generator's obstacle poles are discarded (TOGT does not model them).
    """
    import jax
    import jax.numpy as jnp

    gates_pos, gates_quat, _ = _sampler()(jnp.asarray(GROUND_START, dtype=jnp.float32),
                                          jax.random.PRNGKey(int(seed)))
    gp, gq = np.asarray(gates_pos, dtype=np.float64), np.asarray(gates_quat, dtype=np.float64)
    yaw = 2.0 * np.arctan2(gq[:, 2], gq[:, 3])                # quat = (0, 0, sin(y/2), cos(y/2))
    yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
    return {"seed": int(seed),
            "gates": [{"pos": [float(p[0]), float(p[1]), float(p[2])], "yaw": float(y)}
                      for p, y in zip(gp, yaw)]}


def race_gates(gates: list[dict]) -> list[RaceGate]:
    return [RaceGate(tuple(g["pos"]), yaw=g["yaw"]) for g in gates]


def level2_gates() -> list[dict]:
    """The nominal level-2 gates in the same dict form (the regression target of the tests)."""
    return [{"pos": list(g.pos), "yaw": float(g.yaw)} for g in lsy_level2_race().gates]


# ---- 2. the TOGT track file -----------------------------------------------------------------------
def _normal(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw), np.sin(yaw), 0.0])


def _state_block(name: str, pos) -> str:
    return (f"{name}:\n  pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n  vel: [0.0, 0.0, 0.0]\n"
            "  acc: [0.0, 0.0, 0.0]\n  jer: [0.0, 0.0, 0.0]\n  rot: [1.0, 0.0, 0.0, 0.0]\n"
            "  cthrustmass: 9.81\n  euler: [0.0, 0.0, 0.0]\n")


def _gate_block(key: str, pos, yaw: float) -> str:
    # TOGT's RectanglePrisma normal is its local z; rpy [0, -90, y] maps it to -(cos y, sin y, 0), so
    # y = lsy_yaw_deg + 180 aligns it with the lsy normal (configs/togt/README.md).
    return (f"{key}:\n  type: 'RectanglePrisma'\n  name: 'lsy_gate'\n"
            f"  position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
            f"  rpy: [0.0, -90.0, {(np.degrees(yaw) + 180.0) % 360.0:.3f}]\n"
            f"  width: 0.4\n  height: 0.4\n  marginW: {TUBE_MARGIN}\n  marginH: {TUBE_MARGIN}\n"
            "  length: 0.0\n  midpoints: 0\n  stationary: true\n")


def end_point(gates: list[dict]) -> np.ndarray:
    last = gates[-1]
    return np.asarray(last["pos"], dtype=np.float64) + END_PAST * _normal(last["yaw"])


def tube_yaml(gates: list[dict], init_pos=GROUND_START, end_pos=None) -> str:
    """TOGT track: every gate bracketed by an entry and an exit corridor TUBE_D along its normal
    (upstream's lsy_level2_tube.yaml, generalised to any gate), a ground start, a rest end."""
    end = end_point(gates) if end_pos is None else np.asarray(end_pos, dtype=np.float64)
    names, blocks = [], []
    for i, g in enumerate(gates, start=1):
        p, n = np.asarray(g["pos"], dtype=np.float64), _normal(g["yaw"])
        for suffix, q in (("in", p - TUBE_D * n), ("", p), ("out", p + TUBE_D * n)):
            key = f"G{i}{suffix}"
            names.append(key)
            blocks.append(_gate_block(key, q, g["yaw"]))
    head = ("# Generated by lesson9/track_lib.py: tube corridors around lsy level-3 gates, ground start.\n"
            + _state_block("initState", init_pos) + "\n" + _state_block("endState", end) + "\n"
            + "orders: [" + ", ".join(f"'{n}'" for n in names) + "]\n\n")
    return head + "\n".join(blocks)


# ---- 3. plan it ----------------------------------------------------------------------------------
def run_togt(yaml_text: str, name: str, out_csv: Path, thrust_frac: float = THRUST_FRAC,
             timeout: float = 120.0) -> tuple[bool, str]:
    """Plan `yaml_text` with the Lesson-6 TOGT driver. Returns (ok, message)."""
    if not BIN.exists():
        return False, f"driver missing: {BIN} (bash tasks/racing/code/togt/build.sh)"
    out_csv = Path(out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        params = patched_params("togt", thrust_frac, work, bound_z=(0.0, 3.0))   # ground start: boundZ from 0
        track = work / f"{name}.yaml"
        track.write_text(yaml_text, newline="\n")                                 # TOGT's parser: LF only
        try:
            r = subprocess.run([str(BIN.resolve()), str(params.resolve()), "setups.yaml", str(track.resolve()),
                                str(out_csv), "togt"], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "togt_timeout"
    if r.returncode != 0 or not out_csv.exists():
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:] or [""]
        return False, f"togt_failed(exit {r.returncode}) {tail[0][:80]}"
    return True, "ok"


# ---- 4. is the plan flyable? ----------------------------------------------------------------------
def _speed_thrust(tr) -> tuple[float, float]:
    t = np.arange(tr.lead_in, tr.t_end, 0.002)
    v = float(np.linalg.norm(tr.vel(t), axis=1).max())
    thrust = float(np.linalg.norm(tr.acc(t) + np.array([0.0, 0.0, GRAVITY]), axis=1).max())
    return v, thrust


def diagnose_plan(csv: Path, gates: list[dict], v_cap: float = V_CAP, angle_cap: float = ANGLE_CAP_DEG,
                  arena_margin: float = ARENA_MARGIN, min_margin: float = MIN_FRAME_MARGIN) -> dict:
    """Structural verdict on a plan CSV. `reasons` is empty iff the track passes."""
    rg = race_gates(gates)
    reasons: list[str] = []
    d: dict = {"reasons": reasons}
    try:
        tr = SampledRaceTrajectory(str(csv), gates=rg, obstacles=[])
    except ValueError as e:                                     # a gate is not crossed inside its opening
        reasons.append("gate_missed")
        d["error"] = str(e)[:120]
        return d
    t = np.arange(tr.lead_in, tr.t_end, 0.002)
    pos = tr.pos(t)
    rep = feasibility_report(tr)
    angles = [a["angle_deg"] for a in crossing_angles(tr)]
    frames = [(i + 1, r) for i, rows in enumerate(plane_crossings(tr)) for r in rows if r["verdict"] == "FRAME"]
    v_peak, thrust_peak = _speed_thrust(tr)
    stretch = max(1.0, v_peak / v_cap)
    lo, hi = ARENA_LOW + arena_margin, ARENA_HIGH - arena_margin
    outside = bool((pos[:, 0] < lo[0]).any() or (pos[:, 0] > hi[0]).any()
                   or (pos[:, 1] < lo[1]).any() or (pos[:, 1] > hi[1]).any()
                   or (pos[:, 2] > ARENA_Z_MAX - arena_margin).any())
    gt = np.asarray(tr.gate_times)
    if not (np.diff(gt) > 0).all():
        reasons.append("gate_order")
    if frames:
        reasons.append("frame_crossing")
    if max(angles) > angle_cap:
        reasons.append("angle")
    if outside:
        reasons.append("arena")
    if not rep["gate_crossings_ok"] and "frame_crossing" not in reasons:
        reasons.append("crossings_not_ok")
    # the exact test (contact.py): the level drone box flying the reference, and how much room it has
    t_c = np.arange(tr.lead_in, tr.t_end, 0.005)
    p_c, q_c = tr.pos(t_c), np.tile([0.0, 0.0, 0.0, 1.0], (len(t_c), 1))
    hit = ct.first_contact(t_c, p_c, q_c, rg)
    margin = ct.frame_margin(t_c, p_c, q_c, rg)
    if hit is not None:
        reasons.append("ref_contact")
    elif margin < min_margin:
        reasons.append("frame_margin")
    tr_s = SampledRaceTrajectory(str(csv), gates=rg, obstacles=[], time_scale=stretch)
    v_s, thrust_s = _speed_thrust(tr_s)
    gaps = [float(np.linalg.norm(np.asarray(a["pos"][:2]) - np.asarray(b["pos"][:2])))
            for a, b in zip(gates[:-1], gates[1:])]
    d.update({
        "plan_lap_s": float(tr.gate_times[-1] - tr.lead_in), "plan_gate1_s": float(gt[0] - tr.lead_in),
        "v_peak": v_peak, "thrust_peak": thrust_peak, "thrust_limit": float(rep["thrust_acc_limit"]),
        "max_angle_deg": float(max(angles)), "angles_deg": [round(a, 1) for a in angles],
        "min_gate_gap": min(gaps), "stretch": stretch, "v_peak_stretched": v_s,
        "thrust_peak_stretched": thrust_s, "lap_stretched_s": float(tr.gate_times[-1] - tr.lead_in) * stretch,
        "min_z": float(pos[:, 2].min()), "n_frame_crossings": len(frames),
        "frame_crossings": [{"gate": g, "offset": round(r["offset"], 3), "intended": bool(r["intended"])}
                            for g, r in frames],
        "frame_margin_m": float(margin),
        "ref_contact": None if hit is None else {"gate": hit["gate"], "box": hit["box"],
                                                 "t": round(hit["t"] - tr.lead_in, 2)},
    })
    return d


def build_track(seed: int, plan_dir: Path, name: str, min_margin: float = MIN_FRAME_MARGIN) -> tuple[dict, dict]:
    """layout -> yaml -> TOGT -> verdict. Returns (layout, diag); diag['reasons'] is empty iff it passes."""
    layout = generate_layout(seed)
    ok, msg = run_togt(tube_yaml(layout["gates"]), name, Path(plan_dir) / f"{name}.csv")
    if not ok:
        return layout, {"reasons": [msg.split("(")[0].split(" ")[0]], "error": msg}
    return layout, diagnose_plan(Path(plan_dir) / f"{name}.csv", layout["gates"], min_margin=min_margin)
