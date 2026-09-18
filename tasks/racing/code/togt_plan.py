"""Plan a lap with the TOGT-Planner and diagnose it — Lesson 6 §2.

    python tasks/racing/code/togt_plan.py --track raw                        # zero-thickness gates
    python tasks/racing/code/togt_plan.py --track tube --thrust-frac 0.85    # entry/exit corridors
    python tasks/racing/code/togt_plan.py --track ground --thrust-frac 0.85  # a GROUND start (L8 §3)

Runs in the MAIN venv and needs the driver from `bash tasks/racing/code/togt/build.sh`.
The plan lands in tasks/racing/plans/<name>.csv (git-ignored); the script then loads it
back through crazy_track's `SampledRaceTrajectory` and prints what decides whether a
tracker can fly it:

  * the planned lap (motion onset -> last gate), top speed, thrust demand vs the limit
  * the crossing ANGLE at every gate -- the single number behind "raw plans are
    untrackable": lag x in-plane speed = error in the gate plane
  * every crossing of every gate PLANE (does the path go through a FRAME elsewhere?)
  * the clearance from the four obstacle poles, which TOGT does not model

Two knobs match the parent project's sweep: --thrust-frac (fraction of the platform's
thrust the planner may use; 0.95 is the feasibility bound the lessons use everywhere)
and --margin (shrinks the gate window TOGT may cross: window = 0.5 * (0.4 - margin)).

GROUND-START PLANS (Lesson 8 §3). The three `ground` tracks put `initState` on the floor at
(-1.5, 0.75, 0.05), where the race actually starts, so the plan flies straight into gate 1
instead of spending a 1.5 s takeoff and a second rest-to-rest leg first. `ground` is the bare
one, `ground-b08` and `ground-b08g3` pre-shift the gate references against the vendored
tracker's measured corner cut (read their headers). Those plans need
one thing the parameter files do not give them: TOGT's `boundZ` starts at 0.15 m, which no
trajectory beginning at z = 0.05 can satisfy, so a track whose `initState` z is below 0.15
is planned with `--bound-z 0.0 3.0` automatically. Five optional flags shape such a plan
without a new track file -- `--init-pos`, `--bound-z`, `--end-pos`, `--end-vel`,
`--via-radius` (and `--coast`, which only matters with a non-zero `--end-vel`) -- and
`--hold T` (default 0.2 s) is the rotor spin-up the diagnostics add to the planned lap to get
the ground clock the leaderboard reads. On a ground-start plan the
diagnostics print three more lines: that ground-clock lap, the first half second of the
climb, and the closest passage of every pole with its time, height and speed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_refs import crossing_angles, obstacle_clearance, plane_crossings  # noqa: E402

from crazy_track.trajectories.freestyle import GRAVITY, feasibility_report, lsy_level2_race  # noqa: E402
from crazy_track.trajectories.sampled import SampledRaceTrajectory  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]                    # tasks/racing/code -> rl_quad_control/
VENDORED = ROOT / "tasks" / "racing" / "crazy_track"
CFG = VENDORED / "configs" / "togt"
PLANS = ROOT / "tasks" / "racing" / "plans"
BIN = ROOT / "repos" / "togt-build" / "togt_race"

# configs/togt/*/quad.yaml: the drone is rescaled by k so that mass = 1 kg (see the
# README in that folder for why); per-rotor bounds scale with it.
K = 1.0 / 0.04338
ROTOR_MAX_TRUE = 0.20                                          # N per rotor (THRUST_MAX / 4)
HERE = Path(__file__).resolve().parent
TRACKS = {"raw": ("lsy_level2_track.yaml", 0.14), "tube": ("lsy_level2_tube.yaml", 0.30),
          # Lesson 7: the pole-aware tube (ball vias beside poles 1, 3, 4) lives next to the driver
          "poles": (str(HERE / "togt" / "lsy_level2_tube_poles.yaml"), 0.30),
          # Lesson 8: the same tube started ON THE GROUND -- plain, with gates 1 and 2
          # pre-compensated (b08), and with gate 3 pre-compensated on top of that (the headline)
          "ground": (str(HERE / "togt" / "lsy_level2_ground.yaml"), 0.30),
          "ground-b08": (str(HERE / "togt" / "lsy_level2_ground_b08.yaml"), 0.30),
          "ground-b08g3": (str(HERE / "togt" / "lsy_level2_ground_b08_g3.yaml"), 0.30)}
# a plan whose first point sits below this is a GROUND start: it needs boundZ down to 0,
# the hold instead of a takeoff, and the extra diagnostics (Lesson 8 §3)
GROUND_Z = 0.15
POLE_RADIUS = 0.015


def track_source(track: str, track_yaml: Path | None = None) -> Path:
    """The track yaml a run will read: --track-yaml if given, else the named track."""
    src = Path(track_yaml) if track_yaml is not None else Path(TRACKS[track][0])
    return src if src.is_absolute() else CFG / src


def track_init_z(src: Path) -> float:
    """initState.pos[2] of a track yaml (1.0 for the hover-start tracks, 0.05 on the ground)."""
    m = re.search(r"^initState:[ \t]*\n(?:^[ \t]+.*\n)*?^[ \t]+pos:[ \t]*\[([^\]]*)\]",
                  src.read_text(), re.M)
    return float(m.group(1).split(",")[2]) if m else 1.0


def patched_params(mode: str, frac: float, work: Path,
                   bound_z: tuple[float, float] | None = None) -> Path:
    dst = work / f"cf21b_{mode}"
    shutil.copytree(CFG / f"cf21b_{mode}", dst)
    maxthr = ROTOR_MAX_TRUE * K * frac
    for p in (dst / "quad.yaml", dst / "init" / "planning.yaml", dst / "refine" / "planning.yaml"):
        s = p.read_text()
        s = re.sub(r"^(thrust_max:\s*)[0-9.]+", rf"\g<1>{maxthr:.4f}", s, flags=re.M)
        s = re.sub(r"^(maxThr:\s*)[0-9.]+", rf"\g<1>{maxthr:.4f}", s, flags=re.M)
        if bound_z is not None:                                # ground starts: boundZ down to 0
            s = re.sub(r"^(boundZ:\s*)\[[^\]]*\]", rf"\g<1>[{bound_z[0]:g}, {bound_z[1]:g}]",
                       s, flags=re.M)
        p.write_text(s, newline="\n")                          # TOGT's parser: LF only
    return dst


def _patch_state(s: str, block: str, key: str, vec) -> str:
    """Replace `  key: [...]` inside the top-level `block:` mapping (initState / endState)."""
    pat = re.compile(rf"(^{block}:[ \t]*\n(?:^[ \t]+.*\n)*?^[ \t]+{key}:[ \t]*)\[[^\]]*\]", re.M)
    new = "[" + ", ".join(f"{float(v):g}" for v in vec) + "]"
    s2, n = pat.subn(lambda m: m.group(1) + new, s)
    if n != 1:
        raise ValueError(f"could not patch {block}.{key} (matched {n} times)")
    return s2


def patched_track(track: str, margin: float | None, work: Path, track_yaml: Path | None = None,
                  init_pos=None, end_pos=None, end_vel=None, via_radius: float | None = None) -> Path:
    src = track_source(track, track_yaml)
    s = src.read_text()
    if margin is not None:
        s = re.sub(r"marginW: [0-9.]+", f"marginW: {margin}", s)
        s = re.sub(r"marginH: [0-9.]+", f"marginH: {margin}", s)
    if init_pos is not None:
        s = _patch_state(s, "initState", "pos", init_pos)
    if end_pos is not None:
        s = _patch_state(s, "endState", "pos", end_pos)
    if end_vel is not None:
        s = _patch_state(s, "endState", "vel", end_vel)
    if via_radius is not None:
        s = re.sub(r"^(\s*radius:\s*)[0-9.]+", rf"\g<1>{via_radius:g}", s, flags=re.M)
    p = work / src.name
    p.write_text(s, newline="\n")
    return p


def append_coast(csv: Path, coast: float) -> None:
    """Extend the plan by `coast` seconds of constant-velocity motion (a = 0).

    Only useful with --end-vel: a plan that ends at speed still has a last SAMPLE, and
    SampledRaceTrajectory clamps the position and zeroes the velocity there, so the tracker
    would be asked to stop on the spot at the end. The coast gives it somewhere to go.
    """
    header = csv.read_text().splitlines()[0]
    names = header.split(",")
    d = np.genfromtxt(csv, delimiter=",", skip_header=1)
    if d.ndim == 1:
        d = d[None]
    col = {n: i for i, n in enumerate(names)}
    dt = float(np.median(np.diff(d[:, col["t"]])))
    n = int(round(coast / dt))
    last = d[-1].copy()
    v = last[[col["v_x"], col["v_y"], col["v_z"]]]
    rows = []
    for k in range(1, n + 1):
        r = last.copy()
        r[col["t"]] = last[col["t"]] + k * dt
        r[[col["p_x"], col["p_y"], col["p_z"]]] = last[[col["p_x"], col["p_y"], col["p_z"]]] + v * k * dt
        for c in ("a_lin_x", "a_lin_y", "a_lin_z"):
            if c in col:
                r[col[c]] = 0.0
        rows.append(r)
    out = np.vstack([d, np.asarray(rows)]) if rows else d
    np.savetxt(csv, out, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"  appended {n} coast samples ({coast:.2f} s at v = {np.round(v, 2)} m/s) to {csv.name}")


def diagnose(csv: Path, time_scale: float = 1.0, hold: float = 0.2) -> None:
    ref = lsy_level2_race()
    try:
        tr = SampledRaceTrajectory(str(csv), gates=ref.gates, obstacles=ref.obstacles,
                                   time_scale=time_scale)
    except ValueError as e:                                    # a gate not crossed inside its opening
        print(f"\n[!] {e}\n    The plan is not a lap: a gate is missed (or crossed outside the opening). "
              "Smaller --margin, or a different track/mode.")
        return
    rep = feasibility_report(tr)
    t = np.arange(tr.lead_in, tr.t_end, 0.002)
    thrust = np.linalg.norm(tr.acc(t) + np.array([0.0, 0.0, GRAVITY]), axis=1)
    vmax = float(np.linalg.norm(tr.vel(t), axis=1).max())
    lap = tr.gate_times[-1] - tr.lead_in
    # a plan that starts on the floor is flown with a HOLD, not a takeoff (Lesson 8 §3): the
    # three extra lines below only make sense -- and only appear -- for such a plan.
    p0 = np.asarray(tr.pos(tr.lead_in), dtype=np.float64)
    ground_start = bool(p0[2] < GROUND_Z)
    print(f"\nplan: {csv}" + (f"  (time-stretched x{time_scale:.2f})" if time_scale != 1.0 else ""))
    print(f"  planned lap (motion onset -> last gate) {lap:6.3f} s     incl. the stop at the end {tr.plan_duration:5.2f} s")
    if ground_start:
        print(f"  first point {np.round(p0, 3)} is on the ground: the bridge's RACE_TAKEOFF_T is a pure "
              f"HOLD, so the ground clock reads")
        print(f"      planned lap + hold = {lap:.3f} + {hold:.2f} = {lap + hold:6.3f} s   "
              f"(the race floors it to its 20 ms step)")
        s_first = (0.1, 0.2, 0.3, 0.4, 0.5)
        z_first = [float(tr.pos(tr.lead_in + s)[2]) for s in s_first]
        vz_first = [float(tr.vel(tr.lead_in + s)[2]) for s in s_first]
        thr_first = [float(np.linalg.norm(tr.acc(tr.lead_in + s) + np.array([0.0, 0.0, GRAVITY])))
                     for s in (0.0, 0.1, 0.2, 0.3)]
        print("  first 0.5 s: z = " + " ".join(f"{z:.2f}" for z in z_first)
              + " m at 0.1..0.5 s;  vz = " + " ".join(f"{v:.2f}" for v in vz_first)
              + " m/s;  thrust at 0/0.1/0.2/0.3 s = " + " ".join(f"{a:.1f}" for a in thr_first)
              + " m/s^2   (nothing leaves the floor before 0.14 s -- Lesson 8 §2)")
    print(f"  top speed {vmax:.2f} m/s   thrust demand {thrust.min():.1f}..{thrust.max():.1f} m/s^2 "
          f"(limit {rep['thrust_acc_limit']:.1f})   min z {rep['min_z']:.2f} m")
    offs = " ".join(f"{o:.3f}" for o in rep["gate_max_inplane_offset"])
    print(f"  intended crossings inside the opening: {'yes' if all(o <= rep['gate_clearance_bound'] for o in rep['gate_max_inplane_offset']) else 'NO'}"
          f"  (offsets {offs} m, bound {rep['gate_clearance_bound']:.2f})")
    print("  crossing angle to the gate normal:  "
          + "   ".join(f"G{i + 1} {c['angle_deg']:4.0f} deg ({c['inplane_speed']:.1f} m/s in-plane)"
                       for i, c in enumerate(crossing_angles(tr))))
    bad = [(i + 1, r) for i, rows in enumerate(plane_crossings(tr)) for r in rows if r["verdict"] == "FRAME"]
    if bad:
        print("  [!] the path crosses a gate PLANE through the FRAME zone (0.13-0.43 m from the centre):")
        for i, r in bad:
            print(f"      G{i} at t={r['t'] - tr.lead_in:5.2f} s, {r['offset']:.2f} m from the centre"
                  + ("  <- the intended crossing" if r["intended"] else "  (a second pass, not the intended one)"))
        print("      feasibility_report's gate_crossings_ok is False for this reason; the race would call it a crash.")
    else:
        print("  every gate-plane crossing is either through the opening or clear of the frame")
    pos, vel = tr.pos(t), tr.vel(t)
    clear = obstacle_clearance(pos, tr.obstacles)
    tight = [i + 1 for i, c in enumerate(clear) if c < 0.10]
    print("  obstacle clearance (pole surface):  " + "  ".join(f"pole{i + 1} {c:.2f} m" for i, c in enumerate(clear))
          + (f"   [!] < 0.10 m at pole {tight}: the race would likely score a contact" if tight else ""))
    if ground_start:
        # WHEN the plan passes each pole, and how fast: a ground start flies the first leg low
        # and fast, and the tracker's cut eats 0.05-0.10 m of every one of these margins.
        rows = []
        for k, ob in enumerate(tr.obstacles):
            d = np.linalg.norm(pos[:, :2] - np.asarray(ob[:2], dtype=np.float64), axis=1) - POLE_RADIUS
            d = np.where(pos[:, 2] < ob[2], d, np.inf)         # only while below the pole top
            i = int(d.argmin())
            rows.append((k + 1, float(d[i]), float(t[i] - tr.lead_in), float(pos[i, 2]),
                         float(np.linalg.norm(vel[i]))))
        print("  closest passage of each pole:  "
              + "  ".join(f"pole{k} {c:.2f} m at t={tt:.2f} s (z {z:.2f} m, {v:.1f} m/s)"
                          for k, c, tt, z, v in rows))
    print(f"\nnext:  python tasks/racing/code/race_eval.py --plan {csv.relative_to(ROOT) if csv.is_relative_to(ROOT) else csv} "
          f"--controller mpc --reason \"...\"")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--track", choices=sorted(TRACKS), default="raw",
                   help="raw = the four gates as zero-thickness corridors; tube = each gate "
                        "bracketed by an entry and an exit corridor 0.6 m along its normal (upstream; "
                        "three corridors stand on poles); poles = the pole-aware tube of Lesson 7; "
                        "ground / ground-b08 / ground-b08g3 = the same tube started on the FLOOR, "
                        "plain and with the gate pre-compensations of Lesson 8 §3")
    p.add_argument("--track-yaml", type=Path, default=None,
                   help="your own TOGT track file (Lesson 6 exercise 3 / Lesson 7): overrides --track; "
                        "the plan is named after the file")
    p.add_argument("--thrust-frac", type=float, default=0.95,
                   help="fraction of the platform thrust the planner may use (per-rotor bound)")
    p.add_argument("--margin", type=float, default=None,
                   help="gate margin (window = 0.5*(0.4 - margin)); default 0.14 raw / 0.30 tube")
    p.add_argument("--mode", choices=["togt", "aos"], default="togt",
                   help="togt = planTOGT (used everywhere); aos = the 'true time optimality' refine, "
                        "which drops gates on this track (documented, not recommended)")
    p.add_argument("--bin", type=Path, default=BIN, help="the built togt_race driver")
    p.add_argument("--out-dir", type=Path, default=PLANS)
    p.add_argument("--name", default=None, help="CSV name (default from track/frac/margin)")
    p.add_argument("--diagnose-only", type=Path, default=None,
                   help="skip planning; print the diagnostics of an existing plan CSV")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="with --diagnose-only: report the plan stretched in time")
    # ---- Lesson 8 §3: what a ground start needs (all optional; omitted = Lesson 6/7 behaviour)
    p.add_argument("--init-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="override the track's initState.pos (a ground start: -1.5 0.75 0.05)")
    p.add_argument("--bound-z", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="TOGT's boundZ in the planning parameter files (default: 0.0 3.0 when the "
                        "track's initState z is below 0.15 m, else the parameter files' own values)")
    p.add_argument("--end-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="override the track's endState.pos (H2: a stop further past gate 4)")
    p.add_argument("--end-vel", type=float, nargs=3, default=None, metavar=("VX", "VY", "VZ"),
                   help="override the track's endState.vel (H2: cross gate 4 at speed)")
    p.add_argument("--via-radius", type=float, default=None,
                   help="override every SingleBall via radius of the track (H6: 0.15 -> 0.18/0.20)")
    p.add_argument("--coast", type=float, default=0.0,
                   help="append this many seconds of constant-velocity motion to the CSV "
                        "(only with --end-vel: the plan's last sample would otherwise stop dead)")
    p.add_argument("--hold", type=float, default=0.2,
                   help="the hold (RACE_TAKEOFF_T) the ground-clock lap in the diagnostics assumes")
    args = p.parse_args()

    if args.diagnose_only is not None:
        diagnose(args.diagnose_only, args.time_scale, args.hold)
        return
    if not args.bin.exists():
        sys.exit(f"driver not found: {args.bin}\n   build it once:  bash tasks/racing/code/togt/build.sh")

    margin = args.margin
    base = args.track_yaml.stem if args.track_yaml is not None else args.track
    name = args.name or (f"{base}_f{args.thrust_frac:.2f}"
                         + (f"_m{margin:.2f}" if margin is not None else "")
                         + ("" if args.mode == "togt" else f"_{args.mode}"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = (args.out_dir / f"{name}.csv").resolve()

    # boundZ: a plan that starts on the floor cannot satisfy the parameter files' 0.15 m floor
    bound_z = tuple(args.bound_z) if args.bound_z is not None else None
    if bound_z is None:
        init_z = (float(args.init_pos[2]) if args.init_pos is not None
                  else track_init_z(track_source(args.track, args.track_yaml)))
        if init_z < GROUND_Z:
            bound_z = (0.0, 3.0)
            print(f"ground start (initState z {init_z:g} m): planning with boundZ {list(bound_z)}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        params = patched_params(args.mode, args.thrust_frac, work, bound_z)
        track = patched_track(args.track, margin, work, args.track_yaml,
                              args.init_pos, args.end_pos, args.end_vel, args.via_radius)
        # Absolute paths on purpose: TOGT's loader prefixes a relative params dir twice
        # and then segfaults on empty parameters (documented in configs/togt/README.md).
        cmd = [str(args.bin.resolve()), str(params.resolve()), "setups.yaml", str(track.resolve()),
               str(out_csv), args.mode]
        print("running:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        keep = [ln for ln in (r.stdout + r.stderr).splitlines()
                if re.search(r"Duration|fail|FAIL|not found|saved|segfault", ln, re.I)]
        print("\n".join("  " + ln for ln in keep) or "  (no output)")
        if r.returncode != 0 or not out_csv.exists():
            sys.exit(f"\nplanning failed (exit {r.returncode}). Common causes, see docs/4-troubleshooting.md:\n"
                     "  'Config files not found!'  -> a CRLF checkout of configs/togt (must be LF)\n"
                     "  'Optimizatoin fails!'      -> dynamicConstCheck must stay false; or the margin leaves no window\n"
                     "  a segfault                 -> relative paths, or prism gates (length > 0)")
    if args.coast > 0:
        append_coast(out_csv, args.coast)
    print(f"wrote {out_csv}")
    diagnose(out_csv, hold=args.hold)


if __name__ == "__main__":
    main()
