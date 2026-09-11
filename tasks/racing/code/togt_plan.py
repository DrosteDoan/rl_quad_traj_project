"""Plan a lap with the TOGT-Planner and diagnose it — Lesson 6 §2.

    python tasks/racing/code/togt_plan.py --track raw                        # zero-thickness gates
    python tasks/racing/code/togt_plan.py --track tube --thrust-frac 0.85    # entry/exit corridors

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
TRACKS = {"raw": ("lsy_level2_track.yaml", 0.14), "tube": ("lsy_level2_tube.yaml", 0.30),
          # Lesson 7: the pole-aware tube (ball vias beside poles 1, 3, 4) lives next to the driver
          "poles": (str(Path(__file__).resolve().parent / "togt" / "lsy_level2_tube_poles.yaml"), 0.30)}


def patched_params(mode: str, frac: float, work: Path) -> Path:
    dst = work / f"cf21b_{mode}"
    shutil.copytree(CFG / f"cf21b_{mode}", dst)
    maxthr = ROTOR_MAX_TRUE * K * frac
    for p in (dst / "quad.yaml", dst / "init" / "planning.yaml", dst / "refine" / "planning.yaml"):
        s = p.read_text()
        s = re.sub(r"^(thrust_max:\s*)[0-9.]+", rf"\g<1>{maxthr:.4f}", s, flags=re.M)
        s = re.sub(r"^(maxThr:\s*)[0-9.]+", rf"\g<1>{maxthr:.4f}", s, flags=re.M)
        p.write_text(s, newline="\n")                          # TOGT's parser: LF only
    return dst


def patched_track(track: str, margin: float | None, work: Path, track_yaml: Path | None = None) -> Path:
    src = Path(track_yaml) if track_yaml is not None else Path(TRACKS[track][0])
    if not src.is_absolute():
        src = CFG / src
    s = src.read_text()
    if margin is not None:
        s = re.sub(r"marginW: [0-9.]+", f"marginW: {margin}", s)
        s = re.sub(r"marginH: [0-9.]+", f"marginH: {margin}", s)
    p = work / src.name
    p.write_text(s, newline="\n")
    return p


def diagnose(csv: Path, time_scale: float = 1.0) -> None:
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
    print(f"\nplan: {csv}" + (f"  (time-stretched x{time_scale:.2f})" if time_scale != 1.0 else ""))
    print(f"  planned lap (motion onset -> last gate) {lap:6.3f} s     incl. the stop at the end {tr.plan_duration:5.2f} s")
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
    clear = obstacle_clearance(tr.pos(t), tr.obstacles)
    tight = [i + 1 for i, c in enumerate(clear) if c < 0.10]
    print("  obstacle clearance (pole surface):  " + "  ".join(f"pole{i + 1} {c:.2f} m" for i, c in enumerate(clear))
          + (f"   [!] < 0.10 m at pole {tight}: the race would likely score a contact" if tight else ""))
    print(f"\nnext:  python tasks/racing/code/race_eval.py --plan {csv.relative_to(ROOT) if csv.is_relative_to(ROOT) else csv} "
          f"--controller mpc --reason \"...\"")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--track", choices=sorted(TRACKS), default="raw",
                   help="raw = the four gates as zero-thickness corridors; tube = each gate "
                        "bracketed by an entry and an exit corridor 0.6 m along its normal (upstream; "
                        "three corridors stand on poles); poles = the pole-aware tube of Lesson 7")
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
    args = p.parse_args()

    if args.diagnose_only is not None:
        diagnose(args.diagnose_only, args.time_scale)
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

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        params = patched_params(args.mode, args.thrust_frac, work)
        track = patched_track(args.track, margin, work, args.track_yaml)
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
    print(f"wrote {out_csv}")
    diagnose(out_csv)


if __name__ == "__main__":
    main()
