"""Generate the Lesson 9 track library: the first N seeds that pass the structural filter, in seed order.

    python tasks/racing/code/lesson9/gen_tracks.py --role study --n 15 --seed-start 0
    python tasks/racing/code/lesson9/gen_tracks.py --role dev   --n 3  --seed-start 1000
    python tasks/racing/code/lesson9/gen_tracks.py --role train --n 100000 --seed-start 200000 --max-tries 400 \
        --manifest-part 0          # one seed WINDOW of a parallel run (gen_pool.py launches them all)

Runs in the MAIN venv, in the container (needs the TOGT driver: bash tasks/racing/code/togt/build.sh).
For each candidate seed it builds the lsy level-3 layout, plans a TOGT ground-start tube, and applies
`track_lib.diagnose_plan` (METHODOLOGY.md section 3: a verdict on the PLAN, never on a controller). It
writes

    tasks/racing/code/lesson9/tracks/<role>/track_<seed>.json   passing tracks (small, tracked in git)
    tasks/racing/code/lesson9/tracks/manifest_<role>.csv        EVERY candidate, pass or reject, with reasons
    tasks/racing/plans/lesson9/<role>_<seed>.csv                the TOGT plan (git-ignored, regenerable)
    tasks/racing/figures/lesson9/tracks_<role>.png              top view of the accepted tracks (git-ignored)

Selection is by seed order, so it cannot be cherry-picked; every reject and its reason is in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

import track_lib as tl  # noqa: E402  (sits next to this file)

HERE = Path(__file__).resolve().parent
ROOT = tl.ROOT
PLANS = ROOT / "tasks" / "racing" / "plans" / "lesson9"
FIGS = ROOT / "tasks" / "racing" / "figures" / "lesson9"

COLUMNS = ["role", "seed", "status", "reasons", "plan_lap_s", "v_peak", "thrust_peak", "max_angle_deg",
           "stretch", "lap_stretched_s", "min_gate_gap", "n_frame_crossings", "frame_margin_m"]


def lsy_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT / "repos" / "lsy_drone_racing"), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def plot_overview(role: str, tracks: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from crazy_track.trajectories.sampled import SampledRaceTrajectory

    n = len(tracks)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, trk in zip(axes.ravel(), tracks):
        ax.set_visible(True)
        gates = tl.race_gates(trk["gates"])
        plan = SampledRaceTrajectory(str(ROOT / trk["plan_csv"]), gates=gates, obstacles=[])
        t = np.arange(plan.lead_in, plan.t_end, 0.01)
        p = plan.pos(t)
        ax.add_patch(plt.Rectangle(tl.ARENA_LOW, *(tl.ARENA_HIGH - tl.ARENA_LOW), fill=False, ec="0.6", lw=0.8))
        ax.plot(p[:, 0], p[:, 1], color="#1f77b4", lw=1.2)
        for i, g in enumerate(gates, start=1):
            n_ = g.normal[:2]
            tang = np.array([-n_[1], n_[0]])
            a, b = np.asarray(g.pos[:2]) - 0.36 * tang, np.asarray(g.pos[:2]) + 0.36 * tang
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#d62728", lw=2.2)
            ax.annotate("", xy=np.asarray(g.pos[:2]) + 0.35 * n_, xytext=np.asarray(g.pos[:2]),
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
            ax.text(g.pos[0], g.pos[1] + 0.12, str(i), fontsize=8, ha="center")
        ax.plot(*tl.GROUND_START[:2], "ks", ms=4)
        d = trk["diag"]
        ax.set_title(f"seed {trk['seed']}  lap {d['plan_lap_s']:.2f}s  x{d['stretch']:.2f}", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(-2.7, 2.7)
        ax.set_ylim(-1.7, 1.7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Lesson 9 {role} tracks: gates (red, arrow = pass direction), TOGT plan (blue), start (black)",
                 fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--role", choices=["study", "dev", "train", "val"], required=True)
    ap.add_argument("--n", type=int, required=True, help="number of passing tracks to collect")
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--max-tries", type=int, default=400)
    ap.add_argument("--min-margin", type=float, default=tl.MIN_FRAME_MARGIN,
                    help="required reference frame margin in metres (contact.frame_margin); 0 = only 'no contact'")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--manifest-part", type=int, default=None,
                    help="write manifest_<role>_partNN.csv instead of manifest_<role>.csv, so several seed windows "
                         "of one role can run in parallel without overwriting each other (gen_pool.py merges them)")
    args = ap.parse_args()

    out_dir = HERE / "tracks" / args.role
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = lsy_commit()
    rows, passed = [], []
    seed = args.seed_start
    while len(passed) < args.n and seed < args.seed_start + args.max_tries:
        name = f"{args.role}_{seed:04d}"
        layout, diag = tl.build_track(seed, PLANS, name, args.min_margin)
        ok = not diag["reasons"]
        row = {"role": args.role, "seed": seed, "status": "pass" if ok else "reject",
               "reasons": "+".join(diag["reasons"])}
        for k in COLUMNS[4:]:
            v = diag.get(k)
            row[k] = "" if v is None else round(float(v), 3)
        rows.append(row)
        if ok:
            trk = {"role": args.role, "seed": seed, "gates": layout["gates"],
                   "start": list(tl.GROUND_START), "end": [float(x) for x in tl.end_point(layout["gates"])],
                   "plan_csv": str((PLANS / f"{name}.csv").relative_to(ROOT)),
                   "stretch": diag["stretch"], "thrust_frac": tl.THRUST_FRAC, "diag": diag,
                   "generator": {"fn": "lsy_drone_racing.envs.randomize.build_random_track_fn",
                                 "key": f"jax.random.PRNGKey({seed})", "lsy_commit": commit},
                   "constants": {"v_cap": tl.V_CAP, "angle_cap_deg": tl.ANGLE_CAP_DEG, "min_frame_margin": args.min_margin,
                                 "arena_margin": tl.ARENA_MARGIN, "tube_d": tl.TUBE_D, "end_past": tl.END_PAST}}
            (out_dir / f"track_{seed:04d}.json").write_text(json.dumps(trk, indent=1))
            passed.append(trk)
        print(f"seed {seed:5d}  {row['status']:6s} {row['reasons'] or '':28s} "
              + (f"lap {diag['plan_lap_s']:.2f}s  vmax {diag['v_peak']:.2f}  angle {diag['max_angle_deg']:.0f}deg  "
                 f"stretch {diag['stretch']:.3f}" if "plan_lap_s" in diag else diag.get("error", "")[:60]),
              flush=True)
        seed += 1

    mname = (f"manifest_{args.role}.csv" if args.manifest_part is None
             else f"manifest_{args.role}_part{args.manifest_part:02d}.csv")
    with open(HERE / "tracks" / mname, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n_try, n_pass = len(rows), len(passed)
    why = Counter(r for row in rows if row["status"] == "reject" for r in row["reasons"].split("+"))
    print(f"\n{args.role}: {n_pass} passed of {n_try} candidates (seeds {args.seed_start}..{seed - 1}); "
          f"reject reasons (a reject can have several): {dict(why)}")
    if n_pass < args.n:
        print(f"[!] only {n_pass}/{args.n} found within --max-tries {args.max_tries}")
    if passed and not args.no_plot:
        plot_overview(args.role, passed, FIGS / f"tracks_{args.role}.png")
        print(f"overview: {FIGS / f'tracks_{args.role}.png'}")


if __name__ == "__main__":
    main()
