"""lambda > 0 PREVIEW on the validation pool (Lesson 9, 2026-09-25): does a crossover look plausible at all?

    python tasks/racing/code/lesson9/preview_lambda.py --member v4=robust:<.../datt_ppo_final.zip>   # container

DEVELOPMENT information, not a result: it runs on the 22 `val` tracks (never trained on, not study tracks), a
coarse lambda grid {0.25, 0.5, 0.75, 1.0} of each condition's frozen ceiling (`maxima.json`), and three members --
the RL checkpoint, M1, and M1+L1 (the most disturbance-robust MPC member in calibration; `--only` changes that).
Deterministic conditions (wind_const, payload, mass_mult) fly one seed, stochastic ones (wind_gust, lighthouse,
combined) three. lambda = 0 is flown once, under wind_const (it is identical in every condition).
The reported study uses the 10 untouched study tracks and the full grid; nothing here is tuned to.

Two views per (condition, member, lambda), because RL is only viable on part of the pool at lambda = 0:
  raw        completed laps / laps, over ALL val tracks (what a reader sees first)
  retention  the same, over only the tracks THAT MEMBER completed at lambda = 0 -- how fast it degrades once viable
             (the study's rule: compare where both are viable at lambda = 0)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launcher import launch  # noqa: E402

CONDS = ("wind_const", "payload", "wind_gust", "lighthouse", "mass_mult", "combined")
STOCHASTIC = ("wind_gust", "lighthouse", "combined")
LAMS = [0.25, 0.5, 0.75, 1.0]


def summarise(out_dir: Path, labels: list[str]) -> None:
    rows = []
    for f in sorted(out_dir.glob("laps_val_*.csv")):
        rows += [r for r in csv.DictReader(open(f)) if r["member"] in labels]
    base = {m: {r["track"] for r in rows if r["member"] == m and r["cond"] == "wind_const"
                and float(r["lam"]) == 0.0 and r["completed"] == "1"} for m in labels}
    n_tracks = len({r["track"] for r in rows})
    print(f"\nbase (completed at lambda=0, of {n_tracks} tracks): " + ", ".join(f"{m} {len(base[m])}" for m in labels))
    for view in ("raw", "retention"):
        print(f"\n== {view}: completed laps / laps ==")
        print(f"{'condition':<11} {'member':<8}" + "".join(f"{'lam=' + str(x):>14}" for x in [0.0] + LAMS))
        for cond in CONDS:
            for m in labels:
                cells = []
                for lam in [0.0] + LAMS:
                    c = "wind_const" if lam == 0.0 else cond
                    rs = [r for r in rows if r["member"] == m and r["cond"] == c and abs(float(r["lam"]) - lam) < 1e-9]
                    if view == "retention":
                        rs = [r for r in rs if r["track"] in base[m]]
                    k, n = sum(int(r["completed"]) for r in rs), len(rs)
                    cells.append(f"{k:>3}/{n:<3} {100 * k / n:3.0f}%" if n else "      -     ")
                print(f"{cond:<11} {m:<8}" + "".join(f"{c:>14}" for c in cells))
            print()


def paired_rmse(out_dir: Path, a: str, b_labels: list[str]) -> None:
    """Tracking error, not completion: median RMSE of member `a` against each member in `b_labels`, over the laps BOTH
    completed on the same (track, seed). RMSE of a lap that ended early is censored, so failed laps are excluded --
    which selects survivors: at high lambda the pairs left are the easy laps and n is small, so read cells with n < 5
    as anecdote. `a<b` is the share of pairs where `a` tracked more tightly."""
    R = {}
    for f in sorted(out_dir.glob("laps_val_*.csv")):
        for r in csv.DictReader(open(f)):
            R[(r["cond"], round(float(r["lam"]), 4), r["track"], r["seed"], r["member"])] = r
    print(f"\n== paired tracking RMSE (m) on laps both completed: {a} vs each of {b_labels} ==")
    print(f"{'condition':<11}{'lam':>5}" + "".join(f"  {'n':>3} {a:>7} {b:>7} {a + '<' + b:>9}"[:40] for b in b_labels))
    for cond in CONDS:
        for lam in [0.0] + LAMS:
            if lam == 0.0 and cond != "wind_const":
                continue
            c = "wind_const" if lam == 0.0 else cond
            cells = []
            for b in b_labels:
                pairs = []
                for (cc, ll, t, sd, m), ra in R.items():
                    if cc == c and ll == lam and m == a and ra["completed"] == "1":
                        rb = R.get((c, lam, t, sd, b))
                        if rb and rb["completed"] == "1":
                            pairs.append((float(ra["rmse_3d"]), float(rb["rmse_3d"])))
                if pairs:
                    cells.append(f"  {len(pairs):>3} {np.median([x[0] for x in pairs]):7.3f} {np.median([x[1] for x in pairs]):7.3f} "
                                 f"{np.mean([x[0] < x[1] for x in pairs]):9.2f}")
                else:
                    cells.append(f"  {0:>3} {'-':>7} {'-':>7} {'-':>9}")
            print(f"{cond:<11}{lam:>5}" + "".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--member", action="append", default=[], metavar="LABEL=SPEC", required=True)
    ap.add_argument("--only", default="M1,M1+L1", help="MPC members to fly next to the RL ones")
    ap.add_argument("--tag", default="val_lambda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--paired-rmse", default=None, metavar="LABEL",
                    help="also print paired tracking RMSE of this member against every other member (completed laps only)")
    args = ap.parse_args()

    members = dict(m.split("=", 1) for m in args.member)
    labels = list(members) + [x for x in args.only.split(",") if x]
    out_dir = HERE / "results" / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        seeds = sorted(int(p.stem.split("_")[1]) for p in (HERE / "tracks" / "val").glob("track_*.json"))
        tasks = [{"role": "val", "track": s, "cond": c, "lams": ([0.0] if c == "wind_const" else []) + LAMS,
                  "seeds": 3 if c in STOCHASTIC else 1, "members": members, "only": ",".join(labels),
                  "out": out_dir / f"laps_val_{s:06d}_{c}.csv"} for s in seeds for c in CONDS]
        print(f"{len(tasks)} tasks ({len(seeds)} tracks x {len(CONDS)} conditions), members {labels}", flush=True)
        launch(tasks, workers=args.workers)
    summarise(out_dir, labels)
    if args.paired_rmse:
        paired_rmse(out_dir, args.paired_rmse, [x for x in labels if x != args.paired_rmse])


if __name__ == "__main__":
    main()
