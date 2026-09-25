"""Fly RL checkpoints (and M1) over a track set at lambda=0, in parallel, and summarise (Lesson 9, 2026-09-25).

    python tasks/racing/code/lesson9/eval_pool.py --role val \
        --member v4_16M=robust:/workspace/.../datt_ppo_final.zip --member open=robust:/workspace/.../final.zip --m1

    python tasks/racing/code/lesson9/eval_pool.py --role study --tracks 4,25,93,387,504 --member ...   # selection set

Runs in the container (main venv). One `driver.py` process per track (resumable on its own CSV under
results/eval/<tag>/), `launcher.launch` for the parallelism. `--m1` adds the corrected MPC as the reference
column. Prints per-member completions / gates / mean rmse and, per track, who completed -- the validation pool
(`--role val`) is what RL design decisions are judged on from here on, NOT study tracks (limitation 15).
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


def track_seeds(role: str, only: str | None) -> list[int]:
    if only:
        return [int(x) for x in only.split(",")]
    return sorted(int(p.stem.split("_")[1]) for p in (HERE / "tracks" / role).glob("track_*.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--role", required=True, choices=["study", "dev", "train", "val"])
    ap.add_argument("--tracks", default=None, help="comma-separated seeds (default: every track of the role)")
    ap.add_argument("--member", action="append", default=[], metavar="LABEL=SPEC")
    ap.add_argument("--m1", action="store_true", help="also fly M1 (the corrected MPC) as the reference column")
    ap.add_argument("--tag", default=None, help="results/eval/<tag>/ (default: <role>)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    members = dict(m.split("=", 1) for m in args.member)
    labels = list(members) + (["M1"] if args.m1 else [])
    seeds = track_seeds(args.role, args.tracks)
    out_dir = HERE / "results" / "eval" / (args.tag or args.role)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [{"role": args.role, "track": s, "cond": "wind_const", "lams": [0.0], "seeds": 1, "members": members,
              "only": ",".join(labels), "out": out_dir / f"laps_{args.role}_{s:06d}.csv"} for s in seeds]
    print(f"{len(seeds)} {args.role} tracks x members {labels}", flush=True)
    launch(tasks, workers=args.workers)

    rows = defaultdict(dict)
    for t in tasks:
        if not Path(t["out"]).exists():
            continue
        for r in csv.DictReader(open(t["out"])):
            if float(r["lam"]) == 0.0 and r["member"] in labels:
                rows[r["member"]][t["track"]] = r
    print(f"\n{'member':<22} {'complete':>9} {'gates':>8} {'mean rmse':>10} {'median max_dev':>15}")
    for m in labels:
        rs = list(rows[m].values())
        if not rs:
            print(f"{m:<22} (no rows)")
            continue
        print(f"{m:<22} {sum(int(r['completed']) for r in rs):>4}/{len(rs):<4} "
              f"{sum(int(r['gates_passed']) for r in rs):>4}/{4 * len(rs):<3} "
              f"{np.mean([float(r['rmse_3d']) for r in rs]):>10.4f} {np.median([float(r['max_dev']) for r in rs]):>15.3f}")
    print("\nper track (completed=1):  " + "  ".join(f"{m}" for m in labels))
    for s in seeds:
        cells = []
        for m in labels:
            r = rows[m].get(s)
            cells.append("  -  " if r is None else f"{int(r['completed'])}/{int(r['gates_passed'])}")
        print(f"  {s:>7}  " + "  ".join(f"{c:>8}" for c in cells))


if __name__ == "__main__":
    main()
