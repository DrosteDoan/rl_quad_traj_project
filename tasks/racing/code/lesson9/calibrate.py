"""Phase 5: calibrate the disturbance ceilings on the DEV tracks (METHODOLOGY.md section 4).

    python tasks/racing/code/lesson9/calibrate.py run --workers 8      # flies the scan (about 30-40 min)
    python tasks/racing/code/lesson9/calibrate.py report               # tables + a PROPOSED maxima.json

The rule (user): the ceiling of a disturbance is the level at which at most one controller still persists, where
persisting means completing at least 50 % of the laps pooled over the dev tracks and seeds. Every disturbance is
scanned in units of Lesson 7's value (lam_eff = 1 is Lesson 7's condition); the driver's lam is lam_eff / SCAN_MAX
with the scale set to SCAN_MAX. The proposed scale of a disturbance is the first grid level from which at most one
member persists at every higher level too. Nothing is frozen here: `report` writes `maxima_proposed.json`, and
freezing is copying it to `maxima.json` after the user approves.

Only the MPC family exists at this point (the robust RL seeds are trained on the frozen ceilings), so the rule is
applied to those four members; re-check it once the policies exist. `latency_only` is the delay-only check for the
fixed 1-step Lighthouse latency (METHODOLOGY.md section 4).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import driver as dr  # noqa: E402
import launcher  # noqa: E402

DEV_TRACKS = [100023, 100082, 100092]      # regenerated 2026-09-22 at thrust-frac 0.80
SCAN_MAX = 8.0
GRID_EFF = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]      # lam_eff, units of Lesson 7's value
CONDS = ("wind_const", "payload", "wind_gust", "lighthouse")
STOCHASTIC = ("wind_gust", "lighthouse")
SEEDS = 5
OUT = dr.RESULTS / "calib"


def make_tasks() -> list[dict]:
    tasks = []
    for cond in CONDS:
        for tr in DEV_TRACKS:
            tasks.append({"role": "dev", "track": tr, "cond": cond, "seeds": SEEDS if cond in STOCHASTIC else 1,
                          "lams": [0.0] + [x / SCAN_MAX for x in GRID_EFF], "scales": {cond: SCAN_MAX},
                          "out": OUT / f"laps_dev_{tr}_{cond}.csv"})
    for tr in DEV_TRACKS:
        tasks.append({"role": "dev", "track": tr, "cond": "latency_only", "seeds": 1, "lams": [0.0, 0.5],
                      "out": OUT / f"laps_dev_{tr}_latency_only.csv"})
    return tasks


def load() -> list[dict]:
    rows = []
    for f in sorted(OUT.glob("laps_dev_*.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def report() -> None:
    rows = load()
    if not rows:
        sys.exit(f"no calibration laps in {OUT}")
    members = list(dict.fromkeys(r["member"] for r in rows))
    print(f"{len(rows)} laps; dev tracks {DEV_TRACKS}; members {members}\n")

    # viability at lam = 0 (nominal, one lap per track and member)
    base = defaultdict(list)
    for r in rows:
        if float(r["lam"]) == 0.0 and r["cond"] == "wind_const":
            base[r["member"]].append(int(r["completed"]))
    print("lam = 0 (nominal) completions over the 3 dev tracks: "
          + "  ".join(f"{m} {sum(base[m])}/{len(base[m])}" for m in members) + "\n")

    # the delay-only check
    lat = defaultdict(list)
    for r in rows:
        if r["cond"] == "latency_only" and float(r["lam"]) > 0:
            lat[r["member"]].append(int(r["completed"]))
    if lat:
        print("delay-only check (1-step latency, every error size zero): "
              + "  ".join(f"{m} {sum(lat[m])}/{len(lat[m])}" for m in members)
              + "   (compare with the lam = 0 line above)\n")

    proposed, summary = {}, []
    for cond in CONDS:
        cell = defaultdict(list)
        for r in rows:
            if r["cond"] == cond:
                cell[(round(float(r["lam"]) * SCAN_MAX, 3), r["member"])].append(int(r["completed"]))
        levels = sorted({k[0] for k in cell})
        print(f"== {cond}  (completion, pooled over dev tracks{' and 5 seeds' if cond in STOCHASTIC else ''}; "
              f"lam_eff = 1 is Lesson 7's condition)")
        print(f"{'lam_eff':>8s} " + " ".join(f"{m:>8s}" for m in members) + "   persisting")
        persist = {}
        for lv in levels:
            comp = {m: float(np.mean(cell[(lv, m)])) if cell[(lv, m)] else np.nan for m in members}
            persist[lv] = sum(c >= 0.5 for c in comp.values())
            print(f"{lv:8.2f} " + " ".join(f"{100 * comp[m]:7.0f}%" for m in members) + f"   {persist[lv]}")
            for m in members:
                summary.append({"cond": cond, "lam_eff": lv, "member": m, "completion": comp[m],
                                "laps": len(cell[(lv, m)])})
        pos = [lv for lv in levels if lv > 0]
        ceiling = None
        for i, lv in enumerate(pos):
            if all(persist[x] <= 1 for x in pos[i:]):
                ceiling = lv
                break
        proposed[cond] = ceiling if ceiling is not None else f">{pos[-1]:g}"
        note = ("" if ceiling is None else " (already at the first grid level)" if ceiling == pos[0] else "")
        print(f"-> proposed ceiling: {proposed[cond]}x Lesson 7's value{note}\n")

    with open(OUT / "calibration_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cond", "lam_eff", "member", "completion", "laps"])
        w.writeheader()
        w.writerows(summary)
    scales = {k: v for k, v in proposed.items() if not isinstance(v, str)}
    (HERE / "maxima_proposed.json").write_text(json.dumps(
        {"scales": scales, "unreached": {k: v for k, v in proposed.items() if isinstance(v, str)},
         "rule": "first grid level from which at most one MPC-family member completes >= 50 % of dev laps",
         "dev_tracks": DEV_TRACKS, "grid_lam_eff": GRID_EFF, "status": "PROPOSED, not frozen"}, indent=1))
    print(f"wrote {HERE / 'maxima_proposed.json'} (proposed; copy to maxima.json to freeze)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["run", "report"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.what == "run":
        launcher.launch(make_tasks(), workers=args.workers)
    else:
        report()


if __name__ == "__main__":
    main()
