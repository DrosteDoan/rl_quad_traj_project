"""Run the exact gate-frame contact check on the REFERENCE path of every accepted track.

    python tasks/racing/code/lesson9/check_references.py --role study
    python tasks/racing/code/lesson9/check_references.py --role dev

For each track: does the level drone box flying the TOGT plan touch a frame (`contact.first_contact`), and how
far could the box grow before it did (`contact.frame_margin`, metres per half-extent: roughly how far a
tracker may stray from the plan, in any direction, before lsy would end the race). Phase 1 selected tracks with
a plane-crossing proxy; this is the exact test, and it writes `tracks/reference_margins_<role>.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import contact as ct
import track_lib as tl
from crazy_track.trajectories.sampled import SampledRaceTrajectory

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--role", choices=["study", "dev"], required=True)
    args = ap.parse_args()
    rows = []
    for f in sorted((HERE / "tracks" / args.role).glob("track_*.json")):
        trk = json.loads(f.read_text())
        gates = tl.race_gates(trk["gates"])
        plan = SampledRaceTrajectory(str(tl.ROOT / trk["plan_csv"]), gates=gates, obstacles=[])
        t = np.arange(plan.lead_in, plan.t_end, 0.005)
        pos = plan.pos(t)
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))               # the reference has no attitude: level
        hit = ct.first_contact(t, pos, quat, gates)
        margin = ct.frame_margin(t, pos, quat, gates)
        rows.append({"role": args.role, "seed": trk["seed"], "contact": "" if hit is None else
                     f"G{hit['gate']}-{hit['box']}@{hit['t'] - plan.lead_in:.2f}s",
                     "frame_margin_m": round(margin, 3)})
        print(f"seed {trk['seed']:5d}  {'CONTACT ' + rows[-1]['contact'] if hit else 'clean':22s} "
              f"frame margin {margin * 100:5.1f} cm")
    out = HERE / "tracks" / f"reference_margins_{args.role}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    m = np.array([r["frame_margin_m"] for r in rows])
    print(f"\n{args.role}: {sum(bool(r['contact']) for r in rows)} of {len(rows)} references touch a frame; "
          f"frame margin cm  min {m.min() * 100:.1f}  median {np.median(m) * 100:.1f}  max {m.max() * 100:.1f}")


if __name__ == "__main__":
    main()
