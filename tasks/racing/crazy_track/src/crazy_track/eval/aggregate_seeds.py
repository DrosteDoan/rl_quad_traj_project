"""Aggregate multi-seed benchmark runs into mean +/- std tables.

Usage: python -m crazy_track.eval.aggregate_seeds --prefix ms-
Groups results/*_<prefix>*/summary.csv rows by (cell, controller, trajectory),
where cell = the tag with the trailing seed suffix (-sN) stripped.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from crazy_track.eval.runlog import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="ms-")
    args = parser.parse_args()

    groups = defaultdict(list)
    for d in sorted((REPO_ROOT / "results").glob(f"*_{args.prefix}*")):
        tag = d.name.split("_", 3)[-1]  # strip YYYY-MM-DD_HH-MM-SS_
        cell = re.sub(r"-s\d+$", "", tag)
        f = d / "summary.csv"
        if not f.exists():
            continue
        with open(f) as fh:
            for row in csv.DictReader(fh):
                groups[(cell, row["controller"], row["trajectory"])].append(
                    float(row["rmse_3d"])
                )

    print(f"{'cell':16s} {'controller':12s} {'traj':7s} {'n':>2s} "
          f"{'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s}")
    for (cell, ctrl, traj), vals in sorted(groups.items()):
        v = np.asarray(vals)
        print(f"{cell:16s} {ctrl:12s} {traj:7s} {len(v):2d} "
              f"{v.mean():7.3f} {v.std():7.3f} {v.min():7.3f} {v.max():7.3f}")


if __name__ == "__main__":
    main()
