"""Turn race_eval.py runs into the Lesson-6 comparison table.

    python tasks/racing/code/race_table.py                 # both clocks
    python tasks/racing/code/race_table.py --start ground  # leaderboard clock only
    python tasks/racing/code/race_table.py --csv tasks/racing/figures/race_table.csv

Reads every tasks/racing/crazy_track/results/*_race-eval-* directory (metadata.yaml +
summary.csv), keeps the LATEST run per (clock, plan, controller label), and prints one
table per clock: rows = plans, columns = controllers, cell = race_time in seconds with the
max deviation from the reference in brackets, or "k/4" gates when the lap was not completed.
Runs in the MAIN venv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "tasks" / "racing" / "crazy_track" / "results"


def load_runs() -> dict:
    """(start, plan [cond]) -> label -> list of (gates, race_time, max_dev, dir), one per SEED
    (the latest run per seed wins; a nominal row has one seed)."""
    seen: dict = {}
    for d in sorted(RESULTS.glob("*_race-eval-*")):
        meta_p, sum_p = d / "metadata.yaml", d / "summary.csv"
        if not (meta_p.exists() and sum_p.exists()):
            continue
        cfg = yaml.safe_load(meta_p.read_text()).get("config", {})
        if "start" not in cfg or "plan_name" not in cfg:
            continue
        with open(sum_p) as fh:
            r = next(csv.DictReader(fh), None)
        if r is None:
            continue
        cond = cfg.get("cond", "nominal")
        plan = cfg["plan_name"] if cond == "nominal" else f"{cfg['plan_name']} [{cond}]"
        label = cfg.get("label", r["controller"])
        seen[(cfg["start"], plan, label, int(cfg.get("seed", 0)))] = (
            int(r["gates_passed"]), r["race_time"], float(r["max_ref_dev"]), d.name)
    rows: dict = defaultdict(lambda: defaultdict(list))
    for (start, plan, label, _seed), v in seen.items():
        rows[(start, plan)][label].append(v)
    return rows


def cell(runs) -> str:
    """One seed: race_time (max dev) or k/4 (max dev). Several seeds (gust / Lighthouse
    realisations): mean±std of the completed laps (completed/seeds)."""
    if not runs:
        return "-"
    if len(runs) == 1:
        gates, rt, dev, _ = runs[0]
        return f"{float(rt):.3f} ({dev:.2f})" if gates == 4 and rt != "inf" else f"{gates}/4 ({dev:.2f})"
    import statistics

    ok = [float(rt) for gates, rt, _, _ in runs if gates == 4 and rt != "inf"]
    tag = f"{len(ok)}/{len(runs)}"
    if not ok:
        return f"0/{len(runs)} laps"
    sd = statistics.pstdev(ok) if len(ok) > 1 else 0.0
    return f"{statistics.mean(ok):.3f}±{sd:.3f} ({tag})"


def plan_key(name: str) -> tuple:
    order = {"closed-form": 0, "tube": 1, "raw": 2}
    head = name.split("_")[0].split("-x")[0]
    for k, o in order.items():
        if head.startswith(k):
            return (o, name)
    return (9, name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start", choices=["hover", "ground", "all"], default="all")
    p.add_argument("--csv", type=Path, default=None, help="also write the table as CSV")
    p.add_argument("--cond", default=None,
                   help="only rows of this condition (nominal, wind_const, lighthouse, ...; Lesson 7)")
    args = p.parse_args()

    rows = load_runs()
    if args.cond is not None:
        want = "" if args.cond == "nominal" else f"[{args.cond}]"
        rows = {k: v for k, v in rows.items()
                if (want and k[1].endswith(want)) or (not want and "[" not in k[1])}
    if not rows:
        raise SystemExit(f"no race_eval runs found under {RESULTS}")
    starts = ["hover", "ground"] if args.start == "all" else [args.start]
    out_rows = []
    for start in starts:
        keys = sorted((k for k in rows if k[0] == start), key=lambda k: plan_key(k[1]))
        if not keys:
            continue
        labels = sorted({lab for k in keys for lab in rows[k]})
        title = ("BENCHMARK clock (hover start, motion onset -> last gate)" if start == "hover"
                 else "LEADERBOARD clock (ground start, t = 0 -> last gate; still one deterministic episode)")
        print(f"\n== {title} ==")
        w = max(24, max(len(k[1]) for k in keys) + 2)
        print(f"{'plan':{w}s}" + "".join(f"{lab:>22s}" for lab in labels))
        for k in keys:
            cells = [cell(rows[k].get(lab, [])) for lab in labels]
            print(f"{k[1]:{w}s}" + "".join(f"{c:>22s}" for c in cells))
            out_rows.append({"start": start, "plan": k[1], **dict(zip(labels, cells))})
        print("cell = race_time s (max deviation from the reference, m); k/4 = gates passed when the lap failed;")
        print("       multi-seed conditions: mean±std over completed laps (completed/seeds)")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in out_rows for k in r}, key=lambda k: (k not in ("start", "plan"), k))
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(out_rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
