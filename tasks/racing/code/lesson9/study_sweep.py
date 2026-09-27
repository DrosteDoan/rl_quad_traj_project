"""The study sweep (Lesson 9) -- the one pending experiment. Protocol frozen in METHODOLOGY.md's
"SWEEP PROTOCOL" block (S1-S7) at the very top of that file; this script implements it exactly, and if
the two ever disagree, METHODOLOGY.md is the source of truth (fix whichever is wrong and say why).

NOT yet run. Four phases, in order, each resumable on its own CSVs (driver.py's own rule: a
(lam, seed, member) row already on disk is skipped) so a later phase -- or a rerun after an interruption
-- costs nothing already flown. Every (track, condition) cell is independent by design -- its own output
file, its own flights -- so this is meant to be driven ONE TRACK AND ONE CONDITION AT A TIME (`--track`,
`--condition`), not as one all-in-one invocation across all 10 x 6:

    # every phase needs the same --member / --sensitivity-member flags, repeated identically, because
    # later phases find earlier phases' data by these labels
    M='--member fa0=robust_full:/workspace/.../fa-s0/datt_ppo_final.zip
       --member fa1=robust_full:/workspace/.../fa-s1/datt_ppo_final.zip
       --member fa2=robust_full:/workspace/.../fa-s2/datt_ppo_final.zip
       --sensitivity-member s0=robust:/workspace/.../v4/datt_ppo_final.zip
       --sensitivity-member s1=robust:/workspace/.../s1/datt_ppo_final.zip
       --sensitivity-member s2=robust:/workspace/.../s2/datt_ppo_final.zip'

    python tasks/racing/code/lesson9/study_sweep.py --phase determinism $M
    python tasks/racing/code/lesson9/study_sweep.py --phase coarse     $M --track 747 --condition wind_const --workers 8
    python tasks/racing/code/lesson9/study_sweep.py --phase midpoints  $M --track 747 --condition wind_const --workers 8
    python tasks/racing/code/lesson9/study_sweep.py --phase charts     $M --track 747 --condition wind_const

Omit `--track`/`--condition` to sweep every track/condition in one call (still four separate phase
invocations); `--phase all` runs all four phases for whatever `--track`/`--condition` scope is given, in
one process. `determinism` ignores `--track` (a controller's own determinism is checked once, on one
representative track, regardless of scope) but does respect `--condition` if it narrows to specific
deterministic conditions (a no-op, not an error, if `--condition` names a stochastic one).

S1 (tracks): the 10 untouched study tracks -- 15 study tracks minus the 5-track "selection set"
(4, 25, 93, 387, 504) RL's own v1-v4 development screened on. `--track` restricts to one of them.

S2 (members): the primary comparison is the 5-member MPC family (driver.py's DEFAULT_MEMBERS) plus the
3 `--member` labels (full authority, the reported RL line). The 3 `--sensitivity-member` labels (the
0.7-authority recipe) get their own, separate chart with the same 5 MPC members, never mixed with the
primary 8 on one chart.

S3/S4 (grid and trials): a coarse grid at lambda = 0.0..1.0 in steps of 0.1, flown under EVERY condition
(no cross-condition sharing of lambda = 0 -- each (track, condition) cell is self-contained, and even at
lambda = 0 an escalated, sampling-based member like `mppi_l1` can show seed-to-seed variance from its own
internal randomness, not just from an external disturbance that happens to be off). Stochastic conditions
(wind_gust, lighthouse, combined) get 25 seeds per cell. Deterministic conditions (wind_const, payload,
mass_mult) get 1 seed, except members the `determinism` phase found non-deterministic (expected:
`mppi_l1`), which get 3. `midpoints` then adds a 0.05 lambda point wherever any member's coarse-grid
completion rate changed by >= 0.2 across an interval, at the same per-condition seed rules.

S5 (completion): all 4 gates in order, no contact ever -- driver.py's own `completed` column. This
script's chart phase uses only that; it does not (yet) render the continuous-statistic charts (RMSE,
lap time, max deviation) S6 also specifies -- the renderer is written so adding those is a small
follow-up (mean + raw min/max band), not a rewrite.

S6/S6a (charts): one folder per track, six mini-folders per track (one per condition), each holding its
raw CSV and its rendered chart(s), matplotlib with MPLBACKEND=Agg. Axes: x = completion rate (%), y =
lambda. Every member in a chart's roster is drawn, including a flat 0% line for one that never
completes -- this falls out of the mean computation automatically, nothing is filtered.

S7 (crossover, partially frozen): each chart also carries two bold derived lines -- MPC's best member
(the max over the 5) against the RL group's best member (the max over its 3 seeds; PRIMARY), and against
the RL group's median seed (SENSITIVITY, dashed). Stated asymmetry: MPC's best-of-5 is the best of five
distinct designs; RL's best-of-3 is the best of three seeds of one recipe -- an easier bar, favouring RL
wherever the primary line shows a crossover. The single scalar crossover-location statistic (a margin,
a tie-break rule) remains deferred, per METHODOLOGY.md S7 -- these are lines to look at, not a verdict.
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
import driver as drv  # noqa: E402
import knobs as kb  # noqa: E402
from launcher import launch  # noqa: E402

OUT_ROOT = HERE / "results" / "eval" / "study_sweep"
SELECTION_SET = {4, 25, 93, 387, 504}   # study tracks RL's own v1-v4 development screened on (limitation 15)
STOCHASTIC = ("wind_gust", "lighthouse", "combined")
DETERMINISTIC = ("wind_const", "payload", "mass_mult")
COARSE_LAMS = [round(i / 10, 1) for i in range(11)]     # 0.0, 0.1, ..., 1.0
STOCHASTIC_SEEDS = 25
DET_CHECK_TRACK_LAM = 0.5                # the one cell the determinism phase repeats
DET_CHECK_SEEDS = 2
MPC_LABELS = tuple(drv.DEFAULT_MEMBERS)   # M1, M1+ESO, M1+L1, M1+mass, mppi_l1


def untouched_study_tracks() -> list[int]:
    all_seeds = sorted(int(p.stem.split("_")[1]) for p in (HERE / "tracks" / "study").glob("track_*.json"))
    return [s for s in all_seeds if s not in SELECTION_SET]


def parse_members(pairs: list[str]) -> dict[str, str]:
    return dict(p.split("=", 1) for p in pairs)


# ---- phase: determinism -------------------------------------------------------------------------------------------
def determinism_path() -> Path:
    return OUT_ROOT / "determinism.json"


def run_determinism(tracks: list[int], rl_members: dict[str, str], workers: int, conditions: list[str]) -> None:
    det_conds = [c for c in conditions if c in DETERMINISTIC]
    if not det_conds:
        print("no deterministic condition in --condition's scope -- nothing to check (escalation is a "
              "property of the controller, not the stochastic conditions, so this is a no-op, not an error)")
        return
    labels = MPC_LABELS + tuple(rl_members)
    # one file PER CONDITION -- concurrent tasks must never share an output file (driver.py writes its own
    # header on first open; two processes racing to create the same file corrupt it, as val_lambda_cap07's
    # laps_val_300162_wind_const.csv did earlier in this study)
    outs = {c: OUT_ROOT / f"determinism_check_{c}.csv" for c in det_conds}
    tasks = [{"role": "study", "track": tracks[0], "cond": c, "lams": [DET_CHECK_TRACK_LAM],
              "seeds": DET_CHECK_SEEDS, "members": rl_members, "only": ",".join(labels), "out": outs[c]}
             for c in det_conds]
    launch(tasks, workers=workers)

    rows = defaultdict(list)
    for r in csv.DictReader(open(outs[det_conds[0]])):    # one condition suffices (module docstring, S4)
        rows[r["member"]].append(r)
    found = []
    for label in labels:
        rs = rows.get(label, [])
        if len(rs) < 2:
            continue
        a, b = rs[0], rs[1]
        differs = a["completed"] != b["completed"] or abs(float(a["rmse_3d"] or 0) - float(b["rmse_3d"] or 0)) > 1e-9
        if differs:
            found.append(label)
    p = determinism_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    prior = json.load(open(p)) if p.exists() else {"escalated": [], "checked_conditions": []}
    # ACCUMULATE: a --condition-restricted run must not erase an earlier run's findings
    escalated = sorted(set(prior["escalated"]) | set(found))
    checked = sorted(set(prior["checked_conditions"]) | set(det_conds))
    json.dump({"escalated": escalated, "checked_conditions": checked,
               "checked_track": tracks[0], "checked_lam": DET_CHECK_TRACK_LAM}, open(p, "w"), indent=1)
    print(f"escalated to 3 seeds on deterministic conditions: {escalated or '(none)'} -- wrote {p}")


def load_escalated() -> list[str]:
    p = determinism_path()
    if not p.exists():
        raise SystemExit(f"{p} does not exist -- run --phase determinism first (S4 needs its answer "
                          f"before any deterministic-condition cell can be flown)")
    return json.load(open(p))["escalated"]


# ---- shared task-building for coarse and midpoints ------------------------------------------------------------------
def cell_out(track: int, cond: str) -> Path:
    d = OUT_ROOT / f"track_{track}" / cond
    d.mkdir(parents=True, exist_ok=True)
    return d / "laps.csv"


def cell_tasks(track: int, cond: str, lams: list[float], rl_members: dict[str, str],
               escalated: list[str]) -> tuple[list[dict], list[dict]]:
    """Returns (batch1, batch2) -- two lists to `launch()` ONE AFTER THE OTHER, never together. A
    deterministic cell's "regular" (1 seed) and "escalated" (3 seed) sub-tasks target the SAME output
    file (cell_out), so if both ran in the same `launch()` call they could race to create/append it
    concurrently and corrupt it -- the exact bug found and fixed earlier in this study
    (val_lambda_cap07's laps_val_300162_wind_const.csv). Different (track, cond) cells target different
    files, so batching by "which sub-task number" rather than by cell keeps full cross-cell parallelism
    within each batch."""
    if not lams:
        return [], []
    out = cell_out(track, cond)
    all_labels = MPC_LABELS + tuple(rl_members)
    if cond in STOCHASTIC:
        return [{"role": "study", "track": track, "cond": cond, "lams": lams, "seeds": STOCHASTIC_SEEDS,
                  "members": rl_members, "only": ",".join(all_labels), "out": out}], []
    esc = [x for x in all_labels if x in escalated]
    reg = [x for x in all_labels if x not in escalated]
    batch1, batch2 = [], []
    if reg:
        batch1.append({"role": "study", "track": track, "cond": cond, "lams": lams, "seeds": 1,
                        "members": rl_members, "only": ",".join(reg), "out": out})
    if esc:
        batch2.append({"role": "study", "track": track, "cond": cond, "lams": lams, "seeds": 3,
                        "members": rl_members, "only": ",".join(esc), "out": out})
    return batch1, batch2


# ---- phase: coarse -------------------------------------------------------------------------------------------------
def run_coarse(tracks: list[int], rl_members: dict[str, str], workers: int, conditions: list[str]) -> None:
    escalated = load_escalated() if any(c in DETERMINISTIC for c in conditions) else []
    batch1: list[dict] = []
    batch2: list[dict] = []
    for track in tracks:
        for cond in conditions:
            # lambda = 0 is flown under EVERY condition, not shared across them (each (track, cond) folder must
            # stand alone for --track/--condition to invoke independently); it also is NOT assumed noise-free for
            # an escalated member, since e.g. mppi_l1 samples internally regardless of whether an external
            # disturbance is active, so its own seed-to-seed variance can appear even at lambda = 0
            b1, b2 = cell_tasks(track, cond, COARSE_LAMS, rl_members, escalated)
            batch1 += b1
            batch2 += b2
    print(f"{len(batch1) + len(batch2)} coarse tasks across {len(tracks)} tracks x {len(conditions)} "
          f"conditions ({len(batch1)} regular, {len(batch2)} escalated -- launched in two sequential batches)")
    launch(batch1, workers=workers)
    launch(batch2, workers=workers)


# ---- phase: midpoints ----------------------------------------------------------------------------------------------
def member_completion_by_lam(csv_path: Path, label: str) -> dict[float, float]:
    vals = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        if r["member"] == label:
            vals[round(float(r["lam"]), 6)].append(int(r["completed"]))
    return {lam: float(np.mean(v)) for lam, v in vals.items()}


def needed_midpoints(csv_path: Path, all_labels: tuple[str, ...]) -> list[float]:
    lams = sorted(COARSE_LAMS)
    needed = set()
    for label in all_labels:
        by_lam = member_completion_by_lam(csv_path, label)
        for a, b in zip(lams, lams[1:]):
            if a in by_lam and b in by_lam and abs(by_lam[b] - by_lam[a]) >= 0.2:
                needed.add(round((a + b) / 2, 3))
    return sorted(needed)


def run_midpoints(tracks: list[int], rl_members: dict[str, str], workers: int, conditions: list[str]) -> None:
    escalated = load_escalated() if any(c in DETERMINISTIC for c in conditions) else []
    all_labels = MPC_LABELS + tuple(rl_members)
    batch1: list[dict] = []
    batch2: list[dict] = []
    report = []
    for track in tracks:
        for cond in conditions:
            out = cell_out(track, cond)
            if not out.exists():
                continue   # coarse hasn't been flown for this cell yet; nothing to escalate from
            mids = needed_midpoints(out, all_labels)
            if mids:
                report.append((track, cond, mids))
                b1, b2 = cell_tasks(track, cond, mids, rl_members, escalated)
                batch1 += b1
                batch2 += b2
    for track, cond, mids in report:
        print(f"track {track} {cond}: midpoints {mids}")
    print(f"{len(batch1) + len(batch2)} midpoint tasks ({sum(len(m) for _, _, m in report)} lambda points "
          f"added, {len(batch1)} regular, {len(batch2)} escalated)")
    launch(batch1, workers=workers)
    launch(batch2, workers=workers)


# ---- phase: charts -------------------------------------------------------------------------------------------------
# Three statistics (S5/S6). COMPLETION is the primary outcome: every attempt counts (no censoring), plotted as a
# percentage, mean line only, higher is better. RMSE and LAP TIME are the two continuous statistics S6 also
# specifies: "censored on failed laps" (S5, section 7's "lap time (completers)") -- only completed laps count --
# plotted in their native units, mean line WITH a raw min/max band (S6), lower is better (so a family's "best
# member" is the MINIMUM, the opposite direction from completion's maximum).
class Stat:
    def __init__(self, key: str, column: str, unit: str, scale: float, completed_only: bool, higher_is_better: bool):
        self.key, self.column, self.unit, self.scale = key, column, unit, scale
        self.completed_only, self.higher_is_better = completed_only, higher_is_better


STATS = {
    "completion": Stat("completion", "completed", "completion rate (%)", 100.0, completed_only=False,
                        higher_is_better=True),
    "rmse": Stat("rmse", "rmse_3d", "RMSE (m)", 1.0, completed_only=True, higher_is_better=False),
    "lap_time": Stat("lap_time", "race_time", "lap time (s)", 1.0, completed_only=True, higher_is_better=False),
}


def _raw_by_lam(csv_path: Path, label: str, stat: Stat) -> dict[float, list[float]]:
    vals = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        if r["member"] != label:
            continue
        if stat.completed_only and int(r["completed"]) != 1:
            continue
        v = r[stat.column]
        if v == "" or v is None:
            continue
        vals[round(float(r["lam"]), 6)].append(float(v))
    return dict(vals)


def _mean_series(csv_path: Path, label: str, stat: Stat) -> tuple[list[float], list[float]]:
    by_lam = _raw_by_lam(csv_path, label, stat)
    lams = sorted(by_lam)
    return lams, [stat.scale * float(np.mean(by_lam[l])) for l in lams]


def _band_series(csv_path: Path, label: str, stat: Stat) -> tuple[list[float], list[float], list[float]]:
    """(lams, mins, maxs) of the RAW per-lap values -- only meaningful for a continuous statistic (S6);
    COMPLETION never calls this (S6a: binary, no band -- a literal min/max would be 0%/100% almost always)."""
    by_lam = _raw_by_lam(csv_path, label, stat)
    lams = sorted(by_lam)
    return lams, [stat.scale * min(by_lam[l]) for l in lams], [stat.scale * max(by_lam[l]) for l in lams]


def _family_series(csv_path: Path, labels: tuple[str, ...], how: str, stat: Stat) -> tuple[list[float], list[float]]:
    per_member = {lab: _raw_by_lam(csv_path, lab, stat) for lab in labels}
    lams = sorted(set().union(*[set(d) for d in per_member.values()])) if per_member else []
    best = max if stat.higher_is_better else min
    fn = {"best": best, "median": lambda xs: float(np.median(xs))}[how]
    out = []
    for l in lams:
        # "best"/"median" of family = that fn over each member's OWN mean at this lambda (S7: best/median of
        # seeds, not of individual laps -- a member's mean is what represents it in the family comparison)
        vals = [float(np.mean(per_member[lab][l])) for lab in labels if l in per_member[lab]]
        out.append(stat.scale * fn(vals) if vals else float("nan"))
    return lams, out


def _render(ax_title: str, out_png: Path, csv_path: Path, stat: Stat,
            mpc_labels: tuple[str, ...], rl_labels: tuple[str, ...]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    warm = plt.cm.autumn(np.linspace(0.15, 0.75, len(mpc_labels)))
    cool = plt.cm.winter(np.linspace(0.15, 0.75, len(rl_labels)))
    all_x: list[float] = []
    for lab, colour in zip(mpc_labels + rl_labels, np.concatenate([warm, cool]) if len(mpc_labels + rl_labels) else []):
        x, y = _mean_series(csv_path, lab, stat)
        if not x:
            continue
        ax.plot(y, x, color=colour, lw=1, alpha=0.8, label=lab)
        all_x += y
        if stat.key != "completion":   # S6: continuous stats get a band; completion never does (S6a)
            _, lo, hi = _band_series(csv_path, lab, stat)
            ax.fill_betweenx(x, lo, hi, color=colour, alpha=0.12, lw=0)
            all_x += lo + hi
    best_label = "best" if stat.higher_is_better else "best (lowest)"
    x, y = _family_series(csv_path, mpc_labels, "best", stat)
    if x:
        ax.plot(y, x, color="firebrick", lw=2.5, label=f"MPC {best_label} (primary)"); all_x += y
    x, y = _family_series(csv_path, rl_labels, "best", stat)
    if x:
        ax.plot(y, x, color="navy", lw=2.5, label=f"RL {best_label} (primary)"); all_x += y
    x, y = _family_series(csv_path, rl_labels, "median", stat)
    if x:
        ax.plot(y, x, color="navy", lw=2.5, ls="--", label="RL median (sensitivity)"); all_x += y
    ax.set_xlabel(stat.unit)
    ax.set_ylabel("lambda")
    if stat.key == "completion":
        ax.set_xlim(-2, 102)
    elif all_x:
        pad = 0.05 * (max(all_x) - min(all_x) + 1e-9)
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(ax_title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def run_charts(tracks: list[int], rl_members: dict[str, str], sens_members: dict[str, str],
               conditions: list[str]) -> None:
    rl_labels, sens_labels = tuple(rl_members), tuple(sens_members)
    for track in tracks:
        for cond in conditions:
            csv_path = cell_out(track, cond)
            if not csv_path.exists():
                continue
            d = csv_path.parent
            for key, stat in STATS.items():
                _render(f"track {track} -- {cond} -- primary -- {stat.unit}", d / f"chart_primary_{key}.png",
                        csv_path, stat, MPC_LABELS, rl_labels)
                _render(f"track {track} -- {cond} -- sensitivity (0.7 authority) -- {stat.unit}",
                        d / f"chart_sensitivity_{key}.png", csv_path, stat, MPC_LABELS, sens_labels)
    print(f"charts written under {OUT_ROOT} (3 statistics x 2 (primary/sensitivity) per track/condition)")


# ---- CLI -------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=["determinism", "coarse", "midpoints", "charts", "all"])
    ap.add_argument("--member", action="append", default=[], metavar="LABEL=SPEC", required=True,
                    help="repeat 3x: the full-authority RL seeds (the primary line)")
    ap.add_argument("--sensitivity-member", action="append", default=[], metavar="LABEL=SPEC", required=True,
                    help="repeat 3x: the 0.7-authority RL seeds (the sensitivity line)")
    ap.add_argument("--track", type=int, default=None,
                    help="restrict to ONE track seed, to run track by track (default: all 10 untouched tracks)")
    ap.add_argument("--condition", choices=kb.CONDITIONS, default=None,
                    help="restrict to ONE condition, to run condition by condition (default: all six)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tracks = [args.track] if args.track is not None else untouched_study_tracks()
    conditions = [args.condition] if args.condition else list(kb.CONDITIONS)
    rl_members = parse_members(args.member)
    sens_members = parse_members(args.sensitivity_member)
    if len(rl_members) != 3 or len(sens_members) != 3:
        raise SystemExit("expected exactly 3 --member (full authority) and 3 --sensitivity-member (0.7 authority)")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    phases = ["determinism", "coarse", "midpoints", "charts"] if args.phase == "all" else [args.phase]
    for phase in phases:
        print(f"=== phase: {phase} ===", flush=True)
        if phase == "determinism":
            run_determinism(tracks, {**rl_members, **sens_members}, args.workers, conditions)
        elif phase == "coarse":
            run_coarse(tracks, {**rl_members, **sens_members}, args.workers, conditions)
        elif phase == "midpoints":
            run_midpoints(tracks, {**rl_members, **sens_members}, args.workers, conditions)
        elif phase == "charts":
            run_charts(tracks, rl_members, sens_members, conditions)


if __name__ == "__main__":
    main()
