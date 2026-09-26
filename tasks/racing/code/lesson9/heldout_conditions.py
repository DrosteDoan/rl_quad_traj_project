"""Held-out conditions: is the RL policy's standing specific to the conditions it was trained on? (Lesson 9, 2026-09-25)

    python tasks/racing/code/lesson9/heldout_conditions.py --member v4=robust:<.../datt_ppo_final.zip>      # container

Flies v4 (the RL member, first `--member`), M1 and M1+L1 on the 22 `val` tracks under `knobs.HELD_OUT` -- lift, light,
step_wind, blackout -- at lambda in {0.25, 0.5, 0.75, 1.0}, one seed (all four are deterministic), then computes the
quantities pre-registered in METHODOLOGY.md section 5c: retention (over the tracks each member completed at lambda = 0)
for each held-out condition and its nearest in-distribution ANALOGUE (read from `results/eval/val_lambda`, the lambda
preview), and the difference-in-differences
    DiD(v4 vs X) = (v4_heldout - v4_analogue) - (X_heldout - X_analogue)         (points, same lambda)
for X in {M1, M1+L1}. Rule, per (pair, lambda in {0.5, 1.0}) cell: OVERFIT SIGNATURE if DiD <= -20 against BOTH MPC members,
GENERALISES if |DiD| < 20 against both, else MIXED; the verdict counts the 8 cells (>= 5 overfit / >= 5 generalises / mixed).
Development information on `val`, not a study result. Output: results/eval/val_heldout/summary.txt.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import knobs as kb  # noqa: E402
from launcher import launch  # noqa: E402

LAMS = [0.25, 0.5, 0.75, 1.0]
ANALOGUE = {"lift": "payload", "light": "mass_mult", "step_wind": "wind_const", "blackout": "lighthouse"}
THRESHOLD = 20.0          # points; METHODOLOGY.md section 5c
VERDICT_CELLS = 5         # of 8


def read_rows(d: Path, labels: list[str]) -> list[dict]:
    rows = []
    for f in sorted(d.glob("laps_val_*.csv")):
        rows += [r for r in csv.DictReader(open(f)) if r["member"] in labels]
    return rows


def rate(rows, member, cond, lam, tracks=None):
    rs = [r for r in rows if r["member"] == member and r["cond"] == cond and abs(float(r["lam"]) - lam) < 1e-9
          and (tracks is None or r["track"] in tracks)]
    return (sum(int(r["completed"]) for r in rs), len(rs))


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--member", action="append", default=[], metavar="LABEL=SPEC", required=True,
                    help="the RL member(s); the FIRST is the one the difference-in-differences is computed for")
    ap.add_argument("--only", default="M1,M1+L1", help="MPC members flown next to it (the difference-in-differences comparators)")
    ap.add_argument("--in-dir", default=str(HERE / "results" / "eval" / "val_lambda"), help="in-distribution lambda preview")
    ap.add_argument("--tag", default="val_heldout")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    members = dict(m.split("=", 1) for m in args.member)
    rl = list(members)[0]
    mpc = [x for x in args.only.split(",") if x]
    labels = list(members) + mpc
    out_dir = HERE / "results" / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        seeds = sorted(int(p.stem.split("_")[1]) for p in (HERE / "tracks" / "val").glob("track_*.json"))
        tasks = [{"role": "val", "track": s, "cond": c, "lams": LAMS, "seeds": 1, "members": members,
                  "only": ",".join(labels), "out": out_dir / f"laps_val_{s:06d}_{c}.csv"} for s in seeds for c in kb.HELD_OUT]
        print(f"{len(tasks)} tasks ({len(seeds)} tracks x {len(kb.HELD_OUT)} held-out conditions), members {labels}", flush=True)
        launch(tasks, workers=args.workers)

    held, indist = read_rows(out_dir, labels), read_rows(Path(args.in_dir), labels)
    base = {m: {r["track"] for r in indist if r["member"] == m and r["cond"] == "wind_const" and float(r["lam"]) == 0.0
                and r["completed"] == "1"} for m in labels}
    lines: list[str] = []
    emit = lambda s="": (lines.append(s), print(s))
    emit(f"base (tracks completed at lambda = 0, of the val pool): " + ", ".join(f"{m} {len(base[m])}" for m in labels))

    for view, restrict in (("raw", False), ("retention", True)):
        emit(f"\n== held-out conditions, {view}: completed laps / laps ==")
        emit(f"{'condition':<11} {'member':<8}" + "".join(f"{'lam=' + str(x):>13}" for x in LAMS))
        for c in kb.HELD_OUT:
            for m in labels:
                cells = []
                for lam in LAMS:
                    k, n = rate(held, m, c, lam, base[m] if restrict else None)
                    cells.append(f"{k:>3}/{n:<3} {pct(k, n):3.0f}%")
                emit(f"{c:<11} {m:<8}" + "".join(f"{x:>13}" for x in cells))
            emit()

    emit("== retention (%): analogue (in-distribution, lambda preview) -> held-out, same lambda ==")
    emit(f"{'held-out':<10}{'analogue':<11}{'member':<8}" + "".join(f"{'lam=' + str(x):>15}" for x in LAMS))
    ret = {}
    for c, a in ANALOGUE.items():
        for m in labels:
            cells = []
            for lam in LAMS:
                ka, na = rate(indist, m, a, lam, base[m]); kh, nh = rate(held, m, c, lam, base[m])
                ret[(c, m, lam)] = (pct(ka, na), pct(kh, nh))
                cells.append(f"{pct(ka, na):4.0f} ->{pct(kh, nh):4.0f}")
            emit(f"{c:<10}{a:<11}{m:<8}" + "".join(f"{x:>15}" for x in cells))
        emit()

    emit(f"== difference-in-differences: how many MORE points {rl} loses than an MPC member when the condition goes from its analogue to the held-out one ==")
    emit(f"{'held-out':<10}{'lam':>5}  " + "  ".join(f"{'DiD vs ' + x:>14}" for x in mpc) + "   cell")
    tally = {"overfit": 0, "generalises": 0, "mixed": 0}
    for c in ANALOGUE:
        for lam in (0.5, 1.0):
            a_rl, h_rl = ret[(c, rl, lam)]
            dids = []
            for x in mpc:
                a_x, h_x = ret[(c, x, lam)]
                dids.append((h_rl - a_rl) - (h_x - a_x))
            if all(d <= -THRESHOLD for d in dids):
                cell = "OVERFIT SIGNATURE"; tally["overfit"] += 1
            elif all(abs(d) < THRESHOLD for d in dids):
                cell = "generalises"; tally["generalises"] += 1
            else:
                cell = "mixed"; tally["mixed"] += 1
            emit(f"{c:<10}{lam:>5}  " + "  ".join(f"{d:>+14.0f}" for d in dids) + f"   {cell}")
    verdict = ("OVERFIT SIGNATURE: the RL advantage in the study conditions is substantially specific to them" if tally["overfit"] >= VERDICT_CELLS
               else "GENERALISES: no evidence the advantage is specific to the study conditions" if tally["generalises"] >= VERDICT_CELLS
               else "MIXED: reported per pair")
    emit(f"\ntally over the 8 cells: {tally}  ->  {verdict}")
    for thr in (15.0, 20.0, 25.0):
        t2 = {"overfit": 0, "generalises": 0, "mixed": 0}
        for c in ANALOGUE:
            for lam in (0.5, 1.0):
                a_rl, h_rl = ret[(c, rl, lam)]
                dd = [(h_rl - a_rl) - (ret[(c, x, lam)][1] - ret[(c, x, lam)][0]) for x in mpc]
                t2["overfit" if all(d <= -thr for d in dd) else "generalises" if all(abs(d) < thr for d in dd) else "mixed"] += 1
        emit(f"  threshold sensitivity: at {thr:.0f} points -> {t2}")

    # ---- uncertainty: paired bootstrap over TRACKS (all members are flown on the same 22 tracks) ----
    tracks = sorted({r["track"] for r in indist if r["cond"] == "wind_const" and float(r["lam"]) == 0.0})
    T = {t: i for i, t in enumerate(tracks)}

    def per_track(rows, member, cond, lam):
        v = np.full(len(tracks), np.nan)
        acc = {}
        for r in rows:
            if r["member"] == member and r["cond"] == cond and abs(float(r["lam"]) - lam) < 1e-9 and r["track"] in T:
                acc.setdefault(r["track"], []).append(int(r["completed"]))
        for t, xs in acc.items():
            v[T[t]] = np.mean(xs)
        return v

    b = {m: np.array([1.0 if t in base[m] else 0.0 for t in tracks]) for m in labels}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(tracks), size=(4000, len(tracks)))

    def boot_ret(m, arr):
        num = (b[m][idx] * arr[idx]).sum(1); den = b[m][idx].sum(1)
        return np.where(den > 0, 100.0 * num / np.where(den > 0, den, 1), np.nan)

    emit("\n== difference-in-differences with a paired bootstrap over the 22 tracks (4,000 resamples): point [95% interval] ==")
    emit(f"{'held-out':<10}{'lam':>5}  " + "  ".join(f"{'vs ' + x:>24}" for x in mpc) + "   reading")
    for c, a in ANALOGUE.items():
        for lam in (0.5, 1.0):
            rl_d = boot_ret(rl, per_track(held, rl, c, lam)) - boot_ret(rl, per_track(indist, rl, a, lam))
            cells, flags = [], []
            for x in mpc:
                x_d = boot_ret(x, per_track(held, x, c, lam)) - boot_ret(x, per_track(indist, x, a, lam))
                d = rl_d - x_d
                pt = ((ret[(c, rl, lam)][1] - ret[(c, rl, lam)][0]) - (ret[(c, x, lam)][1] - ret[(c, x, lam)][0]))
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                cells.append(f"{pt:>+6.0f} [{lo:>+5.0f},{hi:>+5.0f}]")
                flags.append("v4 loses more" if hi < 0 else "v4 loses less" if lo > 0 else "unresolved")
            emit(f"{c:<10}{lam:>5}  " + "  ".join(f"{x:>24}" for x in cells) + "   " + " / ".join(flags))

    a25, a75 = ret[("lift", rl, 0.25)], ret[("lift", rl, 0.75)]
    p25, p75 = ret[("lift", rl, 0.25)][0], ret[("lift", rl, 0.75)][0]
    lift_drop, pay_drop = a25[1] - a75[1], a25[0] - a75[0]
    emit(f"\nsecondary (lift edge): {rl} retention fell {lift_drop:+.0f} points from lambda 0.25 to 0.75 under lift, vs {pay_drop:+.0f} under payload; "
         f"pre-registered edge signature needs lift's drop to exceed payload's by >= 25 (here {lift_drop - pay_drop:+.0f})")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
