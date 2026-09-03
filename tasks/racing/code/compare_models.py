"""Race several policies on the SAME LSY protocol and compare them — Lesson 5 §4.

One model's lap time is a number; two models' lap times are a *claim*. This
script makes the comparison honest by construction: every model flies the same
config, the same number of episodes, scored by the race's own `simulate()` —
the exact loop `scripts/evaluate.py` uses — never a re-implementation of it.

Run in the RACE venv, with your completed race_bridge.py already copied into
repos/lsy_drone_racing/lsy_drone_racing/control/ and `control_mode = "attitude"`
set in the config you pass (both from Lesson 3 §4):

    /opt/venvs/race/bin/python tasks/racing/code/compare_models.py \
        --episodes 20 --config level0.toml \
        v5_s0=tasks/racing/crazy_track/results/<run-a>/datt_ppo_final.zip \
        v5_s1=tasks/racing/crazy_track/results/<run-b>/datt_ppo_final.zip \
        v5_s2=tasks/racing/crazy_track/results/<run-c>/datt_ppo_final.zip

Each positional argument is  label=path/to/policy.zip.

Outputs (in --out-dir, default tasks/racing/figures/):
    comparison.csv         per-episode times, every model, every episode
    comparison.png         lap times (dots = episodes, bar = mean over
                           successes) and success rate, side by side
    <label>/flown_ep*.csv  each model's flown paths (for plot_trajectory.py)

What the numbers mean (same rules as the leaderboard, Lesson 3 §3):
  * an episode counts only if ALL gates are passed; failures score None
  * a model below 50 % success is NOT ranked — a fast crash is not a lap
  * mean is over successful episodes only, so always read it NEXT TO the
    success rate, never alone
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

LSY = Path("/workspace/repos/lsy_drone_racing")

# Single-measure charts use one calm hue; identity lives in the axis labels.
BAR = "#5B7FA6"
DOT = "#2B3A4A"


def race(model: Path, episodes: int, config: str, log_dir: Path) -> list:
    """Fly `episodes` races with race_bridge.py driving `model`. Their loop."""
    os.environ["RACE_MODEL"] = str(model.resolve())
    os.environ["RACE_LOG_DIR"] = str(log_dir)
    # Stale flown-lap logs from a previous run would shift the episode indices.
    if log_dir.exists():
        for old in log_dir.glob("flown_ep*.csv"):
            old.unlink()
    sys.path.insert(0, str(LSY / "scripts"))
    from sim import simulate  # lsy's own runner — the protocol, not a copy

    return simulate(config=config, controller="race_bridge.py",
                    n_runs=episodes, render=False)


def summarize(label: str, times: list) -> dict:
    ok = [t for t in times if t is not None]
    n = len(times)
    return {
        "label": label,
        "episodes": n,
        "successes": len(ok),
        "success_rate": len(ok) / n if n else 0.0,
        "mean": float(np.mean(ok)) if ok else float("nan"),
        "std": float(np.std(ok)) if ok else float("nan"),
        "best": float(np.min(ok)) if ok else float("nan"),
        "times": times,
    }


def figure(rows: list[dict], out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))

    fig, (ax_t, ax_s) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 2]})

    # Lap times: every successful episode as a dot, the mean as a thin bar.
    for i, r in enumerate(rows):
        ok = np.array([t for t in r["times"] if t is not None], dtype=float)
        if len(ok):
            jitter = (np.random.default_rng(i).random(len(ok)) - 0.5) * 0.18
            ax_t.plot(np.full(len(ok), x[i]) + jitter, ok, "o", ms=5,
                      color=DOT, alpha=0.55, zorder=3)
            ax_t.bar(x[i], r["mean"], width=0.55, color=BAR, alpha=0.35,
                     zorder=1)
            ax_t.errorbar(x[i], r["mean"], yerr=r["std"], fmt="none",
                          ecolor=DOT, capsize=4, lw=1.2, zorder=4)
            ax_t.annotate(f"{r['mean']:.2f}±{r['std']:.2f}",
                          (x[i], r["mean"]), textcoords="offset points",
                          xytext=(0, 8), ha="center", fontsize=9)
    ax_t.set_xticks(x, labels)
    ax_t.set_ylabel("lap time  (s)   — successful episodes only")
    ax_t.set_title("Lap times (dot = one episode)")
    ax_t.grid(True, axis="y", color="0.92", lw=0.5)
    ax_t.set_axisbelow(True)

    # Success rate: the other half of the score. 50 % is the ranking cutoff.
    rates = [r["success_rate"] * 100 for r in rows]
    ax_s.bar(x, rates, width=0.55, color=BAR)
    ax_s.axhline(50, color="0.15", lw=1.2, ls=":")
    ax_s.annotate("ranking cutoff", (x[-1] + 0.35, 50), fontsize=9,
                  ha="right", va="bottom", color="0.15")
    for i, r in enumerate(rates):
        ax_s.annotate(f"{r:.0f} %", (x[i], r), textcoords="offset points",
                      xytext=(0, 4), ha="center", fontsize=9)
    ax_s.set_xticks(x, labels)
    ax_s.set_ylim(0, 105)
    ax_s.set_ylabel("success rate  (%)")
    ax_s.set_title("Success (20 gates = 1 lap)")
    ax_s.grid(True, axis="y", color="0.92", lw=0.5)
    ax_s.set_axisbelow(True)

    fig.suptitle("Read them TOGETHER: a fast mean at low success is not a result",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    # Fail BEFORE racing for ~10 minutes, not after: the figure needs this.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("matplotlib is missing in this venv — re-run "
                 "tasks/racing/setup.sh (or scripts/setup_python_envs.sh)")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("models", nargs="+",
                   help="label=path/to/policy.zip (one per model)")
    p.add_argument("--episodes", type=int, default=20,
                   help="Episodes per model (the leaderboard uses 20).")
    p.add_argument("--config", default="level0.toml",
                   help="Race config (level0.toml or level1.toml).")
    p.add_argument("--out-dir", type=Path, default=Path("tasks/racing/figures"))
    args = p.parse_args()

    pairs = []
    for spec in args.models:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).parent.name, spec
        model = Path(path)
        if not model.exists():
            sys.exit(f"model not found: {model}")
        pairs.append((label, model))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, model in pairs:
        print(f"\n=== {label}: {args.episodes} episodes on {args.config} ===")
        times = race(model, args.episodes, args.config, args.out_dir / label)
        rows.append(summarize(label, times))

    # ---- table ---------------------------------------------------------------
    print(f"\n{'model':<12} {'mean±std (s)':>16} {'best':>7} {'success':>10}  ranked?")
    for r in rows:
        ranked = "yes" if r["success_rate"] >= 0.5 else "NO — below 50 %"
        print(f"{r['label']:<12} {r['mean']:>9.3f}±{r['std']:.3f} "
              f"{r['best']:>7.3f} {r['successes']:>4d}/{r['episodes']:<3d}  {ranked}")
    print("\nleaderboard: 3.394 s all-time, 3.419 s current (at 100 % success)")

    # ---- csv -----------------------------------------------------------------
    csv_path = args.out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "episode", "time_s", "success"])
        for r in rows:
            for i, t in enumerate(r["times"]):
                w.writerow([r["label"], i, "" if t is None else f"{t:.4f}",
                            int(t is not None)])
    print(f"wrote {csv_path}")

    figure(rows, args.out_dir / "comparison.png")


if __name__ == "__main__":
    main()
