"""Generate the RL training and validation track pools for Lesson 9 (2026-09-25), in parallel seed windows.

    python tasks/racing/code/lesson9/gen_pool.py            # in the container, main venv

WHY. The gate-aware screen (train_gate_aware.py v3) learned its 3 dev tracks (train full-lap rate 0.535, 2/3 in
the harness) but scored 1/6 on unseen tracks, level with the open-space baseline: a transfer gap with a 3-track
pool. This builds a larger pool from the SAME generator and filters as the study tracks (so the training
distribution matches the study distribution by construction, and different instances are what is held out),
and a separate validation pool so RL design decisions no longer have to be judged on study tracks (limitation 15).

SEED WINDOWS, fixed in advance and disjoint from every existing role (study 0..1318, dev 100000+):

    train   200000 .. 205999   15 windows of 400 candidates  -> RL training pool (with the 3 dev tracks)
    val     300000 .. 301999    5 windows of 400 candidates  -> validation: never trained on, never a study track

Every accepted track in a window is kept -- no truncation to a target count, no choosing -- so the pool is
whatever the fixed seed windows yield (~1.1 % of candidates pass, per the study/dev batches). Every candidate and
its reject reasons go to tracks/manifest_<role>.csv (merged from the per-window part files). Each window is one
`gen_tracks.py` process, run with the launcher's environment (JAX on CPU, single-thread limits: the combination
measured in METHODOLOGY.md section 9 to let concurrent workers run at the speed of one).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from launcher import ENV  # noqa: E402

WINDOW = 400
ROLES = {"train": (200000, 15), "val": (300000, 5)}      # role -> (first seed, number of windows)
WORKERS = 8


def run_window(role: str, part: int, seed_start: int) -> tuple[str, int, int]:
    cmd = [sys.executable, str(HERE / "gen_tracks.py"), "--role", role, "--n", "100000", "--seed-start",
           str(seed_start), "--max-tries", str(WINDOW), "--manifest-part", str(part), "--no-plot"]
    log = HERE / "results" / "logs" / f"gen_{role}_part{part:02d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, env={**os.environ, **ENV}, stdout=fh, stderr=subprocess.STDOUT).returncode
    return role, part, rc


def merge(role: str) -> list[dict]:
    parts = sorted((HERE / "tracks").glob(f"manifest_{role}_part*.csv"))
    rows = []
    for p in parts:
        rows += list(csv.DictReader(open(p)))
    rows.sort(key=lambda r: int(r["seed"]))
    if rows:
        with open(HERE / "tracks" / f"manifest_{role}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for p in parts:
        p.unlink()
    return rows


def main() -> None:
    jobs = [(role, k, first + k * WINDOW) for role, (first, n) in ROLES.items() for k in range(n)]
    print(f"{len(jobs)} windows x {WINDOW} candidates, {WORKERS} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for role, part, rc in ex.map(lambda j: run_window(*j), jobs):
            print(f"  {role} window {part:02d}: rc={rc}", flush=True)
    import gen_tracks as gt

    for role in ROLES:
        rows = merge(role)
        passed = sorted(int(r["seed"]) for r in rows if r["status"] == "pass")
        print(f"{role}: {len(passed)} passed of {len(rows)} candidates ({100 * len(passed) / max(1, len(rows)):.2f} %)")
        print(f"  seeds: {passed}")
        tracks = [json.loads((HERE / "tracks" / role / f"track_{s:04d}.json").read_text()) for s in passed]
        if tracks:
            gt.plot_overview(role, tracks, gt.FIGS / f"tracks_{role}.png")


if __name__ == "__main__":
    main()
