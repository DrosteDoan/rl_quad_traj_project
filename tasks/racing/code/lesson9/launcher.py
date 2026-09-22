"""Run driver tasks in parallel worker processes, with the environment that makes that work (Lesson 9).

    from launcher import launch
    launch([{"role": "dev", "track": 100023, "cond": "wind_gust", "lams": [0, 0.1], "seeds": 5,
             "scales": {"wind_gust": 8}, "out": "/workspace/.../laps.csv"}, ...], workers=8)

Measured (METHODOLOGY.md section 9): 4 concurrent workers fly at the speed of one alone ONLY with
`JAX_PLATFORMS=cpu` and single-thread limits; without them the same workers were 15x slower and hung at exit in
the WSL GPU driver. A worker peaks at about 1.2 GB, so about 8-9 fit in 15 GB. Each task is one `driver.py`
process, which is resumable on its own CSV, so a launch can simply be repeated after an interruption.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = {"JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "NUMEXPR_NUM_THREADS": "1", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
       "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}


def task_cmd(t: dict) -> list[str]:
    cmd = [sys.executable, str(HERE / "driver.py"), "--role", t["role"], "--track", str(t["track"]),
           "--cond", t["cond"], "--lams", ",".join(f"{x:.6g}" for x in t["lams"]), "--seeds", str(t["seeds"]),
           "--out", str(t["out"])]
    if t.get("scales"):
        cmd += ["--scales", ",".join(f"{k}={v:g}" for k, v in t["scales"].items())]
    for label, spec in (t.get("members") or {}).items():
        cmd += ["--member", f"{label}={spec}"]
    if t.get("only"):
        cmd += ["--only", t["only"]]
    if t.get("stretch_mult"):
        cmd += ["--stretch-mult", str(t["stretch_mult"])]
    return cmd


def launch(tasks: list[dict], workers: int = 8, log_dir: Path | None = None) -> list[tuple[dict, int]]:
    """Run every task (longest first: the tasks with the most laps), `workers` at a time. Returns (task, rc)."""
    log_dir = Path(log_dir or HERE / "results" / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **ENV}
    order = sorted(tasks, key=lambda t: -(len(t["lams"]) * (t["seeds"] if t["cond"] in ("wind_gust", "lighthouse", "combined") else 1)))

    def run(t: dict) -> tuple[dict, int, float]:
        log = log_dir / (Path(t["out"]).stem + ".log")
        t0 = time.time()
        with open(log, "a") as fh:
            rc = subprocess.run(task_cmd(t), stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        return t, rc, time.time() - t0

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run, t) for t in order]
        for i, f in enumerate(as_completed(futs), start=1):
            t, rc, dt = f.result()
            print(f"[{i}/{len(tasks)}] {Path(t['out']).stem}: rc={rc} in {dt / 60:.1f} min", flush=True)
            results.append((t, rc))
    return results
