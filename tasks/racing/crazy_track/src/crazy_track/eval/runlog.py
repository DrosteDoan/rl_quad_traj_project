"""Run logging: every benchmark run gets a timestamped directory under results/.

Layout:
    results/<YYYY-MM-DD_HH-MM-SS>_<tag>/
        metadata.yaml     # date-time, reason for the run, git commit, config
        summary.csv       # one row per (controller, trajectory): RMSE metrics
        <controller>_<traj>.npz   # full time series: t, pos, ref_pos, action, vel

Reports (human analysis) live separately in reports/ as dated markdown files.
"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


class RunLogger:
    def __init__(self, tag: str, reason: str, config: dict | None = None,
                 results_root: Path | None = None):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        root = results_root or REPO_ROOT / "results"
        # concurrent runs can land in the same second: suffix instead of crash
        for suffix in ("", "-b", "-c", "-d"):
            self.dir = root / f"{stamp}{suffix}_{tag}"
            try:
                self.dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                if suffix == "-d":
                    raise
        self.summary_rows: list[dict] = []
        meta = {
            "datetime": stamp,
            "tag": tag,
            "reason": reason,
            "git_commit": _git_commit(),
            "config": config or {},
        }
        with open(self.dir / "metadata.yaml", "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    def log_rollout(self, controller: str, trajectory: str, data: dict[str, np.ndarray],
                    metrics: dict[str, float]) -> None:
        np.savez_compressed(self.dir / f"{controller}_{trajectory}.npz", **data)
        self.summary_rows.append({"controller": controller, "trajectory": trajectory, **metrics})
        self._write_summary()

    def _write_summary(self) -> None:
        if not self.summary_rows:
            return
        keys = list(self.summary_rows[0].keys())
        with open(self.dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.summary_rows)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() or "no-commit"
    except OSError:
        return "unknown"


def tracking_metrics(pos: np.ndarray, ref: np.ndarray, t: np.ndarray,
                     warmup: float = 1.0) -> dict[str, float]:
    """RMSE metrics as in Table III of arXiv:2311.13081: e (3D) and e_xy.

    The first `warmup` seconds are excluded (initial transient: the drone starts
    at rest while the reference may have nonzero initial velocity).
    """
    m = t >= warmup
    err = pos[m] - ref[m]
    return {
        "rmse_3d": float(np.sqrt(np.mean(np.sum(err**2, axis=-1)))),
        "rmse_xy": float(np.sqrt(np.mean(np.sum(err[:, :2] ** 2, axis=-1)))),
        "max_err": float(np.max(np.linalg.norm(err, axis=-1))),
        "warmup_excluded_s": warmup,
    }
