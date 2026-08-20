"""Figure-5 benchmark runner (stub) + trajectory visualization (working).

Once controllers exist: roll out each controller on N random trajectories per
family (smooth / infeasible), record per-step position error, and produce the
paper's box plots of mean tracking error per episode.

Already usable now — visualize sampled reference trajectories:

    python -m crazy_track.eval.benchmark --preview
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from crazy_track.trajectories import ChainedPolyTrajectory, ZigzagTrajectory


def preview(seed: int = 0, n: int = 3) -> None:
    """Plot n random trajectories of each family (top-down XY view + Z profile)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, cls in enumerate([ChainedPolyTrajectory, ZigzagTrajectory]):
        for _ in range(n):
            traj = cls.random(rng)
            t = np.linspace(0, traj.duration, 1000)
            p = traj.pos(t)
            axes[row, 0].plot(p[:, 0], p[:, 1], lw=1)
            axes[row, 1].plot(t, p[:, 2], lw=1)
        axes[row, 0].set(title=f"{cls.__name__} (XY)", xlabel="x [m]", ylabel="y [m]", aspect="equal")
        axes[row, 1].set(title=f"{cls.__name__} (Z)", xlabel="t [s]", ylabel="z [m]")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="plot sample reference trajectories")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.preview:
        preview(args.seed)
    else:
        raise NotImplementedError("Full benchmark needs controllers + tracking env (see README).")


if __name__ == "__main__":
    main()
