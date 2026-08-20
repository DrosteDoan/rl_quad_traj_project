"""Gymnasium wrapper around crazyflow for DATT-style trajectory tracking.

Design (to implement on the WSL2/Linux side where crazyflow is installed):

- Observation (DATT Sec. IV): position error e(t), velocity, attitude (quat or
  rotation vector), plus a feedforward window of future reference positions
  Trajectory.ref_window(t, horizon, dt) expressed in the body/yaw frame.
- Action: collective thrust + body rates (CTBR), matching crazyflow's
  attitude-rate interface and DATT's action space.
- Reward: -||pos - ref||, optionally with action-smoothness penalty (paper
  uses negative tracking error).
- Episode: fixed length (e.g. 10 s at 50 Hz control), new random trajectory
  each reset (ChainedPolyTrajectory during training, both families at eval).
- Domain randomization hooks (mass, thrust-to-weight, latency) for the
  L1-adaptation experiments.

crazyflow is imported lazily so the rest of the package (trajectories, eval
plotting) stays importable on machines without the simulator.
"""

from __future__ import annotations


def make_tracking_env(*args, **kwargs):
    try:
        import crazyflow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "crazyflow is not installed. Run scripts/setup_env.sh inside WSL2/Ubuntu "
            "(see README) — pip install 'crazy-track[sim]'."
        ) from e
    raise NotImplementedError("Tracking env wiring is the next roadmap step.")
