from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from crazy_track.trajectories import Trajectory


class Controller(ABC):
    """Common interface for all benchmarked controllers (DATT, MPC, L1-MPC, PID).

    Action convention follows DATT and crazyflow: collective thrust + body rates
    (CTBR), i.e. action = [thrust, wx, wy, wz]. The onboard rate loop is assumed
    to be handled by the simulator's low-level controller.
    """

    @abstractmethod
    def reset(self, trajectory: Trajectory) -> None:
        """Called once before each episode with the reference to track."""

    @abstractmethod
    def act(self, state: np.ndarray, t: float) -> np.ndarray:
        """Map full drone state + time to a CTBR action (4,).

        state layout (matches envs.tracking_env): [pos(3), vel(3), quat(4), omega(3)].
        """
