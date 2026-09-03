"""Bridge: run a crazy_track DATT policy inside the LSY race environment.

    ⚠️  SCAFFOLD — this file does not run until you complete `_build_reference`.
    Everything else is written for you; that one method is your design decision
    (Lesson 3 §4), because choosing the reference IS the interesting part of a
    tracking assignment.

Install:
    cp tasks/racing/code/race_bridge.py \
       repos/lsy_drone_racing/lsy_drone_racing/control/

Their loader requires the controller to live in that folder and requires
exactly ONE Controller subclass per file.

Then in repos/lsy_drone_racing/config/level0.toml:

    [controller]
    file = "race_bridge.py"

    [env]
    control_mode = "attitude"

Run (note the race environment's interpreter):

    cd repos/lsy_drone_racing
    /opt/venvs/race/bin/python scripts/sim.py --config level0.toml     # add --render False without a display

Two OPTIONAL environment variables (Lesson 5 uses both):

    RACE_MODEL    path to a policy .zip — overrides MODEL_PATH below, so
                  compare_models.py can race several checkpoints without
                  editing this file
    RACE_LOG_DIR  if set, every episode's flown path is written there as
                  flown_ep<N>.csv (t,x,y,z) — overlay it on the reference
                  with plot_trajectory.py --flown
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from lsy_drone_racing.control.controller import Controller

from crazy_track.controllers.datt import DATTPolicyController
from crazy_track.trajectories.freestyle import FreestyleTrajectory, RaceGate

# Your policy from Lesson 2. Absolute path — the race scripts run from their
# own directory, so a relative path will not resolve. (RACE_MODEL, if set,
# wins over this constant.)
MODEL_PATH = "/workspace/tasks/racing/crazy_track/results/<your-run>_datt-train/datt_ppo_final.zip"


class RaceBridgeController(Controller):
    """Feed the race observation to a DATT tracking policy.

    The mapping is unusually clean because the two projects agree on the action
    format: DATT emits [roll, pitch, yaw, collective_thrust_N], which is exactly
    what the race accepts in `attitude` mode. No conversion, no rescaling.
    Lesson 3 §3 shows how to verify that claim rather than trusting this
    comment — do that once, it is a habit worth having.
    """

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        self.freq = int(config.env.freq)      # 50 Hz at every level
        self._tick = 0

        # The race must run in ATTITUDE mode (Lesson 3 §4: `control_mode =
        # "attitude"` under [env] in the level toml). In "state" mode the env
        # clips our 4-number command against 13-number state bounds and dies
        # with an opaque "Incompatible shapes for broadcasting: (13,), (1, 1, 4)".
        mode = str(config.env.control_mode)
        if mode != "attitude":
            raise RuntimeError(
                f'race_bridge needs control_mode = "attitude" in the race config, got {mode!r}. '
                "Edit [env] control_mode in repos/lsy_drone_racing/config/<level>.toml (Lesson 3 §4)."
            )

        # The race hands us the nominal gate poses up front. At Levels 0 and 1
        # these are also the TRUE poses — which is exactly why those levels are
        # a pure tracking problem (Lesson 3 §1).
        self.gates = [
            RaceGate(tuple(p), yaw=float(y))
            for p, y in zip(self._gate_positions(obs), self._gate_yaws(obs))
        ]
        self.start_pos = np.asarray(obs["pos"], dtype=np.float64).reshape(3)

        self.traj = self._build_reference()
        model_path = os.environ.get("RACE_MODEL", MODEL_PATH)
        self.policy = DATTPolicyController(model_path, control_freq=self.freq)
        self.policy.reset(self.traj)
        self._finished = False

        # Optional flight recorder (Lesson 5): one CSV of (t, x, y, z) per
        # episode, so you can SEE where the flown lap peeled away from the plan.
        # (The race loop creates a FRESH controller every episode, so the file
        # index is derived from what is already on disk, not from a counter.)
        self._log_dir = os.environ.get("RACE_LOG_DIR")
        self._flown: list[tuple[float, float, float, float]] = []

    # ---- reading the race observation ---------------------------------------

    @staticmethod
    def _gate_positions(obs: dict) -> np.ndarray:
        return np.asarray(obs["gates_pos"], dtype=np.float64).reshape(-1, 3)

    @staticmethod
    def _gate_yaws(obs: dict) -> np.ndarray:
        """Gate yaw from its quaternion (gates only rotate about z)."""
        from scipy.spatial.transform import Rotation as R

        quats = np.asarray(obs["gates_quat"], dtype=np.float64).reshape(-1, 4)
        return R.from_quat(quats).as_euler("xyz")[:, 2]

    # ---- YOUR JOB ------------------------------------------------------------

    def _build_reference(self) -> FreestyleTrajectory:
        """Return the trajectory the policy will track.

        TODO (Lesson 3 §4). Your reference must:

          1. START AT `self.start_pos`, which is ON THE GROUND (z ~ 0.01 m).
             The race clock starts here, so takeoff is inside your lap time. A
             reference that begins at hover height silently asks the policy to
             teleport upward — you get a violent first second and lose the time
             anyway. Add an explicit takeoff leg.

          2. PASS THROUGH EVERY GATE in `self.gates`, in order, entering along
             each gate's normal. The ("gate", gate, speed) op does this for you.

          3. BE FEASIBLE. Check with
                 from crazy_track.trajectories.freestyle import feasibility_report
                 rep = feasibility_report(traj)
             and require rep["max_thrust_acc_outside_arc"] <= rep["thrust_acc_limit"]
             and rep["gate_crossings_ok"] before flying. (The combined
             rep["feasible"] flag also demands min_z > 0.15 m — which a RACE
             reference can never satisfy, because rule 1 forces it to start on
             the ground. Judge the takeoff leg by eye instead.) Thrust-to-weight
             is 1.88; a reference demanding more acceleration than that cannot be
             tracked by ANY controller, and training longer will not help.

        Start from `lsy_level2_race()` in
        tasks/racing/crazy_track/src/crazy_track/trajectories/freestyle.py — it already
        routes these four gates (Level 0 and Level 2 share the nominal poses) —
        then prepend the takeoff and tune `cruise`.

        WHERE THE ROUTE LIVES: it is the `ops` list of that function
        (freestyle.py lines 420-430): ("gate", g, speed) entries, two
        ("via", point, velocity) swing-out points — (1.75, 0.5, 0.95) heading north
        after gate 1, (-1.7, -1.1, 0.95) heading east after gate 3 — and a final
        ("hover", point, T). The op grammar is the FreestyleTrajectory docstring
        (line 171); connect() (line 209) turns each op into a quintic segment.
        Copy that list into THIS method (with `self.gates` instead of the pasted
        poses) and edit the vias, gate speeds and cruise HERE — the vendored file
        stays as the lessons cite it. Draw the result before you fly it:

            /opt/venvs/race/bin/python tasks/racing/code/plot_trajectory.py \
                --bridge repos/lsy_drone_racing/lsy_drone_racing/control/race_bridge.py \
                --config level0.toml

        Returns:
            FreestyleTrajectory beginning at self.start_pos.
        """
        raise NotImplementedError(
            "Build your reference trajectory here — Lesson 3 §4. Hint: start "
            "from lsy_level2_race(cruise=...) and prepend a takeoff leg from "
            "self.start_pos."
        )

    # ---- the control loop ----------------------------------------------------

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Return [roll, pitch, yaw, collective_thrust_N]."""
        state = np.concatenate([
            np.asarray(obs["pos"], dtype=np.float64).reshape(3),
            np.asarray(obs["vel"], dtype=np.float64).reshape(3),
            np.asarray(obs["quat"], dtype=np.float64).reshape(4),   # xyzw
            np.asarray(obs["ang_vel"], dtype=np.float64).reshape(3),
        ])
        t = self._tick / self.freq
        if self._log_dir is not None:
            p = np.asarray(obs["pos"], dtype=np.float64).reshape(3)
            self._flown.append((t, p[0], p[1], p[2]))
        return np.asarray(self.policy.act(state, t), dtype=np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        # Stop once the last gate is behind us, so we do not keep flying — and
        # keep the clock running — after the lap is done.
        n_passed = int(obs["n_gates_passed"])
        self._finished = n_passed >= len(np.atleast_1d(obs["gate_sequence"]))
        return self._finished

    def episode_reset(self):
        if self._log_dir is not None and self._flown:
            out = Path(self._log_dir)
            out.mkdir(parents=True, exist_ok=True)
            idx = len(list(out.glob("flown_ep*.csv")))
            np.savetxt(out / f"flown_ep{idx:02d}.csv", np.asarray(self._flown),
                       delimiter=",", header="t,x,y,z", comments="")
            self._flown = []
        self._tick = 0
        self._finished = False
        self.policy.reset(self.traj)
