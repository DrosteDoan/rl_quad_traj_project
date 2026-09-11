"""Bridge: run a crazy_track MODEL-BASED controller (MPC family, MPPI+L1, PID, ADRC)
inside the LSY race environment — Lesson 6 §5.

Same shape as race_bridge.py (Lesson 3), but the tracker is built from a controller
spec instead of a policy .zip, and the reference is COMPLETE: a ground-start takeoff
followed by either the closed-form racing line or a TOGT plan. Nothing here is a
scaffold — the exercise in Lesson 6 is the benchmark, not the bridge.

Install (their loader wants the file in that folder, one Controller subclass per file):

    cp tasks/racing/code/race_bridge_mpc.py repos/lsy_drone_racing/lsy_drone_racing/control/

Run one episode (race venv; the level toml must have control_mode = "attitude"):

    cd repos/lsy_drone_racing
    RACE_CONTROLLER=mpc_offsetfree /opt/venvs/race/bin/python scripts/sim.py \
        --config level0.toml --controller race_bridge_mpc.py --render False

Score it like the leaderboard (20 episodes) and compare the variants in one go:

    /opt/venvs/race/bin/python tasks/racing/code/compare_models.py --bridge race_bridge_mpc.py \
        --episodes 20 --config level0.toml mpc=mpc eso=mpc_offsetfree l1=mpc_l1

Knobs (environment variables):
    RACE_CONTROLLER   mpc (default) | mpc_offsetfree[_w<N>] | mpc_l1[_c<Hz>] | mppi_l1 | pid | adrc
                      | datt:<zip>  (the same specs race_eval.py takes)
    RACE_PLAN         a TOGT plan CSV (Lesson 6 §2); default = the closed-form line
    RACE_TIME_SCALE   stretch a CSV plan in time (default 1.0)
    RACE_CRUISE       cruise of the closed-form line (default 2.5 m/s)
    RACE_TAKEOFF_Z    height of the closed-form line's first point (default 1.0; 0.7 = gate 1's
                      height, the Lesson-3 baseline's takeoff, kinder to a policy)
    RACE_LINE         lsy (default: lsy_level2_race()'s line, which passes ~0.1 m from poles 3
                      and 4) | safe (two extra vias that clear those poles; race_refs.py)
    RACE_TAKEOFF_T    seconds for the rest-to-rest climb from the ground to the plan's
                      first point (default 1.5)
    RACE_START        auto (default) | ground | hover. auto = hover when the race put the drone
                      at hover height (z > 0.5, i.e. a *_hoverstart.toml made by
                      compare_models.py --start hover), else ground. hover = no takeoff leg;
                      the reference holds the plan's first point for RACE_SETTLE seconds
                      (default 1.0: lsy spins the rotors up from rest and the drone sags
                      ~0.5 m first), then the plan moves. Flight time - RACE_SETTLE is the
                      benchmark clock's motion-onset time, inside the race.
    RACE_LOG_DIR      write flown_ep<N>.csv (t,x,y,z) per episode, for plot_trajectory.py
    RACE_CODE_DIR     where race_refs.py lives (default /workspace/tasks/racing/code)
    RACE_VERBOSE      print the plan summary once per episode

The reference is the SAME object race_eval.py --start ground builds (race_refs.py:
GroundStartTrajectory), so a number from that script and a number from this bridge
differ only by what the real environment adds: its 50 Hz loop, its randomisations
(Level 1: mass, inertia, start pose, action noise), and contacts with gates and poles.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from lsy_drone_racing.control.controller import Controller

sys.path.insert(0, os.environ.get("RACE_CODE_DIR", "/workspace/tasks/racing/code"))
from race_refs import GroundStartTrajectory, closed_form_line  # noqa: E402

from crazy_track.trajectories.freestyle import RaceGate, feasibility_report  # noqa: E402


def _drop_loop_warnings(record: logging.LogRecord) -> bool:
    return not record.getMessage().startswith("Controller execution time exceeded")


def make_race_controller(spec: str, freq: int):
    """The vendored benchmark's controller specs, built at the RACE's control rate.

    lissajous_benchmark.make_controller hard-codes 100 Hz; the race steps at 50 Hz
    and the estimators (ESO / L1) integrate with the controller's dt, so the rate
    must be the real one.
    """
    if spec == "mpc":
        from crazy_track.controllers.mpc import MPCController
        return MPCController(control_freq=freq)
    m = re.fullmatch(r"mpc_offsetfree(?:_w(\d+))?", spec)
    if m:
        from crazy_track.controllers.mpc import MPCController
        return MPCController(control_freq=freq, disturbance="eso",
                             eso_w=float(m.group(1)) if m.group(1) else 7.0)
    m = re.fullmatch(r"mpc_l1(?:_c(\d+))?", spec)
    if m:
        from crazy_track.controllers.mpc import MPCController
        return MPCController(control_freq=freq, disturbance="l1",
                             l1_cutoff_hz=float(m.group(1)) if m.group(1) else 4.0)
    if spec == "mppi_l1":
        from crazy_track.controllers.mppi_l1 import MPPIL1Controller
        return MPPIL1Controller(control_freq=freq, seed=int(os.environ.get("RACE_SEED", "0")))
    if spec == "pid":
        from crazy_track.controllers.pid import PIDController
        return PIDController(control_freq=freq)
    if spec == "adrc":
        from crazy_track.controllers.adrc import ADRCController
        return ADRCController(control_freq=freq)
    if spec.startswith("datt:"):
        from crazy_track.controllers.datt import DATTPolicyController
        return DATTPolicyController(spec.split(":", 1)[1], control_freq=freq)
    raise ValueError(f"unknown RACE_CONTROLLER {spec!r} (mpc | mpc_offsetfree | mpc_l1 | mppi_l1 | "
                     "pid | adrc | datt:<zip>)")


class RaceBridgeMPCController(Controller):
    """Feed the race observation to a model-based tracker from the vendored benchmark."""

    def __init__(self, obs: dict, info: dict, config: dict):
        super().__init__(obs, info, config)
        self.freq = int(config.env.freq)          # 50 Hz at every level
        self._tick = 0
        mode = str(config.env.control_mode)
        if mode != "attitude":
            raise RuntimeError(
                f'race_bridge_mpc needs control_mode = "attitude" in the race config, got {mode!r}. '
                "Edit [env] control_mode in repos/lsy_drone_racing/config/<level>.toml (Lesson 3 §4)."
            )
        missing = [k for k in ("gates_pos", "gates_quat", "n_gates_passed", "gate_sequence")
                   if k not in obs]
        if missing:
            raise RuntimeError(
                f"race observation has no {missing}: repos/lsy_drone_racing is older than the pinned "
                "commit (scripts/pins.sh, LSY_REF). Re-run bash tasks/racing/setup.sh (Lesson 3 §4)."
            )
        self.gates = [
            RaceGate(tuple(p), yaw=float(y))
            for p, y in zip(self._gate_positions(obs), self._gate_yaws(obs))
        ]
        self.start_pos = np.asarray(obs["pos"], dtype=np.float64).reshape(3)
        self.traj = self._build_reference()

        self.spec = os.environ.get("RACE_CONTROLLER", "mpc")
        self.ctrl = make_race_controller(self.spec, self.freq)
        self.ctrl.reset(self.traj)
        self._finished = False
        self._solve_ms: list[float] = []
        # An MPC solve can take longer than the 20 ms the race loop has; lsy's sim.py logs a
        # warning on EVERY such step (thousands of lines per evaluation). Drop exactly that
        # record -- the episode stats (Flight time, Gates passed) stay -- and print the
        # measured cost once per episode instead: it is a real finding (Lesson 6 §6).
        for name in ("sim", "__main__"):          # sim.py run as a script, or imported
            logging.getLogger(name).addFilter(_drop_loop_warnings)

        self._log_dir = os.environ.get("RACE_LOG_DIR")
        self._flown: list[tuple[float, float, float, float]] = []

    # ---- reading the race observation ---------------------------------------
    @staticmethod
    def _gate_positions(obs: dict) -> np.ndarray:
        return np.asarray(obs["gates_pos"], dtype=np.float64).reshape(-1, 3)

    @staticmethod
    def _gate_yaws(obs: dict) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R

        quats = np.asarray(obs["gates_quat"], dtype=np.float64).reshape(-1, 4)
        return R.from_quat(quats).as_euler("xyz")[:, 2]

    # ---- the reference: takeoff + (closed-form line | TOGT plan) --------------
    def _build_reference(self):
        start_mode = os.environ.get("RACE_START", "auto")
        if start_mode == "auto":
            start_mode = "hover" if self.start_pos[2] > 0.5 else "ground"
        if start_mode == "hover":
            # lsy starts the rotors from rest: released at hover height the drone sags ~0.5 m
            # in the first 0.4 s and a fast plan reaches gate 1 before it recovers (measured:
            # one lap in five ends on the gate-1 frame). RACE_SETTLE holds the first point for
            # that long first; the race clock then reads RACE_SETTLE more than motion onset.
            takeoff_t = float(os.environ.get("RACE_SETTLE", "1.0"))
            start = None
        else:
            takeoff_t = float(os.environ.get("RACE_TAKEOFF_T", "1.5"))
            start = self.start_pos
        plan_csv = os.environ.get("RACE_PLAN")
        if plan_csv:
            from crazy_track.trajectories.freestyle import LSY_LEVEL2_OBSTACLES
            from crazy_track.trajectories.sampled import SampledRaceTrajectory

            plan = SampledRaceTrajectory(plan_csv, gates=self.gates, obstacles=LSY_LEVEL2_OBSTACLES,
                                         time_scale=float(os.environ.get("RACE_TIME_SCALE", "1.0")))
            plan_name = Path(plan_csv).name
        else:
            cruise = float(os.environ.get("RACE_CRUISE", "2.5"))
            line = os.environ.get("RACE_LINE", "lsy")
            # RACE_TAKEOFF_Z: height of the line's first point above the start (default 1.0,
            # the parent project's hover point; 0.7 = gate 1's height, so the approach to
            # gate 1 needs no descent -- the Lesson-3 baseline's choice)
            z0 = float(os.environ.get("RACE_TAKEOFF_Z", "1.0"))
            plan = closed_form_line(self.gates, cruise=cruise, line=line,
                                    start=(float(self.start_pos[0]), float(self.start_pos[1]), z0))
            plan_name = f"closed-form ({line}) cruise {cruise:g}, from z {z0:g}"
        traj = GroundStartTrajectory(plan, start=start, takeoff_t=takeoff_t)
        rep = feasibility_report(traj)
        if rep["max_thrust_acc_outside_arc"] > rep["thrust_acc_limit"]:
            raise RuntimeError(f"reference demands {rep['max_thrust_acc_outside_arc']:.1f} m/s^2 > "
                               f"{rep['thrust_acc_limit']:.1f}: lower RACE_CRUISE / raise RACE_TIME_SCALE")
        if os.environ.get("RACE_VERBOSE"):
            print(f"[race_bridge_mpc] {plan_name}: {start_mode} start, takeoff {takeoff_t} s -> "
                  f"{np.round(traj.takeoff_target, 2)}, "
                  f"last gate at {traj.gate_times[-1]:.2f} s, duration {traj.duration:.2f} s, "
                  f"thrust {rep['max_thrust_acc_outside_arc']:.1f}/{rep['thrust_acc_limit']:.1f}, "
                  f"gate_crossings_ok {rep['gate_crossings_ok']}")
        return traj

    # ---- the control loop ----------------------------------------------------
    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Return [roll, pitch, yaw, collective_thrust_N] (Lesson 3 §3)."""
        state = np.concatenate([
            np.asarray(obs["pos"], dtype=np.float64).reshape(3),
            np.asarray(obs["vel"], dtype=np.float64).reshape(3),
            np.asarray(obs["quat"], dtype=np.float64).reshape(4),   # xyzw
            np.asarray(obs["ang_vel"], dtype=np.float64).reshape(3),
        ])
        t = self._tick / self.freq
        if self._log_dir is not None:
            self._flown.append((t, state[0], state[1], state[2]))
        t0 = time.perf_counter()
        action = self.ctrl.act(state, t)
        self._solve_ms.append(1e3 * (time.perf_counter() - t0))
        return np.asarray(action, dtype=np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        n_passed = int(obs["n_gates_passed"])
        self._finished = n_passed >= len(np.atleast_1d(obs["gate_sequence"]))
        return self._finished

    def episode_reset(self):
        if self._solve_ms:
            ms = np.asarray(self._solve_ms[1:] or self._solve_ms)   # the first call warms the solver
            print(f"[race_bridge_mpc] {self.spec}: {len(self._solve_ms)} steps, controller "
                  f"{ms.mean():.1f} ms mean / {ms.max():.0f} ms max per 20 ms step")
        if self._log_dir is not None and self._flown:
            out = Path(self._log_dir)
            out.mkdir(parents=True, exist_ok=True)
            idx = len(list(out.glob("flown_ep*.csv")))
            np.savetxt(out / f"flown_ep{idx:02d}.csv", np.asarray(self._flown),
                       delimiter=",", header="t,x,y,z", comments="")
            self._flown = []
        self._solve_ms = []
        self._tick = 0
        self._finished = False
        self.ctrl.reset(self.traj)
