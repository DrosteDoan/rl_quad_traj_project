"""Closed-loop rollout harness: controller vs crazyflow Sim.

Benchmark ground truth: cf21B_500 (Crazyflie 2.1 brushless), first-principles
dynamics with the firmware Mellinger attitude loop at 500 Hz. Our controllers
command [roll, pitch, yaw, collective_thrust_N] at `control_freq` (default 100 Hz).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from crazy_track.controllers.base import Controller
from crazy_track.trajectories import Trajectory

DRONE = "cf21B_500"


def make_sim(dynamics: str = "first_principles", freq: int = 500,
             control: str = "attitude"):
    from crazyflow import Sim
    from crazyflow.dynamics import Dynamics

    return Sim(
        n_worlds=1, n_drones=1, drone=DRONE, dynamics=Dynamics(dynamics),
        control=control, freq=freq, device="cpu",
    )


def set_drone_state(sim, pos: np.ndarray, vel: np.ndarray | None = None) -> None:
    """Teleport the drone to a state (used to start rollouts at the trajectory start)."""
    states = sim.data.states
    pos = jnp.asarray(pos, dtype=jnp.float32).reshape(1, 1, 3)
    vel = jnp.zeros((1, 1, 3)) if vel is None else jnp.asarray(vel, dtype=jnp.float32).reshape(1, 1, 3)
    sim.data = sim.data.replace(states=states.replace(pos=pos, vel=vel))


def get_state(sim) -> np.ndarray:
    """State vector [pos(3), vel(3), quat_xyzw(4), ang_vel(3)] for controllers."""
    s = sim.data.states
    return np.concatenate([
        np.asarray(s.pos[0, 0]), np.asarray(s.vel[0, 0]),
        np.asarray(s.quat[0, 0]), np.asarray(s.ang_vel[0, 0]),
    ]).astype(np.float64)


def apply_force(sim, f: np.ndarray) -> None:
    """Set the world-frame external force on the CoM (crazyflow states.force)."""
    force = jnp.asarray(f, dtype=jnp.float32).reshape(1, 1, 3)
    sim.data = sim.data.replace(states=sim.data.states.replace(force=force))


def rollout(controller: Controller, traj: Trajectory, control_freq: int = 100,
            sim=None, start_at_rest: bool = True, disturbance=None,
            sensor=None) -> dict[str, np.ndarray]:
    """Run one closed-loop episode. Returns time series arrays.

    `sensor` (e.g. LighthouseSensor) filters what the controller sees; metrics
    always use the true state.
    """
    sim = sim or make_sim()
    sim.reset()
    if start_at_rest:
        set_drone_state(sim, traj.pos(0.0))
    controller.reset(traj)
    if disturbance is not None:
        disturbance.reset()
    if sensor is not None:
        sensor.reset()

    n_substeps = sim.freq // control_freq
    n_steps = int(traj.duration * control_freq)
    dt = 1.0 / control_freq

    log = {k: [] for k in ("t", "pos", "vel", "quat", "ref_pos", "ref_vel", "action",
                           "meas_pos")}
    for i in range(n_steps):
        t = i * dt
        state = get_state(sim)
        state_meas = sensor.measure(t, state) if sensor is not None else state
        action = np.asarray(controller.act(state_meas, t), dtype=np.float32)
        log["t"].append(t)
        log["pos"].append(state[:3])
        # what the controller actually saw (== pos without a sensor model): lets a
        # plot show the 34 Hz ZOH staircase and bias offset the estimator fights
        log["meas_pos"].append(np.asarray(state_meas[:3]))
        log["vel"].append(state[3:6])
        log["quat"].append(state[6:10])
        log["ref_pos"].append(traj.pos(t))
        log["ref_vel"].append(traj.vel(t))
        log["action"].append(action)
        if disturbance is not None:
            apply_force(sim, disturbance.force(t, state))
        if sim.control == "force_torque":  # CTBR/acro controllers emit [fc, tx, ty, tz]
            sim.force_torque_control(action.reshape(1, 1, 4))
        elif sim.control == "rotor_vel":  # motor-level controllers emit RPMs
            sim.rotor_vel_control(action.reshape(1, 1, 4))
        else:
            sim.attitude_control(action.reshape(1, 1, 4))
        sim.step(n_substeps)
    return {k: np.asarray(v) for k, v in log.items()}
