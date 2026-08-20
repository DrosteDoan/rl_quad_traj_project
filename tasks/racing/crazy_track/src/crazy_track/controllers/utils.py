"""Shared math for position-level controllers: desired acceleration -> attitude command."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

GRAVITY = 9.81
# cf21B_500 physical limits (drones/params.toml): 4 motors x [thrust_min, thrust_max]
THRUST_MIN = 4 * 0.02136263065537499
THRUST_MAX = 4 * 0.2
MASS = 0.04338
RPY_MAX = 1.0  # rad, safety clip for commanded roll/pitch


# cf21B_500 inertia and CTBR rate-loop constants
J_DIAG = np.array([25e-6, 28e-6, 49e-6])
RATE_MAX = np.array([15.0, 15.0, 5.0])  # rad/s command limits (roll, pitch, yaw)
TORQUE_MAX = np.array([0.008, 0.008, 0.002])  # N*m, from motor-force / arm limits
KW_RATE = 25.0  # rate-loop P gain (1/s)


def ctbr_to_force_torque(thrust: np.ndarray, w_des: np.ndarray,
                         omega: np.ndarray) -> np.ndarray:
    """Body-rate P loop -> [fc, tx, ty, tz] for crazyflow force_torque control.

    torque = J*kw*(w_des - w) + w x (J w) (gyroscopic feedforward). Works on
    batched inputs: thrust (...,), w_des/omega (..., 3) -> (..., 4).
    """
    Jw = J_DIAG * omega
    torque = J_DIAG * KW_RATE * (w_des - omega) + np.cross(omega, Jw)
    torque = np.clip(torque, -TORQUE_MAX, TORQUE_MAX)
    return np.concatenate([np.asarray(thrust)[..., None], torque], axis=-1)


def acc2attitude(a_des: np.ndarray, quat_xyzw: np.ndarray, mass: float = MASS,
                 yaw_des: float = 0.0) -> np.ndarray:
    """Convert desired CoM acceleration to [roll, pitch, yaw, thrust_N] (Mellinger-style).

    thrust = f_des . z_body (current attitude); desired attitude aligns body-z
    with f_des at the desired yaw.
    """
    f_des = mass * (a_des + np.array([0.0, 0.0, GRAVITY]))
    z_body = R.from_quat(quat_xyzw).as_matrix()[:, 2]
    thrust = float(np.clip(np.dot(f_des, z_body), THRUST_MIN, THRUST_MAX))

    z_des = f_des / max(np.linalg.norm(f_des), 1e-6)
    x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    y_des = np.cross(z_des, x_c)
    y_des /= max(np.linalg.norm(y_des), 1e-6)
    x_des = np.cross(y_des, z_des)
    rpy = R.from_matrix(np.stack([x_des, y_des, z_des], axis=-1)).as_euler("xyz")
    rpy[:2] = np.clip(rpy[:2], -RPY_MAX, RPY_MAX)
    return np.array([rpy[0], rpy[1], rpy[2], thrust])
