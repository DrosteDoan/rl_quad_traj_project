"""External disturbance models, applied as world-frame forces on the CoM.

Magnitudes are scaled to the cf21B_500 (m=43.4 g, weight 0.426 N, TWR 1.88).
DATT (arXiv:2310.09053) perturbs with accelerations up to 3.5 m/s^2, i.e.
~0.15 N on this platform — our defaults sit in that range.
"""

from __future__ import annotations

import numpy as np

MASS = 0.04338
G = 9.81
HOVER_THRUST = MASS * G
PROP_RADIUS = 0.0275  # m, cf21B


class Disturbance:
    name = "none"

    def reset(self, rng: np.random.Generator | None = None) -> None:
        pass

    def force(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.zeros(3)


class ConstantWind(Disturbance):
    """Steady wind drag force, default ~2.5 m/s^2 equivalent (60% of DATT's max)."""

    name = "wind_const"

    def __init__(self, force_vec=(0.11, 0.0, 0.0)):
        self.f = np.asarray(force_vec, dtype=np.float64)

    def force(self, t, state):
        return self.f


class GustWind(Disturbance):
    """Unsteady wind: mean + sinusoidal gust + Ornstein-Uhlenbeck turbulence.

    Mimics DATT's real-world fan setup (unsteady, partially periodic flow).
    """

    name = "wind_gust"

    def __init__(self, mean=(0.08, 0.0, 0.0), gust_amp=0.08, gust_hz=0.7,
                 ou_sigma=0.04, ou_tau=0.5, dt=0.01, seed=0):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.gust_amp, self.gust_hz = gust_amp, gust_hz
        self.ou_sigma, self.ou_tau, self.dt = ou_sigma, ou_tau, dt
        self._seed = seed
        self.reset()

    def reset(self, rng=None):
        self.rng = rng or np.random.default_rng(self._seed)
        self.ou = np.zeros(3)

    def force(self, t, state):
        a = self.dt / self.ou_tau
        self.ou += -a * self.ou + self.ou_sigma * np.sqrt(2 * a) * self.rng.normal(size=3)
        gust = self.gust_amp * np.sin(2 * np.pi * self.gust_hz * t) * np.array([1.0, 0.3, 0.0])
        return self.mean + gust + self.ou


class GroundEffect(Disturbance):
    """Cheeseman-Bennett in-ground-effect thrust gain, applied as an upward force.

    T_IGE/T_OGE = 1 / (1 - (R/4z)^2). Only significant below z ~ 4R (~11 cm);
    pair with a low-altitude trajectory (benchmark uses z = 0.08 m).
    """

    name = "ground"

    def force(self, t, state):
        z = max(float(state[2]), PROP_RADIUS / 2)  # avoid singularity below R/4
        ratio = 1.0 / max(1.0 - (PROP_RADIUS / (4 * z)) ** 2, 0.25)
        return np.array([0.0, 0.0, HOVER_THRUST * (ratio - 1.0)])


class Payload(Disturbance):
    """Extra payload as a constant downward force (default 10 g, 23% of weight)."""

    name = "payload"

    def __init__(self, extra_mass: float = 0.010):
        self.f = np.array([0.0, 0.0, -extra_mass * G])

    def force(self, t, state):
        return self.f


SCENARIOS = {
    "none": Disturbance,
    "wind_const": ConstantWind,
    "wind_gust": GustWind,
    "ground": GroundEffect,
    "payload": Payload,
}
