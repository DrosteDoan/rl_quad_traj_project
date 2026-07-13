"""Classical controllers to benchmark against the RL tracker (Lesson 10).

The RL policy is one way to make the drone follow the circle. It is NOT the only
way — engineers have flown drones for decades with **classical** controllers that
use physics and math instead of learning. To know whether our RL brain is any
good, we race it against these classics on the exact same circle.

This file holds the classical controllers. Each one answers the same question the
RL policy answers every step: *"given where I am and where the target is, what
attitude command should I send?"* — so they all plug into the same simulator.

The controllers here
---------------------
* ``PIDController``  — the workhorse of real drones. Turns position error into a
  tilt-and-thrust command with three simple gains (P, I, D). Fast, no model of the
  future.
* ``MPCController``  — "Model-Predictive Control". Looks a few steps into the
  future, and each step solves a little optimization: *what sequence of commands
  keeps me closest to the path?* Uses the path look-ahead, like the RL tracker.

A third baseline, the simulator's built-in **onboard state controller**, lives in
``benchmark_traj.py`` because it uses a different command type (a full-state
setpoint) rather than an attitude command.

All the physical constants are for the Crazyflie 2.1 Brushless (``cf21B_500``).
"""

from __future__ import annotations

import numpy as np

# --- Physical facts about the cf21B_500, read from the simulator's drone params.
GRAVITY = 9.81
DRONE_MASS = 0.04338              # kg
THRUST_MIN = 0.08545             # N  (collective, all 4 rotors)
THRUST_MAX = 0.8                 # N
MAX_TILT = np.pi / 2             # rad, the attitude action bound


# --------------------------------------------------------------------------- ref
class CircleReference:
    """The moving target: where should the drone be at time ``t`` (and how fast)?

    This is the *same* circle the environment flies, written as clean math so a
    classical controller can also ask "where is the path heading next?".
    """

    def __init__(self, radius=0.5, period=6.0, height=1.0, center=(0.0, 0.0)):
        self.radius = float(radius)
        self.period = float(period)
        self.height = float(height)
        self.center = (float(center[0]), float(center[1]))
        self.omega = 2.0 * np.pi / self.period   # angular speed (rad/s)

    def pos(self, t):
        """Target position at time t: a point going around the circle."""
        cx, cy = self.center
        return np.array([cx + self.radius * np.cos(self.omega * t),
                         cy + self.radius * np.sin(self.omega * t),
                         self.height])

    def vel(self, t):
        """How fast the target is moving (its velocity) at time t."""
        return np.array([-self.radius * self.omega * np.sin(self.omega * t),
                         self.radius * self.omega * np.cos(self.omega * t),
                         0.0])

    def acc(self, t):
        """The target's acceleration at time t (it curves, so this isn't zero)."""
        return np.array([-self.radius * self.omega**2 * np.cos(self.omega * t),
                         -self.radius * self.omega**2 * np.sin(self.omega * t),
                         0.0])


# ---------------------------------------------------- accel -> attitude command
def accel_to_attitude(a_des: np.ndarray) -> np.ndarray:
    """Convert a desired acceleration into an attitude command.

    Both classical controllers work out a *desired acceleration* (which way and
    how hard to push the drone). A quadrotor can only push along the direction it
    points, so we turn that acceleration into "tilt this way, push this hard".

    Returns a physical attitude command ``[roll, pitch, yaw, thrust]`` where the
    angles are in radians and thrust is in Newtons. (Sign convention, matching
    Crazyflow + the real firmware: +pitch accelerates +x, +roll accelerates -y.)
    """
    ax, ay, az = a_des
    # Total upward push must beat gravity; az already includes the +g we want.
    az = max(az, 0.1)  # never command zero/negative lift (the drone would fall)
    # Tilt needed to produce the sideways accelerations (small-angle geometry).
    pitch = np.arctan2(ax, az)          # nose down/up -> move in x
    roll = np.arctan2(-ay, az)          # bank -> move in y (note the minus sign)
    roll = float(np.clip(roll, -MAX_TILT, MAX_TILT))
    pitch = float(np.clip(pitch, -MAX_TILT, MAX_TILT))
    # Collective thrust: enough to give acceleration az once we account for tilt
    # (tilting "wastes" some lift, so divide by cos of the tilt angles).
    thrust = DRONE_MASS * az / (np.cos(roll) * np.cos(pitch))
    thrust = float(np.clip(thrust, THRUST_MIN, THRUST_MAX))
    return np.array([roll, pitch, 0.0, thrust], dtype=np.float32)


def normalize_attitude(cmd: np.ndarray) -> np.ndarray:
    """Map a physical ``[roll,pitch,yaw,thrust]`` command to the agent's [-1, 1].

    The environment's ``NormalizeActions`` wrapper expects every controller —
    classical or RL — to speak the same [-1, 1] language, then rescales it back to
    physical units. So the classical controllers normalize here; the round trip is
    exact, so both kinds of controller drive identical physics.
    """
    roll, pitch, yaw, thrust = cmd
    n_roll = roll / MAX_TILT
    n_pitch = pitch / MAX_TILT
    n_yaw = yaw / MAX_TILT
    # thrust in [THRUST_MIN, THRUST_MAX] -> [-1, 1]
    n_thrust = 2.0 * (thrust - THRUST_MIN) / (THRUST_MAX - THRUST_MIN) - 1.0
    return np.clip([n_roll, n_pitch, n_yaw, n_thrust], -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- PID
class PIDController:
    """A cascaded PID position controller (the classic drone autopilot).

    Every step it measures the error (how far the drone is from where the target
    is now) and its rate of change, and mixes them with three gains:

        desired_accel = Kp * (position error)
                      + Kd * (velocity error)
                      + Ki * (accumulated error)
                      + gravity + the target's own acceleration (feed-forward)

    Then ``accel_to_attitude`` turns that into a tilt-and-thrust command. It has no
    idea where the path goes next — it only reacts to the current error — which is
    exactly the classical baseline we want to compare the RL look-ahead against.
    """

    def __init__(self, kp=6.0, kd=4.0, ki=1.0, dt=0.02):
        self.kp, self.kd, self.ki, self.dt = kp, kd, ki, dt
        self.reset()

    def reset(self):
        self._integral = np.zeros(3)

    def act(self, pos, vel, ref_pos, ref_vel, ref_acc) -> np.ndarray:
        """Return a normalized ``[-1, 1]`` attitude action."""
        e_pos = ref_pos - pos
        e_vel = ref_vel - vel
        self._integral += e_pos * self.dt
        self._integral = np.clip(self._integral, -1.0, 1.0)  # anti-windup
        a_des = (self.kp * e_pos + self.kd * e_vel + self.ki * self._integral
                 + ref_acc + np.array([0.0, 0.0, GRAVITY]))
        return normalize_attitude(accel_to_attitude(a_des))


# --------------------------------------------------------------------------- MPC
class MPCController:
    """Model-Predictive Control: plan a few steps ahead, then act.

    Where PID only looks at the *current* error, MPC looks at the next ``horizon``
    points of the path and solves a small optimization each step: pick the sequence
    of accelerations that keeps the drone closest to the upcoming path without
    thrashing the controls. It then applies only the first acceleration (and
    re-plans next step). This "look-ahead" is the same advantage the RL tracker
    gets — so MPC is the classical controller most like our RL brain.

    We model the drone as a simple point mass (position + velocity pushed by an
    acceleration). That's enough for smooth trajectories and keeps the math small.
    """

    def __init__(self, horizon=20, dt=0.02, q_pos=12.0, q_vel=1.0, r_acc=0.01):
        import casadi as ca

        self.ca = ca
        self.N = int(horizon)
        self.dt = float(dt)
        self._build(q_pos, q_vel, r_acc)

    def _build(self, q_pos, q_vel, r_acc):
        ca = self.ca
        N, dt = self.N, self.dt
        opti = ca.Opti()
        # Decision variables: position (3), velocity (3) over the horizon, and the
        # acceleration command (3) at each step.
        P = opti.variable(3, N + 1)
        V = opti.variable(3, N + 1)
        A = opti.variable(3, N)
        # Parameters we fill in each step: the current state and the future path.
        p0 = opti.parameter(3)
        v0 = opti.parameter(3)
        ref = opti.parameter(3, N + 1)   # where the path will be, step by step

        cost = 0
        opti.subject_to(P[:, 0] == p0)
        opti.subject_to(V[:, 0] == v0)
        for k in range(N):
            # Point-mass motion: next pos/vel from current + acceleration.
            opti.subject_to(P[:, k + 1] == P[:, k] + V[:, k] * dt)
            opti.subject_to(V[:, k + 1] == V[:, k] + A[:, k] * dt)
            cost += q_pos * ca.sumsqr(P[:, k] - ref[:, k])      # stay on the path
            # Match the target's SPEED (its position is moving), so we don't lag
            # behind. The reference velocity is how fast the path point itself moves.
            ref_vel = (ref[:, k + 1] - ref[:, k]) / dt
            cost += q_vel * ca.sumsqr(V[:, k] - ref_vel)        # keep up, don't trail
            cost += r_acc * ca.sumsqr(A[:, k])                  # keep it gentle
        cost += q_pos * ca.sumsqr(P[:, N] - ref[:, N])          # terminal on path
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False, "ipopt": {"print_level": 0,
                     "max_iter": 60, "sb": "yes"}})
        self.opti, self.P, self.V, self.A = opti, P, V, A
        self.p0, self.v0, self.ref = p0, v0, ref

    def reset(self):
        pass

    def act(self, pos, vel, ref_future) -> np.ndarray:
        """``ref_future`` is an ``(N+1, 3)`` array of upcoming target positions."""
        ca = self.ca
        self.opti.set_value(self.p0, pos)
        self.opti.set_value(self.v0, vel)
        self.opti.set_value(self.ref, ref_future.T)
        try:
            sol = self.opti.solve()
            a0 = np.array(sol.value(self.A[:, 0])).reshape(3)
        except Exception:
            # If the solver hiccups, fall back to "aim at the next point".
            a0 = 4.0 * (ref_future[0] - pos)
        a_des = a0 + np.array([0.0, 0.0, GRAVITY])   # add the anti-gravity lift
        return normalize_attitude(accel_to_attitude(a_des))
