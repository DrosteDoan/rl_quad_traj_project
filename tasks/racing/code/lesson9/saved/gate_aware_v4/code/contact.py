"""Gate-frame contact check for the harness: does a flown path touch a gate, and when? (Lesson 9 phase 2)

The harness runs a bare crazyflow Sim: there is no gate geometry in it, so nothing can collide. lsy's race
ends the episode at the first MuJoCo contact between the drone's collision BOX and any gate frame
(`race_core.py`: `use_box_collision(sim, True)`; contact if geom distance < 0). This module reproduces that
test on a logged path, from the same numbers lsy uses:

    drone   crazyflow/drones/cf21B_500.xml   col_box     half-extents (0.07, 0.07, 0.02) m, body-fixed
    gate    lsy envs/assets/gate.xml         col_top/bottom/left/right, 0.01 m half-thickness along the
            pass direction; top/bottom span y in [-0.36, 0.36], z in [0.20, 0.36] (and mirrored);
            left/right span y in [0.20, 0.36] (and mirrored), z in [-0.36, 0.36]

Those give Lesson 8 section 1's thresholds for a LEVEL drone: a post is touched at an in-plane offset of
0.13 m at a gate whose normal is world-aligned (G3, G4) and 0.101 m at a 45-degree gate (G1, G2), a rail at
0.18 m vertical. test_contact.py derives them and cross-checks the whole test against MuJoCo itself.

Poles are out of the study (METHODOLOGY.md), so only gate frames are checked. Overlap is the exact
separating-axis test for two oriented boxes; the path is swept: the segment between consecutive samples is
subdivided to <= `sub` metres (a 100 Hz sample is up to 5 cm of travel at 5 m/s; the frame is 2 cm thick).

    hit = first_contact(t, pos, quat, gates)        # None, or {"t", "gate", "box", "pos"}
    m   = frame_margin(t, pos, quat, gates)         # how far the drone box could grow before touching
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

DRONE_HALF = np.array([0.07, 0.07, 0.02])
# (name, centre, half-extents) in the gate frame: x = pass direction (thickness), y = left, z = up
GATE_BOXES = (
    ("top", np.array([0.0, 0.0, 0.28]), np.array([0.01, 0.36, 0.08])),
    ("bottom", np.array([0.0, 0.0, -0.28]), np.array([0.01, 0.36, 0.08])),
    ("left", np.array([0.0, -0.28, 0.0]), np.array([0.01, 0.08, 0.36])),
    ("right", np.array([0.0, 0.28, 0.0]), np.array([0.01, 0.08, 0.36])),
)
NEAR = 0.75          # only poses this close to a gate centre can touch it (frame corner 0.51 m + drone 0.08 m)


def _overlap(c1: np.ndarray, R1: np.ndarray, h1: np.ndarray, c2: np.ndarray, R2: np.ndarray,
             h2: np.ndarray) -> np.ndarray:
    """Exact separating-axis test. Box 1: M poses, centres c1 (M,3), rotations R1 (M,3,3) whose COLUMNS are
    the box axes, half-extents h1 (3,). Box 2: one fixed box (centre c2, rotation R2, half-extents h2).
    True where the interiors overlap (MuJoCo's geom distance < 0)."""
    m = len(c1)
    d = c2[None, :] - c1
    a1 = [R1[:, :, i] for i in range(3)]
    a2 = [np.broadcast_to(R2[:, j], (m, 3)) for j in range(3)]
    axes = a1 + a2
    for i in range(3):
        for j in range(3):
            cr = np.cross(a1[i], a2[j])
            n = np.linalg.norm(cr, axis=1, keepdims=True)
            axes.append(np.where(n > 1e-9, cr / np.maximum(n, 1e-12), 0.0))   # parallel edges: no axis
    separated = np.zeros(m, dtype=bool)
    for a in axes:
        proj1 = np.einsum("mk,mki->mi", a, R1)                # (M,3): a . box-1 axes
        proj2 = a @ R2                                        # (M,3): a . box-2 axes
        r1 = (np.abs(proj1) * h1).sum(axis=1)
        r2 = (np.abs(proj2) * h2).sum(axis=1)
        separated |= np.abs(np.einsum("mk,mk->m", a, d)) > r1 + r2
    return ~separated


def pose_contacts(pos: np.ndarray, quat: np.ndarray, gates, inflate: float = 0.0):
    """Per pose: (touching any gate frame?, gate index (1-based) or 0, box name or ''). pos (M,3),
    quat (M,4) xyzw. `inflate` grows the drone box by that many metres per half-extent."""
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    quat = np.atleast_2d(np.asarray(quat, dtype=np.float64))
    rot = R.from_quat(quat).as_matrix()
    h = DRONE_HALF + float(inflate)
    hit = np.zeros(len(pos), dtype=bool)
    which_gate = np.zeros(len(pos), dtype=int)
    which_box = np.full(len(pos), "", dtype=object)
    for gi, g in enumerate(gates, start=1):
        Rg = np.asarray(g.rotation, dtype=np.float64)
        centre = np.asarray(g.center, dtype=np.float64)
        near = np.linalg.norm(pos - centre, axis=1) < NEAR
        if not near.any():
            continue
        idx = np.flatnonzero(near)
        for name, c_loc, h_loc in GATE_BOXES:
            ov = _overlap(pos[idx], rot[idx], h, centre + Rg @ c_loc, Rg, h_loc)
            new = ov & ~hit[idx]
            hit[idx[new]] = True
            which_gate[idx[new]] = gi
            which_box[idx[new]] = name
    return hit, which_gate, which_box


def _sweep(t: np.ndarray, pos: np.ndarray, quat: np.ndarray, sub: float):
    """Subdivide every step to <= `sub` metres of travel (nlerp for the attitude)."""
    t, pos, quat = (np.asarray(x, dtype=np.float64) for x in (t, pos, quat))
    if len(t) < 2:
        return t, pos, quat
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    n = np.maximum(1, np.ceil(step / sub)).astype(int)
    idx = np.repeat(np.arange(len(step)), n)
    first = np.concatenate([[0], np.cumsum(n)[:-1]])
    s = (np.arange(len(idx)) - np.repeat(first, n)) / np.repeat(n, n)
    q0, q1 = quat[idx], quat[idx + 1]
    q1 = np.where((q0 * q1).sum(axis=1, keepdims=True) < 0, -q1, q1)          # shortest way round
    q = (1 - s)[:, None] * q0 + s[:, None] * q1
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p = pos[idx] + s[:, None] * (pos[idx + 1] - pos[idx])
    tt = t[idx] + s * (t[idx + 1] - t[idx])
    return (np.concatenate([tt, t[-1:]]), np.vstack([p, pos[-1:]]), np.vstack([q, quat[-1:]]))


def first_contact(t, pos, quat, gates, inflate: float = 0.0, sub: float = 0.01) -> dict | None:
    """The first moment the swept drone box touches a gate frame, or None."""
    ts, ps, qs = _sweep(t, pos, quat, sub)
    hit, gate, box = pose_contacts(ps, qs, gates, inflate)
    if not hit.any():
        return None
    i = int(np.argmax(hit))
    return {"t": float(ts[i]), "gate": int(gate[i]), "box": str(box[i]), "pos": [float(x) for x in ps[i]]}


def frame_margin(t, pos, quat, gates, max_margin: float = 0.3, sub: float = 0.01, tol: float = 5e-4) -> float:
    """Smallest growth of the drone box (metres per half-extent) at which the path first touches a frame:
    0.0 if it already does, `max_margin` if it never does within that. Roughly 'how far a tracker may stray
    from this path, in any direction, before the race would end'."""
    ts, ps, qs = _sweep(t, pos, quat, sub)
    if pose_contacts(ps, qs, gates, 0.0)[0].any():
        return 0.0
    if not pose_contacts(ps, qs, gates, max_margin)[0].any():
        return float(max_margin)
    lo, hi = 0.0, float(max_margin)
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if pose_contacts(ps, qs, gates, mid)[0].any():
            hi = mid
        else:
            lo = mid
    return float(hi)
