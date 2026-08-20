"""Render a rollout npz in the crazyflow/MuJoCo scene with the REAL lsy
drone-racing gates (vendored assets, MIT, see assets/lsy_drone_racing/).

Usage: python -m crazy_track.eval.freestyle_sim_gif --npz <rollout.npz>
           --track lsy-level2 --out flight_sim.gif
           [--fps 25] [--slowmo 4] [--width 960] [--height 640]

Kinematic replay: the recorded pos/quat series drives the drone's free
joint directly (no re-simulation), so the GIF shows exactly the logged
flight. Requires an offscreen GL context (MUJOCO_GL=egl or osmesa).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from crazy_track.eval.freestyle_gif import _frame_times
from crazy_track.trajectories.freestyle import TRACKS

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "lsy_drone_racing"


def _build_scene(traj):
    """crazyflow drone scene + lsy gate/obstacle bodies -> compiled MjModel."""
    import mujoco
    from scipy.spatial.transform import Rotation as R

    from crazyflow import Sim
    from crazyflow.dynamics import Dynamics

    sim = Sim(n_worlds=1, n_drones=1, drone="cf21B_500",
              dynamics=Dynamics("first_principles"), control="attitude",
              freq=500, device="cpu")
    spec = sim.spec
    frame = spec.worldbody.add_frame()
    gate_spec = mujoco.MjSpec.from_file(str(ASSETS / "gate.xml"))
    for i, g in enumerate(traj.gates):
        body = frame.attach_body(gate_spec.body("gate"), "", f":{i}")
        body.pos = g.center
        body.quat = R.from_euler("z", g.yaw).as_quat(scalar_first=True)
    obstacle_spec = mujoco.MjSpec.from_file(str(ASSETS / "obstacle.xml"))
    for i, o in enumerate(getattr(traj, "obstacles", [])):
        body = frame.attach_body(obstacle_spec.body("obstacle"), "", f":{i}")
        body.pos = o
    return spec.compile()


def render_sim_gif(npz_path: str, track: str, out: str, fps: int = 25,
                   slowmo: int = 4, width: int = 960, height: int = 640) -> None:
    import mujoco
    from PIL import Image

    data = np.load(npz_path)
    traj = TRACKS[track]()
    t, pos, quat = data["t"], data["pos"], data["quat"]
    windows = [(f["t_rot_start"], f["t_rot_end"]) for f in traj.flips]
    ftimes = _frame_times(float(t[-1]), windows, fps, slowmo)

    model = _build_scene(traj)
    mj_data = mujoco.MjData(model)
    # crazyflow models the drone as a MOCAP body (dynamics live in MJX)
    drone_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone:0")
    mocap_id = model.body_mocapid[drone_body]
    assert mocap_id >= 0, "drone:0 is not a mocap body in this crazyflow version"

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    # chase cam: follow the smoothed drone position (the cf21B is ~10 cm —
    # a static wide shot reduces it to a speck), slow orbit for parallax
    cam.distance, cam.elevation = 2.4, -16.0
    look = pos[0].astype(np.float64).copy()

    frames = []
    for k, ti in enumerate(ftimes):
        i = min(np.searchsorted(t, ti), len(t) - 1)
        mj_data.mocap_pos[mocap_id] = pos[i]
        mj_data.mocap_quat[mocap_id] = np.roll(quat[i], 1)  # xyzw -> wxyz
        mujoco.mj_kinematics(model, mj_data)
        mujoco.mj_camlight(model, mj_data)
        look += 0.08 * (pos[i] - look)  # EMA follow
        cam.lookat[:] = [look[0], look[1], max(0.4, look[2])]
        cam.azimuth = -55.0 + 30.0 * k / len(ftimes)  # slow orbit
        renderer.update_scene(mj_data, camera=cam)
        frames.append(Image.fromarray(renderer.render()))
    renderer.close()

    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames @ {fps} fps, {width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--track", required=True, choices=sorted(TRACKS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--slowmo", type=int, default=4)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()
    render_sim_gif(args.npz, args.track, args.out, args.fps, args.slowmo,
                   args.width, args.height)


if __name__ == "__main__":
    main()
