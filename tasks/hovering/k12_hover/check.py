"""A tiny self-test so students can confirm their setup works.

Run it with::

    python -m k12_hover.check
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    print("Checking your hover setup...\n")

    # 1) The packages we need.
    import jax
    import stable_baselines3 as sb3

    print(f"  Stable-Baselines3 : {sb3.__version__}")
    print(f"  JAX devices       : {jax.devices()}")
    has_gpu = any(d.platform == "gpu" for d in jax.devices())
    print(f"  GPU available     : {'yes 🚀' if has_gpu else 'no (CPU is fine, just slower)'}")

    # 2) Build the hover environment and take a few steps.
    import numpy as np

    from k12_hover import make_sb3_env, target_distance

    env = make_sb3_env(num_envs=4, device="cpu")
    obs = env.reset()
    assert obs.shape == (4, 13), f"unexpected obs shape {obs.shape}"
    action = np.zeros((4, 4), dtype=np.float32)
    action[:, 0] = 0.2
    for _ in range(3):
        obs, reward, dones, infos = env.step(action)
    env.close()
    print(f"  Environment       : ok (saw {obs.shape[1]} numbers, "
          f"started {target_distance(obs).mean() * 100:.0f} cm from target)")

    print("\nEverything works! ✅")
    print("Next: open lessons/01_what_is_rl.md")


if __name__ == "__main__":
    main()
