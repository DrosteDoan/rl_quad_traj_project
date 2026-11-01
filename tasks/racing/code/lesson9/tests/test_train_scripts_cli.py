"""Checks for `train_gate_aware.py`'s CLI-exposed PPO hyperparameters (main venv, in the container):

    python tasks/racing/code/lesson9/test_train_scripts_cli.py        # no pytest needed

MECHANICS ONLY: argparse-level checks that the corrected defaults (2026-09-23, after the pre-gate-aware
rounds 1-3 flew weak across both groups -- METHODOLOGY.md sections 5a/5b) are what `train_gate_aware.py`
actually ships, for every `--group`, and that they keep the PPO minibatch structure (4 minibatches/epoch)
the recipe was tuned around. No env is built, no model is constructed, no `.learn()` anywhere.

`train_robust.py`/`train_contrast.py`/`train_full_authority.py` -- the pre-gate-aware standalone trainers
this file used to check -- were deleted 2026-09-27 as unrepresentative of the study: every reported
checkpoint (0.7-authority and full-authority alike) came from `train_gate_aware.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_gate_aware as tga  # noqa: E402


def test_corrected_defaults():
    for g in ("robust", "robust_full", "contrast", "contrast_force"):
        args = tga.build_parser().parse_args(["--reason", "test", "--group", g])
        assert (args.gamma, args.n_steps, args.batch_size) == (0.99, 512, 2048), g


def test_round_1_3_reproduction_override():
    """The original (weak) round 1-3 hyperparameters still parse, for reproducing that history."""
    args = tga.build_parser().parse_args(["--reason", "test", "--gamma", "0.98",
                                          "--n-steps", "256", "--batch-size", "1024"])
    assert (args.gamma, args.n_steps, args.batch_size) == (0.98, 256, 1024)


def test_minibatch_structure_matches_between_corrected_and_original():
    """n_steps * n_envs / batch_size = 4 minibatches/epoch, unchanged by the fix."""
    corrected = tga.build_parser().parse_args(["--reason", "t"])
    original = tga.build_parser().parse_args(["--reason", "t", "--gamma", "0.98",
                                               "--n-steps", "256", "--batch-size", "1024"])
    assert corrected.n_steps * corrected.n_envs / corrected.batch_size == 4
    assert original.n_steps * original.n_envs / original.batch_size == 4


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
