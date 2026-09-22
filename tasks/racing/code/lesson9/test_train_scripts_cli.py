"""Checks for the training scripts' CLI-exposed hyperparameters (main venv, in the container):

    python tasks/racing/code/lesson9/test_train_scripts_cli.py        # no pytest needed

MECHANICS ONLY: argparse-level checks that the corrected defaults (2026-09-23, after rounds 1-3 flew weak
across both groups) and the round-1-3 reproduction override parse to the right values. No env is built, no
model is constructed, no `.learn()` anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_contrast  # noqa: E402
import train_robust  # noqa: E402


def test_corrected_defaults():
    for mod, label in ((train_robust, "robust"), (train_contrast, "contrast")):
        args = mod.build_parser().parse_args(["--reason", "test"])
        assert (args.gamma, args.n_steps, args.batch_size) == (0.99, 512, 2048), label


def test_round_1_3_reproduction_override():
    for mod, label in ((train_robust, "robust"), (train_contrast, "contrast")):
        args = mod.build_parser().parse_args(["--reason", "test", "--gamma", "0.98",
                                              "--n-steps", "256", "--batch-size", "1024"])
        assert (args.gamma, args.n_steps, args.batch_size) == (0.98, 256, 1024), label


def test_minibatch_structure_matches_between_corrected_and_original():
    """n_steps * n_envs / batch_size = 4 minibatches/epoch, unchanged by the fix."""
    for mod in (train_robust, train_contrast):
        corrected = mod.build_parser().parse_args(["--reason", "t"])
        original = mod.build_parser().parse_args(["--reason", "t", "--gamma", "0.98",
                                                   "--n-steps", "256", "--batch-size", "1024"])
        assert corrected.n_steps * corrected.n_envs / corrected.batch_size == 4
        assert original.n_steps * original.n_envs / original.batch_size == 4


def test_screen_command_reuses_seed_0():
    """The one-seed-pair screen: --seed 0, corrected defaults, both groups -- seed 0 must parse for both
    (train_contrast restricts --seed to {0,1,2})."""
    for mod in (train_robust, train_contrast):
        args = mod.build_parser().parse_args(["--reason", "screen", "--seed", "0"])
        assert args.seed == 0 and (args.gamma, args.n_steps, args.batch_size) == (0.99, 512, 2048)


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
