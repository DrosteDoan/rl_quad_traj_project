# Saved recipe: gate-aware v4 (`s0`)

A frozen record of the training run that produced `s0`, seed 0 of the 0.7 rad learned recipe (also called
`v4`). It is kept exactly as it was saved on 2026-09-24; **do not edit or tidy anything in this folder.**

| file | what it is |
|---|---|
| `code/` | copies of the Lesson 9 code at the time of training. These are not duplicates to clean up: they are the code that actually trained this policy, and their SHA-256 hashes are recorded in `recipe.json`. |
| `ckpt/` | the final model (`datt_ppo_final.zip`, the one the study flies as `s0`) and every 2M-step checkpoint |
| `recipe.json` | the training command, the recipe, the pre-registered viability bar, its validation result, and the hashes of the code and the final checkpoint |
| `run_metadata.yaml` | the training run's configuration, reason and git commit |
| `manifest_train.csv`, `manifest_val.csv` | the track pools it was trained and validated on |
| `lambda_preview_summary.txt` | its first validation-pool preview at λ > 0 |

`tests/test_gate_aware.py` depends on this folder. It executes `code/gate_aware_env.py` next to the live
code and checks, step by step, that the live training environment is still bit-identical to the one that
trained this policy, and it loads `ckpt/datt_ppo_final.zip` to check the same under the policy's own
actions. Changing a file here breaks that check, and with it the evidence that the current code is the code
that produced the result.

The same training run is also in `../../policies/2026-09-24_18-26-35_racing-gate-aware-train/`.
