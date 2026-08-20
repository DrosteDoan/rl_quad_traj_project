# crazy_track

Trajectory-tracking benchmark for the **Crazyflie 2.1 brushless** on the
[crazyflow](https://github.com/utiasDSL/crazyflow) simulator, replicating the
Figure-5 / Table-III Lissajous benchmark of
["Learning to Fly in Seconds" (arXiv:2311.13081)](https://arxiv.org/abs/2311.13081)
with a controller pool that includes a DATT-style learned policy
([arXiv:2310.09053](https://arxiv.org/abs/2310.09053)).

## Benchmark

Figure-eight Lissajous `p(t) = [cos(2*pi*t/T), sin(4*pi*t/T)/2, z]` at
slow (T=15 s), normal (5.5 s), fast (3.5 s), plus an acro tier (2.2 s, at the
platform's feasibility limit) and a vertical-plane variant. Metric: RMSE 3D /
xy (1 s warmup excluded). Sim ground truth: `cf21B_500` (43.4 g, TWR 1.88),
first-principles dynamics + firmware-replica Mellinger attitude loop at 500 Hz.

## Current results (2026-07-22)

Nominal, RMSE 3D (m):

| controller | slow | normal | fast |
|---|---|---|---|
| PID (acc feedforward) | 0.012 | 0.022 | 0.088 |
| ADRC (velocity-ESO, w=7) | 0.012 | 0.034 | 0.089 |
| MPPI+L1 (tuned: N=512, AR(1) noise) | 0.023 | 0.035 | 0.068 |
| MPC (CasADi/ipopt, so_rpy model) | 0.018 | 0.063 | 0.083 |
| Offset-free MPC (+velocity-ESO disturbance state, soft-start) | **0.004** | 0.036 | **0.052** |
| DATT v3 (PPO + L1 obs + perturb training) | 0.021 | 0.048 | 0.099 |
| PID + [xadapt](https://github.com/muellerlab/xadapt_ctrl) low-level (pretrained, unseen airframe) | 0.025 | 0.038 | 0.067 |
| ADRC + xadapt (best overall stack: payload 0.018, gust 0.063) | 0.020 | **0.018** | 0.084 |

Disturbances (normal speed) and Lighthouse-sensor results: see
`reports/2026-07-22_*.md`. Highlights:
- **ADRC** wins all force-disturbance scenarios with clean sensing (wind
  0.025-0.032 vs PID 0.109) but is the most sensor-noise-sensitive.
- **DATT v5** (asymmetric actor-critic + noise domain randomization) is the
  best policy under the realistic deployment condition (Lighthouse + wind:
  0.059±0.002 across 3 training seeds, non-overlapping vs v4's 0.077±0.006;
  statistically tied with the ADRC+xadapt classical stack at 0.060±0.009).
  At 3 training seeds v5 dominates or ties v4 in every measured cell — the
  earlier "v5 trades clean-state performance" reading was single-seed noise.
- **DATT-acro** (CTBR body-rate policy) wins both acro-tier trajectories and
  degrades gracefully where attitude-mode controllers diverge; **acro2.1**
  additionally performs 360-degree roll flips (reward shaping was decisive —
  see reports for the 0-degree-flip failure analysis).
- Idealized-sim absolute numbers are ~5-10x better than the papers' real-world
  values for well-understood reasons (no drag/noise/latency); the Lighthouse
  sensor model (34+-18 Hz, 2-4 cm bias, per arXiv:2104.11523) closes most of
  the gap.

## Setup (Windows: WSL2 Ubuntu required for the simulator)

```powershell
wsl --install -d Ubuntu-24.04      # elevated PowerShell, one-time
```

Inside Ubuntu — if the repo lives on `/mnt/c`, enable metadata mounts once
(editable pip installs fail without it):

```bash
sudo tee -a /etc/wsl.conf >/dev/null <<'EOF'

[automount]
options = "metadata"
EOF
# then from Windows: wsl --terminate Ubuntu-24.04
```

Then:

```bash
./scripts/setup_env.sh                       # venv at ~/venvs/crazy_track
source ~/venvs/crazy_track/bin/activate
python -m pytest tests/                      # sanity check
```

## Running benchmarks

Every run writes a timestamped directory under `results/` with
`metadata.yaml` (date-time, **reason**, git commit, config), per-rollout
`.npz`, `summary.csv`, and tracking plots. `--reason` is mandatory by design.

```bash
# classical pool, 3 speeds
python -m crazy_track.eval.lissajous_benchmark \
  --controllers pid adrc mppi_l1 mpc --reason "..."

# learned policy (attitude-mode) / acro policy (CTBR)
python -m crazy_track.eval.lissajous_benchmark \
  --controllers datt:results/<train-run>/datt_ppo_final.zip --reason "..."

# disturbances: wind_const | wind_gust | ground | payload
# sensor model:  --sensor lighthouse     vertical fig-8:  --vertical
python -m crazy_track.eval.lissajous_benchmark \
  --controllers adrc --disturbance wind_const --sensor lighthouse --reason "..."

# flip maneuvers (CTBR policy)
python -m crazy_track.eval.flip_eval --model results/<run>/datt_ppo_final.zip --reason "..."
```

## Training DATT policies

```bash
python -m crazy_track.training.ppo_train --timesteps 3000000 --reason "..." [flags]
#  (no flag)      v3: L1 obs + force-perturbation training      (43-dim obs)
#  --noisy-sensor v4: + Lighthouse noise on observations
#  --v5           v5: + asymmetric actor-critic + noise DR      (56-dim obs)
#  --v6           v6a: + 4-frame stacked actor obs              (185-dim obs)
#  --ctbr         acro: body-rate actions, refs to the TWR limit
#  --acro2        acro phase 2: + flip primitives               (46-dim obs)
```

## Repository layout

```
src/crazy_track/
  trajectories/    # Lissajous (+vertical/acro), chained-poly, zigzag, flip
  envs/            # crazyflow rollout harness + vectorized RL training env
  controllers/     # PID, ADRC, MPPI+L1, MPC, DATT eval controllers, L1, CTBR utils
  training/        # PPO entrypoint + asymmetric actor-critic policy
  eval/            # benchmark runners, flip eval, run logging
  disturbances.py  # wind / gust / ground-effect / payload models
  sensors.py       # Lighthouse LH2 measurement model (arXiv:2104.11523)
configs/  reports/  results/  scripts/  tests/
papers/
  paper1-benchmark/   # fig-8 controller benchmark: claims, evidence map, models
  paper2-acrobatics/  # acrobatic (flip) controller: story arc, experiments, models
```

This work targets two papers; `papers/<paper>/README.md` is the per-paper
index (scope, claims with statistical evidence, run/model manifests).
`reports/` stays the chronological lab notebook (new reports are prefixed
`p1-`/`p2-`); `results/` run dirs are immutable and referenced by both.

## Roadmap

- [x] Fig-5 replication: 5 controllers x 3 speeds; disturbance + sensor matrices
- [x] DATT v3-v6a study (perturbations, noise, asymmetric AC, frame stacking)
- [x] DATT-acro: CTBR interface, acro-tier + vertical fig-8, flip primitives
- [ ] Flip robustness: higher hover margin, pitch-flip asymmetry, recovery precision
- [x] [xadapt_ctrl](https://github.com/muellerlab/xadapt_ctrl) in the pool:
      payload robustness validated (0.038 vs 0.093 for PID+Mellinger), new
      fast-nominal pool best; needs `git clone` of the repo + `pip install
      onnx onnxruntime` (see `controllers/xadapt.py`)
- [ ] Offset-free MPC; adaptive-bandwidth ESO; multi-seed statistics

## References

- Eschmann, Albani, Loianno — Learning to Fly in Seconds (arXiv:2311.13081)
- Huang et al. — DATT: Deep Adaptive Trajectory Tracking (arXiv:2310.09053)
- Taffanel et al. — Lighthouse Positioning System dataset (arXiv:2104.11523)
- [lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing),
  [learning-to-fly](https://github.com/arplaboratory/learning-to-fly)
