# Lesson 2 — Train a DATT policy

⏱️ ~30 minutes of attention + ~25 minutes of compute for the first real run.
**You finish when** you have your own `datt_ppo_final.zip` and can explain its
learning curves.

> **The one big idea:** training is not the hard part — *knowing whether it
> worked* is. So we start with a deliberately short run, learn to read the
> curves, and only then spend real compute.

---

## 1. Your first run (short, on purpose)

Inside the container, from `/workspace`:

```bash
cd /workspace/tasks/racing/crazy_track && python -m crazy_track.training.ppo_train --timesteps 500000 --v5 --seed 0 --reason "tutorial lesson 2: first short run, learning to read the curves"
```

~7 minutes. Read §2 while it runs.

Three things about that command line:

- **`--v5`** selects the asymmetric actor-critic from Lesson 1 §2 (noisy
  observations for the actor, privileged ones for the critic) plus sensor-noise
  domain randomisation.
- **`--seed 0`** — you will need 1 and 2 as well. Not optional; see §5.
- **`--reason` is required by the code.** Look at
  `tasks/racing/crazy_track/src/crazy_track/training/ppo_train.py:104`:

  ```python
      log = RunLogger(tag="datt-train", reason=args.reason, config=vars(args))
  ```

  Every run writes `metadata.yaml` with your reason, the **git commit**, and the
  full config. The project refuses to let you produce a result you cannot later
  explain. Adopt the habit — in six weeks you will have forty runs and no memory
  of which was which.

Output lands in `tasks/racing/crazy_track/results/<timestamp>_datt-train/`:

```
├── datt_ppo_final.zip   # your policy
├── metadata.yaml        # reason, git commit, full config
└── tb/                  # TensorBoard logs
```

## 2. What PPO is actually doing

`tasks/racing/crazy_track/src/crazy_track/training/ppo_train.py:140`:

```python
        model = PPO(
            policy, env, verbose=1, seed=args.seed,
            n_steps=256, batch_size=1024, learning_rate=3e-4, gamma=0.98,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log.dir / "tb"),
        )
```

The policy is a small network. PPO runs **16 copies of the simulator in
parallel**, collects `n_steps=256` transitions from each, and nudges the network
so actions that turned out better than expected become more likely — while
**clipping** how far the policy may move in a single update. That clip is the
whole trick: it stops one unlucky batch from destroying a working policy.

Two of those numbers are worth understanding rather than accepting:

- **`gamma=0.98`** at 50 Hz means the policy effectively cares about the next
  ~1 second (`1/(1-γ)` = 50 steps). Against a **0.6 s look-ahead window** that
  is a sensible match — the policy is asked to care about roughly the horizon it
  can see. If you later make the drone fly much faster, ask whether both numbers
  should move together.
- **`n_steps=256` × 16 envs = 4096** transitions per update, split into batches
  of 1024.

And the architecture choice, `ppo_train.py:128`:

```python
    if args.v5 or args.v6:
        from crazy_track.training.asymmetric import AsymmetricPolicy

        policy, policy_kwargs = AsymmetricPolicy, {}
    else:
        policy, policy_kwargs = "MlpPolicy", dict(net_arch=[64, 64])
```

`--v5` swaps the standard MLP for the asymmetric policy that splits the 56-number
observation: first 43 to the actor, all 56 to the critic. Everything else is
identical, which is exactly what makes it a clean comparison.

**Leave these hyperparameters alone** until Lesson 4 — every number in the parent
project used them, so your results stay comparable. Then change **one** at a
time.

## 3. Read the curves

```bash
tensorboard --logdir tasks/racing/crazy_track/results --port 6006 --bind_all
```

Open <http://localhost:6006>. Four curves matter:

| curve | healthy | when it is not |
|---|---|---|
| `rollout/ep_rew_mean` | rises, then flattens | flat from step 0 → not learning; falling → LR too high or a reward bug |
| `train/explained_variance` | climbs toward 1 | stuck near 0 → the critic cannot predict returns at all |
| `train/entropy_loss` | rises slowly (less negative) | collapses fast → went deterministic early, stopped exploring |
| `train/approx_kl` | ~0.005–0.02 | spikes → updates too aggressive |

> 🔍 **Look at `explained_variance` first.** It answers "does my critic
> understand this problem *at all*?" — and if the answer is no, nothing
> downstream can work. In our runs it passes 0.9 within the first ~40 k steps.

## 4. The real run

```bash
cd /workspace/tasks/racing/crazy_track && python -m crazy_track.training.ppo_train --timesteps 4000000 --v5 --seed 0 --reason "tutorial lesson 2: reference-length v5 run, seed 0"
```

~50 minutes on CPU. **4 M steps is what the reference policies in the parent
project used** — verified in their `metadata.yaml` — so your numbers are
comparable to theirs.

While it runs, a fair question: *why not train on the race track directly?*
Because a policy trained on one path learns that path, not the skill. Lesson 3
puts your policy on a track it has never seen. If it works there, you have a
tracker. If it only works on its training path, you have an expensive lookup
table.

## 5. Three seeds, and why this is not negotiable

Run the same command with `--seed 1` and `--seed 2`.

Not bureaucracy — measured necessity. Twice in the parent project a confident
single-seed conclusion **inverted** when the third seed arrived:

- A variant "clearly cost clean-state accuracy" — at three seeds that was seed
  luck, and the variant was fine.
- A training recipe gave a perfect result on seed 0 and a **total refusal to
  perform the behaviour at all** on seed 1. Same code, same command line.

If you report a single-seed number as a finding, you are reporting a coin flip.
Report `mean ± std` over three, or say plainly that it is one seed.

## 6. Sanity-check before you race

```bash
cd /workspace/tasks/racing/crazy_track && python -m crazy_track.eval.lissajous_benchmark --controllers datt:results/<your-run>_datt-train/datt_ppo_final.zip --speeds slow normal fast --reason "tutorial lesson 2: does my policy track a figure-8 at all"
```

Flies a figure-8 at three speeds and prints RMSE in metres. Rough expectations
for a healthy v5 policy: a few centimetres at `slow`, growing toward ~0.15 m at
`fast`. Above ~0.3 m at `fast`, something is wrong — go back to your curves.

> 💡 A 500 k-step policy scores tens of metres here. That is not a broken
> evaluation, that is an undertrained policy, and it is worth seeing once so you
> recognise the signature.

---

## 🛠️ Before you move on

1. **Compare your three seeds.** Put the three `fast` RMSE values in a table.
   That spread is your **measurement noise floor** — any improvement in Lesson 4
   smaller than it is not an improvement.
2. **Break it on purpose.** Train 500 k steps with `learning_rate=3e-3`
   (`ppo_train.py:142`). Watch `approx_kl` explode and the reward curve fall
   over. Put it back. Knowing what a diverged run *looks like* saves hours.
3. **Train a `--v2` policy** (no privileged critic, no noise randomisation) and
   compare its figure-8 numbers to your v5. Which is better, and does your
   explanation match Lesson 1 §2?
4. **Check the metadata.** Open a `metadata.yaml`. Could a classmate reproduce
   your run from it alone? That is the standard.

**Next:** [Lesson 3 — Evaluate like the race](03-evaluate-like-the-race.md)
