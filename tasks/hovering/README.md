# 🚁 Task: Teach a Drone to Hover with Reinforcement Learning

Learn **Reinforcement Learning (RL)** by teaching a simulated Crazyflie 2.1
Brushless quadrotor to **hover** — fly to a point in the air and hold still.

This is a **task** inside the shared `rl_quad_control` development environment.
The Docker image is the *infrastructure* (it has Python, CUDA, JAX, MuJoCo,
PyTorch, Stable-Baselines3 …); this folder is just the *content* you work on.

```
        target  ✕
                          🚁  "How do I get to the X and stay there?"
   ──────────────────────────  ground
```

---

## 1. Start the toolbox container (once)

From the **`rl_quad_control/`** folder (see its `README.md` / `docs/` for details):

```bash
# CPU
docker compose up -d
# ...or with an NVIDIA GPU:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Then open a shell inside it:

```bash
docker compose exec dev bash
```

You are now in the container, in `/workspace`, with the `main` Python
environment already active.

## 2. Install the simulator (once)

Inside the container, install Crazyflow into the `main` environment:

```bash
bash tasks/hovering/setup.sh
```

This clones `learnsyslab/crazyflow` into `repos/` and installs it. (If you plan
to use other tasks too, run `scripts/clone_repos.sh` +
`scripts/setup_python_envs.sh` instead — that sets up every task.)

> Run `setup.sh` again if you ever **recreate** the container (e.g. after
> changing a compose file): the cloned `repos/` is kept on your computer, so the
> re-run just reinstalls into the fresh container and is quick.

## 3. Check it works

```bash
cd /workspace/tasks/hovering
python -m k12_hover.check          # prints "Everything works! ✅"
```

---

## 4. The 3-minute taste

```bash
# Watch the pre-trained drone hover
python -m k12_hover.eval_sb3 --model models/hover_PPO.zip

# Train your OWN drone (add --jax-device gpu if the container has a GPU)
python -m k12_hover.train_sb3 --num-envs 64 --timesteps 1000000

# Score it
python -m k12_hover.eval_sb3 --model models/hover_PPO.zip
```

Evaluation prints how many centimeters from the target the drone holds — under
~10 cm is a great hover. Trained models are saved in `models/` (shared with your
computer, so they persist after you exit the container).

### Watch it fly in 3D (optional, Linux)

To open a real 3-D window (drone + red target ball), start the container with the
**GUI override** and allow it to use your display:

```bash
# On the HOST (once per login):
bash scripts/allow_gui.sh            # grants the container access to your screen

# Start the container WITH the gui override (add -f docker-compose.gpu.yml if GPU):
docker compose -f docker-compose.yml -f docker-compose.gui.yml up -d
docker compose -f docker-compose.yml -f docker-compose.gui.yml exec dev bash

# Inside the container:
bash tasks/hovering/setup.sh         # (once per container)
cd tasks/hovering
python -m k12_hover.eval_sb3 --model models/hover_PPO.zip --render
```

A MuJoCo window opens showing the drone hover at the red target. (GUI passthrough
is Linux-only here; on macOS/Windows use the number output above, which works
everywhere. See the toolbox `docs/` for X11 details.)

---

## 5. The lessons (do these in order)

Read the markdown in `lessons/` on your computer; run the commands inside the
container.

| # | Lesson | What you learn |
|---|--------|----------------|
| 1 | `lessons/01_what_is_rl.md` | The big ideas: agent, environment, reward |
| 2 | `lessons/02_build_the_env.md` | How we describe the hover task (`hover_env.py`) |
| 3 | `lessons/03_train_with_sb3.md` | Train a drone and read the numbers |
| 4 | `lessons/04_understand_rewards.md` | Change the reward, change the behavior |
| 5 | `lessons/05_try_other_algorithms.md` | PPO vs A2C vs SAC: different "brains" |
| 6 | `lessons/06_advanced_cleanrl.md` | Peek inside PPO (advanced) |

---

## What's in this folder

```
tasks/hovering/
├── README.md            ← you are here
├── setup.sh             ← installs Crazyflow into the main venv (run once)
├── k12_hover/           ← the code
│   ├── hover_env.py     ← the hover TASK (see / do / reward)   [Lesson 2]
│   ├── train_sb3.py     ← train with Stable-Baselines3 (easy)  [Lesson 3]
│   ├── eval_sb3.py      ← watch & score a trained drone
│   ├── train_cleanrl.py ← PPO written out by hand (advanced)   [Lesson 6]
│   └── check.py         ← "is my setup OK?" self-test
├── lessons/             ← the step-by-step lessons
└── models/              ← trained drones are saved here
```

---

## Notes for teachers

* **The Docker image is the infrastructure, not the task.** You build it once;
  every task (hovering, racing, …) runs in the same container. If a task needs a
  package the image lacks, edit `rl_quad_control/requirements/main.txt` (or the
  `Dockerfile`), rebuild, and share the change back.
* This task uses the `main` environment (Python 3.12). **learnsyslab/crazyflow
  requires Python ≥ 3.12** (it uses `value in EnumClass`, a 3.12 feature), which
  is why the toolbox's `main` venv is 3.12.
* The image pins the JAX stack (`requirements/constraints.txt`) so editable repo
  installs can't upgrade `jaxlib` past the CUDA plugin (which would break GPU
  linear algebra).
