# Vendored: crazy_track @ `58eed32`

This directory is a **vendored snapshot** of the `crazy_track` research code
(`src/`, `configs/`, `tests/`, `pyproject.toml`) at commit
`58eed327120aa9e75dc288a41998ae6e3226373a` (2026-09-09).

History:

| date | commit | why |
|---|---|---|
| 2026-08-21 | `1921aa3` | first snapshot (Lessons 1–5) |
| 2026-09-10 | `58eed32` | Lesson 6: adds `trajectories/sampled.py` (a TOGT plan as a `Trajectory`), `eval/togt_race_eval.py`, the `disturbance="none"/"eso"/"l1"` switch of `controllers/mpc.py` (registered as `mpc_l1`), `--controller <spec>` on both race evaluators, `configs/togt/` (planner parameter sets + the lsy level-2 track / tube yamls) and `tests/test_sampled.py` |

The files the earlier lessons cite by line — `freestyle.py`, `datt_env.py`,
`chained_poly.py`, `ppo_train.py` — are byte-identical between the two snapshots, so
those citations still hold (re-verified 2026-09-10). Lesson 6 cites `mpc.py` and
`sampled.py` at this snapshot.

Why vendored instead of cloned:

1. **The upstream repository is private.** Students cannot `git clone` it, so
   `scripts/clone_repos.sh` cannot fetch it the way it fetches the public
   simulators. Shipping the pinned source inside this repo is what makes the
   racing task self-contained.
2. **The lessons cite file AND line numbers** in this source
   (e.g. `src/crazy_track/envs/datt_env.py:20`). A pinned snapshot cannot
   drift, so those citations stay correct.

Research artifacts (results/, papers/, reports/, publication*/ — several
hundred MB) are deliberately NOT included; only the code the lessons use. The
C++ driver for the TOGT planner (`scripts/togt/` upstream) lives in
`tasks/racing/code/togt/` with a container-aware build script.

Training output written by `crazy_track.eval.runlog.RunLogger` lands in
`results/` **inside this directory** (git-ignored). That is intentional: your
runs stay on your machine and never pollute the repo. `configs/togt/**` must
keep LF line endings (the planner's YAML reader chokes on CR); the repo's
`.gitattributes` enforces that.

If the maintainer bumps this snapshot, re-check the line-number citations
listed in each lesson before committing (`grep -o -E "[a-z_]+\.py:[0-9]+"
tasks/racing/lessons/*.md`).
