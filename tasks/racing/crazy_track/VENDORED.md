# Vendored: crazy_track @ `1921aa3`

This directory is a **vendored snapshot** of the `crazy_track` research code
(`src/`, `configs/`, `tests/`, `pyproject.toml`) at commit
`1921aa3d885a10b391287875b9c960619b1236fa`.

Why vendored instead of cloned:

1. **The upstream repository is private.** Students cannot `git clone` it, so
   `scripts/clone_repos.sh` cannot fetch it the way it fetches the public
   simulators. Shipping the pinned source inside this repo is what makes the
   racing task self-contained.
2. **The lessons cite file AND line numbers** in this source
   (e.g. `src/crazy_track/envs/datt_env.py:20`). A pinned snapshot cannot
   drift, so those citations stay correct.

Research artifacts (results/, papers/, reports/, publication*/ — several
hundred MB) are deliberately NOT included; only the code the lessons use.

Training output written by `crazy_track.eval.runlog.RunLogger` lands in
`results/` **inside this directory** (git-ignored). That is intentional: your
runs stay on your machine and never pollute the repo.

If the maintainer bumps this snapshot, re-check the line-number citations
listed in each lesson before committing.
