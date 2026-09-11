# TOGT-Planner configuration for the cf21B_500 on the lsy level-2 track

Inputs for [TOGT-Planner](https://github.com/FSC-Lab/TOGT-Planner) (pinned at `0ed9afb`,
2026-09-02) driven by `scripts/togt/togt_race`. Results and the full story:
`reports/2026-09-09_p2-togt-planner-benchmark.md`.

```
cf21b_togt/   parameter set for RacePlanner::planTOGT  (upstream "standard" style)
cf21b_aos/    parameter set for RacePlanner::plan      (AOS-TOGT, upstream "cpc" style; not used —
              its refine stage drops gates, see report)
lsy_level2_track.yaml   the four nominal gates, zero-thickness corridors (what TOGT expects)
lsy_level2_tube.yaml    the same gates bracketed by entry/exit corridors 0.6 m along each
                        normal — forces near-normal crossings; THIS is the plan that races
```

## Things that are not obvious

- **The drone is rescaled by k = 1/0.04338 (mass → 1 kg).** Mass, inertia and per-rotor
  thrust bounds are all multiplied by k, which leaves every acceleration, body rate and the
  optimal trajectory unchanged (thrust = m·(a+g)). It is needed because TOGT's thrust
  penalty is evaluated in absolute Newtons and is ~530× too weak for a 43 g drone
  otherwise — the solver then ignores the thrust bounds entirely (measured). True values
  are in the comment header of `quad.yaml`.
- **Thrust bounds are 0.95 × TWR** (per rotor 0.19 N true), the same headroom
  `feasibility_report` enforces. `scripts/togt/sweep_margin.py` varies this.
- **`dynamicConstCheck: false`.** On, L-BFGS does not converge on this track's short
  pieces. Feasibility is verified downstream instead.
- **LF line endings, enforced by `.gitattributes`.** TOGT's YAML reader keeps a trailing CR
  in values; CRLF files fail with "Config files not found!".
- **Gate orientation:** TOGT's `RectanglePrisma` normal is its local z; `rpy: [0, -90, y]`
  maps it to −(cos y, sin y, 0), so `y = lsy_yaw_deg + 180` aligns it with the lsy normal.
- **Margins:** `marginW/H` shrink the *total* width: window = 0.5·(0.4 − margin).
  0.14 → ±0.13 m (drone half-extent + safety), 0.30 → ±0.05 m.
- **Absolute paths** when calling the driver, or use the hardened `togt_race` from
  `scripts/togt/` (it absolutizes). TOGT's loader prefixes the params dir twice otherwise.

## Rebuild the driver (no sudo needed)

```bash
git clone https://github.com/FSC-Lab/TOGT-Planner && git -C TOGT-Planner checkout 0ed9afb
python3 -m venv ~/cmk && ~/cmk/bin/pip install "cmake>=3.25,<4"        # if cmake < 3.25
curl -sL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz
~/cmk/bin/cmake -S eigen-3.4.0 -B eigen-build -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
TOGT_DIR=$PWD/TOGT-Planner EIGEN3_DIR=$PWD/eigen-build CMAKE=~/cmk/bin/cmake bash scripts/togt/build.sh
```
