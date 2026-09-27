# Lesson 9 — Where the crossover is: learned vs model-based racing under graded disturbance

⏱️ ~3 hours of reading; the compute is the expensive part if you run it yourself — flying the
shipped checkpoint against M1 on the validation pool is under 2 minutes on 8 workers, but training
your own copy of the recipe is 5–6 hours (16M steps) or 9+ hours (25M). **You finish when** you have
(a) the shipped `gate_aware_v4` checkpoint flown against M1 on the 22 validation tracks and your own
number next to the pre-registered viability bar, (b) the held-out-condition table read and a verdict
on whether the RL policy's advantage travels beyond what it was trained on, (c) an opinion on the
control-authority question — should MPC be capped to match RL, or RL retrained to match MPC — and
what changes about the answer once you pick one, and (d) a plain-English account of what the *dropped*
control group would have told us, and why held-out conditions are a substitute, not a replacement,
for it.

> **The one big idea:** Lesson 7 asked whether a tracker is precise, robust, or survives contact, at
> one disturbance setting, on one track. This lesson asks a sharper question: as a disturbance grows
> from nothing to a calibrated ceiling, *where* does the ranking between model-based and learned
> trackers flip, and does that crossover point sit in the same place across many track geometries or
> wander with the course. Getting an RL policy that could even race turned out to be most of the
> work — three rebuilds, a dropped control group, and a control-authority argument that overturned its
> own pre-registered rule — and every one of those detours is worth more than the numbers it produced,
> because each is a way an RL-vs-MPC comparison can quietly cheat before a single lambda is swept.
> **The notice that governs every number this lesson produces: the crossover regimes found here are
> not intended to represent universal thresholds for learned or model-based control** — they describe
> this course's tracks, this course's disturbance models, and these specific trackers.

**Prerequisites:** Lesson 7 (the disturbance models, the six-condition matrix, the λ ∈ [0, 1] grading),
Lesson 8 (the corrected MPC family, ground-start plans, the box contact model). Everything runs inside
the `rl_quad_traj` container.

**Receipts.** Every number here is measured and traceable: `tasks/racing/code/lesson9/METHODOLOGY.md`
is the living design document (read it first if a claim here and there disagree — it is updated before
this lesson text catches up to it, and its §13 is a standing errata of every correction made along the
way). The raw per-lap CSVs and their summaries live under `tasks/racing/code/lesson9/results/eval/`,
one directory per experiment; the saved, frozen recipe is `tasks/racing/code/lesson9/saved/gate_aware_v4/`
(code snapshot, checkpoints, sha256s — a regression test reads it directly).

---

## 0. The design: five controllers, six conditions, fifteen tracks

Five model-based controllers, the Lesson 8 corrected family, evaluated with no re-tuning:

| column | what it is | spec |
|---|---|---|
| `M1` | the corrected MPC (Lesson 8's measured model) | `mpcdev:att=sim,drag=0.495,fgain=1.0` |
| `M1+ESO` | M1 + velocity ESO | `...,dist=eso` |
| `M1+L1` | M1 + L1 adaptation | `...,dist=l1` |
| `M1+mass` | M1 + thrust-scale (mass) adaptation | `...,mass=1` |
| `mppi_l1` | sampling-based MPC + L1, reconfigured to an 0.8 s preview horizon | `mppi_l1` |

M1 alone completes all 22 unseen validation tracks (RMSE 0.0525 m) with none of Lesson 8's parameters
touched — the model-based side of this study needed nothing new.

Six disturbance conditions, each graded by λ ∈ [0, 1] against a frozen ceiling (Lesson 7's own values
for four of them, Lesson 8's Level-1 extreme for `mass_mult`): `wind_const`, `payload`, `wind_gust`,
`lighthouse` (sensor noise), `mass_mult`, and `combined` (gust + payload together). Fifteen randomly
generated study tracks carry the headline sweep; a further 111-track `train` pool and 22-track `val`
pool, from disjoint fixed seed windows, exist purely so every RL design decision below can be made
*without* touching the tracks the eventual sweep reports on (METHODOLOGY.md §3).

The learned side is one policy family, developed in several rounds below, and (as of this lesson) two
parallel recipes that differ only in how much roll/pitch authority the policy is allowed to command —
§5 is why that split exists and which one this lesson actually reports.

## 1. Building an RL policy that can race at all

The obvious thing — take the racing-envelope policy Lesson 7 trained, add the six disturbance
conditions as domain randomization, and evaluate it — does not produce a racer. It produces a policy
that tracks a smooth open-space reference about as well as ever and completes 0–2 of 6 screening
tracks. Diagnosing why took four rebuilds, and the diagnosis is the actual content of this section —
not "we trained longer," but successive elimination of confounds that had nothing to do with learned
vs model-based control at all.

**A precision gap, ruled out as the whole story.** Flying the open-space policy and M1 on the same
tracks at the same margins: M1 completes with under 9.3 cm of deviation while the RL policy crashes
4–11× over the available margin. Checking whether this was a train/eval distribution mismatch (flying
the checkpoint inside its own training range, not at the nominal λ = 0 it was screened at) changed
nothing. The reference distribution itself was the suspect: every training episode is a smooth,
randomly-wandering open-space curve (`ChainedPolyTrajectory.random`) — nothing in it ever asks the
policy to thread a narrow, oriented 0.4 m opening with 3.5–5 cm of margin, which is the literal skill
gate racing demands.

**A control-authority gap, checked and ruled out.** Every RL policy in this course, going back to
Lesson 1, has trained under a ±0.7 rad roll/pitch cap inherited from the vendored recipe, while the MPC
family plans with the full ±1.0 rad bound. Widening it to match M1 (one seed, everything else held
equal) plateaued in the same training band and scored a wash against its 0.7 rad twin (8/24 gates vs
10/24). Authority was not the bottleneck *at that budget* — a finding this lesson returns to and
partially reverses in §5, once the recipe could actually race.

**The real gap: training had no gates.** The open-space environment's only failure conditions are the
floor and gross divergence — no gate geometry, no contact, anywhere in training. Three rebuilds later
established viability:

- **v1** re-origined the dev-track references so training started airborne (a literal ground start
  would trip the open-space floor-crash rule). It trained cleanly by every optimizer diagnostic and
  scored **0 of 6** on the screening set, every failure a gate-1 contact at t = 1.2–1.7 s — uniform
  enough to be a train/eval gap rather than generic imprecision. M1, which re-solves from an explicit
  model at every step rather than generalizing from a training distribution, flies the same ground
  launch on three of those tracks with under 9 cm of deviation.
- **v2** removed the gap at its source: real ground-start references (`driver.build_traj`, the same
  trajectory the eval harness constructs) and a floor rule matched to the eval harness's own divergence
  check. Also **0 of 6**, but the failure mode changed — the policy now *swerves around* gate 1 rather
  than crashing into it, missing by 0.4–0.8 m with no contact. That distinction mattered: training had
  never penalized a missed gate at all, only contact and gross divergence, and a contact forfeits on
  the order of 200 units of remaining return against roughly 0.3 for a clean miss. A policy under
  pressure will find the two-orders-of-magnitude cheaper way out every time.
- **v3** closed that loophole — terminate on *any* missed gate, using the eval harness's own
  gate-crossing rule so training and evaluation share one definition of failure, with modest penalties
  (−3.5 for a miss or contact) rather than the vendored −5.0. It learned its own three training tracks
  but transferred to only 1 of 6 screening tracks: viable, but a 3-track training pool is not enough to
  generalize from.
- **v4** attacked pool size and training length together: 111 additional training tracks and a
  separate 22-track validation pool, both from fixed seed windows disjoint from the study tracks, 16M
  steps. A viability bar was written down before this checkpoint was flown: **≥11 of 22 validation
  tracks viable, 7–10 promising, ≤6 no change.** The final checkpoint completed **13 of 22** (68/88
  gates, RMSE 0.187 m against M1's 0.0525 m) — past the bar, and roughly 3.5× looser than M1 wherever
  it does complete. This is the recipe the rest of this lesson calls `v4`, saved at
  `saved/gate_aware_v4/`.

Train your own copy the same way (about 5.7 hours for 16M steps on this machine; the container-only
command, safe to paste into a plain host terminal):

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_gate_aware.py \
  --group robust --reason "Lesson 9: gate-aware recipe, seed 0"'
```

Evaluate it against M1 on the 22 validation tracks (a couple of minutes on 8 workers):

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/eval_pool.py \
  --role val --member v4=robust:<path to your datt_ppo_final.zip> --m1'
```

One seed only clears the viability bar; three seeds of this exact recipe complete **13, 13 and 6** of
22 tracks. A seed spread that wide, on an identical recipe, is well beyond binomial noise — one seed
never characterises this recipe, and the spread turns out to be the single most consistent finding of
this lesson, repeating itself at every later variant.

## 2. The control group that could not be built

`v4` was trained on disturbance ranges matched to 0.8× this lesson's own frozen ceilings. That makes
its robustness impossible to tell apart from "trained on the exam" — an advantage under `wind_const` at
λ = 0.6 might be real controller robustness, or it might be a policy that has simply seen exactly this
force in training. The obvious fix is a control group: the identical gate-aware recipe, but trained on
the *vendored*, unmatched disturbance ranges instead, so any advantage v4 shows beyond the control
group's is attributable to the matching, not to learning in general.

It was built and it never learned to race at all — 0 of 22 validation tracks after 16M steps, most
likely because its vendored sensor and force ranges never hand it an episode easy enough to bootstrap
from. Every rescue considered either narrowed the question (a force-only ablation) or reintroduced the
same confound in a different shape (a range-scaled dose-response still needs the vendored group to
train at all). The control group was dropped as infeasible. This is stated plainly rather than
smoothed over: **the confound is unresolved**, and the RL line in this study is labelled "trained on
ranges matched to the study's own ceilings" everywhere it is reported.

## 3. Held-out conditions: does the advantage travel?

With no control group, the test moved to the other side. Instead of asking "would an untrained policy
do worse under these exact conditions," ask "does a *trained* policy's advantage survive on conditions
it never saw at all." Four new conditions, built and never trained on or calibrated: an upward force
beyond v4's trained range (`lift`, the mirror of `payload`), a *lighter* drone (`light`, the mirror of
`mass_mult` — training only ever made the drone heavier), a wind that switches on mid-lap instead of
being constant from the start (`step_wind`), and a position sensor that freezes for longer than
anything in training (`blackout`). Each has a nearest in-distribution analogue, so the measure is a
difference of differences: how much *more* does RL lose than an MPC member, moving from its trained
condition to the unseen one, at the same λ.

The rule was written before any flight: at λ ∈ {0.5, 1.0}, a condition pair shows the **overfit
signature** if that loss is ≥20 points worse than *both* MPC comparators; it **generalises** if the gap
is under 20 points against both. Eight cells (four conditions × two λ) decide by a simple majority.

The result repeats itself, with variations, across every seed and every recipe this lesson trains:

- **`step_wind` and most of `blackout` transfer.** Losing no more than MPC does when the condition
  changes is the generalising case, and it holds here consistently.
- **A lighter drone at λ = 1 is an overfit cell in every single RL run flown** — 0.7-authority seeds
  and full-authority seeds alike. Training only ever made the drone heavier; the policy has nothing to
  generalise from in the other direction, and it shows.
- **Upward force past the trained edge is usually an overfit cell too**, and the size of the drop
  scales almost exactly with how far past training the condition pushes.
- The aggregate verdict (generalises / mixed / overfit) is **sensitive to which MPC members it's
  compared against** — capping MPC's own authority to match RL (§5) turns several "generalises"
  verdicts into "mixed," because the MPC comparator's own robustness was flattered by extra authority
  in the first place. The per-condition reading is the stable one; the one-word aggregate label is not.
- A post-hoc bootstrap over the 22 tracks resolves only the clearest cells (the lighter-drone one,
  reliably); most of the rest include zero in their 95% interval, meaning "generalises" mostly means
  "not distinguishable from no change" at this sample size, not "shown equal."

Read plainly: **v4's robustness is bounded by what it was trained on, and is not shown to be
exam-specific everywhere else.** That is a narrower claim than either "it generalises" or "it's an
exam artefact," and it is the honest one this design can support.

## 4. Equal authority: whose tuning point wins?

Every RL policy in this course commands roll/pitch at 0.7× the drone's physical limit — a convention
carried unmodified from the very first lesson. The MPC family plans with the full 1.0× bound, because
that is the value Lesson 8's whole tuning — horizon, cost weights, the corrected thrust map — was fit
under. Comparing 0.7-authority RL against 1.0-authority MPC compares two controllers each free to use a
different fraction of the same drone, and it isn't obvious which side that favours until it's measured.

**Capping MPC down, to match RL:** M1 capped to `rp_max=0.7` completes 21 of 22 validation tracks (not
22 — an earlier claim, checked on only 3 tracks, was wrong), and M1+L1's advantage under steady wind
mostly evaporates: 22/22 → 18/22 → 5/22 completions from λ = 0.5 to 1.0 once capped, against 22/22 all
the way to 22/22 uncapped. Most of what looked like "MPC dominates steady forces" turns out to be
authority, not control quality.

**Retraining RL up, to match MPC:** the alternative is to give RL the full 1.0× bound instead and
retrain. The action-magnitude term in the reward operates on the *normalized* policy output, not
physical radians, so widening authority doesn't quietly change what that term penalizes — a real
confound was checked for and ruled out before trusting the comparison. Three seeds at 16M steps gave 10
and 5 of 22 (a third not yet trained); extended to 25M steps, the same three seeds gave **10, 2 and 8**
— one seed (2/22) did not recover with more training and dragged the three-seed mean to 6.7, exactly at
the pre-registered line for "comparatively worse than the 0.7 recipe" (whose own three-seed mean is
10.7).

By the letter of that pre-registered rule, full authority should have stayed a reported sensitivity and
the 0.7 recipe should have stayed primary. It was overridden instead, deliberately and on the record:
M1's own tuning point is 1.0×, so capping it is the less faithful comparison for the model-based side
regardless of which number makes RL look better, and one weak seed dragging a three-seed mean down is
not, by itself, evidence that the recipe is worse. **As of this lesson, the full-authority recipe (25M
steps, seeds 0–2) against the uncapped MPC family is the reported RL line; the 0.7-authority recipe
(16M steps) is the sensitivity line, with its own capped-MPC comparisons already flown.** This is
recorded as a stated deviation from a pre-registered rule, not folded in silently — and the weak seed
stays in the reported spread rather than being replaced.

Train the full-authority recipe the same way, with `--group robust_full`:

```bash
docker exec -it rl_quad_traj bash -lc 'cd /workspace && export JAX_PLATFORMS=cpu && \
  /opt/venvs/main/bin/python tasks/racing/code/lesson9/train_gate_aware.py \
  --group robust_full --reason "Lesson 9: gate-aware recipe, full authority, seed 0"'
```

Held-out verdicts for the full-authority seeds echo §3 exactly: one seed generalises, the other two are
mixed, and the lighter-drone-at-λ=1 overfit cell reappears regardless of authority.

## 5. A gate that isn't where the plan says

Neither controller family in this study senses the gates directly — the MPC family follows a plan, and
RL's observation is a reference preview, not the gate's true pose. A related project's own version of
this lesson builds a policy that *does* observe the gates directly and reports that a perfect
plan-following tracker, under the kind of gate-pose randomization real racing competitions use, clears
all four gates only about a third of the time. That is exactly the condition neither of our controllers
has ever had to face, so it was built: `gate_shift` displaces the *true* gate poses (at those same
randomization amplitudes, scaled by λ) while the plan and everything either controller observes stay
nominal.

The prediction, written down first, was that no crossover should appear — a controller that cannot
sense the gate has no way to react to it moving, so the ordering should be set by tracking precision
alone. That held: MPC is modestly ahead at λ = 0.5 (81–85% retention against 59–64%), and the two
families sit within sampling noise of each other from λ = 0.75 up. The pre-registered surprise
threshold (an RL seed 20 or more points ahead of both MPC members) was not met. What decides a given
lap is whether the *original* plan's line happens to still clear the moved gate — geometry, not
controller family.

## 6. What this does not settle

1. **The training confound is unresolved, not answered.** The control group that would have separated
   "trained on matched ranges" from "learned generic robustness" could not be built. Held-out conditions
   are the substitute, and they show something narrower: robustness is bounded by the training range,
   not that it is absent everywhere else.
2. **Seed variance is the largest effect measured in this lesson, full stop.** The 0.7-authority recipe
   spans 6–13 of 22 validation tracks across three identical-recipe seeds; the full-authority recipe
   spans 2–10. Any single-seed number in this lesson should be read as one draw from that spread, never
   as the recipe's true performance.
3. **The full-authority decision is an argued deviation, not a data-driven one.** The numbers alone
   still favour the 0.7-authority recipe; it was overridden on a fairness argument about which
   controller's tuning point to respect. A reader who disagrees with that argument should read this
   lesson's RL line as the 0.7-authority recipe instead — the sensitivity comparisons for that reading
   already exist.
4. **n = 22 throughout the validation pool** means single retention or difference-in-differences cells
   carry roughly ±10–20 points of sampling noise; only the clearest signals (the lighter-drone overfit
   cell) survive a bootstrap check.
5. **The actual research question has not been run.** Everything above is validation-pool development
   and screening, deliberately kept separate from the 15 study tracks so that no RL design decision
   could leak into the tracks the real sweep reports on. The six-condition × λ-grid sweep, on the ten
   study tracks no RL decision has touched, is the next step and is still open.
6. **The gate-displacement finding is a first pass**, on the validation pool only, against the capped
   MPC comparators, not yet against the study tracks or the full authority/uncapped pairing.

---

## 🛠️ Before you move on

1. **Fly `gate_aware_v4` yourself.** `eval_pool.py --role val --m1 --member v4=robust:saved/gate_aware_v4/ckpt/datt_ppo_final.zip` and check your number against the pre-registered bar (§1) before reading further.
2. **Read one held-out summary end to end.** `results/eval/val_heldout/summary.txt` (or any `val_heldout_*` sibling) has the raw table, the retention table, the difference-in-differences table and the tally. Recompute the verdict for one condition pair by hand from the raw retention numbers, and say in one sentence why the 20-point threshold, not 15, was the one fixed in advance.
3. **Argue the other side of §4.** The lesson reports full authority as primary despite the numbers favouring 0.7-authority. Write the paragraph that would justify the opposite call — keeping 0.7-authority as primary because the rule said so — and say what would have to be true about M1's tuning for your paragraph to be the right one instead.
4. **Design the missing control group.** §2 could not build one. Propose a design that would isolate "trained on matched ranges" from "learned generic robustness" without requiring the control policy to solve the exact same hard bootstrapping problem v4 did, and name the number that would tell you it worked before you spend the compute.
5. **Predict the sweep.** Before it runs: on which of the six conditions do you expect the crossover to sit at the *lowest* λ, and on which study track geometry (if any) do you expect it to differ most from the validation-pool preview in this lesson? Write both down now, dated, so you can be wrong later on the record.

**Back to:** [Lesson 8 — Closing the gap](08-closing-the-gap.md)
