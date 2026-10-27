# Balanced PPO continuation

The requested objective is **balance growth and survival**. V8's original
PPO retention gate remains failed (7 deaths versus 2); its reports are immutable.
The accepted demo remains v3, with v8 available separately as experimental.

**Round completed; no replacement was promoted.** The best split-only
checkpoint reached **65.74 mean final mass / 1 death**, versus **58.21 / 1**
for v8 on the same eight development focus cases. The same-seat v3 control
reached **73.58 / 0**. Growth retention is **89.34%**, below the unchanged
90% gate. Try the clearly marked experimental checkpoint at
**[localhost:8081](http://localhost:8081/)**; accepted v3 stays on port 8080.
All 97 regression tests passed. No fresh final seed reserves were consumed.

## Causal audit

`python -m src.audit_ppo_survival` replays all four fatal candidate-agent cases
from that report. Each original replay reproduces every player's final mass,
captures, deaths and splits. Replacing only the focus controller with v3, or
diagnostically disabling its split bit, reproduces all seven focus deaths at
the same frames. Every death occurred before the focus agent's first split,
with mass below 36 and splitting physically unavailable. Two cases have
identical focus actions for the entire match under all three variants.

This rules out the focus agents' own split choices as the direct cause of
these seven deaths. It does not rule out effects from other agents' splits.
The old mixed-cohort comparison uses different seats for candidate and
baseline; it cannot by itself isolate the causal effect of their controllers.
These consumed seeds are now development diagnostics, never fresh tests.
Raw events and comparison rows: `outputs/balanced_v9/ppo_death_audit.json`.

## Training hypothesis

Several losses involve slow escape or corner traps. Freezing movement and
training only the split head cannot correct movement while splitting is
unavailable. Continue the PPO continuous actor and critic, keeping the v8
split head bit-identical initially. Execution clips actuator bounds and masks
physically impossible splits only. No threat, wall or target rule may rewrite
movement or veto a feasible split.

`src.train_balanced_ppo` uses a 20-second exponential reward discount horizon,
instead of the old trainer's gamma .99 at 30 Hz (about 3.3 seconds), and rewards
log mass gain plus .10 per piece capture minus 3 per full death. Respawn mass
cannot earn positive growth reward. Two learners face two legacy/neural
opponents and three heuristics. 35% of training episodes include a smaller
agent near a corner with an approaching predator. Curriculum placement is
training-only; there is no inference-time escape rule. The initial run uses
120,000 learner transitions, four workers, and checkpoints every 15 updates.
Checkpoints record exact sources, base weights and actual episode seeds.

Pilot: 512 transitions verified finite weights, changed movement weights and
an unchanged split head. This verifies plumbing, not improved performance.

## Prospective evaluation (freeze before inspecting results)

- Training seed family: 6,000,001 + worker*10,000 + episode index.
- Development matches: 6,400,001 + 37*i; consumed freely for selection.
- Fresh final paired matches: 6,500,001 + 37*i, i=0..11.
- Further encounter/league reserve families: 6,600,001 and 6,700,001.
- Each paired seed starts three identical 60-second worlds: four v3 PPO and
  three heuristics as control; then replace only the first focus PPO seat;
  then replace only the second focus PPO seat. All other controllers, seeds,
  positions and masses stay fixed. Seats are shuffled once per seed.
- Compare the two focus cases with their exact control seats. Bootstrap by
  seed (12 independent clusters), not by treating both seats as independent.
- Engineering gate: mean final mass >=90% of paired control, no more than
  two extra full deaths over 24 focus cases, zero physical violations, zero
  nonphysical action overrides. Report captures, splits and confidence
  intervals separately. Passing is a retention gate, not statistical proof
  of better general performance.
- Freeze checkpoint hash, evaluation code and protocol before final seeds.
  Select using development seeds only. Do not promote a failed checkpoint.

This follow-up controls seats more carefully; it does not retroactively
change v8's original failed gate. Preserve accepted weights and reports.
New fully learned movement uses a distinct checkpoint control enum so an
old loader fails explicitly instead of silently restoring steering rules.

## Development results and next hypothesis

The first movement continuation completed 120,832 learner transitions.
Its 30,720-transition checkpoint failed the four-seed, eight-focus-case
development comparison: mean final mass **32.76 versus 73.58**, full deaths
**8 versus 0**, wall-contact frames 603 versus 157. It executed 15 splits
with two recent-projectile captures. Physical violations and nonphysical
action overrides were both zero. None of this is fresh acceptance evidence;
no final seeds were consumed. Raw data: `outputs/balanced_v9/dev_30720.json`.

This exposes substantial dependence on the old controller's steering
assistance. A longer reward horizon alone has not established a successful
replacement. The next hypothesis is movement distillation before further
PPO training, rather than asking PPO to relearn basic navigation from a
poor starting controller.

`src.distill_balanced_movement` fits continuous actions from the v3 neural
controller plus its movement assistance, using the existing v2 demonstration
observations (79,800 training and 21,000 held-out rows). Assistance is a
training-label source only; split labels are excluded, and the split head
is unchanged. It weights overridden training examples fourfold. Sixty
epochs achieved held-out weighted MSE 0.03399; this is a fitting metric,
not match performance. Exact dataset/teacher hashes and original dataset
seed manifest are embedded in the checkpoint.

The v10 continuation starts from that warm-start and uses 240,000 further
learner transitions, the same growth/death reward and 20-second discount
horizon, with new training seeds **6,200,001 + worker*10,000 + episode**.
Development seeds and all predeclared final seed reserves stay unchanged.
The warm-start and later checkpoints must pass match evaluation before
any deployment. The regression suite currently has **94 passing tests**.

| Development checkpoint (same 8 focus cases) | Mean final mass | Full deaths | Wall-contact frames |
| --- | ---: | ---: | ---: |
| Same-seat v3 control | 73.58 | 0 | 157 |
| V9, 30,720 transitions | 32.76 | 8 | 603 |
| V9, 120,832 transitions | 49.32 | 2 | 917 |
| V10 distilled warm-start | 56.64 | 2 | 509 |

These are development comparisons, not independent unseen tests. Neither
the completed v9 run nor the warm-start meets the 90% mass-retention bar
(66.22 on these cases). More training improved v9 survival; distillation
improved the starting controller's development mass and wall contacts.
Neither result establishes a successful deployment replacement.

Audit scope caveat for the initial development JSON format: the
`nonphysical_overrides` counter compares emitted commands with raw neural
commands **only for BalancedPPO**. Its zero for historical assisted
controllers is unmeasured, not evidence of no steering assistance. Those
controllers explicitly retain it. Growth, death, capture, wall-contact and
physical-violation measurements cover every focus case. Final reports must
mark unmeasured control counters unavailable rather than report zero.

## Separate split-learning hypothesis

V10's 61,440-transition checkpoint regressed to 46.02 mass / 3 deaths.
V8 with its original movement assistance reached 58.21 / 1 on the same
cases; it also misses the growth threshold. These remain rejected development
results. The v10 movement run completed 241,664 transitions and failed its
final development comparison, reported below.

A separate controlled branch, `src.train_balanced_splits`, keeps v8's
continuous neural weights and historical movement assistance frozen and
trains **only the split head and critic** with the same 20-second discount
and balanced growth/death reward. This isolates split learning from the
regressing movement replacement. It uses 120,000 learner transitions and
new training seeds **6,300,001 + worker*10,000 + episode**. Actions are still
subject only to physical split limits; movement assistance cannot change
the requested split bit. Explicit movement assistance is permitted in this
branch, as in v8; the user requested learned split risk rather than removal
of every movement rule.

The stochastic Bernoulli policy is centered on the existing .65 deterministic
decision boundary, with temperature .25. Unavailable actions contribute zero
to a policy loss averaged over all transitions. A short initial trial used
per-minibatch eligible-action normalization; it overweighted rare feasible
actions and was aborted. Its source/configuration/log are preserved under
`outputs/balanced_split_v11/aborted_sparse_trial`, never used for acceptance.
The 512-transition plumbing pilot had no eligible splits: its head remained
unchanged, continuous actor stayed frozen and critic updated, as expected.

The evaluator now independently compares the emitted split bit with the
raw neural request for **all** PPO controllers. It measures full command
equality for the fully learned movement variant, and reports `null` for
unmeasured historical movement overrides. Both candidate variants must
retain the neural split bit whenever physically feasible; fully learned
movement must additionally retain both continuous commands. Growth and
death thresholds remain unchanged. This distinguishes the requested split
contract from the optional movement-replacement experiment.

Select checkpoints using development cases only. If a newly trained split
head passes development retention, test the selected frozen checkpoint on
the reserved fresh paired seeds and reactive encounter family. A head that
simply stops splitting must not be presented as improved split capability;
the encounter gate also requires useful initial splits outside the old guard.

V10's final 241,664-transition checkpoint failed at **42.33 mass / 4 deaths**
on the same eight development cases, versus 73.58 / 0. Its raw action contract
passed; better action provenance is not evidence of better gameplay.

The split-only branch encountered float32 Bernoulli KL saturation. The run
was stopped; the last pre-overflow 30,720-transition checkpoint and all its
preceding losses were verified finite. Post-overflow local checkpoints are
explicitly excluded from continuation/acceptance. The numerical repair
uses stable logit-space cross entropy, finite-loss/gradient checks, and two
extreme-logit regression tests. All **97 tests passed**. Stable continuation
fits 90,112 additional transitions from those weights, resetting the
optimizer and using fresh seeds **6,340,001 + worker*10,000 + episode**.
Sources, bad-trial logs and the valid continuation lineage are preserved.

A cheap diagnostic on the consumed v8 observations confirms that the early
split checkpoint is not an all-hold head: 68 proposals, 42.7% initial recall
and 22 known useful initial splits outside the old guard. Precision is only
55.9%, below v8's 73.9%; these classification results do not prove improved
match outcomes or fresh generalization.

## Completed split-only comparison

| Checkpoint, same 8 development focus cases | Mean final mass | Full deaths | Piece captures | Splits |
| --- | ---: | ---: | ---: | ---: |
| Guarded v3, same-seat control | 73.58 | 0 | 13 | 1 |
| Physical-only v8 | 58.21 | 1 | 8 | 22 |
| Split PPO, 30,720 transitions | **65.74** | 1 | 12 | 24 |
| Split PPO, 120,832 valid transitions total | 58.82 | 1 | 8 | 23 |

More training did not improve this checkpoint. The best early split head is
selected for an **experimental preview only**, with no more development
deaths than v8. The mean mass gain versus v8 is +7.53, but its four-seed
bootstrap interval is [-12.89, +33.06], which includes zero. Selection and
testing on these consumed development seeds do not establish generalization.
There were zero recent-projectile captures for both v11 checkpoints in this
comparison: do not claim improved split attacks. Their zero physical
violations and zero changed feasible split requests establish the execution
contract, not safety or combat quality. Historical movement override counts
are unavailable, rather than zero.

The stable continuation's weights and losses are all finite, its continuous
actor is bit-identical to v8, and its split head changed. The original 30,720
checkpoint predates the numerical overflow. Post-overflow trial weights are
excluded from selection. See `outputs/balanced_split_v11/verification.json`,
`selection.json`, `control_world_audit.json`, and the raw `dev_*.json` reports.
The control-world audit confirms identical initial observations and every
control player's gameplay outcomes across all development report versions;
adding raw-action instrumentation did not change the comparison worlds.

The experimental manifest pins the selected PPO, unchanged v8 diffusion and
critic hashes. Its banner says **89.3% growth retention; 90% required; unseen
validation pending**. Live readiness, model hashes, warning and JPEG stream
were verified; port 8080 still serves the exact accepted-v3 model hashes.

```bash
python scripts/play_agar_web.py --port 8081 \
  --policy-dir outputs/balanced_split_v11/experimental

python -m src.evaluate_balanced_ppo \
  --candidate outputs/balanced_split_v11/training/ppo_30720.pt \
  --output /tmp/reproduced_development.json \
  --seeds 6400001 6400038 6400075 6400112 --workers 4
```

## Next useful work and approaches to avoid

- Improve temporal credit assignment before simply extending the same run.
  Gamma gives a 20-second reward discount horizon, but lambda .98 gives
  direct GAE traces about 1.52 seconds; 256-step rollouts span 8.53 seconds.
  Longer consequences currently depend heavily on the learned critic.
  Test longer rollouts and traces against this frozen development benchmark,
  with new training seeds and no change to physical-only split execution.
- Consider longer full-match paired split/hold outcomes for correcting the
  head's false positives. Initial two-second encounter labels alone do not
  establish long-match value. Keep useful split opportunities in the audit
  so an all-hold policy cannot masquerade as better learned splitting.
- Do not remove movement assistance and declare the raw actor improved:
  both full-neural movement trials regressed. Distillation helped the initial
  controller, but subsequent PPO updates did not preserve that improvement.
- Do not change the 90% threshold to turn 89.34% into a pass, promote the
  numerically invalid trial, relabel consumed seeds unseen, or confuse zero
  execution-contract violations with zero risky actions.
- Keep fresh families 6,500,001 (paired), 6,600,001 (encounters) and 6,700,001
  (league) reserved until a candidate clears development. Preserve v3 and
  the original failed-v8 reports. The current result is a measured
  development improvement over v8, not an accepted or proven general gain.
