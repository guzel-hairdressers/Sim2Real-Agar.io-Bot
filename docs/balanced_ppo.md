# Balanced PPO continuation

The requested objective is **balance growth and survival**. V8's original
PPO retention gate remains failed (7 deaths versus 2); its reports are immutable.
The accepted demo remains v3, with v8 available separately as experimental.

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
