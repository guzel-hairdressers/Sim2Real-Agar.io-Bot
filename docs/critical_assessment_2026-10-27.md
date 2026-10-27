# Critical assessment: playable PPO and diffusion agents

Assessed 2026-10-27 at commit `ca3f24d4ed7f62b860a2520e90dcdae65a21987b`. Assessment only: no policy, environment, weights, or server changes; no commit made. Existing untracked experiment outputs were left alone.

## Verdict

There are reproducible correctness bugs, insufficient observations, mismatched training/deployment conditions, and outdated evaluation claims. A person beating the agents does not by itself establish an algorithm failure. The concrete failures below do establish that the current implementation is not a sound test of PPO versus diffusion quality.

Keep the working environment, neural implementations, and useful heuristic infrastructure. Repair correctness and establish an honest benchmark before adding model complexity or spending more training compute. Removing the latest reflex layer outright is not supported by the short ablation.

## Confirmed issues

### 1. Predator evasion in the latest shared reflex cannot execute

`envs/agar_env.py:1015` encodes threat mass ratio as `clip(threat_mass / own_mass, 1, 5) / 5`, bounded by 1. `src/tactical_combat_reflex.py:46` requires that encoded ratio to be at least 1.15. This is impossible for an environment-produced observation.

Reproduced with a 20 kg agent at (0, 0), facing a 40 kg predator at (1.5, 0), with no food or viruses. Encoded threat ratio was 0.4; passing raw action `[0.5, 0, 0]` through the reflex produced `[0.9, 0, 0]`: forward motion toward the predator, no evasive turn. This is a failed tactical decision, not proof that every such encounter ends in death.

### 2. Splitting ignores whether the resulting pieces can eat the target

The reflex splits whenever an aligned target is close and splitting is available (`src/tactical_combat_reflex.py:97`). It does not check post-split piece mass against the actual per-piece predation rules. Its prey-ratio condition also reads an encoded own/prey ratio as though smaller were always more edible.

Reproduced a split command from a 45 kg agent toward a 30 kg target. The launched half is 22.5 kg. It cannot eat that target; the target meets both mass-gap and ratio requirements to eat the half. Split range is also expressed as a fraction of vision radius, rather than a verified interception envelope under drag and target motion. The heuristic teacher has similar unsafe split checks.

### 3. PPO rollout probabilities and update probabilities disagree

`HybridActorCritic.get_action_and_value` computes the old log probability on the unclipped Gaussian sample, but returns only the clipped action. Training stores that clipped action, and `evaluate_actions` computes the new log probability on it (`src/train_champion_ppo.py:112-132`). Consequently, the probability ratio can differ from 1 before any optimizer update.

Using the actual champion checkpoint, seed 739, 8,192 stochastic samples of the 45-versus-30 observation above:

- 3.8696% of unchanged-policy ratios differed from 1 by more than 1e-5.
- 2.9663% already fell outside the PPO [0.8, 1.2] clipping interval.
- Maximum ratio: 870.04.

These percentages are specific to this probe, not an estimate across all training states. The invariant violation is the relevant result. Preserve raw samples for learning while clipping only the environment action, or use a consistently transformed distribution. The PPO implementation reference explains raw-action storage: [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

### 4. Reward shaping repeatedly pays for a past kill and repeatedly penalizes a past death

`infos[pid]['kills']` and `['deaths']` are cumulative counters. The champion trainer adds 4.5 every subsequent step after any kill and subtracts 3 every subsequent step after any death (`src/train_champion_ppo.py:389-398`). Its displayed kill accumulator also repeatedly sums the cumulative count. This substantially changes the intended objective and inflates the training kill log. Use event deltas and distinguish piece consumption from full player elimination.

### 5. Observations hide information needed for tactical decisions

The 38 features omit opponent velocity, opponent heading, and the arrangement/masses of individual opponent pieces. The learned policies are conditioned on a single observation; the PPO has no recurrent state, and diffusion buffers actions rather than observations.

Exact observation equality was reproduced for:

- A visible opponent moving toward versus away from the observer.
- An intact 50 kg opponent versus two 25 kg pieces with the same mass-weighted center, observed by a 35 kg agent. The intact opponent can eat the observer; neither fragment can.

History or recurrence can help estimate motion. Piece-level observations are needed to distinguish the second case reliably. These must remain within the same visibility restrictions for every competitor.

### 6. The documented diffusion critic is a learned heuristic score, not demonstrated outcome prediction

The README reproduction path uses `src/train_pure_diffusion_champion.py:128-164`: targets combine current mass, hand-written threat/wall/split/food terms, and action smoothness. They do not use observed future returns or simulate the candidate trajectory. `src/train_boundary_diffusion.py:75-113` similarly calls its synthetic targets Monte Carlo values despite not computing rollout returns. Its first-action targets are fed through a critic that repeats that action across the horizon.

Best-of-N selection is implemented, but calling it long-horizon value guidance overstates what these targets establish. A longer action sequence alone does not prove foresight. Train a critic on actual returns with appropriate out-of-distribution checks, or use explicitly described short simulated rollouts. Keep a simple behavioral-cloning baseline to establish whether diffusion and candidate ranking help.

Checkpoint files have no training manifest, so the exact provenance of the current critic cannot be certified from the state dictionary alone. This is a verified defect in the provided training/reproduction paths, not a claim that binary inspection reveals its historical training data.

### 7. Training, historical tournaments, and playable deployment differ

- Champion PPO training worker: 6 players, 60 initial pellets, 200 kg world mass, dt=0.12, randomized arena 11.5–17 m.
- Web deployment: 7 players, 180 initial pellets, 550 kg world mass, dt=1/30, 18 m arena.
- README tournament: 6 players, 100 initial pellets, 14 m, default dt=0.1; different diffusion replanning/sampling settings.

The same 16 action steps therefore cover different physical durations. Some dynamics use fixed per-step factors and rewards instead of time-scaled quantities, so changing dt changes more than display smoothness. Train/evaluate the actual deployment configuration, then introduce measured domain variation.

### 8. The headline benchmark does not validate today's agents

`src/benchmark_pure_tournament.py:79-81` and the expert-harvesting script instantiate the older `SuperiorPPOAgent`. The current checkpoint has hybrid continuous/Bernoulli heads and does not load into that architecture. Strict loading reproduced missing/unexpected keys and a 2-versus-3 log-standard-deviation mismatch. The wrapper catches the error and continues with partially initialized parameters.

The saved headline CSV predates the new hybrid checkpoint and the final reflex commit. It may describe an older experiment; it must not be presented as current evidence. Current diffusion inference also defaults to the reflex, despite this script calling itself pure neural.

Other evaluation weaknesses: fixed family seating; correlated players within each match treated as individual samples for intervals/tests; no checkpoint/configuration hashes; global random yaw means `env.reset(seed=x)` alone does not reproduce observations; diffusion's seed does not seed its Torch sampler. A circling proxy based on mean absolute steering is not a measurement of actual circular trajectories.

### 9. Existing tests miss these failures

30 unittest-discovered tests passed. The 5 plain-function virus tests were then invoked separately and passed. Pytest is absent from the shared venv; no packages were installed.

The existing diffusion tests emitted a critic checkpoint shape mismatch while still passing. There are no existing regression checks for the shared reflex's normalized-threat contract or PPO's unchanged-policy ratio invariant. The environment's sustained-winding branch requires 20 actions, while its action deque holds only 5, making that branch unreachable too.

## Short ablation: keep useful evidence, avoid overstating it

Six diagnostic matches used the actual web configuration, 900 steps = 30 simulated seconds each. Three initial seeds: 72701, 72719, 72733. Repeated with the reflex enabled and bypassed. NumPy and Torch were explicitly reseeded per match; policy-family seating rotated by two positions per seed. Each match had 2 PPO, 2 diffusion, and 3 heuristic agents; a survivor heuristic substituted for the human. Diffusion used the web settings: 16-step plan, execution horizon 3, 5 DDIM steps, 4 candidates. CPU inference with one Torch thread. Total runtime: 47.91 s.

| Mode | PPO piece kills / full deaths | Diffusion piece kills / full deaths | Heuristic piece kills / full deaths |
| --- | ---: | ---: | ---: |
| Current reflex | 0 / 1 | 2 / 1 | 0 / 0 |
| Raw networks | 2 / 5 | 3 / 5 | 9 / 0 |

With the reflex enabled, it changed all 10,800 neural action vectors. It supplied all three controls without consulting the neural action on 582 observations; elsewhere it still replaced thrust/split and often mixed steering with a food rule. There were 202 nearby-threat observations and zero evasion-branch activations.

This small experiment supports preserving useful behavior while repairing the layer. It does not establish a human win rate, a final policy ranking, or a statistically reliable improvement. Match length, unequal family population, and a heuristic human substitute limit interpretation. Raw records are in `outputs/critical_assessment_2026-10-27/diagnostic_baseline.json`.

## What is worth retaining

- Separate actor/critic networks and a discrete split head are reasonable foundations.
- A true neural diffusion denoiser, DDIM sampling, action buffering, and candidate ranking exist; these are not just labels over a scripted bot.
- Mass conservation, speed inversion, split pieces, viruses, and heuristic profiles provide a useful testbed. Existing physics tests pass within their scope.
- The shared reflex provides a behavior baseline with better survival in this short ablation, despite its serious bugs. Retain explicit raw-policy versus assisted-policy comparisons.

## Proposed repair and commit checkpoints

1. **Evaluation and environment contract.** Deterministic RNGs; strict versioned checkpoint loading; shared train/play configuration and physical time; tests for normalized features, split eligibility, approaching threats, corners, fragments, and reset boundaries. Record checkpoint hashes and actual outcomes. Fail visibly on missing/incompatible weights.
2. **PPO correctness.** Consistent action probabilities; event-delta rewards; deliberate handling of time truncation and respawn boundaries. Add motion history or recurrence plus visible piece-level features. Retain only justified safety overrides and make interventions measurable.
3. **Train and test PPO.** Use the deployment mechanics and a diverse opponent league including old checkpoints and scripted exploiters. Randomize starting seats and relevant domains. Select checkpoints on separate validation seeds; reserve final test seeds. Measure survival, mass gain, piece kills, player eliminations, attack success, unsafe split rate, wall traps, and runtime. Include pursuit, retreat, baited split, and virus/corner exploit tests.
4. **Rebuild the diffusion comparison.** Collect corrected demonstrations without chunks crossing resets/respawns; train/evaluate on held-out episodes. Compare behavioral cloning, unguided diffusion, and outcome-guided diffusion under the same observations, opponents, and compute constraints. Do not assume the more complex method wins.
5. **Commit measured checkpoints.** Separate commits for evaluation/correctness, PPO retraining/results, and diffusion/results. Include reproducible commands, hashes, and before/after outcomes. No claim of unbeatable play; require exploit regression results and improvement across held-out matches, not just one favorable video.

## Original robot-navigation code is a separate, unvalidated claim

The repository still includes the original robot environment. `src/evaluate_robustness.py` evaluates `LightweightHeuristicPolicy`, not trained domain-randomized PPO versus nominal PPO as its docstring suggests. The delay queues inject delay+1 actions even when configured as zero delay. The NumPy PPO implementation in `src/train.py` uses `ratio == surr1` to gate its actor gradient, which does not implement the clipped objective derivative. These files are not used by the playable Agar arena. Neither the arena comparisons nor this heuristic robustness sweep demonstrate real robot sim-to-real transfer.

## Audited checkpoint identities

SHA-256, from the files loaded during this assessment:

```text
agar_ppo_champion.pt
a5a4e685ac1ffd6531544f799cc5aad7601a7288b51d42326078b6d5d677df4a
agar_diffusion_champion.pt
47b41d669255142592c8e787f065d32b11484bbd83d6f2fabd8f5d7485b81bff
agar_diffusion_critic_champion.pt
1c71c9e36b036e64b0b5ca8d90cda37f58dce0657c1c2b13474d3e6caa24bab6
```
