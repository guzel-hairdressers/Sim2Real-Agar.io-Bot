# Current-model benchmark

This benchmark asks what the deployed agents do in the same arena. It contains **no previous-version policies**. The models are the accepted split-combat checkpoints from `3e105b9`, unchanged during this evaluation.

![Growth, death-free runs, captures and inference latency](media/showcase/benchmarks.png)

## Mixed-agent matches

Twelve independent matches, each 1,800 steps / 60 simulator seconds. Roster: **2 current PPO + 2 current diffusion + 3 heuristics** (apex, hunter, survivor). Seats are shuffled from each seed. Agents do not cooperate. All seven agents act at every 1/30-second timestep; no control observations are skipped for recording.

| Measurement | PPO | Diffusion | Heuristic |
| --- | ---: | ---: | ---: |
| Individual agent runs | 24 | 24 | 36 |
| Initial mass, mean | 14.02 | 13.88 | 14.00 |
| Final mass, mean | **51.74** | **47.57** | 31.49 |
| 95% match-bootstrap interval for final mass | 45.56–57.57 | 40.77–55.51 | 27.68–36.18 |
| Agent runs with zero full deaths | **20/24 (83.3%)** | **21/24 (87.5%)** | 22/36 (61.1%) |
| Full deaths, including repeated deaths after respawn | 6 | 5 | 20 |
| Opponent pieces captured, total | 17 | 21 | 15 |
| Opponent pieces captured per agent-minute | 0.708 | 0.875 | 0.417 |
| Executed splits, total | 1 | 1 | 12 |
| Recent-projectile piece captures, total | 1 | 1 | 1 |
| Matches whose final-mass leader used this policy | 8/12 | 3/12 | 1/12 |
| Policy inference, mean | 0.279 ms | 0.525 ms | 0.019 ms |
| Mean of per-match inference p95 values | 0.442 ms | 1.521 ms | 0.028 ms |

**Definitions and interpretation:**

- A piece capture consumes one opponent piece; it need not eliminate that opponent's remaining pieces. A full death means losing all pieces. Automatic respawns are counted, not discarded.
- A zero-death run spans the whole 60-second match. It does not mean the agent took no damage or lost no pieces.
- Captures per agent-minute divide by the full simulation duration, including respawns: 24 PPO minutes, 24 diffusion minutes, 36 heuristic minutes.
- Final-mass intervals resample the **12 independent match-level group means**, not individual frames or correlated agents. The chart's shaded time-series bands use the same match-level resampling. They are pointwise intervals, not simultaneous confidence bands.
- One individual is the final-mass leader in each match. The policy groups have unequal slot counts (2/2/3); the leader row is descriptive, not an equal-size team tournament win rate. The agents are not teams.
- Mass is a simulator quantity. Growth, capture and survival are different objectives; no single score is presented as general intelligence or real-world navigation accuracy.

The arena is 18 × 18 units, with 180 initial food pellets, 7 virus hazards, a 550-unit world mass budget, up to 4 pieces per player, and 70-dimensional local observations. [The exact configuration](../envs/arena_config.py) is also saved in the report. PPO and diffusion use the same safety layer; heuristic agents use their own rules. These are deployed-system measurements, not an ablation of pure neural control.

## Hardware and timing

Measured on this user's **macOS ARM64 Mac**, CPU inference, PyTorch 2.8.0, NumPy 1.26.4, Python 3.9.6, with two evaluation workers and one Torch thread per worker. Other processes, including the local preview, may run concurrently. Inference timings measure `policy.predict`, including safety checks; they exclude rendering and simulator stepping. The diffusion mean includes inexpensive buffered-action frames and more expensive replanning frames.

Mean processing time for a complete seven-agent simulation step was **12.73 ms**, averaged across the twelve matches. Evaluation took **140.39 seconds** of wall time with two workers. Neither number is a measured browser playback frame rate. The physics timestep is fixed at 30 Hz regardless of evaluation speed; new demonstration footage is sampled at 15 frames per simulator second and encoded at 30 fps for 2× playback. Each full video runs for 60 seconds and covers 120 simulator seconds. The recordings are separate from this benchmark.

## Separate split-opportunity test

The previously recorded physical test uses the **same current model hashes** and is shown as an absolute capability measurement. It is distinct from the mixed-agent matches above.

| Measurement | PPO | Diffusion |
| --- | ---: | ---: |
| Valid randomized encounters | 996 | 996 |
| Known beneficial split opportunities | 125 | 126 |
| Beneficial opportunities captured with a projectile | **120/125 (96.0%)** | **119/126 (94.4%)** |
| Encounters with executed attacks | 186 | 175 |
| Encounters with projectile target capture | 184 | 172 |
| Executed splits violating the current guard | 0 | 0 |

A beneficial opportunity requires a physical split rollout that captures with a launched piece at least three frames earlier than holding, without immediate attacker damage. Four initial allocations above the curriculum mass budget were excluded. Evaluation tracks actual consumption events from recently launched pieces, not ordinary body-contact kills. Zero guard violations establishes consistency with that guard; it is not a guarantee that every dangerous attack is prevented. See [the test rows](../outputs/split_combat_v3/combat_acceptance.json) and [full split methodology](split_combat_results_2026-10-27.md).

## Reproduce

From the repository root, using the installed project environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m src.benchmark_showcase \
  --workers 2 --output outputs/showcase_recheck/benchmark.json
```

The default seeds are `4100001 + 37*i`, for `i=0..11`: `4100001, 4100038, 4100075, 4100112, 4100149, 4100186, 4100223, 4100260, 4100297, 4100334, 4100371, 4100408`. They were fresh at evaluation and are now public. Use a new seed family for future model development; do not tune against these and continue calling them unseen.

The harness checks the accepted deployment hashes before loading. The report records all player outcomes, per-second mass curves, configuration, model/source hashes, and timing context. The first seed also emits a spectator trace for media rendering. There is no hand-controlled player or older model in this benchmark. Cross-hardware bit-for-bit replay and identical timings are not guaranteed.

For an exact check of the shipped record, use `python scripts/verify_showcase.py`. Rerunning inference is a new measurement; keep the shipped report for provenance.

## Limits

This is a small, fixed-environment simulator evaluation. It does not test other physics, learned teamwork, adversarial human play, new map families, or physical robots. Close predator and corner traps remain known failures. Splits are useful in the controlled test but rare in mixed matches. The front-page demo illustrates behavior, not a representative sample of all failure modes. No benchmark here establishes sim-to-real transfer or superiority to external state-of-the-art methods.

Raw artifacts: [all match rows](../outputs/showcase_v1/benchmark.json), [recorded trace](../outputs/showcase_v1/demo_trace.json.gz), [media hashes](../outputs/showcase_v1/media_manifest.json), [verification](../outputs/showcase_v1/verification.json). Historical comparisons are retained in the [checkpoint history](checkpoint_history_2026-10-27.md), separate from these current-capability measurements.
