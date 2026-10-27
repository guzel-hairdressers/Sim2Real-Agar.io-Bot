# Learned split risk with physical constraints

The new execution path lets the **model** decide whether a risky split is worthwhile. It enforces only the simulator's `can_split` flag: enough mass in an eligible piece, remaining piece capacity, and the split cooldown. Nearby predators, walls, viruses, target mass, attack angle and estimated interception success cannot veto a model split request.

This does **not** remove the existing steering/thrust interventions near immediate hazards. Those affect movement; the requested split bit is restored afterward and cannot be overwritten by those rules. Thus this is a learned split decision with physical constraints, rather than a claim that every part of movement is free of hand-written logic.

## Learning signal

Each training scene is rolled out twice with the same starting state and controlled random seeds: split immediately, or hold. Opponents remain active and reactive. Subsequent split commands are held during these paired trials so the measured difference isolates the initial decision.

Each branch runs **60 steps / 2 seconds**, versus the previous split curriculum's 18 steps / 0.6 seconds. It measures the actor's final mass, mean trajectory mass, piece captures and full deaths. The utility is:

```text
0.6 × normalized final-mass gain
+ 0.4 × normalized mean trajectory-mass gain
+ 0.06 × opponent pieces captured
− 2.0 × full deaths
```

Mass gains are normalized by the actor's initial mass. A positive label requires split utility to exceed hold utility by 0.03. This reward function defines the training objective; it is not an execution rule about which nearby objects permit a split. The deployed model receives its usual local numeric observation, without access to counterfactual simulator outcomes.

The curriculum includes varied masses and relative velocities, third-party hunters, off-axis targets, walls, viruses, distant targets, and already-split states. Invalid initial world-mass allocations are excluded. Initial physical inability to split is a valid negative; no heuristic threat-distance or angle rule labels the remaining cases.

## Training and model selection

- 4,000 requested initial scenes, seeds `4800001..4804000`; entire scenes are divided 80/20 into fitting and development sets.
- A separate learned binary head per algorithm; PPO navigation/value weights, diffusion movement weights, and the return critic remain frozen.
- **240 epochs**, versus the previous 80. Longer training alone is not presumed better: the selected checkpoint maximizes measured development action advantage, breaking ties by weighted prediction loss.
- Minibatches preserve the observed case frequency. The pilot's artificial positive/negative balancing made useful splits appear too common, encouraging excessive actions. Loss weights account for the magnitude of actual outcome differences.
- No final-test outcomes are used for fitting or model selection.

The small pilot was rejected: removing risk vetoes while using an insufficiently trained head caused extra splits and worse utility. Its weights are not the accepted deployment. The larger curriculum adds already-split states and uses the observed class prior to address that failure without restoring nonphysical action vetoes.

## Full-match corrective learning

The first longer run (`learned_risk_v4`) was **not accepted**. On 187 valid development scenes its mean utility difference from guarded v3 was −0.0138 for PPO and −0.0321 for diffusion. In a small two-match development check diffusion finished with mean mass 29.34 versus 60.13 for v3. More captures did not compensate for mass lost after repeated splits. These are development results, not final test results.

Corrective training (`learned_risk_v5`, then revised checkpoint selection in `learned_risk_v7`) adds counterfactual queries at actual physically possible split requests in **24 ordinary seven-agent matches**, seeds `4830001..4830024`. Each query clones the entire world, all controller states, pending action buffers and random generators. It compares splitting now with holding for **120 steps / four seconds**. Other agents keep reacting; the queried actor's subsequent split commands are held in both branches to isolate the immediate decision. The query cannot mutate the real match, and its NumPy random state is restored before normal play continues.

Full episodes, rather than adjacent observations, are separated into fitting and development sets (19 fitting, five development episodes). Fitting retains the original curriculum, replays actual-match corrections six times and a completed prefix of exploratory on-policy scenes twice. Replay multipliers are sampling weights, not counts of independent examples. The exploratory prefix is saved as `generic_corrections_used.json` so an interrupted exploratory collection cannot silently change the training data. Both heads are fitted for another 240 epochs; movement/value/critic weights remain frozen.

Utility-only checkpoint selection improved isolated encounters but still hurt diffusion mass retention in development matches. Selecting solely by prediction loss instead chose an underfit diffusion head with no validation split proposals; that experiment was stopped. The final selection criterion maximizes development action advantage among checkpoints whose weighted prediction loss is within 25% of the best observed loss. This balances useful decisions and probability calibration; it never imposes a runtime split veto. The 25% tolerance was chosen during development, before final evaluation.

This remains outcome-supervised learning of the split heads. It is not additional end-to-end PPO optimization or diffusion-policy training. Longer learning targets the requested split behavior while preserving the accepted movement baseline.

## Second on-policy generation

V7 still failed six-match development retention: PPO had six full deaths versus two for v3, while diffusion retained 85.6% of baseline final mass. V8 adds twelve new matches using v7 collection policies, seeds `4840001..4840012`. Split/hold branches run **240 steps / eight seconds**, covering the earliest piece-merge cooldown. Eligible hold states are also sampled every ten seconds, so learning can recover missed opportunities as well as discourage harmful requests. These are data-collection intervals, not execution rules. Other players remain reactive.

The earlier 227 full-match queries remain in replay, separated by their original episode partition. New episodes have their own fitting/validation partition (nine fitting, three validation). The small curated query buffers and the exploratory prefix are versioned; the large initial buffer remains ignored. Refitting uses the frozen collection buffers with `--reuse-data`, so metadata and source hashes describe the final fitting invocation.

The v7 final evaluation was abandoned after development failure, without reading a result report. V8 reserves new final seed families and binds the declared protocol hash into its fitted checkpoints. See [the complete development record](learned_risk_development.md).

## Controlled randomness and resuming collection

The environment has its own cloned random generator, but virus-fragment launch speed also uses global NumPy randomness. Each split/hold branch now resets that generator as well as the policy/opponent seeds. A regression test exercises actual virus fragmentation and verifies that outcomes no longer depend on the incoming global random state.

An interrupted collection retained **2,300 complete non-virus cases**. Their observation vectors were checked against the current scene generator, and eight policy/case checks verified invariance to the incoming global NumPy seed. All cases containing viruses were discarded and regenerated with controlled branch seeds, along with unfinished scenes. Rows are sorted canonically before fitting. [Resume audit](../outputs/learned_risk_v4/resume_audit.json).

## Validation gates

The [protocol](../outputs/learned_risk_v8/acceptance_protocol.declared.json) was saved before final evaluation:

1. **Fresh reactive encounters:** 600 requested cases starting at `5200001`. Measure actual free-running policy utility, captures, full deaths and splits against accepted v3 policies. Require at least 550 valid cases and 50 beneficial opportunities per algorithm, mean utility difference ≥−0.03, at most `max(2, floor(0.01 × valid cases))` additional deaths, zero physical violations, and zero nonphysical vetoes. Require at least ten measured useful initial split choices outside the previous heuristic guard.
2. **Navigation retention:** 12 fresh 60-second matches per algorithm, seeds `5300001 + 37*i`. Two candidates, two accepted v3 agents, three heuristics, shuffled seats. Require mean final mass ≥90% of baseline and no more than two extra total full deaths. This is an engineering tolerance, not statistical noninferiority.
3. **Current-policy league:** 12 independent matches starting at `5400001`, with two current PPO, two current diffusion and three heuristic agents. This supplies absolute current-capability measurements separately from the retention comparison.
4. **Promotion:** recompute gates from rows; verify declared seeds, model/source hashes, unchanged movement weights and the explicit `learned_risk_physical_only` mode. Historical risk-guarded reports cannot validate this execution path.

## Reproduce

From the repository root, with the project Python environment:

```bash
python -m src.train_learned_risk --cases 4000 --epochs 240 --workers 4

# Resume an interrupted collection; all virus cases are regenerated.
# Use --reuse-data instead if collection is complete and only fitting changes.
python -m src.train_learned_risk --cases 4000 --epochs 240 --workers 4 --resume-data

# Full-match corrective learning, after the base collection and fit finish.
python -m src.refine_match_risk --base outputs/learned_risk_v4 \
  --output outputs/learned_risk_v8 --policy-directory outputs/learned_risk_v7 \
  --prior-corrections outputs/learned_risk_v7/counterfactuals.jsonl \
  --episodes 12 --seed 4840001 --horizon 240 --sample-holds \
  --epochs 240 --workers 4 --selection calibrated_utility
# Add --reuse-data to fit from the versioned v8 query buffer instead of collecting.

python -m src.evaluate_learned_risk --directory outputs/learned_risk_v8 \
  --output outputs/learned_risk_v8/outcomes.json
python -m src.evaluate_risk_league --directory outputs/learned_risk_v8 \
  --mode retention --algorithm ppo --output outputs/learned_risk_v8/ppo_retention.json
python -m src.evaluate_risk_league --directory outputs/learned_risk_v8 \
  --mode retention --algorithm diffusion --output outputs/learned_risk_v8/diffusion_retention.json
python -m src.evaluate_risk_league --directory outputs/learned_risk_v8 \
  --mode league --output outputs/learned_risk_v8/league.json
python scripts/promote_learned_risk.py
```

Initial training inputs are reproducible from the scene generator; the large initial raw buffer is ignored by Git. Small curated full-match query buffers are versioned for audit and refitting. Preserve shipped reports when rerunning. Once published, these test seeds are public; future development needs a fresh final seed family. The remaining limitations include short outcome horizons, clipped observation features, fixed arena physics, imperfect movement and no demonstrated robot transfer.


## Fresh v8 results and playable candidate

**V8 is experimental and was not promoted.** Both encounter tests and diffusion retention passed. PPO failed the predeclared survival gate, despite higher final mass. The normal port-8080 preview and published showcase remain on accepted v3.

The physical-only candidate is available locally at **[localhost:8081](http://localhost:8081/)**. Reproduce it with:

```bash
python scripts/play_agar_web.py --port 8081 \
  --policy-dir outputs/learned_risk_v8/experimental
```

The experimental page carries a warning and `/status` reports both its split-control mode and failed acceptance status. The [experimental manifest](../outputs/learned_risk_v8/experimental/deployment.json) verifies the exact weights without claiming acceptance.

### Fresh reactive encounters

600 scenes were requested; 585 were valid per policy. The initial decision target is measured over two seconds; training also includes longer four/eight-second consequences. Initial precision/recall are classification measurements for that target, not game win rates.

| Measurement | PPO | Diffusion |
| --- | ---: | ---: |
| Known beneficial initial opportunities | 89 | 92 |
| Initial precision | 73.9% | 69.8% |
| Initial recall | 38.2% | 40.2% |
| Useful initial splits outside old heuristic guard | 23 | 23 |
| Mean utility difference vs guarded v3 | -0.0029 | +0.0045 |
| Physical violations | 0 | 0 |
| Nonphysical split vetoes | 0 | 0 |

Utility-difference confidence intervals include zero for both policies. Passing this engineering gate does not establish a general performance improvement. [All encounter rows](../outputs/learned_risk_v8/outcomes.json).

### Twelve-match retention per policy

Two candidates, two guarded-v3 baselines and three reactive heuristics compete in each 60-second match, with shuffled seats. The predeclared gate requires candidate mean final mass ≥90% of baseline and no more than two extra total full deaths.

| Measurement | PPO candidate / baseline | Diffusion candidate / baseline |
| --- | ---: | ---: |
| Mean final mass | 62.31 / 57.37 | 54.00 / 41.57 |
| Full deaths | 7 / 2 | 3 / 1 |
| Piece captures | 36 / 23 | 19 / 8 |
| Executed splits | 40 / 1 | 34 / 0 |
| Acceptance | **Failed: 7 deaths exceeds allowed 4** | **Passed: 3 deaths equals allowed 3** |

Diffusion's pass permits two additional deaths; it is not a survival-improvement claim. Match-bootstrap final-mass difference intervals include zero. [PPO report](../outputs/learned_risk_v8/ppo_retention.json) · [Diffusion report](../outputs/learned_risk_v8/diffusion_retention.json).

### Absolute candidate league

This separate twelve-match league uses two current PPO, two current diffusion and three heuristic agents, with no older policies. Each policy acts at every 30 Hz timestep. Agents compete individually.

| Measurement | PPO | Diffusion | Heuristics |
| --- | ---: | ---: | ---: |
| Agent runs | 24 | 24 | 36 |
| Mean final mass | 58.53 | 51.19 | 27.71 |
| Piece captures | 35 | 18 | 9 |
| Full deaths | 4 | 1 | 29 |
| Executed splits | 51 | 42 | 6 |
| Recent-projectile captures | 1 | 0 | 0 |
| Matches with a final-mass leader | 10 | 2 | 0 |

| Death-free agent runs | 20/24 | 23/24 | 16/36 |

Most captures were body contact. The mass-growth objective also rewards mobility/foraging; 51/42 learned splits produced only 1/0 recent-projectile captures in this league, so reliable split attacks remain a limitation. A piece capture is not necessarily a full elimination. Respawns count. The policy groups have unequal slot counts (2/2/3), so final-mass leaders are descriptive rather than an equal-team tournament win rate.

[Full league data, timelines, source/model hashes and configuration](../outputs/learned_risk_v8/league.json). Inference averaged 2.39 ms for PPO and 4.46 ms for diffusion during substantial parallel CPU contention; [timing context](../outputs/learned_risk_v8/timing_context.json) records the other workers. These are wall-clock measurements, not isolated kernel speed or browser FPS.

Weights/protocol were frozen at `cc07030` before these tests; [immutability audit](../outputs/learned_risk_v8/immutability_audit.json). Test seeds are now public and must not be reused as unseen tests after further tuning. The promotion tests verify that editing summary flags cannot hide failed rows. All 88 regression tests passed with one Torch/OpenMP thread, matching the server and evaluation workers.
