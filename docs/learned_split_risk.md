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

The next candidate (`learned_risk_v5`) adds counterfactual queries at actual physically possible split requests in **24 ordinary seven-agent matches**, seeds `4830001..4830024`. Each query clones the entire world, all controller states, pending action buffers and random generators. It compares splitting now with holding for **120 steps / four seconds**. Other agents keep reacting; the queried actor's subsequent split commands are held in both branches to isolate the immediate decision. The query cannot mutate the real match, and its NumPy random state is restored before normal play continues.

Full episodes, rather than adjacent observations, are separated into fitting and development sets (19 fitting, five development episodes). Fitting retains the original curriculum, replays actual-match corrections six times and a completed prefix of exploratory on-policy scenes twice. Replay multipliers are sampling weights, not counts of independent examples. The exploratory prefix is saved as `generic_corrections_used.json` so an interrupted exploratory collection cannot silently change the training data. Both heads are fitted for another 240 epochs; movement/value/critic weights remain frozen.

This remains outcome-supervised learning of the split heads. It is not additional end-to-end PPO optimization or diffusion-policy training. Longer learning targets the requested split behavior while preserving the accepted movement baseline.

## Controlled randomness and resuming collection

The environment has its own cloned random generator, but virus-fragment launch speed also uses global NumPy randomness. Each split/hold branch now resets that generator as well as the policy/opponent seeds. A regression test exercises actual virus fragmentation and verifies that outcomes no longer depend on the incoming global random state.

An interrupted collection retained **2,300 complete non-virus cases**. Their observation vectors were checked against the current scene generator, and eight policy/case checks verified invariance to the incoming global NumPy seed. All cases containing viruses were discarded and regenerated with controlled branch seeds, along with unfinished scenes. Rows are sorted canonically before fitting. [Resume audit](../outputs/learned_risk_v4/resume_audit.json).

## Validation gates

The [protocol](../outputs/learned_risk_v4/acceptance_protocol.json) was saved before final evaluation:

1. **Fresh reactive encounters:** 600 requested cases starting at `4900001`. Measure actual free-running policy utility, captures, full deaths and splits against accepted v3 policies. Require at least 550 valid cases and 50 beneficial opportunities per algorithm, mean utility difference ≥−0.03, at most `max(2, floor(0.01 × valid cases))` additional deaths, zero physical violations, and zero nonphysical vetoes. Require at least ten measured useful initial split choices outside the previous heuristic guard.
2. **Navigation retention:** 12 fresh 60-second matches per algorithm, seeds `5000001 + 37*i`. Two candidates, two accepted v3 agents, three heuristics, shuffled seats. Require mean final mass ≥90% of baseline and no more than two extra total full deaths. This is an engineering tolerance, not statistical noninferiority.
3. **Current-policy league:** 12 independent matches starting at `5100001`, with two current PPO, two current diffusion and three heuristic agents. This supplies absolute current-capability measurements separately from the retention comparison.
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
  --output outputs/learned_risk_v5 --episodes 24 --epochs 240 --workers 4

python -m src.evaluate_learned_risk --directory outputs/learned_risk_v5 \
  --output outputs/learned_risk_v5/outcomes.json
python -m src.evaluate_risk_league --directory outputs/learned_risk_v5 \
  --mode retention --algorithm ppo --output outputs/learned_risk_v5/ppo_retention.json
python -m src.evaluate_risk_league --directory outputs/learned_risk_v5 \
  --mode retention --algorithm diffusion --output outputs/learned_risk_v5/diffusion_retention.json
python -m src.evaluate_risk_league --directory outputs/learned_risk_v5 \
  --mode league --output outputs/learned_risk_v5/league.json
python scripts/promote_learned_risk.py
```

Training inputs are reproducible from the scene generator; raw JSONL buffers are ignored by Git. Preserve shipped reports when rerunning. Once published, these test seeds are public; future development needs a fresh final seed family. The remaining limitations include short outcome horizons, clipped observation features, fixed arena physics, imperfect movement and no demonstrated robot transfer.
