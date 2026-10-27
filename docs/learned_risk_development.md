# Physical-only split development record

These are development experiments, not final held-out tests. All candidates execute requested splits whenever the simulator permits them. None adds a predator-distance, angle, target-mass, wall or virus veto. Continuous movement assistance remains active.

| Candidate | Development encounters | Mean utility difference vs guarded v3, PPO / diffusion | Six-match mean final mass, candidate / baseline | Decision |
| --- | --- | --- | --- | --- |
| v4: larger initial curriculum | 187 valid per policy | −0.01384 / −0.03208 | Only two matches were run; diffusion 29.34 / 60.13 | Reject |
| v5: four-second, actual-match corrections; maximize development utility | 187 valid per policy | +0.00704 / +0.00077 | PPO 54.99 / 50.84; diffusion 43.95 / 50.96 | Reject: diffusion retained 86.3% of baseline mass; PPO had five full deaths vs two |
| v6: same data; minimize prediction loss | Aborted before completion | Not measured to completion | Not measured to completion | Reject: selected diffusion checkpoint proposed no splits on validation observations |
| v7: maximize utility within 25% of minimum prediction loss | 94 valid per policy | +0.01400 / −0.00328 | PPO 54.28 / 55.54; diffusion 45.06 / 52.63 | Reject: diffusion retained 85.6% of baseline mass; PPO had six full deaths vs two |
| v8: second on-policy generation; eight-second outcomes and hold-state exploration | Evaluated separately | See candidate reports | See candidate reports | Await final acceptance |

The development encounter seed family starts at `4851001`; the six-match development retention seeds are `4950001 + 37*i`, `i=0..5`. Repeated use of these seeds makes them development data. The incomplete v7 final encounter evaluation was stopped after its development retention failure, before any aggregate or per-case outcome report was read. V8 reserves completely different final seeds (`5200001`, `5300001`, `5400001` families). No statement of statistical noninferiority is made from these small development samples.

The initial 4,000 requested scenes yielded 3,897 valid scene pairs. Corrective data contain 227 actual split requests in 24 ordinary seven-agent matches (135 PPO, 92 diffusion) plus 360 exploratory states. Entire episodes/scenes separate fitting and validation. Replay weighting does not create independent samples. Only split heads change; navigation/value/critic weights remain frozen.

V8 collects a second generation from v7 policies in twelve new matches, seeds `4840001..4840012`, with eight-second split/hold branches. It samples every actual physically eligible split request and eligible hold states at ten-second intervals, up to 24 queries per episode. That cap and interval limit training collection cost; they never limit runtime actions. Earlier four-second corrections remain in replay, with their original episode partition. The new horizon reaches the simulator's earliest eight-second piece-merge cooldown, although pieces need not actually merge then. The small full-match query datasets are versioned; the much larger initial raw buffer remains reproducible and ignored.

Longer training is not automatically better. Utility-only selection chose late checkpoints after prediction loss had increased substantially. Loss-only selection preferred a nearly constant negative predictor. The v7 criterion retains observed useful decisions while limiting probability overfitting. This is a development-time model selection rule, not an action filter.

The simulator's virus fragments use global NumPy randomness in addition to the environment RNG. Paired trials control both; a regression test exercises fragmentation. Full-match corrective queries clone controller buffers and RNGs and restore the live rollout's random state after each pair. These details prevent a spurious split advantage caused by a different random draw or an accidentally mutated world.

See [training and final acceptance protocol](learned_split_risk.md), [rejected initial data and results](../outputs/learned_risk_v4/README.md), and the versioned candidate reports for measured rows. Published old v3 capture-opportunity percentages measure a different risk-guarded system and cannot validate physical-only execution.
