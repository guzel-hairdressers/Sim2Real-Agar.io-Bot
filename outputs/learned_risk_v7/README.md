# Rejected development candidate and v8 collection policy

The calibrated-utility checkpoint selector retained useful isolated split decisions, but six-match development retention still failed. PPO mean final mass was 54.28 versus 55.54 for guarded v3, with six full deaths versus two. Diffusion mean final mass was 45.06 versus 52.63, with zero full deaths versus one. These are development results, not final acceptance. There is deliberately no deployment manifest.

The incomplete final encounter evaluation was stopped after this development failure, before a result report was produced or read. V8 uses entirely new final seed families. See `final_evaluation_aborted.json`.

These weights are retained because v8 collects its next on-policy correction set from them. Training source is archived in checkpoint `34249a5`; current collection code subsequently adds longer horizons. `counterfactuals.jsonl` is the small 227-query set originally collected with v4 policies, not a new independent dataset. The 360 exploratory states are in `generic_corrections_used.json`. Later source changes deliberately prevent current promotion of these historical weights. See [the experiment record](../../docs/learned_risk_development.md).
