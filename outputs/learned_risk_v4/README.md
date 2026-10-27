# Experimental base; not a deployment

These physical-only split heads are the base for the full-match corrective training in `learned_risk_v5`. They were rejected on development behavior, especially diffusion's mass retention. There is deliberately no deployment manifest here.

`data_audit.json` describes the 3,897 valid initial scenes; `training_config.json` binds the weights to sources and data. `development_outcomes.json` and the two development retention files are exploratory results, not held-out acceptance reports. Their source hashes describe the code at evaluation time and need not match later evaluator defaults.

Reproduce the initial collection and fit with `python -m src.train_learned_risk --cases 4000 --epochs 240 --workers 4`. Raw counterfactual JSONL is ignored by Git; it must be regenerated before refitting. See [the learning protocol](../../docs/learned_split_risk.md).
