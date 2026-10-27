# Evaluated experimental physical-only candidate

**Not promoted.** Both fresh encounter tests and diffusion retention passed. PPO failed the predeclared survival gate: seven full deaths versus two for guarded v3 (maximum allowed four). Its mean final mass increased from 57.37 to 62.31. Diffusion mean final mass increased from 41.57 to 54.00, with three deaths versus one (within the two-extra-death tolerance). These are twelve-match results per policy; they do not establish universal improvement.

Both learned split heads use only physical execution limits: mass, piece capacity and cooldown. Movement/value/critic weights remain unchanged. Requested splits cannot be vetoed by nearby predators, walls, viruses, target mass or angle. Heads were trained for 240 epochs and selected by development utility within a prediction-loss bound. The 110 new eight-second on-policy queries supplement 227 earlier four-second queries and 3,897 initial scenes.

Weights and protocol were frozen at `cc07030` before fresh tests; `immutability_audit.json` verifies unchanged bytes. Test seed families are 5200001 (encounters), 5300001 (retention), and 5400001 (current-policy league). They are now public and cannot serve as fresh tests after future tuning.

Use [the explicitly experimental playable manifest](experimental/README.md) for a local trial at port 8081. There is deliberately no accepted deployment manifest in this directory. The normal default remains accepted v3. See [method and results](../../docs/learned_split_risk.md) and [development failures](../../docs/learned_risk_development.md).
