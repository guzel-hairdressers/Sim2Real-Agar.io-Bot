# Arena v2 results — 2026-10-27

Changes were accepted only after both policies passed the predeclared mass-gain/survival gate on fresh final seeds. This is an improvement checkpoint, not a claim that the navigation/combat problem is solved.

## Final protocol

12 matches per algorithm; 1,800 control steps per match at dt=1/30 (60 simulated seconds). Each match has two candidate agents, two original agents of the same algorithm, and the original apex/hunter/survivor heuristic profiles. Candidate and baseline seats are shuffled together with a seeded permutation. Starting masses are accounted for using mass gained. All competitors experience one shared environment; no candidate receives invisible opponent data.

Final seeds: 1010001, 1010038, 1010075, 1010112, 1010149, 1010186, 1010223, 1010260, 1010297, 1010334, 1010371, 1010408.

The 810xxx seeds were used for development. An earlier 910xxx PPO run exposed wall contacts and was subsequently treated as development; it is not the final acceptance sample. The final 1010xxx seeds were evaluated after freezing the clearance revision and selected checkpoints. Diffusion used the cosine/normalized/averaged model with return-critic candidate selection. No model was changed after viewing the final sample.

Matches, rather than individual players inside a correlated match, are the bootstrap unit. The gate requires a positive lower endpoint of the match-bootstrap 95% interval for mean mass-gain difference and no increase in mean full deaths. Wall contacts, piece kills, and runtime are also reported. This gate measures improvement over the original policies, not absolute tactical competence.

## Final measurements

| Algorithm | New/old mean final mass | Mass-gain advantage (95% interval) | Pair-average gain wins | New/old full deaths | New/old wall contacts | New/old mean inference |
| --- | --- | --- | --- | --- | --- | --- |
| PPO | 71.01 / 28.04 | 42.78 (21.11–60.71) | 10/12 | 2 / 74 | 1302 / 1813 | 0.22 / 0.19 ms |
| Diffusion | 68.42 / 24.58 | 43.65 (29.79–57.05) | 12/12 | 0 / 56 | 581 / 184 | 0.49 / 0.44 ms |

Deaths include repeated full eliminations after respawn; piece kills are counted separately. “Gain wins” compare two-agent group averages and are not outright individual match victories. Inference timing includes the policy wrapper and safety filter on this Mac under concurrent evaluation load; it does not measure rendering or whole-system FPS.

Raw reports: [PPO](../outputs/arena_v2/ppo_acceptance.json), [diffusion](../outputs/arena_v2_cosine/diffusion_acceptance.json). Reports record source/configuration and checkpoint hashes. The frozen baseline uses the original champion weights, original ca3f24d reflex, and original sampling settings. Its Torch random stream is explicitly seeded for repeatability; it is not a bit-for-bit replay of the original unseeded browser. Both versions are evaluated in the corrected shared environment, so the comparison isolates policy packages under one environment rather than comparing two differently timed game versions.

## Training and ablations

- Data: 24 episodes of 600 steps, with an episode-level 19/5 training/validation split. Training has 79,800 state/action samples and 25,876 valid action chunks; validation has 21,000 samples and 6,792 chunks. Chunks do not cross respawns. Half of training resets sample larger initial cells while retaining the world mass budget.
- PPO: 15 imitation epochs, then 121,344 learner transitions across four arenas with three learner-controlled seats each. Other seats contain a corrected teacher, two heuristic profiles and an original PPO. Gaussian samples are stored without clipping; safety intervention is a deterministic environment-side action transformation. Training truncations bootstrap the actual final state but stop GAE recursion across resets; player deaths stop bootstrap into the respawn.
- Diffusion: the first linear-schedule candidate was only marginally better; its unguided variant failed the development gate. The accepted candidate uses 80 cosine-schedule epochs, symmetric actions, parameter averaging, and the same demonstrations. This follows the cosine scheduler approach in the [official Diffusion Policy configuration](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_ddim_hybrid_workspace.yaml). We retain the small existing denoiser architecture; this is not a claim to reproduce that project.
- The critic targets actual discounted 16-step simulator rewards, not synthetic first-frame scores. Validation MSE was 0.336 versus 0.597 for a constant training-mean predictor. This is supervised generalization on demonstration-like trajectories; candidate-action extrapolation remains a limitation.
- On four development matches, imitation-only PPO had a mean mass-gain advantage of 30.20 versus 36.75 for imitation-plus-PPO. This small ablation does not establish that the PPO stage alone explains the final improvement.
- The cosine diffusion variant had development advantages of 22.97 with the critic and 15.13 without it. The unguided interval crossed zero. Four matches are insufficient to establish a broad superiority claim for critic guidance; the guided version was separately accepted on 12 fresh matches.
- Raw-policy bypass is available through `--raw`; neither the deployed PPO nor deployed diffusion is advertised as a pure neural-only controller. The safety filter preserves ordinary actions and intervenes near observable danger or rejects unsafe splits.

## Exploit diagnostics and remaining failures

Five scripted cases, four fixed perturbations each, run for up to ten seconds with no food. These are deliberately harsher than the normal training game and are diagnostic regression cases, not a calibrated human benchmark. Both algorithms avoided all four split-bait deaths (originals died in all four) but still died in all twelve close-range head-on/flanking/corner predator cases. Overall candidate deaths were 12/20 versus 16/20 for the originals. PPO eliminated the opponent in 8 cases versus 4; diffusion did so in 6 versus 4.

These failures are retained in [PPO encounter records](../outputs/arena_v2/ppo_encounters.json) and [diffusion encounter records](../outputs/arena_v2_cosine/diffusion_encounters.json). Larger anticipation distances, turning-path selection, escape hysteresis and trajectory-based hazard prediction are sensible next experiments. They have not been implemented or claimed solved in this checkpoint.

Diffusion still has more wall contacts than its original baseline (581 versus 184 across the final sample). PPO has fewer in the final sample (1,302 versus 1,813), but still too many. Individual split pieces can hit a wall before a mass-center-based clearance estimate predicts contact. Better whole-body clearance and short-horizon escape planning remain useful follow-ups.

Other limits: normalized mass features saturate at 150; the observation contains nearest tactical pieces rather than a full set of all visible piece arrangements; the critic was trained on demonstration chunks, not a broad counterfactual action dataset; opponents are a limited league rather than a human population. There is no real-robot transfer result.

## Delivery and provenance

54 tests passed with `python -m unittest discover -s tests -v`, including the five existing plain-function physics tests now included in discovery. Regression tests cover the probability-ratio invariant, event counters, motion/piece observability, hidden opponents, deterministic resets/sampling, split bait, braking, time-limit/death GAE boundaries, strict missing checkpoints, cosine terminal noise, and the loaded web models/render/control lifecycle.

The browser preview at localhost:8080 was visually verified. A read-only live check measured 27.13 control steps/second under the current load, against a 30 Hz target; the video stream delivered valid 1200×675 frames. The serving verification JSON records the exact running model hashes. The current user play session is left running; its reset/control state was not changed during the final documentation/commit work. A local versioned manifest verifies all model hashes at startup. Missing or incompatible deployment checkpoints stop startup rather than silently producing random policies.

The training teacher and filter were frozen before the later deployment clearance revision. `benchmarks/training_tactics_v2_initial.py` and `--teacher-version initial_v2` preserve that training behavior. Replaying the first recorded episode produced exactly the stored 4,200 state/action observations. [Dataset manifest](../outputs/arena_v2/dataset_manifest.json) records its checksum, shapes and teacher hash. Full bitwise training replay on a different platform is not guaranteed.

The current README replaces unsupported current-version claims; the old README is retained separately as historical material. Historical weights and unrelated existing experiment outputs are retained. No external publication or push is part of this commit.
