# Sim2Real Agar Control

**Learned navigation and decision-making in a competitive, partially observed world.**

Agents steer through an Agar-style arena, collect food, grow, avoid hazards, and decide when to split to capture smaller opponents. The project brings **PPO**, **action diffusion**, and **rule-based opponents** into the same simulator—and lets you play against the trained agents in a browser.

[![Watch the agents navigate and compete](docs/media/showcase/gameplay.gif)](docs/media/showcase/gameplay.mp4)

**[Full 60-second gameplay video](docs/media/showcase/gameplay.mp4)** · [Screenshots and demo gallery](docs/media.md) · [Benchmark protocol and raw results](docs/benchmarks.md) · [Run locally](#play-against-the-agents)

*Actual simulator recording at real-time speed. Blue: PPO; green: diffusion; amber: heuristic opponents; pink: virus hazards. The dashed circle marks one agent's sensing range. The animation is a ten-second excerpt; the linked video contains the entire match.*

## What this project explores

The arena makes navigation harder than steering toward a waypoint: agents have inertia, their speed changes with mass, nearby opponents can be threats or targets, and splitting changes both mobility and vulnerability. Each policy sees a **70-value local observation**, rather than the full map shown to spectators.

- **Continuous control:** choose thrust and steering at a 30 Hz simulation timestep.
- **Limited sensing:** react to visible opponent pieces, relative motion, food, hazards, and boundaries.
- **Discrete decisions:** learn when a split can capture a target without making the launched half too small.
- **Interactive evaluation:** play alongside two PPO agents, two diffusion agents, and two heuristic opponents.

The current deliverable is a **simulation research testbed**. Despite the historical “Sim2Real” name, physical robot transfer has not been demonstrated.

## See what the agents do

![A complete arena beside one agent's local sensing footprint and current state](docs/media/showcase/overview.png)

The left panel is the spectator's full arena. The right panel illustrates the local sensing footprint used to construct one agent's observation; the policy receives numeric features, not this picture. The table records mass, piece captures, full deaths, and actual split executions.

[![Current PPO and diffusion agents choose a split attack or preserve their mass](docs/media/showcase/split_decisions.gif)](docs/media/showcase/split_decisions.mp4)

**[Split-decision video](docs/media/showcase/split_decisions.mp4)** — a controlled demonstration of both current policies. A 90-unit actor can halve itself and still capture a 20-unit target. A 45-unit actor holds its split against a 30-unit target because its launched half would be too small. Targets are stationary; the video is slowed to **1/6 speed**, with simulator time shown. Split commands come from the deployed policies.

## Current capabilities: benchmarks

These results describe the **current models**, with no historical agents in the match roster. Twelve fresh 60-second matches each contain two PPO agents, two diffusion agents, and three heuristic opponents, with shuffled seats. Agents compete individually; sharing a policy does not make them a team.

| Metric | PPO | Diffusion | Heuristic opponents |
| --- | ---: | ---: | ---: |
| Agent runs | 24 | 24 | 36 |
| Mean final mass, simulation units | **51.7** | **47.6** | 31.5 |
| Runs with no full death | **20/24 (83.3%)** | **21/24 (87.5%)** | 22/36 (61.1%) |
| Opponent pieces captured per agent-minute | 0.71 | **0.88** | 0.42 |
| Executed splits, total | 1 | 1 | 12 |
| Mean policy inference on this Mac, CPU | 0.279 ms | 0.525 ms | 0.019 ms |

Inference includes safety checks and was measured under parallel evaluation load; it excludes rendering and is not browser FPS. A full death means losing all pieces; respawns remain part of the match. Mass is a simulator quantity, not a physical measurement.

![Current-agent growth, death-free runs, capture rate, and separate split-opportunity evaluation](docs/media/showcase/benchmarks.png)

A separate held-out split test contains **996 valid encounters per learned policy**. PPO captured **120/125 beneficial opportunities (96.0%)** and diffusion **119/126 (94.4%)** with a launched piece. Neither executed a split that violated the current safety checks. These percentages apply to known beneficial opportunities in that test; they are **not game win rates or general combat accuracy**.

The agents still split rarely in ordinary matches. Corner escapes, close predator traps, and wall contacts remain weaknesses; skilled humans can beat them. Twelve simulator seeds do not establish broad generalization. See [definitions, confidence intervals, hardware context, and reproduction commands](docs/benchmarks.md).

## How it works

```mermaid
flowchart LR
    S[Local observation: 70 values] --> P[PPO: continuous actions]
    S --> D[Diffusion: candidate action sequences]
    D --> C[Return critic selects a sequence]
    S --> B[Learned binary split head]
    P --> G[Observation-based safety layer]
    C --> G
    B --> G
    G --> E[Shared arena physics]
    E --> S
```

PPO uses imitation initialization followed by reinforcement learning. Diffusion learns short action sequences from demonstrations; a return critic ranks candidates, and the agent executes a short prefix before replanning. Both use separately trained binary split decisions and a safety layer that can reject unsafe splits or alter movement near immediate hazards. Splitting is supervised using paired physical trials in which splitting captures a target earlier than holding.

The models share the same observation contract and physics. Benchmark runs record seeds, individual agent outcomes, model hashes, and source hashes. The exported demonstration is rendered from the first benchmark seed, chosen before its outcome, rather than from a selected winning run.

## Play against the agents

From the repository root, using Python 3.9+:

```bash
python -m pip install -r requirements.txt
python scripts/play_agar_web.py --host 127.0.0.1 --port 8080
```

Open **[localhost:8080](http://localhost:8080/)**. Move the mouse to steer, hold **W** to sprint, press **Space** to split, and press **R** to reset the match.

Accepted weights are included in the repository. The server checks the deployment manifest and rejects missing, incompatible, or checksum-mismatched models. `/status` exposes readiness, model hashes, and the control timestep. The GitHub videos are recordings; the interactive game runs locally.

## Reproduce and explore

| Resource | What it contains |
| --- | --- |
| [Demo gallery](docs/media.md) | Gameplay and split videos, animations, screenshots, and recording provenance |
| [Current benchmarks](docs/benchmarks.md) | Absolute metrics, test definitions, raw data, and commands |
| [Navigation training and checkpoint history](docs/checkpoint_history_2026-10-27.md) | Training recipes and historical comparisons |
| [Split training and limitations](docs/split_combat_results_2026-10-27.md) | Paired split/hold curriculum, acceptance gates, and failure modes |
| [Current benchmark data](outputs/showcase_v1/benchmark.json) | All twelve match results, time series, seeds, and hashes |
| [Training and simulator source](src/) | PPO, diffusion, inference wrappers, benchmarks, and diagnostics |

Run existing regression checks with `python -m unittest discover -s tests -v`. To recreate the media, install the optional [media-build dependencies](requirements-media.txt) and follow the [gallery instructions](docs/media.md#rebuild-the-showcase).
