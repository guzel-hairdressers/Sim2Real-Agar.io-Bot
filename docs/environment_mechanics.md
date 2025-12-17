# Multi-Body Cellular Simulation Environment: Mechanics & Physics Specification

This document details the physical engine, spatial food potential field, continuous collision models, and multi-body dynamics of the continuous Agar.io simulation testbed.

---

## 1. Kinematics & Body-Frame Coordinate Invariance

Each agent controls a cellular body with continuous unicycle kinematics in an enclosed arena $\Omega = [-L/2, L/2]^2$ ($L = 20.0\text{ m}$):

$$
\dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \omega
$$

### Mass-Radius Relationship
Cell radius $r$ scales with square-root mass, assuming constant unit area density $\rho = 1.0\text{ kg/m}^2$:

$$
r = \sqrt{\frac{M}{\pi}}
$$

### Inertia & Terminal Speed
Heavier cells encounter greater hydrodynamic drag. Maximum linear velocity scales inversely with mass:

$$
v_{\max}(M) = v_0 \cdot \left(\frac{M_0}{M}\right)^{0.40}
$$

Small fragments move rapidly ($v \approx 4.8\text{ m/s}$), while large predators move slower ($v \approx 1.2\text{ m/s}$) but possess a wider capture radius.

---

## 2. Closed Thermodynamic Mass Conservation

In standard browser implementations, food spawning is open-loop and cells accumulate mass indefinitely, leading to runaway starvation or unbounded growth. This engine implements a strict **closed thermodynamic mass invariant** ($M_{\text{total}} \equiv 400.0\text{ kg}$):

$$
M_{\text{total}} = \sum_{i \in \mathcal{P}} M_i + \sum_{f \in \mathcal{F}} m_{\text{pellet}} + M_{\text{reserve}} \equiv 400.0\text{ kg}
$$

### Metabolic Decay Recycling
Cellular metabolic loss drains continuous mass from larger cells:

$$
\frac{dM}{dt} = -\alpha M^{1.15}
$$

This mass does not vanish; it drains into a conserved mass reserve pool ($M_{\text{reserve}}$).

### Kill Splatter Dispersion
When a predator swallows a prey cell of mass $m$, $70\%$ is directly absorbed into the predator's body, while the remaining $30\%$ explodes outward as a ring of high-speed nutritious food pellets, fertilizing the combat zone and attracting nearby foragers.

### Numerical Invariance
Across 1,000 continuous simulation steps, cumulative mass drift is bounded to floating-point precision ($|\Delta M| < 3.5 \times 10^{-13}\text{ kg}$).

---

## 3. High-Resolution Spatial Potential Field Grid (50 × 50)

Rather than scattering pellets uniformly, food replenishment samples positions from a dynamic 2D potential field $\mathcal{P}(x, y)$ computed over a $50 \times 50$ spatial grid:

$$
\mathcal{P}(x, y) = S(x, y) \cdot \left[ F_{\text{corridors}}(x, y) + F_{\text{center}}(x, y) + \epsilon_{\text{ambient}} \right]
$$

### Clearance Shadow ($S$)
Zero food can spawn within or immediately adjacent to any player's cell, preventing static camping:

$$
S(x, y) = \begin{cases} 0, & \text{if } \exists i \text{ such that } \|(x, y) - \mathbf{p}_i\| \le r_i + 0.3\text{m} \\ 1, & \text{otherwise} \end{cases}
$$

### Inverse-Mass Inter-Agent Corridors ($F_{\text{corridors}}$)
For every pair of competing agents $(i, j)$, a Gaussian food corridor is projected along the connecting line segment, biased toward the smaller agent:

$$
\mathbf{c}_{ij} = w_i \mathbf{p}_i + (1 - w_i) \mathbf{p}_j, \quad w_i = \frac{1/M_i}{1/M_i + 1/M_j}
$$

Food clusters along these contested pathways, creating high-stakes hunting lanes.

### Central Oasis Potential ($F_{\text{center}}$)
A stationary Gaussian distribution centered at $(0, 0)$ with standard deviation $\sigma = 4.0\text{ m}$ draws agents out of perimeter boundaries into open-field encounters.

---

## 4. Multi-Body Splitting & Re-Merge Dynamics

### 4.1 4-Piece Tactical Cap
Historical Agar.io permitted up to $2^4 = 16$ pieces. In a competitive 6-player arena with $M_0 = 14.0\text{ kg}$ and $400.0\text{ kg}$ total ecosystem mass:
- Splitting into 16 creates $< 1.0\text{ kg}$ fragments, causing severe sensory degradation and vulnerability.
- Enforcing **`max_pieces = 4`** (supporting double-splits $1 \to 2 \to 4$) guarantees each fragment retains $\ge 25\%$ of player mass, maintaining predator lethality while doubling the spatial capture envelope.

### 4.2 Inward Gravitational Re-Merge
Upon splitting, the forward sub-piece is projected at launch speed $v_{\text{launch}} = 4.8\text{ m/s}$ and an 8.0-second re-merge cooldown ($\tau_{\text{cooldown}} = 8.0\text{s}$) begins:

1. **Cooldown Phase**: Sub-pieces exert soft-body elastic contact repulsion ($d_{ij} < r_i + r_j$), moving in formation without self-collision.
2. **Attraction Phase ($\tau \le 0$)**: Inward mutual gravitational acceleration draws sub-pieces toward the player's center of mass $\mathbf{c}_{\text{player}}$:

$$
\mathbf{a}_{\text{gravity}, i} = 1.2 \cdot \frac{\mathbf{c}_{\text{player}} - \mathbf{p}_i}{\max(0.1, \|\mathbf{c}_{\text{player}} - \mathbf{p}_i\|)}
$$

3. **Snap-Merge**: When sub-pieces touch ($d_{ij} \le r_i + r_j$), they merge with exact mass and linear momentum conservation:

$$
M_{\text{merged}} = m_i + m_j, \quad \mathbf{v}_{\text{merged}} = \frac{m_i \mathbf{v}_i + m_j \mathbf{v}_j}{m_i + m_j}
$$

---

## 5. Continuous Swept-Sphere Collision & Predation Rules

### 5.1 Swept-Sphere Continuous Collision
High-speed split lunges ($4.8\text{ m/s}$, moving up to $0.48\text{ m}$ per $\Delta t = 0.1\text{s}$ step) cause discrete collision tests to suffer from tunneling. Swept-sphere segment projection computes the continuous closest approach between segment $\overline{\mathbf{p}_{t-1}\mathbf{p}_t}$ and target centers, preventing tunneling artifacts.

### 5.2 Decoupled Sub-Piece Predation Criterion
Predation is evaluated pairwise across individual sub-pieces rather than aggregate player mass:

$$
\text{Predation Condition}: \quad m_{\text{pred}} - m_{\text{prey}} \ge 6.0\text{ kg} \quad \wedge \quad \frac{m_{\text{pred}}}{m_{\text{prey}}} \ge 1.20
$$

An intact $45\text{ kg}$ piece from a smaller player will cleanly consume a $25\text{ kg}$ fragment of an otherwise split $80\text{ kg}$ giant.
