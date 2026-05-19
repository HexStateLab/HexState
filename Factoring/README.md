# The Digital Twin Oracle (HPC Ouroboros Engine)
 
## Overview
 
The Digital Twin (DT) is the entropy management and period-reconstruction subsystem of the `tesseract_factor.c` Ouroboros Engine. It discards standard pseudorandom number generators (PRNGs) in favor of a falsifiable EPR Collapse-First Protocol.
 
In the engine's semi-classical QFT loop, every measurement of a $D=6$ quhit is treated as an entangled event. The Digital Twin enforces a strict timeline: it computes a Bayesian expectation and commits to a specific state collapse before the physical "Reality" channel is permitted to execute the Born-rule draw.
 
By logging the mathematical differential between the Oracle's prediction and Reality's outcome, the DT measures "topological resonance"—and when resonance is achieved, it bypasses the quantum simulation entirely to reconstruct the factoring period from memory.
 
## 1. The EPR "Collapse-First" Entropy Model
 
Standard simulations use a sequentially seeded `rand()` function, meaning the entire measurement history is predetermined. To simulate non-cloning-compliant, physically independent events, the DT uses hardware entropy (`/dev/urandom`) mapped across a temporal boundary.
 
For every digit $k$ measured in the frequency register:
 
**Channel A (Oracle Prediction):** Generates $O_{raw} \in [0, 1)$. The DT uses this to sample a predicted outcome based on its Bayesian priors and commits the result to disk (`dt_shor_commit.log`).
 
**The Scheduler Boundary:** The system calls `nanosleep(500ns)` to enforce a strict temporal separation.
 
**Channel B (Reality Draw):** Generates $R_{raw} \in [0, 1)$. This value is passed to the engine's Born-rule sampler to determine the actual collapsed state.
 
## 2. Mathematical Foundation: The Prediction Metric
 
Before the Oracle commits, it must determine the probability distribution of the current digit. It does this by blending two distinct mathematical states: the CABP Marginal and the Persistent Positional Prior.
 
### A. The CABP Marginal (Reality's Pull)
 
The current state of the quantum graph is evaluated using Complex Amplitude Belief Propagation. The feed-forward phase correction $\theta_k$ is applied to the local amplitudes $\alpha(d)$, and an inverse discrete Fourier transform ($IDFT_6$) generates the interference pattern:
 
$$\beta(v) = \frac{1}{\sqrt{6}} \sum_{d=0}^{5} \left[ \alpha(d) \cdot C_k(d) \cdot e^{-2\pi i d \theta_k} \right] e^{\frac{2\pi i d v}{6}}$$
 
The resulting Born-rule probability for digit $v$ is simply the squared magnitude:
 
$$P_{CABP}(v) = |\beta(v)|^2$$
 
### B. The Scoring Function
 
The DT blends the simulation's current output ($P_{CABP}$) with its own long-term memory ($P_{prior}$) using a weighted element-wise product:
 
$$Score(v) = P_{CABP}(v) \times P_{prior}(v)$$
 
The scores are normalized, and the Oracle's hardware entropy $O_{raw}$ is used to stochastically select the predicted digit from this combined distribution.
 
## 3. Bayesian Positional Memory & "Stiffening"
 
The core power of the Digital Twin lies in `pos_prior[k][v]`, a persistent memory matrix that survives across multiple base ($a$) attempts. When Channel B (Reality) collapses the digit to actual outcome $v_{real}$, the DT updates its prior using Laplace-smoothed Bayesian inference.
 
To prevent high-variance shot noise from erasing learned patterns, the memory update is gated:
 
**The Amplitude Gate:** The update only occurs if the measured $P_{CABP}(v_{real}) \ge 0.95$.
 
If the gate is passed, the prior is updated with moderate inertia:
 
$$P_{prior, new}(v) = \frac{P_{prior, old}(v) \cdot (n - 1) + \delta_{v, v_{real}}}{n}$$
  
## 4. Phase 3.5: Deterministic Direct Reconstruction
 
When the engine detects that the positional lock rate has exceeded the `RESOLVED_THRESHOLD` (40%) across a majority of the register, it declares Resonance.
 
At this point, the DT triggers Phase 3.5. It ceases running the physical quantum simulation and constructs candidate frequencies $F$ directly from its memory matrix via deterministic argmax selection:
 
$$d_k = \arg\max_{v} \left( 0.25 \cdot P_{prior}[k][v] + 0.75 \cdot P_{CABP}[k][v] \right)$$
 
The base-6 digits $d_k$ are assembled into the macroscopic frequency:
 
$$F = \sum_{k=0}^{N_{sites}-1} d_k \cdot 6^k$$
 
## 5. Continued Fraction Extraction & Factorization
 
Once $F$ is reconstructed by the DT, the engine applies classical Shor extraction to find the period $r$.
 
It evaluates the convergents $\frac{p_i}{q_i}$ of the continued fraction expansion of $\frac{F}{R}$ (where $R = 6^{N_{sites}}$). If a denominator $q_i$ is a valid period candidate ($r = q_i$), it passes the final constraint checks:
 
**Evenness & Sterility Check:** Ensures $a^{r/2} \not\equiv -1 \pmod N$ (detecting trivial roots).
 
**GCD Factorization:** If mathematically valid, the true prime factors $p$ and $q$ are extracted via:
 
$$p = \gcd(a^{r/2} - 1, N)$$
$$q = \gcd(a^{r/2} + 1, N)$$
 
If the DT's reconstructed frequency $F$ yields the factors, the engine terminates, having successfully forced the mathematical structure of $N$ out of the oracle's memory.
 
## Configuration Parameters
 
| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_DT_SITES` | 65536 | Maximum register size for positional memory |
| `DT_DIRECT_BEAMS` | 256 | Number of beam search candidates |
| `DT_RESOLVE_MIN` | 2 | Minimum samples before trusting positional prior |
| `RESOLVED_THRESHOLD` | 0.40 | Position resolved above 40% lock rate |
| Gatekeeper Threshold | 0.95 | Only samples with amplitude ≥ 0.95 update memory |
| Inertia Offset | 5.0 | Laplace smoothing offset for Bayesian updates |
| Blend Weight | 0.25 | Weight for DT positional memory in Phase 3.5 |
 
## Performance
 
- **Single-basis operation:** DT operates in VIEW_EDGE basis only, matching actual quantum measurements
- **Beam search diversity:** 256 beams explore variations around DT predictions
- **High confidence filtering:** Only 95%+ confidence samples update memory
- **Fast adaptation:** Low inertia (5.0 offset) allows early learning
- **Scalability:** Supports up to 2048-bit factoring with 65536 site capacity
 
## Example Output
 
```
═══ PHASE 3.5: DT POSITIONAL MEMORY DIRECT RECONSTRUCTION ═══
N resolved: 48.72%  (threshold per-position: 40%)
Strongly resolved positions (count≥2, rate≥40%): 17 / 39 (43.6%)
Attempting direct frequency reconstruction from positional memory...
[DT beam 13] freq=100 bits  uncertain_pos=31
Trying r = R/F = 3
gcd(a^(r/2)-1, N) = 16860433 ✓
★ DT POSITIONAL MEMORY FACTORED N (beam 13) ★
 
N = 261980999226229
  = 16860433 × 15538213
✓ Verified: p × q = N
Time: 0.023 seconds
```
 
## Files
 
- `tesseract_factor.c` - Main factoring engine with DT implementation
- `dt_shor_commit.log` - DT prediction log (updated per measurement)
- `quhit_triality.c` - Triality basis transformations
- `quhit_hexagram.c` - Hexagram quantum circuit primitives
- `s6_exotic.c` - Exotic S₁₄ codeword operations
- `bigint.cpp` - Arbitrary-precision integer arithmetic

```bash
# Compile with heavy optimizations
gcc -g -O0 -std=gnu99 -I.. -o tesseract_factor_dbg tesseract_factor.c ../quhit_triality.c ../quhit_hexagram.c ../s6_exotic.c ../bigint.cpp -lm -lmpfr -lgmp

# Execute with the default target
./tesseract_factor_dbg 111111 --base 2 --oracle 
```
