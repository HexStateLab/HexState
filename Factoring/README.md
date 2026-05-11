# Ouroboros Engine: Holographic Phase Contraction (HPC) Simulator

The Ouroboros Engine is an experimental, highly optimized C implementation of a semi-classical factoring heuristic. It explores post-quantum computational scaling by replacing dense tensor networks with a linear-time Holographic Phase Contraction (HPC) graph.

By operating entirely on $D=6$ quhits (`TrialityQuhits`) and mapping entanglement strictly to CZ phase edges, the engine completely bypasses standard wavefunction collapse. The state vector is never materialized; the entanglement lives intrinsically within the graph structure.

## ⚠️ Theoretical Bounding & Safety Disclaimer

**This engine is a classical simulation of a quantum heuristic, not a production cryptanalysis tool.** While it introduces radical efficiencies in graph traversal and precision management, it remains bounded by the laws of classical information theory.

As the target composite $N$ scales toward cryptographically relevant sizes (e.g., 2048-bit), the required sample complexity and graph bridging degrade. This engine poses **zero threat**(Maybe LOL) to modern RSA or TLS infrastructure. It is released strictly as a research tool for quantum information theorists, systems architects, and topologists studying measurement-induced phase transitions and AdS/CFT constraint mapping.

---

## Key Architectural Breakthroughs

### 1. The Holographic Bulk (AdS/CFT Mapping)
The engine abandons traditional all-to-all dense matrix simulation. Instead, it builds a topological hexagram cycle where phase kicks are applied to "boundary" sites (Sites 0 & 1), while unmeasured "bulk" sites (Sites 2-5) act as a resonant cavity. The unmeasured bulk holds the entanglement weights stable, preventing boundary frustration and anchoring the complex-domain Belief Propagation.

### 2. Implicit Approximate QFT (AQFT) via IEEE 754
In massive quantum simulations, continuous floating-point rounding noise usually destroys Shor interference peaks. The Ouroboros Engine naturally weaponizes the IEEE 754 double-precision limit. By allowing the hardware to natively truncate phase corrections smaller than machine epsilon across distant quhits, the engine implicitly implements Coppersmith’s Approximate QFT, acting as a perfect bandwidth-limited high-pass filter.

### 3. Cross-Base Carmichael Harvester
Instead of discarding "failed" quantum runs that yield fractured or partial periods, the engine treats every run as a valid fractional sample of the Carmichael Function, $\lambda(N)$. Using a cross-base LCM accumulator (`lcm(a,b) = (a*b)/gcd(a,b)`), the global period is incrementally stitched together from fragmentary data across multiple bases, dismantling the exponential sample-complexity wall of standard Shor implementations.

### 4. Automated Topological Resonance Probe
The engine does not blindly guess bases. Using the `auto_a` routine, it actively pings the manifold with the first 30 prime string candidates, measures the Network Residual Limit, and dynamically sorts the queue to find the resonant frequency of the graph before committing to the heavy continued-fraction expansion loop.

---

## Quick Start

The engine requires zero dependencies outside of standard `mpfr` and `gmp` libraries for the pre-computation phase. The heavy inner loops execute entirely via lightweight 64-bit floating-point math.

```bash
# Compile with heavy optimizations
gcc -O2 -std=gnu99 -I.. -o tesseract_factor tesseract_factor.c ../quhit_triality.c ../quhit_hexagram.c ../s6_exotic.c ../bigint.cpp -lm -lmpfr -lgmp

# Execute with the default target
./factor
```
