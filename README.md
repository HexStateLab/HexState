<p align="center">
HEAP COMPILE. Otherwise you will get segfault!
</p>

<h1 align="center">HexState</h1>

<p align="center">
  <strong>A D=6 Hexagonal Quantum Simulator with Native Triality Views<br/>and Sparse Magic Pointer Tensor Network Overlays</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#core-engine">Core Engine</a> •
  <a href="#hpc-graph">HPC Graph</a> •
  <a href="#triality">Triality</a> •
  <a href="#s6-exotic">S₆ Exotic</a> •
  <a href="#tensor-networks">Tensor Networks</a> •
  <a href="#dynamics">Dynamics</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#api-reference">API Reference</a>
</p>

<p align="center">
  <code>Pure C99</code> · <code>No Dependencies</code> · <code>Consumer Hardware</code> · <code>10^199 Dimensions in Kilobytes</code>
</p>

---

## Table of Contents

1. [Overview & Philosophy](#1-overview--philosophy)
2. [Quickstart](#2-quickstart)
3. [Architecture](#3-architecture)
4. [Core Engine — Quhits, Pairs & Registers](#4-core-engine)
5. [HPC Phase Graph — Holographic Phase Computation](#5-hpc-phase-graph)
6. [Triality Layer — Three Views, One Truth](#6-triality-layer)
7. [S₆ Outer Automorphism & Exotic Invariants](#7-s₆-outer-automorphism)
8. [Tensor Network Overlays — MPS through 6D](#8-tensor-network-overlays)
9. [Entropy-Adaptive Dynamics & Oracles](#9-entropy-adaptive-dynamics)
10. [Benchmarks, Examples & API Reference](#10-benchmarks--api-reference)

---

## 1. Overview & Philosophy

HexState is a quantum simulation engine built from a single observation: **the number 6 is special**.

The symmetric group S₆ is the *only* symmetric group possessing an outer automorphism. This means that D=6 quantum systems — **quhits** (quantum hexits) — carry structural information that no other dimension can express. HexState exploits this uniqueness at every layer of its architecture.

### Why D=6?

| Property | D=2 (Qubit) | D=3 (Qutrit) | D=6 (Quhit) |
|---|---|---|---|
| Basis states | 2 | 3 | **6** |
| Outer automorphism | ✗ | ✗ | **✓ (unique in all of mathematics)** |
| Native triality | ✗ | ✗ | **✓ (Edge/Vertex/Diagonal)** |
| CMY channels | ✗ | ✗ | **✓ (3 qubit subspaces)** |
| Synthemes | 0 | 0 | **15** |
| Synthematic totals | 0 | 0 | **6** |
| DFT order | 2 | 3 | **4 (DFT₆⁴ = I)** |

### Design Principles

1. **Algorithmic over Brute-Force**: Never materialize the full state vector. A 256-quhit system has 6²⁵⁶ ≈ 10¹⁹⁹ amplitudes — more than atoms in the observable universe. HexState represents this in kilobytes.

2. **Lazy Evaluation**: Gates accumulate as metadata. Amplitudes are only computed when measured. Most operations are O(D) or O(1), not O(D²).

3. **Sparse by Default**: The Magic Pointer register encodes only non-zero amplitudes via 128-bit packed basis states. A 100-trillion-quhit register fits in RAM if the state is sparse.

4. **Structure-Aware**: The S₆ outer automorphism, triality views, and synthematic decomposition are not afterthoughts — they are the foundation. Every gate, every measurement, every optimization exploits D=6 structure.

5. **Consumer Hardware**: Pure C99 with no external dependencies beyond libc/libm. Runs on any system with GCC and `__int128` support. No GPU required. No MPI required. No quantum hardware required.

### What Can You Simulate?

- **Condensed matter**: Fermi-Hubbard models on kagome/hexagonal lattices
- **Cryptography**: RSA modulus structural analysis via oracle interference
- **Quantum chemistry**: Molecular orbital simulations with 6-level atoms
- **Quantum information**: Entanglement dynamics, Bell inequality violations
- **Topological phases**: Exotic invariant detection across phase transitions
- **Tensor networks**: MPS, PEPS, and higher-D TNS with automatic bond management
- **Custom physics**: Any Hamiltonian expressible in the D=6 basis


---

## 2. Quickstart

### Requirements

- **GCC** with `__int128` support (GCC 4.6+, Clang 3.1+)
- **libc** and **libm** (standard C library)
- **SSE2** (any x86-64 processor made after 2003)
- Optional: **OpenMP** for parallel tensor network operations

### Building

```bash
# Clone the repository
git clone https://github.com/yourname/HexState.git
cd HexState

# Build a benchmark (e.g., Fermi-Hubbard on kagome lattice)
gcc -O2 -march=native -std=gnu99 \
    3.8-benchmark/fermi_hubbard_kagome.c \
    quhit_triality.c quhit_hexagram.c s6_exotic.c bigint.c \
    -lm -o fermi_hubbard

# Run it
ulimit -s unlimited   # Required for large static arrays
./fermi_hubbard
```

### Minimal Example — 4 Entangled Quhits

```c
#include "hpc_graph.h"
#include "s6_exotic.h"

int main(void) {
    /* One-time initialization */
    triality_exotic_init();
    s6_exotic_init();

    /* Create 4 quhits in an HPC phase graph */
    HPCGraph *g = hpc_create(4);

    /* Put all sites into uniform superposition via DFT₆ */
    for (int i = 0; i < 4; i++)
        triality_dft(&g->locals[i]);

    /* Entangle into a ring: 0↔1↔2↔3↔0 */
    hpc_cz(g, 0, 1);
    hpc_cz(g, 1, 2);
    hpc_cz(g, 2, 3);
    hpc_cz(g, 3, 0);

    /* Check the exotic invariant (D=6 unique!) */
    double delta = hpc_exotic_invariant(g);
    printf("Exotic invariant Δ = %.6f\n", delta);

    /* Measure site 0 — collapses and absorbs phases into partners */
    uint32_t outcome = hpc_measure(g, 0, 0.42);
    printf("Site 0 collapsed to |%u⟩ = %s\n", outcome,
           (char*[]){"Edge","Vertex","Diagonal","Folded","Exotic","Tetra"}[outcome]);

    /* Print statistics */
    hpc_print_stats(g);

    hpc_destroy(g);
    return 0;
}
```

### Build Commands by Use Case

```bash
# HPC-only (most simulations) — minimal build
gcc -O2 -march=native myfile.c quhit_triality.c s6_exotic.c -lm

# With hexagram duality
gcc -O2 -march=native myfile.c quhit_triality.c quhit_hexagram.c s6_exotic.c bigint.c -lm

# Full build (all overlays)
gcc -O2 -march=native -fopenmp myfile.c \
    quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \
    quhit_register.c quhit_substrate.c quhit_triality.c \
    quhit_triadic.c quhit_lazy.c quhit_calibrate.c \
    quhit_hexagram.c quhit_svd_gate.c s6_exotic.c bigint.c \
    mps_overlay.c peps_overlay.c peps3d_overlay.c \
    peps4d_overlay.c peps5d_overlay.c peps6d_overlay.c \
    -lm -msse2

# Template (all sections demonstrated)
gcc -O2 -march=native 3.8-benchmark/hexstate_template.c \
    quhit_triality.c quhit_hexagram.c s6_exotic.c bigint.c -lm
```

> **⚠️ Important**: Large simulations use static arrays that may exceed the default stack size. Always run `ulimit -s unlimited` before executing, or compile with `-Wl,-z,stacksize=67108864`.


---

## 3. Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER APPLICATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  HPC Phase   │  │   Triadic    │  │   Tensor Network         │  │
│  │  Graph       │  │   3-Body     │  │   Overlays               │  │
│  │              │  │   Joints     │  │                          │  │
│  │  O(N+E)      │  │   216 amps   │  │  MPS(1D) → PEPS(2D)     │  │
│  │  holographic │  │   CMY chan.  │  │  → TNS(3D) → ... → 6D   │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│         │                 │                        │                │
│  ┌──────┴─────────────────┴────────────────────────┴─────────────┐  │
│  │                    TRIALITY LAYER                              │  │
│  │                                                                │  │
│  │  TrialityQuhit: 6 views × 6 amplitudes                       │  │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐            │  │
│  │  │  Edge   │ │  Vertex  │ │ Diagonal │ │ Folded │ ...        │  │
│  │  │ (comp.) │ │ (Fourier)│ │ (F²)     │ │ (anti- │            │  │
│  │  │ O(D)    │ │ O(D)     │ │ O(D)     │ │ podal) │            │  │
│  │  └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬───┘            │  │
│  │       └───── lazy DFT₆ ────────┘            │                │  │
│  │                                               │                │  │
│  │  ┌────────────────────┐  ┌───────────────────┐                │  │
│  │  │  S₆ Exotic Engine  │  │  Hexagram Duality │                │  │
│  │  │  720 permutations  │  │  H₆ transform     │                │  │
│  │  │  15 synthemes      │  │  Chirality ±1      │                │  │
│  │  │  Δ invariant       │  │  Kramers-Wannier   │                │  │
│  │  └────────────────────┘  └───────────────────┘                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    CORE ENGINE                                 │  │
│  │                                                                │  │
│  │  Quhit (6 amps) → QuhitPair (36 amps) → QuhitRegister        │  │
│  │                                           (128-bit packed,     │  │
│  │                                            sparse Magic Ptrs)  │  │
│  │                                                                │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │  │
│  │  │ DFT₆     │ │ CZ       │ │ Phase    │ │ Born Measurement │  │  │
│  │  │ X, Z     │ │ Bell     │ │ Unitary  │ │ Partial Trace    │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  ENTROPY-ADAPTIVE DYNAMICS (DynChain / DynLattice)             │  │
│  │  6 Oracles: Prediction · Correlation · Convergence ·           │  │
│  │             Phase Boundary · Site Weight · Ouroboros            │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### File Organization

```
HexState/
├── quhit_engine.h          # Core: Quhit, QuhitPair, QuhitRegister, QuhitEngine
├── quhit_core.c            # Lifecycle, initialization
├── quhit_gates.c           # DFT₆, X, Z, phase, arbitrary unitary
├── quhit_measure.c         # Born sampling, partial trace
├── quhit_entangle.c        # Bell pairs, product pairs, CZ
├── quhit_register.c        # Magic Pointer sparse state vectors
├── quhit_substrate.c       # 128-bit basis packing (__int128)
│
├── quhit_triality.h        # Triality: 6 views, lazy DFT₆, enhancement flags
├── quhit_triality.c        # View conversion, optimal-view gate routing
│
├── hpc_graph.h             # HPC: Holographic Phase Graph (header-only, inline)
├── hpc_contract.h          # Syntheme-aware bond encoding
│
├── s6_exotic.h             # S₆: outer automorphism, synthemes, Δ invariant
├── s6_exotic.c             # Permutation tables, exotic gates, tomography
│
├── quhit_triadic.h         # 3-body: TriadicJoint (216 amps), CMY channels
├── quhit_triadic.c         # Triadic gates, marginals, entropy
│
├── quhit_hexagram.h        # Hexagram: edge-model duality, chirality
├── quhit_hexagram.c        # H₆ transform, path shift, triad gate
│
├── quhit_lazy.c            # Lazy evaluation: Heisenberg-picture accumulation
├── quhit_calibrate.c       # Self-test and calibration
├── quhit_svd_gate.c        # SVD-based gate decomposition
│
├── mps_overlay.h/c         # 1D MPS (χ=256)
├── peps_overlay.h/c        # 2D PEPS (χ=512)
├── peps3d_overlay.h/c      # 3D TNS (χ=256)
├── peps4d_overlay.h/c      # 4D TNS (χ=128)
├── peps5d_overlay.h/c      # 5D TNS (χ=64)
├── peps6d_overlay.h/c      # 6D TNS (χ=32) — the native dimension
├── triality_overlay.h      # Shared triality sidecar for overlays
│
├── quhit_peps_grow.h       # DynLattice: entropy-adaptive growth engine
├── quhit_dyn_integrate.h   # DynChain: 1D dynamics with 6 oracles
│
├── bigint.h/c              # Arbitrary-precision integer support
│
├── 3.8-benchmark/          # Example simulations and benchmarks
│   ├── fermi_hubbard_kagome.c    # Condensed matter physics
│   ├── test_triality_entangle.c  # Entanglement verification
│   ├── rsa_oracle_sim.c          # Cryptographic oracle harvest
│   └── hexstate_template.c       # Comprehensive API template
│
└── README.md               # This file
```

### Memory Model

HexState uses three complementary memory strategies:

| Layer | Storage | Scaling | Use Case |
|---|---|---|---|
| **Quhit** | 12 doubles (96 bytes) | O(1) per site | Individual qudit states |
| **QuhitPair** | 72 doubles (576 bytes) | O(1) per pair | Pairwise entanglement |
| **QuhitRegister** | Sparse hash map | O(nnz) | Large-scale computation |
| **HPC Graph** | Local states + edges | O(N + E) | Holographic simulation |
| **TriadicJoint** | 432 doubles (3,456 bytes) | O(1) per triple | Three-body entanglement |
| **Tensor Network** | Register per site | O(N × χ^d) | Structured lattice sims |

The key insight: **entanglement is stored as graph edges, not as exponential state vectors.** A 256-site system with 255 CZ edges requires ~100 KB, not the 10¹⁹⁹ bytes a full state vector would need.


---

## 4. Core Engine

> **Header**: `quhit_engine.h` · **Sources**: `quhit_core.c`, `quhit_gates.c`, `quhit_measure.c`, `quhit_entangle.c`, `quhit_register.c`

The core engine provides three abstraction levels for quantum state management.

### 4.1 Quhit — The Quantum Hexit

A single D=6 quantum digit with 6 complex amplitudes:

```
|ψ⟩ = α₀|0⟩ + α₁|1⟩ + α₂|2⟩ + α₃|3⟩ + α₄|4⟩ + α₅|5⟩
```

```c
// Initialization
uint32_t q = quhit_init(eng);              // |0⟩
uint32_t q = quhit_init_plus(eng);         // (1/√6) Σ|k⟩
uint32_t q = quhit_init_basis(eng, k);     // |k⟩

// Gates
quhit_apply_dft(eng, q);                   // DFT₆: |k⟩ → (1/√6)Σ ω^(jk)|j⟩
quhit_apply_idft(eng, q);                  // Inverse DFT₆
quhit_apply_x(eng, q);                     // X: |k⟩ → |k+1 mod 6⟩
quhit_apply_z(eng, q);                     // Z: |k⟩ → ω^k|k⟩
quhit_apply_phase(eng, q, phases);         // Custom diagonal phases
quhit_apply_unitary(eng, q, U_re, U_im);   // Arbitrary 6×6 unitary

// Measurement
uint32_t outcome = quhit_measure(eng, q);   // Born sampling → {0,...,5}
double p = quhit_prob(eng, q, k);           // P(outcome = k)
quhit_inspect(eng, q, &snap);              // Non-destructive snapshot
```

### 4.2 QuhitPair — Pairwise Entanglement

Two entangled quhits share a 6×6 = 36-amplitude joint state |ψ⟩ = Σ α_{jk} |j,k⟩:

```c
// Create entangled pairs
quhit_entangle_bell(eng, qa, qb);      // Maximally entangled: (1/√6)Σ|k,k⟩
quhit_entangle_product(eng, qa, qb);   // Separable: |ψ_a⟩⊗|ψ_b⟩

// Two-qudit gates
quhit_apply_cz(eng, qa, qb);           // CZ: |j,k⟩ → ω^(jk)|j,k⟩

// Disentangle
quhit_disentangle(eng, qa, qb);        // Partial trace → product states
```

### 4.3 QuhitRegister — 100-Trillion Scale

Sparse state vectors encoded via **Magic Pointers**: 128-bit packed basis states where only non-zero amplitudes are stored.

```c
// Create a register of N quhits
int reg = quhit_reg_init(eng, chunk_id, n_quhits, D);

// Register-level operations
quhit_reg_apply_dft(eng, reg, qubit_pos);
quhit_reg_apply_cz(eng, reg, pos_a, pos_b);
quhit_reg_apply_unitary_pos(eng, reg, pos, U_re, U_im);

// Measurement
uint64_t outcome = quhit_reg_measure(eng, reg, qubit_pos);

// State vector inspection
double total = quhit_reg_sv_total_prob(eng, reg);  // Should be ≈ 1.0
```

**Basis encoding**: For N quhits at positions p₀, p₁, ..., p_{N-1} with values v₀, v₁, ..., v_{N-1}:

```
basis_index = v₀ × 6⁰ + v₁ × 6¹ + ... + v_{N-1} × 6^{N-1}
```

This 128-bit index is stored as a `__int128` and used as the key in a sparse hash map. Only basis states with non-zero amplitude consume memory.


---

## 5. HPC Phase Graph

> **Header**: `hpc_graph.h` (header-only, all inline)

The Holographic Phase Computation graph is the **primary simulation engine** for most HexState workloads. It replaces exponential state vectors with a polynomial graph representation.

### 5.1 The Key Idea

Instead of storing the full state vector ψ(i₁, i₂, ..., iₙ) with D^N amplitudes, the HPC graph factors it as:

```
ψ(i₁, ..., iₙ) = [∏ₖ aₖ(iₖ)] × [∏_edges wₑ(iₐ, iᵦ)]
```

- **Local states** `aₖ(iₖ)`: 6 complex amplitudes per site (stored as `TrialityQuhit`)
- **Phase edges** `wₑ(iₐ, iᵦ)`: 6×6 phase matrix per edge

**Memory**: O(N × 12 doubles + E × 72 doubles) instead of O(6^N × 2 doubles)

For a 256-site system: **~100 KB** instead of **10¹⁹⁹ bytes**.

### 5.2 Edge Types

| Type | Phase Matrix | Fidelity | Cost |
|---|---|---|---|
| **CZ** | `w(a,b) = ω^(ab)` | Exact (1.0) | O(1) to add |
| **Phase** | General `w(a,b)` | Approximate | O(36) to add |
| **Syntheme** | S₆-structured | Approximate | O(36) to add |

CZ edges are the workhorse — they encode the standard controlled-phase gate exactly and can be compacted: `n` CZ edges between the same pair merge to `ω^(n·a·b)`, and cancel entirely when `n ≡ 0 mod 6`.

### 5.3 Core API

```c
// Lifecycle
HPCGraph *g = hpc_create(n_sites);
hpc_destroy(g);
hpc_grow_sites(g, new_n);          // Add sites dynamically

// Local gates — modify ONLY the local state, NEVER edges
hpc_set_local(g, site, re, im);    // Set amplitudes directly
hpc_dft(g, site);                  // DFT₆
hpc_phase(g, site, phi_re, phi_im);// Phase gate
hpc_shift(g, site, delta);         // Cyclic shift

// Entangling gates — add edges
hpc_cz(g, site_a, site_b);        // Exact CZ edge
hpc_general_2site(g, a, b, w_re, w_im);  // General phase edge

// Edge management
hpc_compact_edges(g);              // Merge redundant CZ edges

// Amplitude evaluation — O(N + E) per query
hpc_amplitude(g, indices, &re, &im);     // ψ(i₁,...,iₙ)
double p = hpc_probability(g, indices);   // |ψ|²
double m = hpc_marginal(g, site, value);  // P(site = value)

// Measurement — collapses site, absorbs phases into partners
uint32_t outcome = hpc_measure(g, site, random_01);

// Observables
double S = hpc_entropy_cut(g, cut_after);     // Bipartition entropy
double delta = hpc_exotic_invariant(g);       // Mean Δ across sites
double norm = hpc_norm_sq(g);                 // Σ|ψ|² (small N only!)

// Diagnostics
hpc_print_stats(g);
hpc_print_state(g, "label");
```

### 5.4 How Measurement Works

When site `k` is measured with outcome `v`:

1. **Local collapse**: site `k`'s state becomes `|v⟩`
2. **Phase absorption**: for each edge touching site `k`, the partner's state is updated:
   ```
   partner[j] *= w(v, j)   // Absorb the phase
   ```
3. **Edge removal**: all edges touching site `k` are deleted
4. **Result**: site `k` is now disentangled from the rest of the graph

This is **measurement-induced disentanglement** — the graph shrinks with every measurement.

### 5.5 Performance Characteristics

| Operation | Cost | Notes |
|---|---|---|
| `hpc_cz` | O(1) | Just appends an edge |
| `hpc_dft` | O(D²) = O(36) | Local DFT₆ on one site |
| `hpc_phase` | O(D) = O(6) | Diagonal gate |
| `hpc_amplitude` | O(N + E) | Full amplitude evaluation |
| `hpc_marginal` | O(degree) | Uses adjacency list |
| `hpc_measure` | O(degree) | Removes edges touching site |
| `hpc_compact_edges` | O(E) | One-pass merge |


---

## 6. Triality Layer

> **Header**: `quhit_triality.h` · **Source**: `quhit_triality.c`

Triality is the heart of HexState's performance model. Every quantum state is simultaneously represented in **six different bases**, and gates automatically execute in whichever view is cheapest.

### 6.1 The Six Views

| View | Index | Basis | Relation to Edge | Gate Cost |
|---|---|---|---|---|
| **Edge** | 0 | Computational |  Identity | Phase: O(D), DFT: O(D²) |
| **Vertex** | 1 | Fourier | DFT₆ | X (shift): O(D) |
| **Diagonal** | 2 | Double-Fourier | DFT₆² | Interference: O(D) |
| **Folded** | 3 | Antipodal pairs | Hadamard on 0↔3,1↔4,2↔5 | Fold gates: O(18) |
| **Exotic** | 4 | S₆-parameterized | Syntheme projection | Exotic gates: O(D) |
| **Tetra** | 5 | DFT₆ eigenbasis | Eigendecomposition | Eigenstate DFT: O(D) |

The DFT₆ connects the first three views cyclically: Edge →^{DFT₆} Vertex →^{DFT₆} Diagonal →^{DFT₆} Edge (since DFT₆ has order 4, three applications return to start with a phase factor).

### 6.2 Lazy View Conversion

Views are computed **on demand**. Each view has a dirty bit:

```c
// Only converts if the vertex view is stale
triality_ensure_view(&q, VIEW_VERTEX);

// Read amplitudes (ensures first, then returns pointer)
const double *re = triality_view_re(&q, VIEW_VERTEX);
```

When a gate modifies one view, it marks others as dirty. The conversion only happens when another view is actually needed — often never.

### 6.3 Enhancement Flags

Three flags enable fast paths that skip unnecessary computation:

- **Eigenstate Phase-Lock** (`eigenstate_class ≥ 0`): DFT₆ eigenstates convert between views in O(1) via phase multiplication, not O(D²) matrix-vector multiply.

- **Subspace Confinement** (`active_mask`): A 6-bit mask tracking which basis states have non-zero amplitude. A state confined to 2 of 6 basis states skips 4/6 of all loop iterations.

- **Real-Valued Fast Path** (`real_valued`): When all imaginary parts are zero, complex multiplication reduces to real multiplication — 2× speedup on all gates.

```c
triality_refresh_flags(&q);   // Detect and set all three flags
```

### 6.4 Key Operations

```c
// Lifecycle
triality_init(&q);                     // |0⟩
triality_init_basis(&q, k);            // |k⟩
triality_copy(&dst, &src);             // Deep copy

// Gates (optimal view selected automatically)
triality_dft(&q);                      // DFT₆
triality_idft(&q);                     // Inverse DFT₆
triality_x(&q);                        // X (cyclic shift +1)
triality_z(&q);                        // Z (phase ω^k)
triality_shift(&q, delta);             // Generalized X (shift by δ)
triality_phase(&q, phi_re, phi_im);    // Diagonal phase gate
triality_phase_single(&q, k, re, im);  // Phase on single basis state — O(1)!
triality_cz(&q1, &q2);                // CZ between two quhits
triality_unitary(&q, view, U_re, U_im); // Arbitrary 6×6 in specified view

// Triality rotation — O(1), zero arithmetic cost!
triality_rotate(&q);       // Edge→Vertex→Diagonal→Edge (relabeling only)
triality_rotate_inv(&q);

// Folded view
triality_fold(&q);         // Antipodal Hadamard: pairs (0↔3, 1↔4, 2↔5)
triality_unfold(&q);
triality_ensure_view_via_fold(&q, VIEW_VERTEX);  // O(18) instead of O(36)

// Tetrahedral eigenbasis
triality_ensure_tetra(&q);
triality_dft_via_tetra(&q);  // O(D) DFT for eigenstates

// Measurement
int outcome = triality_measure(&q, VIEW_EDGE, &rng);
triality_probabilities(&q, VIEW_EDGE, probs);

// Diagnostics
triality_print(&q, "label");
triality_stats_print();
```

### 6.5 The Folded View Optimization

The folded view exploits the antipodal symmetry of the hexagon. Basis states are paired: {0↔3, 1↔4, 2↔5}. Each pair undergoes a 2×2 Hadamard transform:

```
|k_fold⟩ = (|k⟩ + |k+3⟩) / √2    (vesica)
|k+3_fold⟩ = (|k⟩ - |k+3⟩) / √2  (wave)
```

This decomposes the 6×6 DFT₆ into three 2×2 Hadamards (O(18)) followed by a permutation (O(6)), instead of a full 6×6 matrix multiply (O(36)). A **50% speedup** on DFT₆-heavy workloads.


---

## 7. S₆ Outer Automorphism

> **Header**: `s6_exotic.h` · **Source**: `s6_exotic.c`

The symmetric group S₆ (all 720 permutations of 6 elements) possesses a mathematical miracle: it is the **only** symmetric group with an outer automorphism. This automorphism φ swaps:

- Transpositions (ab) ↔ Triple transpositions (ab)(cd)(ef)
- 3-cycles (abc) ↔ Double 3-cycles (abc)(def)

HexState exploits this unique structure as a quantum resource.

### 7.1 Synthemes and Totals

A **syntheme** is a partition of {0,1,2,3,4,5} into 3 unordered pairs. There are exactly **15 synthemes**:

```
{01|23|45}, {01|24|35}, {01|25|34},
{02|13|45}, {02|14|35}, {02|15|34},
{03|12|45}, {03|14|25}, {03|15|24},
{04|12|35}, {04|13|25}, {04|15|23},
{05|12|34}, {05|13|24}, {05|14|23}
```

A **synthematic total** is a maximal set of 5 mutually disjoint synthemes. There are exactly **6 totals** — matching the dimension D=6. This is not a coincidence; it is the outer automorphism in action.

### 7.2 The Exotic Invariant Δ

The exotic invariant measures how much a quantum state exploits the D=6 structure:

```
Δ(ψ) = (1/720) Σ_{σ ∈ S₆} |⟨ψ|σ|ψ⟩ - ⟨ψ|φ(σ)|ψ⟩|²
```

- **Δ = 0**: The state is "automorphism-transparent" — it could be simulated with qubits
- **Δ > 0**: The state is "hexagonally polarized" — it carries structure unique to D=6

```c
s6_exotic_init();                              // One-time table setup

double delta = s6_exotic_invariant(re, im);    // Full Δ computation

// Cheap approximation (48× faster)
double witness = s6_cross_syntheme_witness(re, im);

// Per-conjugacy-class breakdown (11 classes in S₆)
double class_deltas[11];
s6_exotic_fingerprint(re, im, class_deltas);
```

### 7.3 Syntheme-Parameterized Folding

Each syntheme defines a different way to fold the 6-dimensional state into 3 pairs:

```c
// Fold via syntheme #7 (one of 15 possible decompositions)
s6_fold_syntheme(re, im, fold_re, fold_im, 7);

// Unfold (lossless round-trip)
s6_unfold_syntheme(fold_re, fold_im, out_re, out_im, 7);

// Find the optimal syntheme for a given active_mask
int best = s6_optimal_syntheme(active_mask);

// Find the minimum-entropy syntheme
int min_s = s6_min_entropy_syntheme(re, im);
```

### 7.4 Dual Measurement

Every measurement can be performed in both the **standard** (computational) basis and the **exotic** (φ-image) basis simultaneously:

```c
int std_outcome, exo_outcome;
std_outcome = triality_measure_dual(&q, VIEW_EDGE, syntheme, &rng, &exo_outcome);
```

The exotic outcome carries information about the S₆ structure that the standard outcome cannot access.

### 7.5 Tomographic Reconstruction

A quantum state can be reconstructed from 5 fold projections of a single synthematic total:

```c
double fidelity = s6_total_tomography(total_idx, fold_data_re, fold_data_im,
                                       recon_re, recon_im);
```

This provides an alternative to standard quantum state tomography that exploits the D=6 structure for fewer measurements.

---

## 8. Tensor Network Overlays

> **Headers**: `mps_overlay.h`, `peps_overlay.h`, `peps3d_overlay.h`, ..., `peps6d_overlay.h`

HexState provides tensor network overlays from 1D to 6D, all built on top of the Magic Pointer register substrate. Each site's tensor **is** a QuhitRegister — no classical tensor arrays are ever allocated.

### 8.1 MPS — 1D Matrix Product States (χ=256)

```c
MpsChain *mps = mps_init(L);                    // L-site chain

// State preparation
mps_set_product_state(mps, site, amps_re, amps_im);

// Gate helpers
double U_re[36], U_im[36], G_re[1296], G_im[1296];
mps_build_dft6(U_re, U_im);                    // Single-site DFT₆
mps_build_cz(G_re, G_im);                      // Two-site CZ (36×36)

// Gate application
mps_gate_1site(mps, site, U_re, U_im);         // Local gate
mps_gate_bond(mps, site, G_re, G_im);          // Bond gate (triggers SVD)
mps_gate_1site_all(mps, U_re, U_im);           // Batch: all sites
mps_gate_bond_all(mps, G_re, G_im);            // Batch: all bonds
mps_trotter_step(mps, G_re, G_im);             // Even/odd sweep

// Observables
double probs[6];
mps_local_density(mps, site, probs);
int outcome = mps_measure_site(mps, site);

mps_free(mps);
```

Each site tensor has 3 indices: `|k, α, β⟩` where `k ∈ [0,6)` is physical and `α, β ∈ [0, χ)` are bond indices.

### 8.2 PEPS — 2D Projected Entangled Pair States (χ=512)

```c
PepsGrid *peps = peps_init(Lx, Ly);

peps_set_product_state(peps, x, y, amps_re, amps_im);

peps_gate_1site(peps, x, y, U_re, U_im);
peps_gate_horizontal(peps, x, y, G_re, G_im);  // Bond: (x,y)↔(x+1,y)
peps_gate_vertical(peps, x, y, G_re, G_im);    // Bond: (x,y)↔(x,y+1)
peps_trotter_step(peps, G_re, G_im);

peps_local_density(peps, x, y, probs);
int out = peps_measure_site(peps, x, y);

peps_free(peps);
```

Each site tensor has 5 indices: `|k, u, d, l, r⟩` (physical + up/down/left/right bonds). Red-black checkerboard parallelism is used for batch operations with OpenMP.

### 8.3 Higher Dimensions (3D–6D)

| Dimension | Type | Init | χ | Indices per Site |
|---|---|---|---|---|
| 3D | `Tns3dGrid` | `tns3d_init(Lx,Ly,Lz)` | 256 | 7: `k,u,d,l,r,f,b` |
| 4D | `Tns4dGrid` | `tns4d_init(Lx,Ly,Lz,Lw)` | 128 | 9: `k` + 8 bonds |
| 5D | `Tns5dGrid` | `tns5d_init(Lx,Ly,Lz,Lw,Lv)` | 64 | 11: `k` + 10 bonds |
| 6D | `Tns6dGrid` | `tns6d_init(Lx,Ly,Lz,Lw,Lv,Lu)` | 32 | 13: `k` + 12 bonds |

All follow the same API pattern:
```c
tns3d_gate_1site(grid, x, y, z, U_re, U_im);
tns3d_gate_x(grid, x, y, z, G_re, G_im);      // Bond along x-axis
tns3d_gate_y(grid, x, y, z, G_re, G_im);      // Bond along y-axis
tns3d_gate_z(grid, x, y, z, G_re, G_im);      // Bond along z-axis
tns3d_local_density(grid, x, y, z, probs);
tns3d_measure_site(grid, x, y, z);
tns3d_free(grid);
```

The 6D overlay is the **native dimension** — matching the D=6 quhit with 6 spatial axes, each with 2 bond directions (12 bonds + 1 physical = 13 indices per tensor).


---

## 9. Entropy-Adaptive Dynamics

> **Headers**: `quhit_dyn_integrate.h`, `quhit_peps_grow.h`

The dynamics engine provides **breathing lattices** — simulation grids that grow and contract in real-time based on entanglement entropy. Six predictive oracles guide the decisions.

### 9.1 DynChain — 1D Adaptive Chain

```c
DynChain dc = dyn_chain_create(max_sites);

// Configure
dc.grow_threshold = 0.5;       // Expand when boundary entropy > this
dc.contract_threshold = 0.1;   // Contract when tail entropy < this
dc.min_active = 4;             // Never shrink below this

// Seed active region
dyn_chain_seed(&dc, start, end);
```

### 9.2 The Six Oracles

| Oracle | Purpose | Output |
|---|---|---|
| **1. Entropy Prediction** | Linear extrapolation of per-site entropy trends | Predicted H for next step |
| **2. Mutual Information** | Jensen-Shannon divergence between all site pairs | Optimal coupling pair |
| **3. Convergence Horizon** | Detects oscillation vs. convergence vs. stagnation | State flag |
| **4. Phase Boundary** | Finds sharpest entropy gradient (structural edge) | Site index |
| **5. Site Weight** | Information weight = 1 - H/H_max per site | Contraction priority |
| **6. Ouroboros** | Self-referential Boltzmann scoring of oracles 1–5 | Gate recommendation |

### 9.3 The Ouroboros Self-Optimization

Oracle 6 is special: it observes the *other* oracles and learns from their performance. Key discoveries from the closed time-like curve (CTC) convergence loop:

- **DFT-family gates** reduce entropy ~22% vs identity/Z₆/exotic gates
- **Correlation oracle** (Oracle 2) provides ~6% additional entropy reduction beyond gate-only optimization
- **Hold topology** is preferred over aggressive grow/contract
- **Optimal β = 2.42** for Boltzmann oracle weighting

```c
// Standard step (uses oracles 1–5)
dyn_chain_step(&dc);

// Ouroboros-optimized step (uses all 6 oracles)
dyn_chain_ouroboros_step(&dc);

// Query the Ouroboros recommendations
int gate = dyn_chain_recommended_gate(&dc);   // 0–5: which gate to apply
int top = dyn_chain_top_oracle(&dc);          // Which oracle is most valuable
```

### 9.4 Higher-D DynLattice

```c
DynLattice *dl = dyn_peps2d_create(Lx, Ly);        // 2D
DynLattice *dl = dyn_tns3d_create(Lx, Ly, Lz);     // 3D
DynLattice *dl = dyn_tns4d_create(Lx, Ly, Lz, Lw); // 4D
// ... up to 6D

// Activity guard — place before every gate!
if (dyn_peps2d_active(dl, x, y)) {
    peps_gate_1site(grid, x, y, U_re, U_im);
}

// Feed entropy from observables
peps_local_density(grid, x, y, probs);
dyn_peps2d_entropy(dl, x, y, probs, D);

// Step the growth engine
dyn_lattice_step(dl);
```

### 9.5 Convergence States

```
CONVERGING  → entropy monotonically decreasing → keep going
OSCILLATING → entropy bouncing > 3 times       → contract to stabilize
STAGNANT    → entropy change < ε for 4+ epochs → grow to bring fresh sites
```

---

## 10. Benchmarks & API Reference

### 10.1 Included Benchmarks

| Benchmark | File | Sites | Description |
|---|---|---|---|
| **Fermi-Hubbard Kagome** | `fermi_hubbard_kagome.c` | 96 | Condensed matter on hexagonal-symmetry lattice |
| **Triality Entangle** | `test_triality_entangle.c` | 8 | Entanglement verification and phase graph test |
| **RSA Oracle** | `rsa_oracle_sim.c` | 256 | RSA-2048 modulus structural fingerprinting |
| **Template** | `hexstate_template.c` | various | All 10 subsystems demonstrated |

### 10.2 Performance Results

From `fermi_hubbard_kagome.c` (96 sites, 10 Trotter steps):

```
╔═════════════════════════════════════════════════════╗
║  Holographic Phase Graph Statistics                ║
╠═════════════════════════════════════════════════════╣
║  Sites:                   96                       ║
║  Total edges:            285                       ║
║  CZ (exact):             285                       ║
║  Memory:              ~48 KB                       ║
║  Full SV:         10^74 bytes (impossible)          ║
╚═════════════════════════════════════════════════════╝
```

**48 KB** to represent a system that would classically require **10⁷⁴ bytes**.

### 10.3 Complete API Quick Reference

#### HPC Graph (`hpc_graph.h`)
```
hpc_create(n)              hpc_destroy(g)            hpc_grow_sites(g, n)
hpc_set_local(g, s, r, i)  hpc_dft(g, s)             hpc_phase(g, s, r, i)
hpc_shift(g, s, δ)         hpc_cz(g, a, b)           hpc_general_2site(g, ...)
hpc_compact_edges(g)       hpc_amplitude(g, idx, ...)  hpc_probability(g, idx)
hpc_marginal(g, s, v)      hpc_measure(g, s, r)       hpc_entropy_cut(g, cut)
hpc_exotic_invariant(g)    hpc_norm_sq(g)             hpc_print_stats(g)
```

#### Triality (`quhit_triality.h`)
```
triality_init(&q)          triality_init_basis(&q, k)  triality_copy(&d, &s)
triality_dft(&q)           triality_idft(&q)           triality_x(&q)
triality_z(&q)             triality_shift(&q, δ)       triality_phase(&q, r, i)
triality_phase_single(...)  triality_cz(&a, &b)        triality_unitary(...)
triality_rotate(&q)        triality_fold(&q)           triality_unfold(&q)
triality_ensure_view(...)  triality_measure(...)        triality_probabilities(...)
triality_print(&q, label)  triality_stats_print()
```

#### S₆ Exotic (`s6_exotic.h`)
```
s6_exotic_init()                   s6_exotic_invariant(re, im)
s6_exotic_fingerprint(re, im, d)   s6_dual_probabilities(re, im, s, e)
s6_fold_syntheme(...)              s6_unfold_syntheme(...)
s6_optimal_syntheme(mask)          s6_cross_syntheme_witness(re, im)
s6_total_tomography(...)           s6_apply_exotic_gate(...)
```

#### Core Engine (`quhit_engine.h`)
```
quhit_engine_init(eng)     quhit_engine_destroy(eng)
quhit_init(eng)            quhit_init_plus(eng)       quhit_init_basis(eng, k)
quhit_apply_dft(eng, q)   quhit_apply_cz(eng, a, b)  quhit_apply_phase(eng, q, p)
quhit_measure(eng, q)     quhit_inspect(eng, q, &s)   quhit_prob(eng, q, k)
quhit_reg_init(eng, ...)  quhit_reg_apply_dft(...)    quhit_reg_measure(...)
```

### 10.4 Writing Your Own Simulation

The fastest path to a working simulation:

1. **Copy the template**: `cp 3.8-benchmark/hexstate_template.c my_sim.c`
2. **Pick your engine**: Most simulations only need the HPC graph (§2)
3. **Encode your problem**: Map your Hamiltonian to D=6 phase gates + CZ entanglement
4. **Trotter evolve**: Alternate local gates and CZ gates
5. **Measure**: Extract observables via `hpc_marginal` or `hpc_measure`
6. **Analyze**: Check exotic invariants for D=6-specific structure

```c
// Minimal simulation skeleton
HPCGraph *g = hpc_create(N);

for (int i = 0; i < N; i++)
    triality_dft(&g->locals[i]);     // Superposition

for (int step = 0; step < depth; step++) {
    for (int i = 0; i < N; i++)
        hpc_phase(g, i, phi_re, phi_im);  // On-site Hamiltonian
    for (int i = 0; i < N-1; i++)
        hpc_cz(g, i, i+1);               // Coupling
    if (step % 3 == 0)
        hpc_compact_edges(g);              // Prevent edge bloat
}

for (int i = 0; i < N; i++) {
    double probs[6];
    for (int v = 0; v < 6; v++)
        probs[v] = hpc_marginal(g, i, v);
    // ... analyze probs ...
}

hpc_destroy(g);
```

---

## License

MIT License. See `LICENSE` for details.

## Citation

None needed

---
