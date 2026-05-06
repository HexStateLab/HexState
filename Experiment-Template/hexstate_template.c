/*
 * ╔═══════════════════════════════════════════════════════════════════════════════╗
 * ║  hexstate_template.c — Generic HexState Quantum Operations Template         ║
 * ╠═══════════════════════════════════════════════════════════════════════════════╣
 * ║                                                                             ║
 * ║  A comprehensive, runnable template demonstrating every major subsystem     ║
 * ║  of the HexState D=6 quantum simulator. Each section is self-contained     ║
 * ║  and heavily commented so it can be copy-pasted into your own simulation.   ║
 * ║                                                                             ║
 * ║  Subsystems demonstrated:                                                   ║
 * ║    §1  Core Engine        — quhits, registers, basic gates, measurement    ║
 * ║    §2  HPC Phase Graph    — holographic phase encoding, CZ edges           ║
 * ║    §3  Triality Layer     — Edge/Vertex/Diagonal views, optimal-view gates ║
 * ║    §4  S₆ Exotic          — outer automorphism, synthemes, Δ invariant     ║
 * ║    §5  Three-Body (Triadic) — 216-amplitude joints, CMY channels           ║
 * ║    §6  Hexagram Duality   — edge-model quhits, chirality, H₆ transform    ║
 * ║    §7  Lazy Evaluation    — Heisenberg-picture gate accumulation           ║
 * ║    §8  Tensor Networks    — MPS(1D), PEPS(2D), higher-D overlays           ║
 * ║    §9  Entropy Dynamics   — DynChain, 6 oracles, Ouroboros self-tuning     ║
 * ║    §10 Full Pipeline      — Combining all subsystems in a real workflow    ║
 * ║                                                                             ║
 * ║  Build (all sections):                                                      ║
 * ║    gcc -O2 -march=native -o hexstate_template hexstate_template.c \         ║
 * ║        quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \       ║
 * ║        quhit_register.c quhit_substrate.c quhit_triality.c \               ║
 * ║        quhit_triadic.c quhit_lazy.c quhit_calibrate.c \                    ║
 * ║        quhit_dyn_integrate.c quhit_peps_grow.c quhit_hexagram.c \          ║
 * ║        quhit_svd_gate.c s6_exotic.c bigint.c \                             ║
 * ║        mps_overlay.c peps_overlay.c peps3d_overlay.c peps4d_overlay.c \     ║
 * ║        peps5d_overlay.c peps6d_overlay.c \                                  ║
 * ║        -lm -fopenmp -msse2                                                  ║
 * ║                                                                             ║
 * ║  HEAP COMPILE! Large static arrays require:                                 ║
 * ║    ulimit -s unlimited   (before running)                                   ║
 * ║                                                                             ║
 * ╚═══════════════════════════════════════════════════════════════════════════════╝
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ── All HexState headers ── */
#include "quhit_engine.h"        /* Core engine: quhits, registers, gates       */
#include "hpc_graph.h"           /* HPC phase graph: holographic entanglement   */
#include "hpc_contract.h"        /* HPC contraction utilities                   */
#include "quhit_triality.h"      /* Triality quhit: 3+2 views, lazy DFT₆       */
#include "quhit_triadic.h"       /* Three-body entanglement: 216-amp joints     */
#include "s6_exotic.h"           /* S₆ outer automorphism: synthemes, Δ         */
#include "quhit_hexagram.h"      /* Hexagram duality: edge-model quhits         */
#include "mps_overlay.h"         /* 1D MPS tensor network (χ=256)               */
#include "peps_overlay.h"        /* 2D PEPS tensor network (χ=512)              */
#include "peps3d_overlay.h"      /* 3D tensor network (χ=256)                   */
#include "quhit_dyn_integrate.h" /* Entropy-adaptive dynamics, 6 oracles        */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define D 6  /* Quhit dimension — ALWAYS 6 in HexState */

/* ═══════════════════════════════════════════════════════════════════════════════
 * COMMON UTILITIES — PRNG, timing, state/gate construction helpers
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ── xoshiro256** PRNG ── */
static uint64_t rng_s[4];

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
    uint64_t r = rotl64(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0]; rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2]; rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t; rng_s[3] = rotl64(rng_s[3], 45);
    return r;
}

static double rng_uniform(void) {
    return (double)(rng_next() >> 11) / (double)(1ULL << 53);
}

static void rng_seed(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}

/* ── Timing ── */
static struct timespec _t0;
static void tic(void) { clock_gettime(CLOCK_MONOTONIC, &_t0); }
static double toc(void) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - _t0.tv_sec) + (t1.tv_nsec - _t0.tv_nsec) / 1e9;
}

/* ── Normalize a D=6 state vector in-place ── */
static void normalize6(double *re, double *im) {
    double n = 0;
    for (int k = 0; k < D; k++) n += re[k]*re[k] + im[k]*im[k];
    n = sqrt(n);
    if (n > 1e-30) for (int k = 0; k < D; k++) { re[k] /= n; im[k] /= n; }
}

/* ── Build DFT₆ matrix (6×6 complex) ── */
static void build_dft6(double *U_re, double *U_im) {
    double s = 1.0 / sqrt(6.0);
    for (int j = 0; j < D; j++)
    for (int k = 0; k < D; k++) {
        double a = 2.0 * M_PI * j * k / (double)D;
        U_re[j*D+k] = s * cos(a);
        U_im[j*D+k] = s * sin(a);
    }
}

/* ── Build diagonal phase gate from angle array ── */
static void build_phase_gate(double *U_re, double *U_im, const double *angles) {
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));
    for (int j = 0; j < D; j++) {
        U_re[j*D+j] = cos(angles[j]);
        U_im[j*D+j] = sin(angles[j]);
    }
}

/* ── Build CZ two-site gate (36×36 matrix) ── */
static void build_cz_gate(double *G_re, double *G_im) {
    int D2 = D * D;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));
    for (int j = 0; j < D; j++)
    for (int k = 0; k < D; k++) {
        int idx = (j*D+k)*D2 + (j*D+k);
        double a = 2.0 * M_PI * (j * k) / (double)D;
        G_re[idx] = cos(a);
        G_im[idx] = sin(a);
    }
}

/* ── Print a D=6 probability distribution ── */
static void print_probs(const char *label, const double *probs) {
    printf("  %s: [", label);
    for (int k = 0; k < D; k++) printf("%.4f%s", probs[k], k < 5 ? ", " : "");
    printf("]\n");
}

/* ── Print a D=6 state vector ── */
static void print_state(const char *label, const double *re, const double *im) {
    printf("  %s: [", label);
    for (int k = 0; k < D; k++)
        printf("%.3f%+.3fi%s", re[k], im[k], k < 5 ? ", " : "");
    printf("]\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §1  CORE ENGINE — Individual quhits, entangled pairs, registers
 *
 * The fundamental layer. Each quhit has 6 complex amplitudes (|0⟩...|5⟩).
 * Entangled pairs store 36 joint amplitudes. Registers scale to 100T quhits
 * via sparse Magic Pointer encoding (only non-zero entries stored).
 *
 * Key types:  QuhitEngine, Quhit, QuhitPair, QuhitRegister, QuhitSnapshot
 * Key files:  quhit_engine.h, quhit_core.c, quhit_gates.c, quhit_measure.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_core_engine(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §1  CORE ENGINE — Quhits, Pairs, Registers               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── 1.1 Engine lifecycle ──
     * The engine is a large struct (static arrays). Must heap-allocate. */
    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);
    printf("  Engine initialized: max %d quhits, %d pairs, %d registers\n",
           MAX_QUHITS, MAX_PAIRS, MAX_REGISTERS);

    /* ── 1.2 Create individual quhits ──
     * quhit_init()      → |0⟩ (computational ground state)
     * quhit_init_plus() → |+⟩ = (1/√6) Σ|k⟩ (uniform superposition)
     * quhit_init_basis() → |k⟩ (specific basis state) */
    uint32_t q0 = quhit_init(eng);              /* |0⟩ */
    uint32_t q1 = quhit_init_plus(eng);          /* |+⟩ */
    uint32_t q2 = quhit_init_basis(eng, 3);      /* |3⟩ */
    printf("  Created quhits: q0=%u (|0⟩), q1=%u (|+⟩), q2=%u (|3⟩)\n", q0, q1, q2);

    /* ── 1.3 Single-quhit gates ──
     * All gates operate on 6-dimensional amplitudes. */

    /* DFT₆ (Quantum Fourier Transform for D=6):
     * |k⟩ → (1/√6) Σ_j ω^(jk) |j⟩  where ω = e^{2πi/6} */
    quhit_apply_dft(eng, q0);
    printf("  Applied DFT₆ to q0: |0⟩ → |+⟩ (uniform superposition)\n");

    /* Inverse DFT₆: undoes the forward transform */
    quhit_apply_idft(eng, q0);
    printf("  Applied IDFT₆ to q0: back to |0⟩\n");

    /* X gate (cyclic shift): |k⟩ → |k+1 mod 6⟩ */
    quhit_apply_x(eng, q0);
    printf("  Applied X to q0: |0⟩ → |1⟩\n");

    /* Z gate (phase): |k⟩ → ω^k |k⟩ */
    quhit_apply_z(eng, q1);
    printf("  Applied Z to q1: phases applied per basis state\n");

    /* Custom phase gate: specify 6 phases */
    double phases[6] = {0, M_PI/6, M_PI/3, M_PI/2, 2*M_PI/3, 5*M_PI/6};
    quhit_apply_phase(eng, q1, phases);
    printf("  Applied custom phase gate to q1\n");

    /* Arbitrary 6×6 unitary: provide real and imaginary parts (36 doubles each) */
    double U_re[36], U_im[36];
    build_dft6(U_re, U_im);
    quhit_apply_unitary(eng, q2, U_re, U_im);
    printf("  Applied arbitrary 6×6 unitary to q2\n");

    /* ── 1.4 Entanglement (pairwise) ──
     * Creates a 36-amplitude joint state. Bell pair = maximally entangled. */
    uint32_t qa = quhit_init(eng);
    uint32_t qb = quhit_init(eng);
    quhit_entangle_bell(eng, qa, qb);
    printf("  Created Bell pair: qa=%u ↔ qb=%u (36 joint amplitudes)\n", qa, qb);

    /* Product state entanglement (separable, just bookkeeping) */
    uint32_t qc = quhit_init_plus(eng);
    uint32_t qd = quhit_init_basis(eng, 2);
    quhit_entangle_product(eng, qc, qd);
    printf("  Created product pair: qc=%u ⊗ qd=%u\n", qc, qd);

    /* CZ gate on entangled pair: |a,b⟩ → ω^(a·b) |a,b⟩ */
    quhit_apply_cz(eng, qa, qb);
    printf("  Applied CZ gate to Bell pair (qa, qb)\n");

    /* Disentangle (return to product states via partial trace) */
    quhit_disentangle(eng, qc, qd);
    printf("  Disentangled qc ⊗ qd\n");

    /* ── 1.5 Measurement ──
     * Born-rule sampling: outcome ∈ {0,1,2,3,4,5} with P(k) = |⟨k|ψ⟩|²
     * Destructive: collapses the state. */

    /* Non-destructive inspection first */
    QuhitSnapshot snap;
    quhit_inspect(eng, q1, &snap);
    printf("  Inspected q1: entropy=%.4f bits, purity=%.4f, entangled=%d\n",
           snap.entropy, snap.purity, snap.is_entangled);
    printf("  Probabilities: [");
    for (int k = 0; k < D; k++) printf("%.3f%s", snap.probs[k], k<5?", ":"");
    printf("]\n");

    /* Individual probability query */
    double p3 = quhit_prob(eng, q1, 3);
    printf("  P(q1 = 3) = %.4f\n", p3);

    /* Destructive measurement */
    uint32_t outcome = quhit_measure(eng, q1);
    printf("  Measured q1 → %u (collapsed)\n", outcome);

    /* ── 1.6 Registers (100T-scale) ──
     * Sparse state vectors via Magic Pointers. Only non-zero amplitudes stored.
     * Basis states are 128-bit packed: Σ q_k × D^k */
    int reg = quhit_reg_init(eng, /*chunk_id=*/0, /*n_quhits=*/8, /*dim=*/D);
    printf("  Created register: %d quhits, dim=%d, reg_idx=%d\n", 8, D, reg);

    /* Entangle all quhits in register (GHZ-like) */
    quhit_reg_entangle_all(eng, reg);
    printf("  Entangled all register quhits\n");

    /* Register gates */
    quhit_reg_apply_dft(eng, reg, 0);           /* DFT on qubit 0 */
    quhit_reg_apply_cz(eng, reg, 0, 1);         /* CZ on qubits 0,1 */
    quhit_reg_apply_unitary_pos(eng, reg, 2, U_re, U_im); /* Unitary on qubit 2 */
    printf("  Applied DFT(0), CZ(0,1), U(2) on register\n");

    /* Register measurement */
    uint64_t reg_outcome = quhit_reg_measure(eng, reg, 0);
    printf("  Measured register qubit 0 → %lu\n", (unsigned long)reg_outcome);

    /* State vector access */
    double total_p = quhit_reg_sv_total_prob(eng, reg);
    printf("  Register total probability: %.6f (should be ~1.0)\n", total_p);

    /* ── 1.7 Self-test ── */
    printf("  Running self-test... ");
    int ok = quhit_self_test();
    printf("%s\n", ok ? "PASS ✓" : "FAIL ✗");

    /* ── Cleanup ── */
    quhit_engine_destroy(eng);
    free(eng);
    printf("  Engine destroyed.\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §2  HPC PHASE GRAPH — The Holographic Phase Computation model
 *
 * The HPC graph is the primary way to run quantum simulations in HexState.
 * Instead of materializing the exponential state vector, entanglement is
 * encoded as weighted phase edges in a graph:
 *
 *   ψ(i₁,...,iₙ) = [Π_k a_k(i_k)] × [Π_edges w_e(i_a, i_b)]
 *
 * CZ edges are EXACT (fidelity = 1.0). General edges are approximated.
 * Local gates modify only the local TrialityQuhit — edges are NEVER touched.
 * Correlations are revealed at measurement time via marginal contraction.
 *
 * Key types:  HPCGraph, HPCEdge, HPCGateEntry
 * Key file:   hpc_graph.h
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_hpc_graph(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §2  HPC PHASE GRAPH — Holographic Phase Computation       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── 2.1 Create the graph ──
     * Each site gets a TrialityQuhit initialized to |0⟩.
     * Edges, adjacency lists, and gate log are auto-allocated. */
    int N = 8;  /* Number of sites (quhits) */
    HPCGraph *g = hpc_create(N);
    printf("  Created HPC graph: %d sites\n", N);
    printf("  Hilbert space: 6^%d = %llu dimensions\n", N, 1ULL); /* conceptual */
    printf("  Memory: O(N + E) ≈ %lu bytes (not 6^N)\n",
           (unsigned long)(N * sizeof(TrialityQuhit) + sizeof(HPCGraph)));

    /* ── 2.2 Set local states ──
     * Each site's amplitudes are set independently. */
    for (int i = 0; i < N; i++) {
        double re[6] = {0}, im[6] = {0};
        /* Custom initial state: weighted superposition */
        for (int k = 0; k < D; k++) {
            re[k] = cos(2.0 * M_PI * k * (i + 1) / (double)D) + 0.1;
            im[k] = sin(2.0 * M_PI * k * (i + 1) / (double)D) * 0.3;
        }
        normalize6(re, im);
        hpc_set_local(g, i, re, im);
    }
    printf("  Set custom local states for all %d sites\n", N);

    /* Alternative: DFT to create uniform superposition */
    hpc_dft(g, 0);  /* Site 0 → DFT₆|ψ⟩ (Fourier-transformed) */
    printf("  Applied DFT₆ to site 0\n");

    /* ── 2.3 Local gates (modify only local state, never edges) ── */

    /* Phase gate: |k⟩ → e^{iφ_k} |k⟩ — diagonal in computational basis */
    double phi_re[6], phi_im[6];
    for (int k = 0; k < D; k++) {
        double angle = M_PI * k / 3.0;
        phi_re[k] = cos(angle);
        phi_im[k] = sin(angle);
    }
    hpc_phase(g, 1, phi_re, phi_im);
    printf("  Applied phase gate to site 1\n");

    /* Shift gate: |k⟩ → |k+δ mod 6⟩ — cyclic permutation */
    hpc_shift(g, 2, 1);  /* Shift by 1 */
    printf("  Applied shift(+1) to site 2\n");

    /* ── 2.4 CZ entanglement — The exact phase edge ──
     * CZ adds edge: w(a,b) = ω^(a·b)  where ω = e^{2πi/6}
     * Fidelity is ALWAYS 1.0. No truncation. No SVD. */
    hpc_cz(g, 0, 1);
    hpc_cz(g, 1, 2);
    hpc_cz(g, 2, 3);
    printf("  Created CZ chain: 0↔1↔2↔3 (%lu edges)\n", g->n_edges);

    /* Ring topology */
    for (int i = 3; i < N - 1; i++)
        hpc_cz(g, i, i + 1);
    hpc_cz(g, N - 1, 0);  /* Close the ring */
    printf("  Completed ring: %lu CZ edges\n", g->n_edges);

    /* ── 2.5 More local gates AFTER entanglement ──
     * KEY INSIGHT: local gates NEVER destroy edges.
     * The entanglement persists through any local operation. */
    hpc_dft(g, 3);
    hpc_phase(g, 4, phi_re, phi_im);
    printf("  Applied more local gates — edges still alive: %lu\n", g->n_edges);

    /* ── 2.6 Edge compaction ──
     * Multiple CZ edges between same pair merge:
     * n CZ edges → ω^(n·a·b). If n≡0 mod 6: edge cancels entirely. */
    hpc_cz(g, 0, 1);  /* Second CZ on same pair */
    printf("  Added second CZ(0,1) — edges before compaction: %lu\n", g->n_edges);
    hpc_compact_edges(g);
    printf("  After compaction: %lu edges\n", g->n_edges);

    /* ── 2.7 Amplitude evaluation ──
     * ψ(i₁,...,iₙ) = [Π_k a_k(i_k)] × [Π_edges ω^(i_a·i_b)]
     * Cost: O(N + E) per evaluation */
    uint32_t indices[8] = {0, 1, 2, 0, 1, 2, 0, 1};
    double amp_re, amp_im;
    hpc_amplitude(g, indices, &amp_re, &amp_im);
    double prob = amp_re * amp_re + amp_im * amp_im;
    printf("  ψ(0,1,2,0,1,2,0,1): amp = %.6f%+.6fi, |amp|² = %.8f\n",
           amp_re, amp_im, prob);

    /* Direct probability */
    double p = hpc_probability(g, indices);
    printf("  P(0,1,2,0,1,2,0,1) = %.8f\n", p);

    /* ── 2.8 Marginal probabilities ──
     * P(site_k = v) = Σ_{others} |ψ(i₁,...,iₙ)|²
     * Uses per-site adjacency lists: O(degree) not O(E). */
    printf("  Marginals for site 0: [");
    for (int v = 0; v < D; v++) {
        double mv = hpc_marginal(g, 0, v);
        printf("%.4f%s", mv, v < 5 ? ", " : "");
    }
    printf("]\n");

    /* ── 2.9 Born-rule measurement ──
     * Collapses site to |outcome⟩. Absorbs CZ phases into partners.
     * Edges touching measured site are REMOVED (phase absorbed). */
    uint64_t edges_before = g->n_edges;
    uint32_t m0 = hpc_measure(g, 0, rng_uniform());
    printf("  Measured site 0 → %u (edges: %lu → %lu)\n",
           m0, edges_before, g->n_edges);

    /* After measurement, partner states are updated with absorbed phases */
    printf("  Site 1 state (phase-absorbed): [");
    for (int k = 0; k < D; k++)
        printf("%.3f%+.3fi%s", g->locals[1].edge_re[k],
               g->locals[1].edge_im[k], k < 5 ? ", " : "");
    printf("]\n");

    /* ── 2.10 Entropy estimates ── */
    double S = hpc_entropy_cut(g, N / 2);
    printf("  Entropy at midpoint cut: %.4f bits\n", S);

    /* ── 2.11 Exotic invariant across all sites ── */
    double delta = hpc_exotic_invariant(g);
    printf("  Average exotic invariant Δ: %.6f %s\n",
           delta, delta > 0 ? "(hexagonally polarized)" : "(transparent)");

    /* ── 2.12 Graph statistics ── */
    hpc_print_stats(g);
    hpc_print_state(g, "Final state");

    /* ── 2.13 Growing the graph (add sites dynamically) ── */
    hpc_grow_sites(g, N + 4);
    printf("  Grew graph: %lu sites (added 4 new sites initialized to |0⟩)\n",
           g->n_sites);
    hpc_cz(g, N, N + 1);
    printf("  Added CZ between new sites\n");

    /* ── Cleanup ── */
    hpc_destroy(g);
    printf("  Graph destroyed.\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §3  TRIALITY LAYER — Three views of the same quantum state
 *
 * Every state is simultaneously held in Edge/Vertex/Diagonal views (+ Folded,
 * Exotic, Tetra). Gates execute in their cheapest view automatically.
 * DFT₆ has order 4: DFT₆⁴ = I. Views convert lazily on demand.
 *
 * Key optimizations:
 *   - Eigenstate Phase-Lock: O(1) view conversion for DFT₆ eigenstates
 *   - Subspace Confinement: active_mask skips zero basis states
 *   - Real-Valued Fast Path: 2× savings when imaginary parts are zero
 *   - Folded View: O(18) DFT₆ via antipodal fold (vs O(36))
 *
 * Key type:   TrialityQuhit
 * Key file:   quhit_triality.h, quhit_triality.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_triality(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §3  TRIALITY LAYER — Edge/Vertex/Diagonal Views          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    triality_stats_reset();

    /* ── 3.1 Lifecycle ── */
    TrialityQuhit q;
    triality_init(&q);                      /* |0⟩ in all views */
    printf("  Initialized triality quhit to |0⟩\n");

    TrialityQuhit q2;
    triality_init_basis(&q2, 3);            /* |3⟩ */
    printf("  Initialized q2 to |3⟩\n");

    TrialityQuhit q_copy;
    triality_copy(&q_copy, &q);             /* Deep copy */

    /* ── 3.2 View management ──
     * Views: VIEW_EDGE(0), VIEW_VERTEX(1), VIEW_DIAGONAL(2),
     *        VIEW_FOLDED(3), VIEW_EXOTIC(4), VIEW_TETRA(5) */

    /* Ensure a specific view is computed (lazy: only converts if dirty) */
    triality_ensure_view(&q, VIEW_VERTEX);
    printf("  Ensured vertex view (lazy DFT₆ conversion)\n");

    /* Read view amplitudes (read-only, ensures first) */
    const double *v_re = triality_view_re(&q, VIEW_VERTEX);
    const double *v_im = triality_view_im(&q, VIEW_VERTEX);
    print_state("Vertex view", v_re, v_im);

    /* Force sync all views from primary */
    triality_sync_all(&q);
    printf("  Synced all 6 views from primary\n");

    /* ── 3.3 Optimal-view gates ── */

    /* Phase gate: diagonal in EDGE view → O(D) */
    double phi_re[6] = {1,1,1,1,1,1}, phi_im[6] = {0};
    for (int k = 0; k < D; k++) {
        double angle = M_PI * k / 3.0;
        phi_re[k] = cos(angle); phi_im[k] = sin(angle);
    }
    triality_phase(&q, phi_re, phi_im);
    printf("  Phase gate applied (Edge view, O(D))\n");

    /* Single-phase: apply e^{iφ} to just one basis state → O(1) */
    triality_phase_single(&q, 2, cos(M_PI/4), sin(M_PI/4));
    printf("  Single-phase on |2⟩ (O(1))\n");

    /* Z gate: |k⟩ → ω^k|k⟩ — diagonal in Edge view → O(D) */
    triality_z(&q);
    printf("  Z gate applied\n");

    /* X gate: |k⟩ → |k+1 mod 6⟩ — diagonal in VERTEX view → O(D) */
    triality_x(&q);
    printf("  X gate applied (Vertex view, O(D))\n");

    /* Shift gate: generalized X with arbitrary delta */
    triality_shift(&q, 3);  /* |k⟩ → |k+3 mod 6⟩ */
    printf("  Shift(+3) applied\n");

    /* DFT₆: rotates edge→vertex→diagonal→edge */
    triality_dft(&q);
    printf("  DFT₆ applied (view rotation)\n");

    /* Inverse DFT₆ */
    triality_idft(&q);
    printf("  IDFT₆ applied\n");

    /* Arbitrary unitary in a specific view */
    double U_re[36], U_im[36];
    build_dft6(U_re, U_im);
    triality_unitary(&q, VIEW_EDGE, U_re, U_im);
    printf("  Arbitrary 6×6 unitary in Edge view\n");

    /* ── 3.4 CZ gate between two triality quhits ── */
    triality_cz(&q, &q2);
    printf("  CZ gate between q and q2 (Edge view, O(D))\n");

    /* ── 3.5 Measurement ── */
    uint64_t rng = 0xDEADBEEF;
    int outcome = triality_measure(&q2, VIEW_EDGE, &rng);
    printf("  Measured q2 in Edge basis → %d\n", outcome);

    /* Probabilities (no collapse) */
    double probs[6];
    triality_probabilities(&q, VIEW_EDGE, probs);
    print_probs("P(q, Edge)", probs);

    /* ── 3.6 Triality rotation — O(1) relabeling ── */
    triality_rotate(&q);       /* Edge→Vertex→Diagonal→Edge */
    printf("  Triality rotation: views relabeled (O(1), zero cost)\n");
    triality_rotate_inv(&q);   /* Undo */

    /* ── 3.7 Folded view — O(18) intermediate DFT₆ ── */
    triality_fold(&q);
    printf("  Folded: antipodal pairs (0↔3, 1↔4, 2↔5) Hadamard-mixed\n");
    triality_unfold(&q);
    printf("  Unfolded back to edge view\n");

    /* Convert Edge→Vertex via folded intermediate (O(18) vs O(36)) */
    triality_ensure_view_via_fold(&q, VIEW_VERTEX);
    printf("  View conversion via fold path (O(18))\n");

    /* ── 3.8 Enhancement flags ── */
    int eig = triality_detect_eigenstate(&q);
    printf("  Eigenstate class: %d (%s)\n", eig,
           eig >= 0 ? "DFT₆ eigenstate detected" : "not an eigenstate");

    triality_update_mask(&q);
    printf("  Active mask: 0x%02X (%d of 6 basis states active)\n",
           q.active_mask, q.active_count);

    triality_detect_real(&q);
    printf("  Real-valued: %s\n", q.real_valued ? "YES (2× fast path)" : "NO");

    triality_refresh_flags(&q);  /* Update all enhancement flags at once */

    /* ── 3.9 Tetrahedral eigenbasis ── */
    triality_ensure_tetra(&q);
    printf("  Tetrahedral eigenbasis computed (DFT₆ eigenspace decomposition)\n");
    triality_dft_via_tetra(&q);
    printf("  DFT₆ via tetra: O(D) instead of O(D²)\n");

    /* ── 3.10 Print and stats ── */
    triality_print(&q, "Final state");
    triality_stats_print();
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §4  S₆ OUTER AUTOMORPHISM — The only symmetric group with one
 *
 * S₆ (720 elements) has a unique outer automorphism φ that swaps:
 *   Transpositions (ab) ↔ Triple transpositions (ab)(cd)(ef)
 *   3-cycles (abc) ↔ Double 3-cycles (abc)(def)
 *
 * 15 synthemes: all ways to partition {0,...,5} into 3 unordered pairs
 * 6 synthematic totals: maximal sets of 5 disjoint synthemes
 * Exotic invariant Δ(ψ): measures hexagonal polarization (D=6 uniqueness)
 *
 * Key type:   S6Perm, S6Syntheme
 * Key file:   s6_exotic.h, s6_exotic.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_s6_exotic(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §4  S₆ OUTER AUTOMORPHISM — Exotic Quantum Structure     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── 4.1 Initialization (must call once) ── */
    s6_exotic_init();
    printf("  S₆ exotic tables initialized (720 elements, 15 synthemes)\n");

    /* ── 4.2 Permutation operations ── */
    S6Perm sigma = s6_from_int(42);    /* Arbitrary permutation */
    S6Perm phi_sigma = s6_apply_phi(sigma);  /* Apply outer automorphism */
    printf("  σ = S6[42]: [%d,%d,%d,%d,%d,%d]\n",
           sigma.p[0], sigma.p[1], sigma.p[2], sigma.p[3], sigma.p[4], sigma.p[5]);
    printf("  φ(σ):       [%d,%d,%d,%d,%d,%d]\n",
           phi_sigma.p[0], phi_sigma.p[1], phi_sigma.p[2],
           phi_sigma.p[3], phi_sigma.p[4], phi_sigma.p[5]);

    S6Perm composed = s6_compose_perm(sigma, phi_sigma);
    S6Perm inv = s6_inverse(sigma);
    int fp = s6_fixed_points(sigma);
    printf("  σ has %d fixed points, σ⁻¹·σ = identity: %s\n",
           fp, s6_perm_eq(s6_compose_perm(inv, sigma), S6_IDENTITY) ? "YES" : "NO");
    (void)composed;

    /* ── 4.3 Work with a quantum state ── */
    double re[6] = {0.5, 0.3, 0.1, 0.4, 0.2, 0.6};
    double im[6] = {0.1, -0.2, 0.3, 0.0, -0.1, 0.15};
    normalize6(re, im);
    print_state("Test state", re, im);

    /* ── 4.4 Exotic invariant Δ ──
     * Δ = 0: automorphism-transparent (generic, could be qubits)
     * Δ > 0: hexagonally polarized (structure unique to D=6)
     * Cost: O(720 × D) */
    double delta = s6_exotic_invariant(re, im);
    printf("  Exotic invariant Δ = %.6f %s\n", delta,
           delta > 0 ? "→ HEXAGONALLY POLARIZED (D=6 unique)" : "→ transparent");

    /* ── 4.5 Exotic fingerprint — per-conjugacy-class breakdown ── */
    double class_deltas[11];
    s6_exotic_fingerprint(re, im, class_deltas);
    printf("  Exotic fingerprint (11 conjugacy classes of S₆):\n    [");
    for (int i = 0; i < 11; i++) printf("%.4f%s", class_deltas[i], i<10?", ":"");
    printf("]\n");

    /* ── 4.6 Dual probabilities — standard AND exotic bases ── */
    double probs_std[6], probs_exo[6];
    s6_dual_probabilities(re, im, probs_std, probs_exo);
    print_probs("Standard probs", probs_std);
    print_probs("Exotic probs  ", probs_exo);

    /* ── 4.7 Exotic entropy ΔS = S_standard - S_exotic ── */
    double dS = s6_exotic_entropy(re, im, 0);
    printf("  Exotic entropy ΔS = %.4f (>0: more ordered in exotic channel)\n", dS);

    /* ── 4.8 Syntheme-parameterized fold ──
     * 15 different ways to pair the 6 basis states */
    double fold_re[6], fold_im[6];
    s6_fold_syntheme(re, im, fold_re, fold_im, 7);  /* Syntheme #7 */
    printf("  Folded via syntheme #7: vesica + wave decomposition\n");
    s6_unfold_syntheme(fold_re, fold_im, fold_re, fold_im, 7);
    printf("  Unfolded back (lossless)\n");

    /* Optimal syntheme for a given active mask */
    int opt = s6_optimal_syntheme(0x3F);  /* All 6 active */
    printf("  Optimal syntheme for mask 0x3F: #%d\n", opt);

    /* ── 4.9 Exotic gate — apply φ(σ) instead of σ ── */
    double out_re[6], out_im[6];
    s6_apply_exotic_gate(re, im, out_re, out_im, sigma);
    printf("  Applied exotic gate φ(σ₄₂)\n");

    /* ── 4.10 Measurement basis selection ── */
    int basis = s6_optimal_measure_basis(re, im);
    printf("  Optimal measurement basis: %s\n",
           basis < 0 ? "standard (computational)" : "exotic syntheme");

    /* ── 4.11 Cross-syntheme entanglement witness ──
     * Cheap Δ approximation: O(90) vs O(4320) for full Δ */
    double witness = s6_cross_syntheme_witness(re, im);
    printf("  Cross-syntheme witness: %.6f (≈Δ, 48× cheaper)\n", witness);

    /* ── 4.12 Minimum-entropy syntheme ── */
    int min_s = s6_min_entropy_syntheme(re, im);
    printf("  Minimum-entropy syntheme: #%d\n", min_s);

    /* ── 4.13 Synthematic total tomography ──
     * Reconstruct state from 5 fold projections of one total */
    double fold_data_re[5][6], fold_data_im[5][6];
    for (int s = 0; s < 5; s++) {
        int synth_idx = s6_totals[0][s];  /* Total #0's synthemes */
        s6_fold_syntheme(re, im, fold_data_re[s], fold_data_im[s], synth_idx);
    }
    double recon_re[6], recon_im[6];
    double fid = s6_total_tomography(0, fold_data_re, fold_data_im,
                                      recon_re, recon_im);
    printf("  Tomographic reconstruction fidelity (Total #0): %.6f\n", fid);

    /* ── 4.14 Triality quhit exotic integration ── */
    TrialityQuhit tq;
    triality_init(&tq);
    triality_dft(&tq);  /* Non-trivial state */

    triality_exotic_init();  /* One-time exotic engine init */

    triality_set_exotic_syntheme(&tq, 5);  /* Use syntheme #5 */
    triality_fold_syntheme(&tq, 5);
    triality_unfold_syntheme(&tq, 5);
    printf("  Triality exotic fold/unfold via syntheme #5\n");

    /* Exotic gate on triality quhit */
    triality_exotic_gate(&tq, sigma);
    printf("  Applied exotic gate on triality quhit\n");

    /* Dual CZ: standard CZ + exotic channel info */
    TrialityQuhit tq2;
    triality_init_basis(&tq2, 2);
    double dist = triality_cz_dual(&tq, &tq2);
    printf("  Dual CZ distance (standard vs exotic): %.6f\n", dist);

    /* Dual measurement */
    uint64_t rng = 0xCAFE;
    int exo_outcome;
    int std_outcome = triality_measure_dual(&tq, VIEW_EDGE, 0, &rng, &exo_outcome);
    printf("  Dual measurement: standard=%d, exotic=%d\n", std_outcome, exo_outcome);

    /* Cached exotic invariant (avoids recompute if state unchanged) */
    triality_init_basis(&tq2, 4);
    double cached_delta = triality_exotic_invariant_cached(&tq2);
    printf("  Cached Δ(|4⟩) = %.6f\n", cached_delta);

    /* Exotic rotation (full Aut(S₆) ≅ S₆ ⋊ Z₂) */
    triality_rotate_exotic(&tq2);
    printf("  Exotic rotation: cycles through synthematic views\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §5  THREE-BODY ENTANGLEMENT (TRIADIC) — Three mirrors, one point
 *
 * Entanglement is natively three-body. The TriadicJoint holds 6³ = 216
 * complex amplitudes for three entangled quhits |a,b,c⟩.
 *
 * CMY channels: C={0,1}, M={2,3}, Y={4,5} — three qubit subspaces
 * inside every D=6 quhit. Each channel entangles independently.
 *
 * Memory: 216 × 16 = 3,456 bytes per triple (fits in L1 cache)
 *
 * Key types:  TriadicJoint, QuhitTriple
 * Key file:   quhit_triadic.h, quhit_triadic.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_triadic(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §5  THREE-BODY ENTANGLEMENT — Triadic Joints (216 amps)  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── 5.1 Triadic Bell state — (1/√6) Σ|k,k,k⟩ ──
     * Maximum three-body agreement: all collapse to same outcome */
    TriadicJoint j;
    triad_bell(&j);
    printf("  Created triadic Bell: (1/√6) Σ|k,k,k⟩\n");
    printf("  Total probability: %.6f (should be 1.0)\n", triad_total_prob(&j));

    /* Check diagonal structure */
    printf("  Bell diagonal amplitudes:\n");
    for (int k = 0; k < D; k++) {
        int idx = TRIAD_IDX(k, k, k);
        printf("    |%d,%d,%d⟩: %.6f%+.6fi\n", k, k, k, j.re[idx], j.im[idx]);
    }

    /* ── 5.2 CMY Bell state — channel-wise GHZ ── */
    TriadicJoint j_cmy;
    triad_cmy_bell(&j_cmy);
    printf("\n  Created CMY Bell state (channel-correlated GHZ)\n");

    /* ── 5.3 Product state — |ψ_a⟩ ⊗ |ψ_b⟩ ⊗ |ψ_c⟩ ── */
    double a_re[6]={1,0,0,0,0,0}, a_im[6]={0};  /* |0⟩ */
    double b_re[6], b_im[6];                      /* |+⟩ */
    for (int k=0;k<D;k++){b_re[k]=1.0/sqrt(6);b_im[k]=0;}
    double c_re[6]={0,0,0,1,0,0}, c_im[6]={0};  /* |3⟩ */

    TriadicJoint j_prod;
    triad_product(&j_prod, a_re, a_im, b_re, b_im, c_re, c_im);
    printf("  Created product state: |0⟩ ⊗ |+⟩ ⊗ |3⟩\n");

    /* ── 5.4 Marginal probabilities ── */
    double probs_a[6], probs_b[6], probs_c[6];
    triad_marginal_a(&j, probs_a);
    triad_marginal_b(&j, probs_b);
    triad_marginal_c(&j, probs_c);
    print_probs("Bell marginal A", probs_a);
    print_probs("Bell marginal B", probs_b);
    print_probs("Bell marginal C", probs_c);

    /* ── 5.5 Three-body CZ gate ──
     * |a,b,c⟩ → ω^(a·b·c) |a,b,c⟩ */
    triad_apply_cz3(&j_prod);
    printf("\n  Applied triadic CZ3 to product state\n");
    printf("  Total prob after CZ3: %.6f\n", triad_total_prob(&j_prod));

    /* ── 5.6 Channel CZ — entangle within one CMY channel ── */
    triad_apply_channel_cz(&j_prod, 0);  /* C channel only */
    triad_apply_channel_cz(&j_prod, 1);  /* M channel only */
    triad_apply_channel_cz(&j_prod, 2);  /* Y channel only */
    printf("  Applied channel CZ to all 3 CMY channels\n");

    /* ── 5.7 Local gates on triadic state ──
     * Apply a 6×6 unitary to one quhit while others stay fixed */
    double U_re[36], U_im[36];
    build_dft6(U_re, U_im);

    triad_gate_a(&j, U_re, U_im);   /* DFT₆ on quhit A */
    triad_gate_b(&j, U_re, U_im);   /* DFT₆ on quhit B */
    triad_gate_c(&j, U_re, U_im);   /* DFT₆ on quhit C */
    printf("  Applied DFT₆ to all three quhits (A, B, C) independently\n");

    /* ── 5.8 Channel-local gates (2×2 within one CMY channel) ──
     * 9× fewer multiplies than full 6×6 gate */
    double hadamard[8] = {
        0.7071067811865476, 0.0,    /*  1/√2, 0 */
        0.7071067811865476, 0.0,    /*  1/√2, 0 */
        0.7071067811865476, 0.0,    /*  1/√2, 0 */
       -0.7071067811865476, 0.0     /* -1/√2, 0 */
    };
    triad_channel_gate_a(&j, 0, hadamard);  /* Hadamard on C channel of A */
    triad_channel_gate_b(&j, 1, hadamard);  /* Hadamard on M channel of B */
    triad_channel_gate_c(&j, 2, hadamard);  /* Hadamard on Y channel of C */
    printf("  Channel-local Hadamard gates (2×2, 9× cheaper)\n");

    /* CMY composite gate: apply different 2×2 to each channel of quhit A */
    double phase_gate[8] = {1,0, 0,0, 0,0, 0,1};  /* diag(1, i) */
    triad_cmy_gate_a(&j, hadamard, phase_gate, hadamard);
    printf("  CMY composite gate on A: H_C, Phase_M, H_Y\n");

    /* Channel DFT₂ on all channels simultaneously */
    triad_channel_dft_a(&j);
    printf("  Channel DFT₂ on all 3 channels of A\n");

    /* Channel phase gate */
    triad_channel_phase_a(&j, 1, M_PI / 4.0);  /* e^{iπ/4} on M channel */
    printf("  Channel phase(π/4) on M channel of A\n");

    /* ── 5.9 Face switching — O(0) cost ── */
    triad_face_rotate_a(&j);        /* C→M→Y→C */
    printf("  Face rotation: C→M→Y→C (zero multiplies)\n");
    triad_face_rotate_back_a(&j);   /* Undo */
    triad_face_swap_a(&j, 0, 2);    /* Swap C↔Y */
    printf("  Face swap: C↔Y\n");

    /* ── 5.10 Multi-face readout — averaged across A,B,C ── */
    double mf_probs[6];
    triad_multiface_readout(&j, mf_probs);
    print_probs("Multi-face avg", mf_probs);

    /* ── 5.11 Entanglement entropy ── */
    double S = triad_entropy_a(&j);
    printf("  Entanglement entropy S_A = %.4f bits\n", S);

    /* ── 5.12 Renormalize (fix numerical drift) ── */
    triad_renormalize(&j);
    printf("  Renormalized: total prob = %.6f\n", triad_total_prob(&j));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §6  HEXAGRAM DUALITY — Edge-model quhits (Kramers-Wannier dual)
 *
 * The hexagram quhit stores amplitudes on 6 LINE SEGMENTS of the
 * unicursal hexagram — the edge dual of the triality (vertex) quhit.
 *
 * Lines alternate: diameter (through center) and outer (boundary edge).
 * Chirality is intrinsic: two orientations = two mirror tetrahedra.
 * The H₆ transform converts vertex ↔ hexagram (NOT the DFT₆).
 *
 * Key type:   HexagramQuhit
 * Key file:   quhit_hexagram.h, quhit_hexagram.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_hexagram(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §6  HEXAGRAM DUALITY — Edge-Model Quhits                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── 6.1 Initialize H₆ tables (call once) ── */
    hexagram_init_tables();
    printf("  H₆ transform tables initialized\n");

    /* ── 6.2 Lifecycle ── */
    HexagramQuhit hq;
    hexagram_init(&hq);  /* |ℓ₀⟩ with positive chirality */
    printf("  Initialized hexagram quhit to |ℓ₀⟩ (chirality: +1)\n");

    /* From specific line segment */
    HexagramQuhit hq2;
    hexagram_init_line(&hq2, 3, CHIRALITY_NEG);  /* |ℓ₃⟩, negative chirality */
    printf("  Created |ℓ₃⟩ with negative chirality (mirror tetrahedron)\n");

    /* From vertex-basis state vector */
    double v_re[6] = {0.5, 0.3, 0.1, 0.4, 0.2, 0.6};
    double v_im[6] = {0.1, -0.2, 0.3, 0.0, -0.1, 0.15};
    normalize6(v_re, v_im);
    HexagramQuhit hq3;
    hexagram_init_from_vertex(&hq3, v_re, v_im, CHIRALITY_POS);
    printf("  Created hexagram from vertex state via H₆ transform\n");

    /* ── 6.3 Line metadata ── */
    printf("  Line types and colors:\n");
    for (int k = 0; k < D; k++) {
        printf("    ℓ%d: %s, %s (%s)\n", k,
               hexagram_line_type(k) == LINE_DIAMETER ? "diameter" : "outer  ",
               (int[]){0,2,1,0,2,1}[k] == 0 ? "Cyan   " :
               (int[]){0,2,1,0,2,1}[k] == 1 ? "Magenta" : "Yellow ",
               hexagram_line_name(k));
    }

    /* ── 6.4 Native hexagram gates ── */

    /* Path shift: |ℓ_k⟩ → |ℓ_{k+δ}⟩ — O(D), diagonal in hexagram basis */
    hexagram_path_shift(&hq, 2);
    printf("  Path shift +2: advance along unicursal path\n");

    /* Per-line phase: |ℓ_k⟩ → e^{iφ_k}|ℓ_k⟩ */
    double hphi_re[6], hphi_im[6];
    for (int k = 0; k < D; k++) {
        double a = M_PI * k / 3.0;
        hphi_re[k] = cos(a); hphi_im[k] = sin(a);
    }
    hexagram_phase(&hq, hphi_re, hphi_im);
    printf("  Per-line phase gate applied\n");

    /* Diameter-only phase (targets ℓ₀,ℓ₂,ℓ₄) */
    hexagram_diameter_phase(&hq, cos(M_PI/3), sin(M_PI/3));
    printf("  Diameter phase: through-center lines only\n");

    /* Outer-only phase (targets ℓ₁,ℓ₃,ℓ₅) */
    hexagram_outer_phase(&hq, cos(M_PI/6), sin(M_PI/6));
    printf("  Outer phase: boundary edges only\n");

    /* Chirality flip: reverse path orientation (involution) */
    printf("  Chirality before flip: %+d\n", hq.chirality);
    hexagram_flip(&hq);
    printf("  Chirality after flip:  %+d (mirror tetrahedron)\n", hq.chirality);
    hexagram_flip(&hq);  /* flip∘flip = identity */

    /* Triad gate: cycle diameters ℓ₀↔ℓ₂↔ℓ₄, outers ℓ₁↔ℓ₃↔ℓ₅ */
    hexagram_triad(&hq);
    printf("  Triad rotation: 3-fold symmetry of hexagram\n");
    hexagram_triad_inv(&hq);  /* undo */

    /* ── 6.5 Entanglement — center-crossing interaction ── */
    hexagram_cross(&hq, &hq2);
    printf("  Center-crossing entanglement between hq and hq2\n");

    /* ── 6.6 Measurement ── */
    double hprobs[6];
    hexagram_probabilities(&hq3, hprobs);
    print_probs("Hexagram probs", hprobs);

    uint64_t rng = 0xBEEF;
    int houtcome = hexagram_measure(&hq, &rng);
    printf("  Measured hexagram → ℓ%d\n", houtcome);

    /* ── 6.7 Interconversion (vertex ↔ hexagram) ── */
    hexagram_ensure_vertex(&hq3);
    const double *vrec_re = hexagram_vertex_re(&hq3);
    const double *vrec_im = hexagram_vertex_im(&hq3);
    print_state("Recovered vertex", vrec_re, vrec_im);

    /* Convert triality ↔ hexagram */
    TrialityQuhit tq;
    triality_init_basis(&tq, 1);
    triality_dft(&tq);
    HexagramQuhit hq_from_tri;
    triality_to_hexagram(&tq, &hq_from_tri);
    printf("  Converted triality → hexagram via H₆\n");

    hexagram_to_triality(&hq_from_tri, &tq);
    printf("  Converted hexagram → triality via H₆†\n");

    /* ── 6.8 Diagnostics ── */
    hexagram_print(&hq3, "hq3 state");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §7  LAZY EVALUATION — Heisenberg-picture gate accumulation
 *
 * Amplitudes are NEVER touched until measurement. Gates accumulate as
 * diagonal phase vectors in a segment chain:
 *   state → F^pre0 · D0 → F^pre1 · D1 → ... → F^trailing
 *
 * DFT₆⁴ = I, so DFT counts reduce mod 4. Same-view gates fuse.
 * When segments overflow, an Oracle matrix absorbs the chain.
 *
 * Key type:   LazyTrialityQuhit
 * Key file:   quhit_triality.h (bottom section), quhit_lazy.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_lazy(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §7  LAZY EVALUATION — Heisenberg-Picture Gates           ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── 7.1 Lifecycle ── */
    LazyTrialityQuhit lq;
    ltri_init(&lq);           /* |0⟩ (frozen initial state) */
    printf("  Initialized lazy quhit to |0⟩ (amplitudes frozen)\n");

    LazyTrialityQuhit lq2;
    ltri_init_basis(&lq2, 3); /* |3⟩ */
    printf("  Initialized lq2 to |3⟩\n");

    /* ── 7.2 Gate accumulation — O(D) per gate, ZERO view conversions ── */

    /* Z gate: accumulates as diagonal phase segment */
    ltri_z(&lq);
    printf("  Z gate accumulated (segment count: %d)\n", lq.n_segments);

    /* X gate: equivalent to DFT + phase + IDFT, accumulated lazily */
    ltri_x(&lq);
    printf("  X gate accumulated\n");

    /* Shift by delta */
    ltri_shift(&lq, 2);
    printf("  Shift(+2) accumulated\n");

    /* DFT₆: increments trailing DFT counter (mod 4, since DFT⁴=I) */
    ltri_dft(&lq);
    printf("  DFT₆ accumulated (trailing_dfts: %d)\n", lq.trailing_dfts);
    ltri_dft(&lq);
    ltri_dft(&lq);
    ltri_dft(&lq);  /* 4 DFTs = identity, should cancel */
    printf("  4 DFTs accumulated (trailing_dfts: %d, should be 0 mod 4)\n",
           lq.trailing_dfts);

    /* Inverse DFT: decrements counter */
    ltri_idft(&lq);
    printf("  IDFT₆ accumulated (trailing_dfts: %d)\n", lq.trailing_dfts);

    /* Custom phase vector */
    double phi_re[6], phi_im[6];
    for (int k = 0; k < D; k++) {
        double angle = M_PI * k / 6.0;
        phi_re[k] = cos(angle); phi_im[k] = sin(angle);
    }
    ltri_phase(&lq, phi_re, phi_im);
    printf("  Custom phase accumulated\n");

    /* Apply many gates — they fuse into segments */
    for (int i = 0; i < 20; i++) {
        ltri_z(&lq);
        ltri_dft(&lq);
        ltri_phase(&lq, phi_re, phi_im);
    }
    printf("  Applied 60 more gates (fused into %d segments)\n", lq.n_segments);
    printf("  Gates fused: %lu, Segments created: %lu\n",
           lq.gates_fused, lq.segments_created);

    /* ── 7.3 Materialization — only happens when needed ── */

    /* Materialize: apply accumulated transform, get edge-view amplitudes */
    double out_re[6], out_im[6];
    ltri_materialize(&lq, out_re, out_im);
    printf("  Materialized (materializations: %lu)\n", lq.materializations);
    print_state("Materialized", out_re, out_im);

    /* Force materialize into a full TrialityQuhit (for CZ operations) */
    TrialityQuhit tq_out;
    ltri_force_materialize(&lq2, &tq_out);
    printf("  Force-materialized lq2 into TrialityQuhit\n");
    triality_print(&tq_out, "Force-materialized lq2");

    /* ── 7.4 Measurement — materialize + Born sample ── */
    LazyTrialityQuhit lq3;
    ltri_init(&lq3);
    ltri_dft(&lq3);   /* Create superposition */
    ltri_z(&lq3);
    ltri_phase(&lq3, phi_re, phi_im);

    uint64_t rng = 0xFACE;
    int outcome = ltri_measure(&lq3, VIEW_EDGE, &rng);
    printf("  Lazy measurement → %d\n", outcome);

    /* ── 7.5 Stats ── */
    ltri_stats_print(&lq);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §8  TENSOR NETWORK OVERLAYS — MPS(1D), PEPS(2D), TNS(3D+)
 *
 * All overlays use Magic Pointer registers as tensors. Sites are
 * TriOverlaySite with triality sidecar for gate routing.
 *
 * MPS:  |k,α,β⟩   3 indices, χ=256, 1D chains
 * PEPS: |k,u,d,l,r⟩  5 indices, χ=512, 2D grids
 * 3D:   7 indices, χ=256
 *
 * Key files: mps_overlay.h/.c, peps_overlay.h/.c, peps3d_overlay.h/.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_tensor_networks(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §8  TENSOR NETWORKS — MPS, PEPS, higher-D overlays      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ══════════════════════════════════════════════════════════════════
     * 8.1  MPS — 1D Matrix Product State (χ=256)
     * ══════════════════════════════════════════════════════════════════ */
    printf("  ── 8.1 MPS (1D Chain, χ=256) ──\n\n");

    int L_mps = 16;
    MpsChain *mps = mps_init(L_mps);
    printf("  Created MPS chain: L=%d, χ=%llu\n", L_mps, MPS_CHI);

    /* Set product states */
    for (int i = 0; i < L_mps; i++) {
        double re[6] = {0}, im[6] = {0};
        re[i % D] = 1.0;  /* Cycle through basis states */
        mps_set_product_state(mps, i, re, im);
    }
    printf("  Set product states: site k → |k mod 6⟩\n");

    /* Build standard gates */
    double U_re[36], U_im[36];
    mps_build_dft6(U_re, U_im);    /* DFT₆ single-site gate */
    printf("  Built DFT₆ single-site gate\n");

    double G_re[36*36], G_im[36*36];
    mps_build_cz(G_re, G_im);      /* CZ two-site gate */
    printf("  Built CZ two-site gate (36×36)\n");

    /* Apply single-site gate */
    mps_gate_1site(mps, 0, U_re, U_im);
    printf("  Applied DFT₆ to site 0\n");

    /* Apply bond gate (two-site, triggers SVD truncation) */
    mps_gate_bond(mps, 0, G_re, G_im);
    printf("  Applied CZ bond gate on sites 0-1 (SVD truncation)\n");

    /* Batch operations */
    mps_gate_1site_all(mps, U_re, U_im);  /* DFT₆ on all sites */
    printf("  Applied DFT₆ to ALL sites\n");

    mps_gate_bond_all(mps, G_re, G_im);   /* CZ on all adjacent pairs */
    printf("  Applied CZ to ALL adjacent bonds\n");

    /* Trotter step (combined bond gates with even/odd sweep) */
    mps_trotter_step(mps, G_re, G_im);
    printf("  Trotter step complete\n");

    /* Observables */
    double probs[6];
    mps_local_density(mps, L_mps / 2, probs);
    print_probs("MPS mid-chain density", probs);

    /* Measurement */
    int mps_out = mps_measure_site(mps, 0);
    printf("  Measured MPS site 0 → %d\n", mps_out);

    mps_free(mps);
    printf("  MPS chain freed.\n\n");

    /* ══════════════════════════════════════════════════════════════════
     * 8.2  PEPS — 2D Projected Entangled Pair States (χ=512)
     * ══════════════════════════════════════════════════════════════════ */
    printf("  ── 8.2 PEPS (2D Grid, χ=512) ──\n\n");

    int Lx = 4, Ly = 4;
    PepsGrid *peps = peps_init(Lx, Ly);
    printf("  Created PEPS grid: %d×%d = %d sites, χ=%llu\n",
           Lx, Ly, Lx*Ly, PEPS_CHI);

    /* Set product states */
    for (int y = 0; y < Ly; y++)
    for (int x = 0; x < Lx; x++) {
        double re[6] = {0}, im[6] = {0};
        re[(x + y) % D] = 1.0;
        peps_set_product_state(peps, x, y, re, im);
    }
    printf("  Set product states on %d×%d grid\n", Lx, Ly);

    /* Single-site gate */
    peps_gate_1site(peps, 1, 1, U_re, U_im);
    printf("  Applied DFT₆ to site (1,1)\n");

    /* Horizontal bond gate (between (x,y) and (x+1,y)) */
    peps_gate_horizontal(peps, 0, 0, G_re, G_im);
    printf("  Applied horizontal CZ gate at (0,0)↔(1,0)\n");

    /* Vertical bond gate (between (x,y) and (x,y+1)) */
    peps_gate_vertical(peps, 0, 0, G_re, G_im);
    printf("  Applied vertical CZ gate at (0,0)↔(0,1)\n");

    /* Batch operations */
    peps_gate_1site_all(peps, U_re, U_im);
    peps_gate_horizontal_all(peps, G_re, G_im);
    peps_gate_vertical_all(peps, G_re, G_im);
    printf("  Batch: DFT₆ all + horizontal CZ all + vertical CZ all\n");

    /* Trotter step */
    peps_trotter_step(peps, G_re, G_im);
    printf("  PEPS Trotter step complete\n");

    /* Local density */
    peps_local_density(peps, Lx/2, Ly/2, probs);
    print_probs("PEPS center density", probs);

    /* Measurement */
    int peps_out = peps_measure_site(peps, 0, 0);
    printf("  Measured PEPS site (0,0) → %d\n", peps_out);

    peps_free(peps);
    printf("  PEPS grid freed.\n\n");

    /* ══════════════════════════════════════════════════════════════════
     * 8.3  3D Tensor Network (χ=256)
     * ══════════════════════════════════════════════════════════════════ */
    printf("  ── 8.3 TNS 3D (χ=256) ──\n\n");

    int L3 = 3;
    Peps3dGrid *tns3 = peps3d_init(L3, L3, L3);
    printf("  Created 3D TNS grid: %d×%d×%d = %d sites\n",
           L3, L3, L3, L3*L3*L3);

    /* 1-site gate */
    peps3d_gate_1site(tns3, 0, 0, 0, U_re, U_im);
    printf("  Applied 1-site gate at (0,0,0)\n");

    /* 2-site bond gates along each axis */
    peps3d_gate_x(tns3, 0, 0, 0, G_re, G_im);   /* x-axis bond */
    peps3d_gate_y(tns3, 0, 0, 0, G_re, G_im);   /* y-axis bond */
    peps3d_gate_z(tns3, 0, 0, 0, G_re, G_im);   /* z-axis bond */
    printf("  Applied bond gates along x, y, z axes at origin\n");

    /* Local density */
    peps3d_local_density(tns3, 1, 1, 1, probs);
    print_probs("3D TNS center density", probs);

    /* Measurement */
    int tns3_out = peps3d_measure_site(tns3, 0, 0, 0);
    printf("  Measured 3D site (0,0,0) → %d\n", tns3_out);

    peps3d_free(tns3);
    printf("  3D TNS grid freed.\n");

    /* NOTE: 4D, 5D, 6D follow the exact same pattern:
     * peps4d_init(Lx,Ly,Lz,Lw) / peps4d_gate_1site / peps4d_gate_{x,y,z,w}
     * peps5d_init(Lx,Ly,Lz,Lw,Lv) / ...
     * tns6d_init(Lx,Ly,Lz,Lw,Lv,Lu) / tns6d_gate_{x,y,z,w,v,u}
     * All use the same API pattern with increasing coordinate dimensions. */
    printf("\n  Higher-D overlays (4D-6D) follow identical API patterns.\n");
    printf("  See: peps4d_overlay.h, peps5d_overlay.h, peps6d_overlay.h\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §9  ENTROPY-ADAPTIVE DYNAMICS — DynChain with 6 Oracles
 *
 * A breathing lattice that grows or contracts based on real-time
 * entanglement entropy. Five oracles + Ouroboros self-optimization:
 *
 *   1. Entropy Gradient    — linear prediction of site entropy trends
 *   2. Mutual Information  — inter-site correlation (optimal coupling)
 *   3. Convergence Horizon — oscillation/stagnation detection
 *   4. Phase Boundary      — sharpest entropy gradient detection
 *   5. Site Weight          — information weight ranking
 *   6. Ouroboros           — self-referential feedback (β=2.42)
 *
 * Key type:   DynChain, DynLattice
 * Key file:   quhit_dyn_integrate.h, quhit_dyn_integrate.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_dynamics(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §9  ENTROPY DYNAMICS — DynChain + 6 Oracles              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── 9.1 Create DynChain ── */
    int max_sites = 64;
    DynChain dc = dyn_chain_create(max_sites);
    printf("  Created DynChain: max_sites=%d\n", max_sites);

    /* Configure thresholds */
    dc.grow_threshold = 0.5;      /* Expand when boundary entropy > 0.5 */
    dc.contract_threshold = 0.1;  /* Contract when tail entropy < 0.1 */
    dc.min_active = 4;            /* Never shrink below 4 sites */
    printf("  Thresholds: grow=%.1f, contract=%.1f, min=%d\n",
           dc.grow_threshold, dc.contract_threshold, dc.min_active);

    /* Seed active region */
    dyn_chain_seed(&dc, 10, 30);
    printf("  Seeded active region: [%d, %d] (%d sites)\n",
           dc.active_start, dc.active_end, dyn_chain_active_length(&dc));

    /* ── 9.2 Feed entropy data ──
     * In a real simulation, these come from mps_local_density / peps_local_density */
    for (int s = dc.active_start; s <= dc.active_end; s++) {
        /* Simulate entropy profile: high in center, low at edges */
        double t = (double)(s - dc.active_start) / dyn_chain_active_length(&dc);
        double H = 2.0 * sin(M_PI * t);  /* Peaked in middle */
        double probs[6] = {0};
        /* Fake probability distribution with target entropy */
        for (int k = 0; k < D; k++) probs[k] = 1.0 / D + 0.1 * sin(k + s);
        double sum = 0; for (int k=0;k<D;k++) { if(probs[k]<0.01)probs[k]=0.01; sum+=probs[k]; }
        for (int k=0;k<D;k++) probs[k]/=sum;
        dyn_chain_update_entropy(&dc, s, probs, D);
    }
    printf("  Fed entropy data for %d active sites\n", dyn_chain_active_length(&dc));

    /* ── 9.3 Oracle 1: Entropy Prediction ── */
    dyn_chain_record_entropy(&dc);
    /* Simulate a few epochs of history */
    for (int epoch = 0; epoch < 5; epoch++) {
        for (int s = dc.active_start; s <= dc.active_end; s++) {
            double probs[6];
            for (int k=0;k<D;k++) probs[k] = 1.0/D + 0.05*sin(k+s+epoch);
            double sum=0; for(int k=0;k<D;k++){if(probs[k]<0.01)probs[k]=0.01;sum+=probs[k];}
            for(int k=0;k<D;k++) probs[k]/=sum;
            dyn_chain_update_entropy(&dc, s, probs, D);
        }
        dyn_chain_record_entropy(&dc);
    }
    dyn_chain_predict_entropy(&dc);
    printf("  Oracle 1 (Entropy Prediction): %d epochs recorded, predictions ready\n",
           dc.history_cursor);
    printf("    Predicted entropy at start: %.4f, at end: %.4f\n",
           dc.entropy_predicted[dc.active_start],
           dc.entropy_predicted[dc.active_end]);

    /* ── 9.4 Oracle 2: Mutual Information ── */
    double *marginals = (double*)calloc(max_sites * D, sizeof(double));
    for (int s = dc.active_start; s <= dc.active_end; s++)
        for (int k = 0; k < D; k++)
            marginals[s * D + k] = 1.0/D + 0.1*sin(k+s);
    dyn_chain_mutual_info(&dc, marginals, D);
    int ci, cj;
    dyn_chain_best_coupling(&dc, &ci, &cj);
    printf("  Oracle 2 (Mutual Info): best coupling pair = (%d, %d), JSD = %.4f\n",
           ci, cj, dc.max_correlation);
    free(marginals);

    /* ── 9.5 Oracle 3: Convergence Horizon ── */
    int conv = dyn_chain_check_convergence(&dc);
    printf("  Oracle 3 (Convergence): state = %s\n",
           conv == DYN_CONVERGING  ? "CONVERGING" :
           conv == DYN_OSCILLATING ? "OSCILLATING" :
                                     "STAGNANT");

    /* ── 9.6 Oracle 4: Phase Boundary ── */
    int pb = dyn_chain_phase_boundary(&dc);
    printf("  Oracle 4 (Phase Boundary): sharpest gradient at site %d (%.4f)\n",
           pb, dc.phase_gradient);

    /* ── 9.7 Oracle 5: Site Weights ── */
    dyn_chain_compute_weights(&dc, D);
    int weakest = dyn_chain_weakest_site(&dc);
    printf("  Oracle 5 (Site Weights): weakest site = %d (weight=%.4f)\n",
           weakest, dc.site_weight[weakest]);

    /* ── 9.8 Standard step (uses oracles 1-5) ── */
    int prev_start = dc.active_start, prev_end = dc.active_end;
    dyn_chain_step(&dc);
    printf("  Step: [%d,%d] → [%d,%d] (epoch %u, grew=%u, contracted=%u)\n",
           prev_start, prev_end, dc.active_start, dc.active_end,
           dc.epoch, dc.grow_events, dc.contract_events);

    /* ── 9.9 Oracle 6: Ouroboros Self-Optimization ── */
    dyn_chain_rank_oracles(&dc);
    int top = dyn_chain_top_oracle(&dc);
    int gate = dyn_chain_recommended_gate(&dc);
    printf("  Oracle 6 (Ouroboros): β=%.2f, top oracle=%d, recommended gate=%d\n",
           dc.ouroboros_beta, top, gate);

    /* Ouroboros-optimized step (replaces standard step) */
    dyn_chain_ouroboros_step(&dc);
    printf("  Ouroboros step: [%d,%d], epoch=%u, gate_rec=%d\n",
           dc.active_start, dc.active_end, dc.epoch,
           dyn_chain_recommended_gate(&dc));

    /* ── 9.10 Activity queries ── */
    printf("  Site 15 active: %s\n", dyn_chain_is_active(&dc, 15) ? "YES" : "NO");
    printf("  Site 50 active: %s\n", dyn_chain_is_active(&dc, 50) ? "YES" : "NO");

    /* ── 9.11 Higher-D dynamics (convenience wrappers) ── */
    printf("\n  Higher-D DynLattice convenience constructors:\n");
    printf("    dyn_peps2d_create(Lx, Ly)           → 2D PEPS\n");
    printf("    dyn_tns3d_create(Lx, Ly, Lz)        → 3D TNS\n");
    printf("    dyn_tns4d_create(Lx, Ly, Lz, Lw)    → 4D TNS\n");
    printf("    dyn_tns5d_create(Lx, Ly, Lz, Lw, Lv) → 5D TNS\n");
    printf("    dyn_tns6d_create(Lx, Ly, Lz, Lw, Lv, Lu) → 6D TNS\n");

    /* Cleanup */
    dyn_chain_free(&dc);
    printf("  DynChain freed.\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * §10  FULL PIPELINE — Combining HPC + Triality + S₆ + Dynamics
 *
 * A realistic simulation template showing how subsystems compose:
 *   1. Create HPC graph with N sites
 *   2. Prepare initial states (problem-specific encoding)
 *   3. Trotter evolution loop:
 *      a. Local phase gates (on-site terms)
 *      b. DFT₆ (Fourier mixing)
 *      c. CZ entanglement (nearest-neighbor coupling)
 *      d. Edge compaction (merge redundant CZ edges)
 *      e. Selective measurement (Born sampling)
 *      f. Observable extraction
 *   4. Post-processing: statistics, exotic invariants, entropy
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void demo_full_pipeline(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  §10  FULL PIPELINE — Realistic Simulation Template       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ═══ Configuration ═══ */
    int N = 32;             /* Number of sites (quhits) */
    int depth = 10;         /* Trotter circuit depth */
    double dt = 0.1;        /* Trotter time step */
    double coupling = 1.0;  /* Coupling strength */

    /* ═══ Phase 1: Setup ═══ */
    tic();
    HPCGraph *g = hpc_create(N);

    /* Prepare initial state: alternating |0⟩ and DFT|0⟩ */
    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            /* Computational basis */
            double re[6] = {1, 0, 0, 0, 0, 0}, im[6] = {0};
            hpc_set_local(g, i, re, im);
        } else {
            /* Superposition via DFT */
            hpc_dft(g, i);
        }
    }
    printf("  Setup: %d sites, depth=%d, dt=%.2f (%.3f s)\n",
           N, depth, dt, toc());

    /* ═══ Phase 2: Trotter Evolution ═══ */
    printf("  Running Trotter evolution...\n");
    tic();

    for (int step = 0; step < depth; step++) {
        /* 2a. On-site phase gates (Hamiltonian diagonal terms) */
        for (int i = 0; i < N; i++) {
            double phi_re[6], phi_im[6];
            for (int k = 0; k < D; k++) {
                double angle = -coupling * dt * k * k / (double)D;
                phi_re[k] = cos(angle);
                phi_im[k] = sin(angle);
            }
            hpc_phase(g, i, phi_re, phi_im);
        }

        /* 2b. DFT₆ — move to Fourier basis for hopping */
        for (int i = 0; i < N; i++)
            hpc_dft(g, i);

        /* 2c. CZ entanglement — nearest-neighbor coupling */
        for (int i = 0; i < N - 1; i++)
            hpc_cz(g, i, i + 1);

        /* 2d. Edge compaction — prevent edge accumulation */
        if ((step + 1) % 3 == 0)
            hpc_compact_edges(g);

        /* 2e. Selective measurement (stochastic Born sampling) */
        for (int i = 0; i < N; i++) {
            if (rng_uniform() < 0.05) {  /* 5% measurement rate */
                hpc_measure(g, i, rng_uniform());
                /* Re-initialize measured site (optional) */
                hpc_dft(g, i);
            }
        }
    }

    double evo_time = toc();
    printf("  Evolution complete: %.3f s (%d steps × %d sites)\n",
           evo_time, depth, N);
    printf("  Edges: %lu (CZ:%lu, Phase:%lu)\n",
           g->n_edges, g->cz_edges, g->phase_edges);

    /* ═══ Phase 3: Observables ═══ */
    printf("\n  ── Observables ──\n");

    /* 3a. Per-site marginal probabilities */
    printf("  Site marginals (sampled):\n");
    for (int i = 0; i < N; i += N/4) {
        printf("    Site %2d: [", i);
        for (int v = 0; v < D; v++) {
            double p = hpc_marginal(g, i, v);
            printf("%.3f%s", p, v < 5 ? ", " : "");
        }
        printf("]\n");
    }

    /* 3b. Entanglement entropy at various cuts */
    printf("  Entropy profile:\n");
    for (int cut = 0; cut < N; cut += N/8) {
        double S = hpc_entropy_cut(g, cut);
        printf("    Cut after site %2d: S = %.4f bits\n", cut, S);
    }

    /* 3c. Exotic invariant profile */
    double total_delta = 0;
    for (int i = 0; i < N; i++)
        total_delta += triality_exotic_invariant_cached(&g->locals[i]);
    printf("  Mean exotic invariant Δ = %.6f\n", total_delta / N);

    /* 3d. Triality statistics */
    printf("  Triality view conversion stats:\n");
    triality_stats_print();

    /* ═══ Phase 4: Final statistics ═══ */
    printf("\n  ── Performance Summary ──\n");
    hpc_print_stats(g);

    double hilbert_log = N * log10(6.0);
    unsigned long mem = (unsigned long)(N * sizeof(TrialityQuhit) +
                         g->n_edges * sizeof(HPCEdge) + sizeof(HPCGraph));
    printf("  Hilbert space: 6^%d ≈ 10^%.0f dimensions\n", N, hilbert_log);
    printf("  Memory used: %lu bytes (~%.1f KB)\n", mem, mem / 1024.0);
    printf("  Time per Trotter step: %.4f ms\n", evo_time / depth * 1000);

    hpc_destroy(g);
    printf("  Pipeline complete.\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — Run all demos
 * ═══════════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  HEXSTATE QUANTUM OPERATIONS TEMPLATE                              ║\n");
    printf("║  D=6 Hexagonal Quantum Simulator — All Subsystems                  ║\n");
    printf("║  Pure C · No Dependencies · Consumer Hardware                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    /* One-time global initialization */
    rng_seed((uint64_t)time(NULL));
    s6_exotic_init();
    triality_exotic_init();
    hexagram_init_tables();
    triality_stats_reset();

    /* Run all section demos */
    demo_core_engine();       /* §1: Core Engine */
    demo_hpc_graph();         /* §2: HPC Phase Graph */
    demo_triality();          /* §3: Triality Layer */
    demo_s6_exotic();         /* §4: S₆ Exotic */
    demo_triadic();           /* §5: Three-Body Entanglement */
    demo_hexagram();          /* §6: Hexagram Duality */
    demo_lazy();              /* §7: Lazy Evaluation */
    demo_tensor_networks();   /* §8: Tensor Networks */
    demo_dynamics();          /* §9: Entropy Dynamics */
    demo_full_pipeline();     /* §10: Full Pipeline */

    printf("\n╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  ALL SECTIONS COMPLETE                                             ║\n");
    printf("║                                                                     ║\n");
    printf("║  To use this template:                                             ║\n");
    printf("║    1. Copy the section(s) you need into your own .c file           ║\n");
    printf("║    2. Modify the state preparation for your problem                ║\n");
    printf("║    3. Adjust the gate sequence (Trotter circuit) for your Hamiltonian║\n");
    printf("║    4. Extract the observables relevant to your physics             ║\n");
    printf("║                                                                     ║\n");
    printf("║  Most simulations only need §2 (HPC) + §4 (S₆) + §10 (Pipeline)  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    return 0;
}
