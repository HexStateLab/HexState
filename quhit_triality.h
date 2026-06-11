/*
 * quhit_triality.h — The Triality Quhit
 *
 * A new quantum primitive based on the CMY geometric principle:
 * three mutually-defining views (Edge/Vertex/Diagonal) where each
 * view's structure IS the other views' structure in a different role.
 *
 *   Edge of A = Vertex of B = Diagonal of C  (cyclic)
 *
 * The triality quhit stores state in all three views with lazy
 * conversion. Gates automatically execute in their cheapest view:
 *   Phase gates   → Edge view     O(D)
 *   Shift gates   → Vertex view   O(D)
 *   Conjugate ops → Diagonal view O(D)
 *   General       → any view      O(D²)
 *
 * Average gate cost: O(12) instead of O(36). 3× free speedup.
 *
 */

#ifndef QUHIT_TRIALITY_H
#define QUHIT_TRIALITY_H

#include <stdint.h>
#include "s6_exotic.h"

#define TRI_D 6

/* ═══════════════════════════════════════════════════════════════════════
 * VIEW IDENTIFIERS
 * ═══════════════════════════════════════════════════════════════════════ */

#define VIEW_EDGE     0   /* Computational basis — Yellow square */
#define VIEW_VERTEX   1   /* Fourier basis (DFT₆) — Cyan square */
#define VIEW_DIAGONAL 2   /* Conjugate Fourier (DFT₆²) — Magenta square */
#define VIEW_FOLDED   3   /* Antipodal fold: Stage 1 of factored DFT₆ */
#define VIEW_EXOTIC   4   /* Exotic fold: syntheme-parameterized (outer automorphism) */
#define VIEW_TETRA    5   /* Tetrahedral eigenbasis: DFT₆ eigenspace decomposition */

/* Dirty bitmask: bit 0-5 for each view */
#define DIRTY_EDGE     0x01
#define DIRTY_VERTEX   0x02
#define DIRTY_DIAGONAL 0x04
#define DIRTY_FOLDED   0x08
#define DIRTY_EXOTIC   0x10
#define DIRTY_TETRA    0x20
#define DIRTY_ALL      0x3F

/* ═══════════════════════════════════════════════════════════════════════
 * THE TRIALITY QUHIT
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Three views of the same quantum state */
    double edge_re[TRI_D],   edge_im[TRI_D];     /* |ψ⟩ in computational basis */
    double vertex_re[TRI_D], vertex_im[TRI_D];    /* |ψ⟩ in Fourier basis */
    double diag_re[TRI_D],   diag_im[TRI_D];      /* |ψ⟩ in conjugate basis */
    double folded_re[TRI_D], folded_im[TRI_D];    /* Antipodal fold intermediate */
    double exotic_re[TRI_D], exotic_im[TRI_D];    /* Exotic fold (alt syntheme) */
    double tetra_re[TRI_D],  tetra_im[TRI_D];     /* DFT₆ eigenbasis coefficients */
    int    exotic_syntheme;                        /* Which syntheme to use for exotic view */

    uint8_t dirty;      /* Which views are stale (bits 0-3) */
    uint8_t primary;    /* Which view was last written (0/1/2/3) */

    /* ── Enhancement flags ── */
    int8_t  eigenstate_class;  /* -1=unknown, 0..3=DFT₆ eigenvalue {1,-1,i,-i} */
    uint8_t active_mask;       /* Bitmask of non-zero basis states (6 bits) */
    uint8_t active_count;      /* popcount(active_mask), 1..6 */
    uint8_t real_valued;       /* 1 if all imaginary parts are zero */

    /* ── Exotic invariant cache (Fix #5) ── */
    double  cached_delta;          /* Cached exotic invariant Δ */
    double  cached_fingerprint[11];/* Cached conjugacy-class deltas */
    uint8_t delta_valid;           /* 1 if cached values are up-to-date */
} TrialityQuhit;

/* ═══════════════════════════════════════════════════════════════════════
 * TRIALITY PAIR — Two entangled triality quhits
 * Each partner contributes a different view to the joint state.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    double joint_re[TRI_D * TRI_D];
    double joint_im[TRI_D * TRI_D];
    int    view_a;  /* which view partner A contributes */
    int    view_b;  /* which view partner B contributes */
} TrialityPair;

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

/* Initialize to |0⟩ with all views clean */
void triality_init(TrialityQuhit *q);

/* Initialize to basis state |k⟩ */
void triality_init_basis(TrialityQuhit *q, int k);

/* Copy */
void triality_copy(TrialityQuhit *dst, const TrialityQuhit *src);

/* ═══════════════════════════════════════════════════════════════════════
 * VIEW MANAGEMENT — Lazy DFT₆ conversion
 * ═══════════════════════════════════════════════════════════════════════ */

/* Ensure a specific view is up-to-date (converts from primary if dirty) */
void triality_ensure_view(TrialityQuhit *q, int view);

/* Force recompute all views from primary */
void triality_sync_all(TrialityQuhit *q);

/* Get read-only access to a view (ensures it first) */
const double *triality_view_re(TrialityQuhit *q, int view);
const double *triality_view_im(TrialityQuhit *q, int view);

/* ═══════════════════════════════════════════════════════════════════════
 * OPTIMAL-VIEW GATES — O(D) when gate matches view
 * ═══════════════════════════════════════════════════════════════════════ */

/* Phase gate: |k⟩ → e^{iφₖ}|k⟩ — diagonal in EDGE view, O(D) */
void triality_phase(TrialityQuhit *q, const double *phi_re, const double *phi_im);

/* Single-phase: |k⟩ → e^{iφ}|k⟩, all others unchanged — O(1) */
void triality_phase_single(TrialityQuhit *q, int k, double phi_re, double phi_im);

/* Z gate: |k⟩ → ω^k |k⟩ — diagonal in EDGE view, O(D) */
void triality_z(TrialityQuhit *q);

/* Shift gate: |k⟩ → |k+δ mod D⟩ — diagonal in VERTEX view, O(D) */
void triality_shift(TrialityQuhit *q, int delta);

/* X gate: |k⟩ → |k+1 mod D⟩ — diagonal in VERTEX view, O(D) */
void triality_x(TrialityQuhit *q);

/* DFT₆: rotates edge→vertex→diagonal→edge — view rotation, O(D²) once */
void triality_dft(TrialityQuhit *q);

/* Inverse DFT₆ */
void triality_idft(TrialityQuhit *q);

/* General unitary in a specific view — O(D²) */
void triality_unitary(TrialityQuhit *q, int view,
                      const double *U_re, const double *U_im);

/* ═══════════════════════════════════════════════════════════════════════
 * CZ GATE — O(D) in edge view (diagonal)
 * ═══════════════════════════════════════════════════════════════════════ */

void triality_cz(TrialityQuhit *a, TrialityQuhit *b);

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT — O(D) via cached view
 * ═══════════════════════════════════════════════════════════════════════ */

/* Measure in a specific view basis. Returns outcome 0..D-1. Collapses state. */
int triality_measure(TrialityQuhit *q, int view, uint64_t *rng_state);

/* Probability distribution in a view — O(D), no collapse */
void triality_probabilities(TrialityQuhit *q, int view, double *probs);

/* ═══════════════════════════════════════════════════════════════════════
 * TRIALITY ROTATION — The geometric heart
 * ═══════════════════════════════════════════════════════════════════════ */

/* Rotate the role assignment: Edge→Vertex→Diagonal→Edge
 * This is a FREE operation — it just relabels which view is which.
 * No amplitudes are modified. O(1). */
void triality_rotate(TrialityQuhit *q);

/* Inverse rotation: Diagonal→Vertex→Edge→Diagonal. O(1). */
void triality_rotate_inv(TrialityQuhit *q);

/* ═══════════════════════════════════════════════════════════════════════
 * S₆ OUTER AUTOMORPHISM — Exotic Extensions
 *
 * S₆ is the ONLY symmetric group with a non-trivial outer automorphism.
 * These functions exploit this D=6-unique structure.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Initialize the exotic engine (builds φ table). Call once at startup. */
void triality_exotic_init(void);

/* Set which syntheme the exotic view uses (default: 0 = {(01),(23),(45)}) */
void triality_set_exotic_syntheme(TrialityQuhit *q, int syntheme_idx);

/* Fold using any of the 15 synthemes instead of the default antipodal */
void triality_fold_syntheme(TrialityQuhit *q, int syntheme_idx);
void triality_unfold_syntheme(TrialityQuhit *q, int syntheme_idx);

/* Apply exotic gate: uses φ(σ) instead of σ. O(D). */
void triality_exotic_gate(TrialityQuhit *q, S6Perm sigma);

/* Dual CZ: standard CZ + exotic channel information. Returns the
 * statistical distance between standard and exotic channels. */
double triality_cz_dual(TrialityQuhit *a, TrialityQuhit *b);

/* Measure in the exotic fold basis. Returns outcome 0..D-1. */
int triality_measure_exotic(TrialityQuhit *q, int syntheme_idx, uint64_t *rng_state);

/* Dual measurement: returns both standard and exotic outcomes.
 * Exotic outcome is in *exotic_outcome. Standard is returned. */
int triality_measure_dual(TrialityQuhit *q, int view, int exotic_syntheme,
                          uint64_t *rng_state, int *exotic_outcome);

/* 6-fold rotation: cycles through all 6 synthematic views.
 * Standard rotate: Edge→Vertex→Diagonal→Edge (3-cycle, views 0→1→2→0)
 * Exotic rotate:   Also cycles the exotic syntheme through its total.
 * This accesses the full Aut(S₆) ≅ S₆ ⋊ Z₂ structure. */
void triality_rotate_exotic(TrialityQuhit *q);

/* Probabilities in both standard and exotic bases — no collapse */
void triality_dual_probabilities(TrialityQuhit *q, int view,
                                 double *probs_std, double *probs_exo);

/* ═══════════════════════════════════════════════════════════════════════
 * GEOMETRIC COSMOLOGY ENHANCEMENTS
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Enhancement 1: Folded View ── */
/* Fold: pair antipodal vertices (0↔3, 1↔4, 2↔5) via Hadamard.
 * This is Stage 1 of the factored DFT₆ (Cooley-Tukey 6=2×3).
 * vesica[k] = (ψ[k] + ψ[k+3]) / √2  (k=0,1,2)
 * wave[k]   = (ψ[k] - ψ[k+3]) / √2  (k=0,1,2) */
void triality_fold(TrialityQuhit *q);
void triality_unfold(TrialityQuhit *q);

/* Convert Edge↔Vertex via the folded intermediate (O(18) vs O(36)) */
void triality_ensure_view_via_fold(TrialityQuhit *q, int target_view);

/* ── Enhancement 5: Tetrahedral Eigenbasis ── */
/* Decompose state into DFT₆ eigenspaces {λ=1(×2), λ=-1(×2), λ=i, λ=-i}.
 * Once cached, all view conversions and DFT/IDFT gates become O(D). */
void triality_ensure_tetra(TrialityQuhit *q);

/* Convert from tetra cache to any standard view — O(D²) but avoids
 * needing a clean standard view as starting point */
void triality_tetra_to_view(TrialityQuhit *q, int target_view);

/* DFT₆ via tetra: multiply each eigencomponent by λ — O(D) */
void triality_dft_via_tetra(TrialityQuhit *q);
void triality_idft_via_tetra(TrialityQuhit *q);

/* Cached exotic invariant — returns Δ without recomputing if state is unchanged */
double triality_exotic_invariant_cached(TrialityQuhit *q);
void   triality_exotic_fingerprint_cached(TrialityQuhit *q, double *deltas);

/* Invalidate exotic cache (called internally after state-modifying operations) */
void triality_invalidate_exotic_cache(TrialityQuhit *q);

/* ── Enhancement 2: Eigenstate Detection ── */
/* Detect if state is a DFT₆ eigenstate. Sets eigenstate_class.
 * Returns eigenstate_class (0..3) or -1 if not an eigenstate. */
int triality_detect_eigenstate(TrialityQuhit *q);

/* Clear eigenstate flag (call when non-diagonal gate is applied) */
void triality_clear_eigenstate(TrialityQuhit *q);

/* ── Enhancement 3: Subspace Confinement ── */
/* Recompute active_mask and active_count from current edge amplitudes */
void triality_update_mask(TrialityQuhit *q);

/* ── Enhancement 4: Real-Valued Detection ── */
/* Detect and set real_valued flag from current edge amplitudes */
void triality_detect_real(TrialityQuhit *q);

/* ── Combined: refresh all enhancement flags ── */
void triality_refresh_flags(TrialityQuhit *q);

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

/* Print state in all three views */
void triality_print(TrialityQuhit *q, const char *label);

/* View conversion count (for benchmarking) */
typedef struct {
    uint64_t edge_to_vertex;
    uint64_t edge_to_diag;
    uint64_t vertex_to_edge;
    uint64_t vertex_to_diag;
    uint64_t diag_to_edge;
    uint64_t diag_to_vertex;
    uint64_t edge_to_folded;
    uint64_t folded_to_vertex;
    uint64_t gates_edge;    /* gates executed in edge view */
    uint64_t gates_vertex;  /* gates executed in vertex view */
    uint64_t gates_diag;    /* gates executed in diagonal view */
    uint64_t rotations;     /* O(1) triality rotations */
    uint64_t eigenstate_skips;   /* view conversions skipped by eigenstate flag */
    uint64_t mask_skips;         /* operations skipped by active_mask */
    uint64_t real_fast_path;     /* operations using real-valued fast path */
    uint64_t exotic_folds;       /* exotic syntheme fold operations */
    uint64_t exotic_gates;       /* exotic-automorphism gate applications */
    uint64_t dual_measurements;  /* dual standard+exotic measurements */
    uint64_t tetra_conversions;  /* view conversions via tetrahedral eigenbasis */
    uint64_t tetra_dft_skips;    /* DFT/IDFT operations done via tetra O(D) path */
} TrialityStats;

extern TrialityStats triality_stats;
void triality_stats_reset(void);
void triality_stats_print(void);

/* ═══════════════════════════════════════════════════════════════════════
 * LAZY TRIALITY QUHIT — Heisenberg Picture
 *
 * Amplitudes are NEVER touched until measurement.
 * Gates accumulate as diagonal phase vectors.
 * DFTs accumulate as a counter between segments.
 *
 * Chain: state → F^pre0 · D0 → F^pre1 · D1 → ... → F^trailing
 * F⁴ = I, so each count is mod 4. Pure DFT sequences cancel.
 * Same-view consecutive gates fuse into one D. O(D) per gate.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* The frozen initial state — set once at init */
    double state_re[TRI_D], state_im[TRI_D];

    /* Transformation chain: array of segments.
     * Each segment has a pre_dfts count (0-3 DFTs before its diagonal)
     * and a diagonal phase vector applied in edge view. */
    #define MAX_LAZY_SEGMENTS 64
    struct {
        double diag_re[TRI_D];  /* Diagonal phase vector */
        double diag_im[TRI_D];
        int    pre_dfts;        /* 0-3 DFTs to apply BEFORE this diagonal (F^4=I) */
    } segments[MAX_LAZY_SEGMENTS];
    int n_segments;
    int trailing_dfts;          /* DFTs after the last segment (accumulated) */

    /* Oracle: cross-batch composite matrix.
     * When segments overflow, instead of materializing, the Oracle
     * compiles the chain into a 6×6 matrix and absorbs it here.
     * At final materialize: state = oracle_M · initial_state, then
     * any remaining segments are applied on top. */
    double oracle_M_re[TRI_D][TRI_D];
    double oracle_M_im[TRI_D][TRI_D];
    int oracle_active;          /* 1 if oracle_M contains data */

    /* Stats */
    uint64_t gates_fused;       /* Gates absorbed into existing segment */
    uint64_t segments_created;  /* New segments started */
    uint64_t materializations;  /* Times state was materialized */
} LazyTrialityQuhit;

/* Lifecycle */
void ltri_init(LazyTrialityQuhit *q);
void ltri_init_basis(LazyTrialityQuhit *q, int k);

/* Gates — O(D) each, zero view conversions */
void ltri_z(LazyTrialityQuhit *q);
void ltri_x(LazyTrialityQuhit *q);
void ltri_shift(LazyTrialityQuhit *q, int delta);
void ltri_dft(LazyTrialityQuhit *q);
void ltri_idft(LazyTrialityQuhit *q);
void ltri_phase(LazyTrialityQuhit *q, const double *phi_re, const double *phi_im);
void ltri_phase_single(LazyTrialityQuhit *q, int k, double phi_re, double phi_im);

/* Materialize — apply accumulated transform, return edge-view amplitudes */
void ltri_materialize(LazyTrialityQuhit *q, double *out_re, double *out_im);

/* Force materialize — compile oracle + apply chain, producing a TrialityQuhit.
 * Use this when a two-body operation (CZ) needs actual amplitudes. */
void ltri_force_materialize(LazyTrialityQuhit *q, TrialityQuhit *out);

/* Measure — materialize + Born sample */
int ltri_measure(LazyTrialityQuhit *q, int view, uint64_t *rng_state);

/* Stats */
void ltri_stats_print(const LazyTrialityQuhit *q);

/* ═══════════════════════════════════════════════════════════════════════
 * HEXAGRAM INTERCONVERSION
 * Convert between triality (vertex model) and hexagram (edge model).
 * Requires quhit_hexagram.h and hexagram_init_tables() called first.
 * ═══════════════════════════════════════════════════════════════════════ */

struct HexagramQuhit;  /* forward declaration */

/* Convert triality quhit → hexagram quhit via H₆ transform */
void triality_to_hexagram(TrialityQuhit *src, struct HexagramQuhit *dst);

/* Convert hexagram quhit → triality quhit via H₆† transform */
void hexagram_to_triality(struct HexagramQuhit *src, TrialityQuhit *dst);

#endif /* QUHIT_TRIALITY_H */
