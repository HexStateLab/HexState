/*
 * hpc_graph.h — The Holographic Phase Graph
 *
 * The Devil's alternative to SVD.
 *
 * SVD reaches into the interior of a tensor and numerically discovers
 * structure. O(n³). Dense. Bulk-seeking.
 *
 * HPC works from the surface: entanglement is encoded as weighted phase
 * edges in a graph. Amplitudes are computed on demand via O(N+E) graph
 * traversal. The state vector is never materialized.
 *
 * Core formula:
 *   ψ(i₁,...,iₙ) = [Π_k a_k(i_k)] × [Π_edges w_e(i_a, i_b)]
 *
 * For CZ edges: w_e(a,b) = ω^(a·b)  — EXACT, fidelity = 1.0
 * For general edges: w_e(a,b) = arbitrary 6×6 phase matrix — bounded fidelity
 * For syntheme edges: w_e determined by S₆ syntheme projector — O(1) lookup
 *
 * This is an extension of magic_pointer.h that supports:
 *   - Weighted phase edges (not just CZ)
 *   - Syntheme metadata per edge
 *   - Fidelity tracking
 *   - On-demand marginal probabilities
 */

#ifndef HPC_GRAPH_H
#define HPC_GRAPH_H

#include "quhit_triality.h"
#include "s6_exotic.h"
#include "born_rule.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define HPC_D           6       /* Physical dimension per site           */
#define HPC_INIT_EDGES  4096    /* Initial edge capacity (grows)         */
#define HPC_INIT_LOG    8192    /* Initial gate log capacity (grows)     */

/* ω = exp(2πi/6) roots of unity — precomputed */
static const double HPC_W6_RE[6] = {
    1.0, 0.5, -0.5, -1.0, -0.5, 0.5
};
static const double HPC_W6_IM[6] = {
    0.0, 0.866025403784438647, 0.866025403784438647,
    0.0, -0.866025403784438647, -0.866025403784438647
};

/* ═══════════════════════════════════════════════════════════════════════
 * EDGE TYPES — The Devil has more than one handshake
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    HPC_EDGE_CZ,        /* Exact CZ: w(a,b) = ω^(a·b), fidelity=1.0     */
    HPC_EDGE_PHASE,     /* General phase: w(a,b) = arbitrary 6×6 matrix  */
    HPC_EDGE_SYNTHEME,  /* Syntheme-projected: w from S₆ syntheme        */
    HPC_EDGE_PERMUTE    /* Permutation: |i⟩_a|j⟩_b → |i'⟩_a|j'⟩_b       */
} HPCEdgeType;

/* ═══════════════════════════════════════════════════════════════════════
 * WEIGHTED PHASE EDGE — One entangling interaction on the surface
 *
 * For CZ edges, only type + site indices are used.
 * For general/syntheme edges, the full 6×6 phase matrix is stored.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    HPCEdgeType type;
    uint64_t    site_a;         /* First site index                      */
    uint64_t    site_b;         /* Second site index                     */

    union {
        struct {
            double w_re[HPC_D][HPC_D];   /* PHASE/SYNTHEME: 6×6 complex weight */
            double w_im[HPC_D][HPC_D];   /* PHASE/SYNTHEME: imaginary part     */
        };
        struct {
            uint32_t perm_target[HPC_D][HPC_D]; /* fwd: (a,b) → packed target   */
            uint32_t perm_source[HPC_D][HPC_D]; /* inv: (a,b) → packed source   */
        };
    };

    /* Syntheme metadata (only for SYNTHEME type).
     * For CZ: syntheme_id stores CZ multiplicity m. */
    uint8_t     syntheme_id;    /* Which of 15 synthemes (0-14)          */
    uint8_t     total_id;       /* Which of 6 synthematic totals (0-5)   */

    /* Quality metric */
    double      fidelity;       /* 1.0 = lossless, 0.0 = total loss     */
} HPCEdge;



/* ═══════════════════════════════════════════════════════════════════════
 * GATE LOG ENTRY — Recording what was applied
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    HPC_GATE_LOCAL_DFT,
    HPC_GATE_LOCAL_PHASE,
    HPC_GATE_LOCAL_SHIFT,
    HPC_GATE_LOCAL_UNITARY,
    HPC_GATE_CZ,
    HPC_GATE_GENERAL_2SITE,
    HPC_GATE_PERMUTE,
    HPC_GATE_INIT
} HPCGateType;

typedef struct {
    HPCGateType type;
    uint64_t    site_a;
    uint64_t    site_b;         /* Only for 2-site gates                 */
    double      params[12];     /* Gate-specific parameters              */
    double      fidelity;       /* Encoding fidelity for this gate       */
} HPCGateEntry;

/* ═══════════════════════════════════════════════════════════════════════
 * PER-SITE ADJACENCY LIST — O(degree) edge lookup
 *
 * Each site maintains a list of edge indices that touch it.
 * This is the optimization that turns O(N×E) → O(N×degree) = O(N).
 * ═══════════════════════════════════════════════════════════════════════ */

#define HPC_ADJ_INIT 16  /* Initial adjacency list capacity per site */

typedef struct {
    uint64_t *edge_ids;  /* Indices into the graph's edge array         */
    uint64_t  count;     /* Number of edges touching this site          */
    uint64_t  capacity;  /* Allocated capacity                          */
} HPCAdjList;

/* ═══════════════════════════════════════════════════════════════════════
 * HPC GRAPH — The Devil's state representation
 *
 * This struct IS the state. The 6^N state vector does not exist.
 * Entanglement is a graph. Amplitudes are computed on demand.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* ── Sites ── */
    uint64_t        n_sites;
    TrialityQuhit  *locals;         /* Per-site local states             */

    /* ── Phase Graph ── */
    uint64_t        n_edges;
    uint64_t        edge_cap;
    HPCEdge        *edges;          /* Weighted phase edge list          */

    /* ── Adjacency Lists ── O(1) per-site edge lookup */
    HPCAdjList     *adj;            /* Per-site adjacency lists          */

    /* ── Gate Log ── */
    uint64_t        n_log;
    uint64_t        log_cap;
    HPCGateEntry   *gate_log;

    /* ── Statistics ── */
    uint64_t        amp_evals;      /* Amplitude evaluations performed   */
    uint64_t        prob_evals;     /* Probability evaluations           */
    uint64_t        measurements;   /* Measurements performed            */
    uint64_t        cz_edges;       /* Number of exact CZ edges          */
    uint64_t        phase_edges;    /* Number of general phase edges     */
    uint64_t        syntheme_edges; /* Number of syntheme-encoded edges  */
    uint64_t        perm_edges;     /* Number of permutation edges       */
    double          min_fidelity;   /* Worst fidelity across all edges   */
    double          avg_fidelity;   /* Average fidelity                  */
} HPCGraph;

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

static inline HPCGraph *hpc_create(uint64_t n_sites)
{
    HPCGraph *g = (HPCGraph *)calloc(1, sizeof(HPCGraph));
    if (!g) return NULL;

    g->n_sites = n_sites;
    g->locals = (TrialityQuhit *)calloc(n_sites, sizeof(TrialityQuhit));
    if (!g->locals) { free(g); return NULL; }

    for (uint64_t i = 0; i < n_sites; i++)
        triality_init(&g->locals[i]);

    g->edge_cap = (n_sites < HPC_INIT_EDGES) ? n_sites * 2 + 16 : HPC_INIT_EDGES;
    g->edges = (HPCEdge *)calloc(g->edge_cap, sizeof(HPCEdge));
    g->n_edges = 0;

    /* Initialize per-site adjacency lists */
    g->adj = (HPCAdjList *)calloc(n_sites, sizeof(HPCAdjList));
    for (uint64_t i = 0; i < n_sites; i++) {
        g->adj[i].capacity = HPC_ADJ_INIT;
        g->adj[i].edge_ids = (uint64_t *)calloc(HPC_ADJ_INIT, sizeof(uint64_t));
        g->adj[i].count = 0;
    }

    g->log_cap = HPC_INIT_LOG;
    g->gate_log = (HPCGateEntry *)calloc(g->log_cap, sizeof(HPCGateEntry));
    g->n_log = 0;

    g->min_fidelity = 1.0;
    g->avg_fidelity = 1.0;

    return g;
}

static inline void hpc_destroy(HPCGraph *g)
{
    if (!g) return;
    if (g->adj) {
        for (uint64_t i = 0; i < g->n_sites; i++)
            free(g->adj[i].edge_ids);
        free(g->adj);
    }
    free(g->locals);
    free(g->edges);
    free(g->gate_log);
    free(g);
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL: grow arrays
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_grow_edges(HPCGraph *g)
{
    if (g->n_edges < g->edge_cap) return;
    g->edge_cap *= 2;
    g->edges = (HPCEdge *)realloc(g->edges, g->edge_cap * sizeof(HPCEdge));
}

/* Grow the graph to accommodate new_n_sites total sites.
 * Reallocates locals[] and adj[] arrays, initializes new entries.
 * If new_n_sites <= g->n_sites, this is a no-op. */
static inline void hpc_grow_sites(HPCGraph *g, uint64_t new_n_sites)
{
    if (new_n_sites <= g->n_sites) return;

    g->locals = (TrialityQuhit *)realloc(g->locals,
                                          new_n_sites * sizeof(TrialityQuhit));
    g->adj = (HPCAdjList *)realloc(g->adj,
                                    new_n_sites * sizeof(HPCAdjList));

    /* Initialize the new sites */
    for (uint64_t i = g->n_sites; i < new_n_sites; i++) {
        triality_init(&g->locals[i]);
        g->adj[i].capacity = HPC_ADJ_INIT;
        g->adj[i].edge_ids = (uint64_t *)calloc(HPC_ADJ_INIT, sizeof(uint64_t));
        g->adj[i].count = 0;
    }

    g->n_sites = new_n_sites;
}

static inline void hpc_grow_adj(HPCAdjList *a)
{
    if (a->count < a->capacity) return;
    a->capacity *= 2;
    a->edge_ids = (uint64_t *)realloc(a->edge_ids,
                                       a->capacity * sizeof(uint64_t));
}

static inline void hpc_adj_add(HPCGraph *g, uint64_t site, uint64_t edge_id)
{
    HPCAdjList *a = &g->adj[site];
    hpc_grow_adj(a);
    a->edge_ids[a->count++] = edge_id;
}

static inline void hpc_adj_remove(HPCGraph *g, uint64_t site, uint64_t edge_id)
{
    HPCAdjList *a = &g->adj[site];
    for (uint64_t i = 0; i < a->count; i++) {
        if (a->edge_ids[i] == edge_id) {
            a->edge_ids[i] = a->edge_ids[--a->count];
            return;
        }
    }
}

/* Replace one edge ID with another in a site's adjacency list */
static inline void hpc_adj_replace(HPCGraph *g, uint64_t site,
                                    uint64_t old_id, uint64_t new_id)
{
    HPCAdjList *a = &g->adj[site];
    for (uint64_t i = 0; i < a->count; i++) {
        if (a->edge_ids[i] == old_id) {
            a->edge_ids[i] = new_id;
            return;
        }
    }
}

static inline void hpc_grow_log(HPCGraph *g)
{
    if (g->n_log < g->log_cap) return;
    g->log_cap *= 2;
    g->gate_log = (HPCGateEntry *)realloc(g->gate_log,
                                           g->log_cap * sizeof(HPCGateEntry));
}

static inline void hpc_log_gate(HPCGraph *g, HPCGateEntry entry)
{
    hpc_grow_log(g);
    g->gate_log[g->n_log++] = entry;
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL: update fidelity statistics
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_update_fidelity_stats(HPCGraph *g)
{
    if (g->n_edges == 0) {
        g->min_fidelity = 1.0;
        g->avg_fidelity = 1.0;
        return;
    }
    double sum = 0.0;
    double min_f = 1.0;
    for (uint64_t e = 0; e < g->n_edges; e++) {
        double f = g->edges[e].fidelity;
        sum += f;
        if (f < min_f) min_f = f;
    }
    g->min_fidelity = min_f;
    g->avg_fidelity = sum / g->n_edges;
}

/* ═══════════════════════════════════════════════════════════════════════
 * LOCAL GATES — Absorbed into the local quhit state
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_set_local(HPCGraph *g, uint64_t site,
                                  const double re[6], const double im[6])
{
    TrialityQuhit *q = &g->locals[site];
    for (int i = 0; i < HPC_D; i++) {
        q->edge_re[i] = re[i];
        q->edge_im[i] = im[i];
    }
    q->primary = VIEW_EDGE;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;
    triality_update_mask(q);

    HPCGateEntry entry = { .type = HPC_GATE_INIT, .site_a = site,
                           .fidelity = 1.0 };
    for (int i = 0; i < 6; i++) entry.params[i] = re[i];
    hpc_log_gate(g, entry);
}

static inline void hpc_dft(HPCGraph *g, uint64_t site)
{
    triality_dft(&g->locals[site]);
    triality_update_mask(&g->locals[site]);
    HPCGateEntry entry = { .type = HPC_GATE_LOCAL_DFT, .site_a = site,
                           .fidelity = 1.0 };
    hpc_log_gate(g, entry);
}

static inline void hpc_phase(HPCGraph *g, uint64_t site,
                              const double phi_re[6], const double phi_im[6])
{
    triality_phase(&g->locals[site], phi_re, phi_im);
    triality_update_mask(&g->locals[site]);
    HPCGateEntry entry = { .type = HPC_GATE_LOCAL_PHASE, .site_a = site,
                           .fidelity = 1.0 };
    for (int i = 0; i < 6; i++) entry.params[i] = phi_re[i];
    hpc_log_gate(g, entry);
}

static inline void hpc_shift(HPCGraph *g, uint64_t site, int delta)
{
    triality_shift(&g->locals[site], delta);
    triality_update_mask(&g->locals[site]);
    HPCGateEntry entry = { .type = HPC_GATE_LOCAL_SHIFT, .site_a = site,
                           .fidelity = 1.0 };
    entry.params[0] = (double)delta;
    hpc_log_gate(g, entry);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CZ GATE — The Devil's perfect handshake
 *
 * CZ is EXACT in HPC: no truncation, no approximation, no SVD.
 * The entanglement is recorded as a phase edge: w(a,b) = ω^(a·b).
 * Fidelity = 1.0. Always. This is the Devil at full power.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_cz(HPCGraph *g, uint64_t site_a, uint64_t site_b)
{
    hpc_grow_edges(g);

    uint64_t eid = g->n_edges;
    HPCEdge *e = &g->edges[eid];
    memset(e, 0, sizeof(HPCEdge));
    e->type = HPC_EDGE_CZ;
    e->site_a = site_a;
    e->site_b = site_b;
    e->fidelity = 1.0;
    e->syntheme_id = 1;  /* Use syntheme_id to store CZ multiplicity m */
    /* Phase matrix not stored — implicitly ω^(m·a·b) */

    g->n_edges++;
    g->cz_edges++;

    /* Maintain adjacency lists */
    hpc_adj_add(g, site_a, eid);
    hpc_adj_add(g, site_b, eid);

    HPCGateEntry entry = {
        .type = HPC_GATE_CZ,
        .site_a = site_a, .site_b = site_b,
        .fidelity = 1.0
    };
    hpc_log_gate(g, entry);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GENERAL 2-SITE GATE — Encoded as a weighted phase edge
 *
 * For a general 2-site gate G acting on sites (a,b):
 *   The gate creates entanglement that we encode as a phase matrix.
 *   G|ψ_a⟩|ψ_b⟩ = Σ_{j,k} G_{(j,k),(m,n)} ψ_a(m) ψ_b(n) |j⟩|k⟩
 *
 * We decompose G into: (local on a) × (phase edge) × (local on b)
 * The phase edge captures the entangling component.
 *
 * For CZ: this decomposition is EXACT (CZ is already in this form).
 * For general gates: this is the syntheme approximation (lossy).
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_general_2site(HPCGraph *g, uint64_t site_a,
                                      uint64_t site_b,
                                      const double *G_re, const double *G_im)
{
    /* G is a 36×36 matrix (D²×D² = 36×36) in row-major order.
     * G[(j*D+k)*D*D + (m*D+n)] = G_{(j,k),(m,n)}
     *
     * Phase edge extraction:
     * For each (j,k), compute the dominant phase of G_{(j,k),(j,k)}.
     * This captures the diagonal (phase) part of the interaction.
     * Off-diagonal terms are absorbed into local state updates. */

    hpc_grow_edges(g);

    uint64_t eid = g->n_edges;
    HPCEdge *e = &g->edges[eid];
    memset(e, 0, sizeof(HPCEdge));
    e->type = HPC_EDGE_PHASE;
    e->site_a = site_a;
    e->site_b = site_b;

    /* Extract diagonal phases: w(j,k) = G_{(j,k),(j,k)} / |G_{(j,k),(j,k)}| */
    double max_mag = 0.0;
    double fidelity_sum = 0.0;
    int fidelity_count = 0;

    for (int j = 0; j < HPC_D; j++) {
        for (int k = 0; k < HPC_D; k++) {
            int idx = (j * HPC_D + k) * HPC_D * HPC_D + (j * HPC_D + k);
            double g_re = G_re[idx];
            double g_im = G_im[idx];
            double mag = sqrt(g_re * g_re + g_im * g_im);

            if (mag > 1e-15) {
                e->w_re[j][k] = g_re / mag;
                e->w_im[j][k] = g_im / mag;
            } else {
                e->w_re[j][k] = 1.0;
                e->w_im[j][k] = 0.0;
            }

            if (mag > max_mag) max_mag = mag;

            double row_norm2 = 0.0;
            for (int m = 0; m < HPC_D; m++) {
                for (int n = 0; n < HPC_D; n++) {
                    int ridx = (j * HPC_D + k) * HPC_D * HPC_D + (m * HPC_D + n);
                    row_norm2 += G_re[ridx] * G_re[ridx] + G_im[ridx] * G_im[ridx];
                }
            }
            if (row_norm2 > 1e-30) {
                fidelity_sum += (g_re * g_re + g_im * g_im) / row_norm2;
                fidelity_count++;
            }
        }
    }

    e->fidelity = (fidelity_count > 0) ? fidelity_sum / fidelity_count : 0.0;

    g->n_edges++;
    g->phase_edges++;

    /* Maintain adjacency lists */
    hpc_adj_add(g, site_a, eid);
    hpc_adj_add(g, site_b, eid);

    hpc_update_fidelity_stats(g);

    HPCGateEntry entry = {
        .type = HPC_GATE_GENERAL_2SITE,
        .site_a = site_a, .site_b = site_b,
        .fidelity = e->fidelity
    };
    hpc_log_gate(g, entry);
}

/* ═══════════════════════════════════════════════════════════════════════
 * PERMUTATION EDGE — Non-diagonal site-to-site index permutation
 *
 * A permutation edge encodes a bijection on the 36 index pairs of
 * two host sites: |i⟩_a|j⟩_b → |i'⟩_a|j'⟩_b.
 *
 * Unlike phase edges (which multiply amplitudes), permutation edges
 * REARRANGE probability mass between basis configurations. This makes
 * them the native representation for gates like CNOT, Toffoli, and
 * modular multiplication that cannot be expressed as diagonal phases.
 *
 * Amplitude evaluation with permutation edges:
 *   ψ_after(i) = ψ_before(π⁻¹(i))
 *
 * where π⁻¹ maps each target index pair (t_a, t_b) back to the source
 * pair (s_a, s_b) that π sends to (t_a, t_b).
 *
 * Permutation edges are applied in the order they appear in the edge
 * array. During backward iteration in hpc_amplitude(), each permutation
 * edge transforms the current indices via its inverse mapping before
 * earlier edges and local amplitudes are evaluated.
 *
 * Packing: each entry stores packed = (a_idx << 3) | b_idx
 * Values 0-5 fit in 3 bits; max packed value = (5<<3)|5 = 45.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_permute(HPCGraph *g, uint64_t site_a, uint64_t site_b,
                                const uint32_t target[36])
{
    /* target is a flat array of 36 packed targets:
     * target[i_a * 6 + i_b] = (i'_a << 3) | i'_b
     * for source indices i_a, i_b ∈ [0,5].
     *
     * Must be a bijection on the 36 configurations. */

    hpc_grow_edges(g);

    uint64_t eid = g->n_edges;
    HPCEdge *e = &g->edges[eid];
    memset(e, 0, sizeof(HPCEdge));
    e->type = HPC_EDGE_PERMUTE;
    e->site_a = site_a;
    e->site_b = site_b;
    e->fidelity = 1.0;

    /* Store forward mapping and compute inverse */
    int inv_set[36] = {0};

    for (int i_a = 0; i_a < HPC_D; i_a++) {
        for (int i_b = 0; i_b < HPC_D; i_b++) {
            int idx = i_a * HPC_D + i_b;
            uint32_t packed = target[idx];
            e->perm_target[i_a][i_b] = packed;

            /* Verify indices are in range */
            int ta = (packed >> 3) & 7;
            int tb = packed & 7;
            if (ta >= HPC_D || tb >= HPC_D) {
                fprintf(stderr, "hpc_permute: invalid target (%d,%d) "
                        "at source (%d,%d)\n", ta, tb, i_a, i_b);
                e->perm_target[i_a][i_b] = packed; /* store anyway */
            }

            /* Build inverse: for target (ta,tb), source is (i_a,i_b) */
            int tidx = ta * HPC_D + tb;
            if (inv_set[tidx]) {
                fprintf(stderr, "hpc_permute: not bijective — "
                        "multiple sources map to (%d,%d)\n", ta, tb);
            }
            inv_set[tidx] = 1;
            e->perm_source[ta][tb] = (uint32_t)(i_a << 3) | i_b;
        }
    }

    g->n_edges++;
    g->perm_edges++;

    /* Maintain adjacency lists */
    hpc_adj_add(g, site_a, eid);
    hpc_adj_add(g, site_b, eid);

    hpc_update_fidelity_stats(g);

    HPCGateEntry entry = {
        .type = HPC_GATE_PERMUTE,
        .site_a = site_a, .site_b = site_b,
        .fidelity = 1.0
    };
    hpc_log_gate(g, entry);
}

/* ═══════════════════════════════════════════════════════════════════════
 * THE MAGIC: Amplitude Evaluation
 *
 * ψ(i₁,...,iₙ) = [Π_k a_k(i_k)] × [Π_edges w_e(i_a, i_b)]
 *
 * Cost: O(N + E) — linear in sites + edges
 * Memory: O(1) additional
 *
 * For CZ edges: w_e(a,b) = ω^(a·b)  — precomputed lookup, no math
 * For PHASE/SYNTHEME edges: w_e(a,b) from stored 6×6 matrix
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_amplitude(const HPCGraph *g,
                                  const uint32_t *indices,
                                  double *out_re, double *out_im)
{
    /* Phase 1: Apply inverse permutations to indices — O(E)
     *
     * Permutation edges are applied in the order they were added
     * (forward in the edge array). During evaluation, we undo them
     * in REVERSE order, transforming the requested indices back to
     * the "pre-permutation" configuration.
     *
     * Algorithm: iterate edges from newest to oldest.
     * Each PERMUTE edge maps current indices at its sites backward
     * through its inverse (perm_source), so that subsequent edges
     * and local amplitudes see the pre-permutation indices. */

    uint32_t ti[128];  /* transformed indices */
    uint64_t n_sites = g->n_sites;
    /* Bounds check: we use VLA for n_sites > 128 */
    uint32_t *transformed = ti;
    int use_heap = 0;
    if (n_sites > 128) {
        transformed = (uint32_t *)malloc(n_sites * sizeof(uint32_t));
        use_heap = 1;
    }
    memcpy(transformed, indices, n_sites * sizeof(uint32_t));

    for (int64_t e = (int64_t)g->n_edges - 1; e >= 0; e--) {
        const HPCEdge *edge = &g->edges[e];
        if (edge->type != HPC_EDGE_PERMUTE) continue;

        uint64_t a = edge->site_a;
        uint64_t b = edge->site_b;
        uint32_t packed = edge->perm_source[transformed[a]][transformed[b]];
        transformed[a] = (packed >> 3) & 7;
        transformed[b] = packed & 7;
    }

    /* Phase 2: Product of local amplitudes at transformed indices — O(N) */
    double re = 1.0, im = 0.0;
    for (uint64_t k = 0; k < n_sites; k++) {
        uint32_t idx = transformed[k];
        TrialityQuhit *q = (TrialityQuhit*)&g->locals[k];
        triality_ensure_view(q, VIEW_EDGE);
        double a_re = q->edge_re[idx];
        double a_im = q->edge_im[idx];
        double new_re = re * a_re - im * a_im;
        double new_im = re * a_im + im * a_re;
        re = new_re;
        im = new_im;
    }

    /* Phase 3: Phase edge accumulation at transformed indices — O(E)
     * Skip PERMUTE edges (already handled). */
    for (uint64_t e = 0; e < g->n_edges; e++) {
        const HPCEdge *edge = &g->edges[e];
        if (edge->type == HPC_EDGE_PERMUTE) continue;

        uint32_t ia = transformed[edge->site_a];
        uint32_t ib = transformed[edge->site_b];

        double wr, wi;

        if (edge->type == HPC_EDGE_CZ) {
            uint32_t m = edge->syntheme_id ? edge->syntheme_id : 1;
            uint32_t phase_idx = (m * ia * ib) % HPC_D;
            wr = HPC_W6_RE[phase_idx];
            wi = HPC_W6_IM[phase_idx];
        } else {
            wr = edge->w_re[ia][ib];
            wi = edge->w_im[ia][ib];
        }

        double new_re = re * wr - im * wi;
        double new_im = re * wi + im * wr;
        re = new_re;
        im = new_im;
    }

    *out_re = re;
    *out_im = im;
    ((HPCGraph *)g)->amp_evals++;

    if (use_heap) free(transformed);
}

/* ═══════════════════════════════════════════════════════════════════════
 * PROBABILITY — |ψ(i₁,...,iₙ)|²
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double hpc_probability(const HPCGraph *g,
                                      const uint32_t *indices)
{
    double re, im;
    hpc_amplitude(g, indices, &re, &im);
    ((HPCGraph *)g)->prob_evals++;
    return re * re + im * im;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MARGINAL PROBABILITY — P(site_k = v)
 *
 * Uses per-site adjacency lists for O(degree) edge lookup.
 * Only enumerates sites connected by edges to site k.
 * Disconnected sites contribute 1.0 (they're normalized independently).
 *
 * OPTIMIZED: O(degree) edge lookup via adjacency list.
 * Old version: O(E) scan → O(N×E) = O(N²) total.
 * New version: O(degree) lookup → O(N×degree) = O(N) for bounded-degree lattices.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double hpc_marginal(const HPCGraph *g,
                                   uint64_t site, uint32_t value)
{
    const HPCAdjList *adj = &g->adj[site];

    /* Product state: no edges touching this site */
    if (adj->count == 0) {
        TrialityQuhit *q = (TrialityQuhit*)&g->locals[site];
        triality_ensure_view(q, VIEW_EDGE);
        return q->edge_re[value] * q->edge_re[value] +
               q->edge_im[value] * q->edge_im[value];
    }

    /* Find unique connected sites via adjacency list — O(degree) */
    uint64_t connected[128];
    uint64_t conn_edge_ids[512];  /* Edge IDs in connected subsystem */
    uint64_t n_connected = 0;
    uint64_t n_conn_edges = 0;

    for (uint64_t i = 0; i < adj->count; i++) {
        uint64_t eid = adj->edge_ids[i];
        const HPCEdge *edge = &g->edges[eid];
        uint64_t partner = (edge->site_a == site) ? edge->site_b : edge->site_a;

        /* Add edge to subsystem edge list */
        if (n_conn_edges < 512)
            conn_edge_ids[n_conn_edges++] = eid;

        /* Add partner to connected list (dedup) */
        int found = 0;
        for (uint64_t c = 0; c < n_connected; c++)
            if (connected[c] == partner) { found = 1; break; }
        if (!found && n_connected < 128)
            connected[n_connected++] = partner;
    }

    /* ═══ Fast Path for High-Degree Nodes (O(degree) vs O(6^degree)) ═══
     * For n_connected > 6, the exponential enumeration stalls. We switch to
     * a neighbor-factored approximation which is exact for trees and CZ gates. */
    if (n_connected > 6) {
        TrialityQuhit *q_site = (TrialityQuhit*)&g->locals[site];
        triality_ensure_view(q_site, VIEW_EDGE);
        double prob = q_site->edge_re[value] * q_site->edge_re[value] +
                      q_site->edge_im[value] * q_site->edge_im[value];

        for (uint64_t i = 0; i < adj->count; i++) {
            uint64_t eid = adj->edge_ids[i];
            const HPCEdge *edge = &g->edges[eid];

            /* Permutation edges can't be factored — skip in fast path.
             * The resulting marginal is an approximation. */
            if (edge->type == HPC_EDGE_PERMUTE) continue;

            uint64_t partner = (edge->site_a == site) ?
                                edge->site_b : edge->site_a;

            TrialityQuhit *q_p = (TrialityQuhit*)&g->locals[partner];
            triality_ensure_view(q_p, VIEW_EDGE);

            double edge_factor = 0.0;
            for (int k = 0; k < HPC_D; k++) {
                double wr, wi;
                if (edge->type == HPC_EDGE_CZ) {
                    uint32_t m = edge->syntheme_id ? edge->syntheme_id : 1;
                    uint32_t pidx = (m * value * k) % HPC_D;
                    wr = HPC_W6_RE[pidx];
                    wi = HPC_W6_IM[pidx];
                } else if (edge->site_a == site) {
                    wr = edge->w_re[value][k];
                    wi = edge->w_im[value][k];
                } else {
                    wr = edge->w_re[k][value];
                    wi = edge->w_im[k][value];
                }

                double pk = q_p->edge_re[k] * q_p->edge_re[k] +
                            q_p->edge_im[k] * q_p->edge_im[k];
                double w_mag2 = wr * wr + wi * wi;
                edge_factor += pk * w_mag2;
            }
            prob *= edge_factor;
        }
        return prob;
    }

    /* Also find edges between connected partners (not touching site)
     * by scanning adjacency lists of connected sites — O(degree²) */
    for (uint64_t c = 0; c < n_connected; c++) {
        const HPCAdjList *padj = &g->adj[connected[c]];
        for (uint64_t i = 0; i < padj->count; i++) {
            uint64_t eid = padj->edge_ids[i];
            const HPCEdge *edge = &g->edges[eid];
            uint64_t sa = edge->site_a, sb = edge->site_b;
            if (sa == site || sb == site) continue;  /* Already counted */

            /* Check if both ends are in connected set */
            int a_in = 0, b_in = 0;
            for (uint64_t c2 = 0; c2 < n_connected; c2++) {
                if (connected[c2] == sa) a_in = 1;
                if (connected[c2] == sb) b_in = 1;
            }
            if (a_in && b_in) {
                /* Dedup edge */
                int dup = 0;
                for (uint64_t e2 = 0; e2 < n_conn_edges; e2++)
                    if (conn_edge_ids[e2] == eid) { dup = 1; break; }
                if (!dup && n_conn_edges < 512)
                    conn_edge_ids[n_conn_edges++] = eid;
            }
        }
    }

    /* ═══ Component 4: Δ-Gated Fast Path ═══
     * Instead of enumerating all D^n_connected configurations,
     * only enumerate basis states that have nonzero amplitude
     * (tracked by active_mask). For states confined to k of 6
     * basis states, this reduces from 6^n to k^n configs.
     *
     * From the Faustian Pact: Δ≈0 states use fewer basis states,
     * making this optimization most effective when it matters most. */

    /* Build per-partner active state lists */
    uint32_t partner_active[128][6];
    uint32_t partner_active_count[128];
    uint64_t n_configs = 1;

    for (uint64_t c = 0; c < n_connected; c++) {
        TrialityQuhit *q_c = (TrialityQuhit*)&g->locals[connected[c]];
        triality_ensure_view(q_c, VIEW_EDGE);
        uint8_t mask = q_c->active_mask ? q_c->active_mask : 0x3F;
        int cnt = 0;
        for (int k = 0; k < HPC_D; k++)
            if (mask & (1 << k)) partner_active[c][cnt++] = k;
        partner_active_count[c] = cnt;
        n_configs *= cnt;
    }

    /* Check if any permutation edges exist in the connected subgraph */
    int has_perm_in_subgraph = 0;
    for (uint64_t ei = 0; ei < n_conn_edges && !has_perm_in_subgraph; ei++)
        if (g->edges[conn_edge_ids[ei]].type == HPC_EDGE_PERMUTE)
            has_perm_in_subgraph = 1;

    double total_prob = 0.0;

    if (has_perm_in_subgraph) {
        /* ── Permutation-aware enumeration ──
         *
         * When permutation edges exist in the subgraph, the amplitude
         * at each enumerated configuration differs from the standard
         * product formula. We must transform indices through inverse
         * permutations before evaluating local amplitudes and phase
         * edge weights.
         *
         * Algorithm per configuration:
         *   1. Build {site→value} mapping for all subgraph sites
         *   2. Apply inverse permutations in reverse edge order
         *      (newest first) to get "source" indices
         *   3. Evaluate local amplitudes at source indices
         *   4. Evaluate phase edge weights at source indices
         *   5. Skip permutation edges in phase multiplication
         */
        uint64_t all_sites[128];
        int n_all = 1 + (int)n_connected;
        all_sites[0] = site;
        for (uint64_t c = 0; c < n_connected; c++)
            all_sites[1 + c] = connected[c];

        for (uint64_t cfg = 0; cfg < n_configs; cfg++) {
            uint32_t partner_vals[128];
            uint64_t tmp = cfg;
            for (uint64_t c = 0; c < n_connected; c++) {
                uint32_t idx_in_active = tmp % partner_active_count[c];
                partner_vals[c] = partner_active[c][idx_in_active];
                tmp /= partner_active_count[c];
            }

            /* Step 1: build current value mapping */
            uint32_t cur_vals[128];
            cur_vals[0] = value;
            for (uint64_t c = 0; c < n_connected; c++)
                cur_vals[1 + c] = partner_vals[c];

            /* Step 2: apply inverse permutations in reverse edge order */
            for (int64_t ei = (int64_t)n_conn_edges - 1; ei >= 0; ei--) {
                const HPCEdge *edge = &g->edges[conn_edge_ids[ei]];
                if (edge->type != HPC_EDGE_PERMUTE) continue;

                uint64_t a = edge->site_a;
                uint64_t b = edge->site_b;

                /* Find current values of a and b */
                uint32_t ca = 0, cb = 0;
                for (int i = 0; i < n_all; i++) {
                    if (all_sites[i] == a) ca = cur_vals[i];
                    if (all_sites[i] == b) cb = cur_vals[i];
                }

                uint32_t packed = edge->perm_source[ca][cb];
                uint32_t sa_val = (packed >> 3) & 7;
                uint32_t sb_val = packed & 7;

                /* Update values */
                for (int i = 0; i < n_all; i++) {
                    if (all_sites[i] == a) cur_vals[i] = sa_val;
                    if (all_sites[i] == b) cur_vals[i] = sb_val;
                }
            }

            /* Step 3: local amplitudes at source indices */
            TrialityQuhit *q_site = (TrialityQuhit*)&g->locals[site];
            triality_ensure_view(q_site, VIEW_EDGE);
            double amp_re = q_site->edge_re[cur_vals[0]];
            double amp_im = q_site->edge_im[cur_vals[0]];

            for (uint64_t c = 0; c < n_connected; c++) {
                TrialityQuhit *q_p = (TrialityQuhit*)&g->locals[connected[c]];
                triality_ensure_view(q_p, VIEW_EDGE);
                double p_re = q_p->edge_re[cur_vals[1 + c]];
                double p_im = q_p->edge_im[cur_vals[1 + c]];
                double new_re = amp_re * p_re - amp_im * p_im;
                double new_im = amp_re * p_im + amp_im * p_re;
                amp_re = new_re;
                amp_im = new_im;
            }

            /* Step 4: phase edges only (skip PERMUTE) */
            for (uint64_t ei = 0; ei < n_conn_edges; ei++) {
                const HPCEdge *edge = &g->edges[conn_edge_ids[ei]];
                if (edge->type == HPC_EDGE_PERMUTE) continue;

                uint64_t sa = edge->site_a, sb = edge->site_b;
                uint32_t va = 0, vb = 0;

                /* Resolve source values for both endpoints */
                for (int i = 0; i < n_all; i++) {
                    if (all_sites[i] == sa) va = cur_vals[i];
                    if (all_sites[i] == sb) vb = cur_vals[i];
                }

                double wr, wi;
                if (edge->type == HPC_EDGE_CZ) {
                    uint32_t m = edge->syntheme_id ? edge->syntheme_id : 1;
                    uint32_t phase_idx = (m * va * vb) % HPC_D;
                    wr = HPC_W6_RE[phase_idx];
                    wi = HPC_W6_IM[phase_idx];
                } else {
                    wr = edge->w_re[va][vb];
                    wi = edge->w_im[va][vb];
                }

                double new_re = amp_re * wr - amp_im * wi;
                double new_im = amp_re * wi + amp_im * wr;
                amp_re = new_re;
                amp_im = new_im;
            }

            total_prob += amp_re * amp_re + amp_im * amp_im;
        }
    } else {
        /* ── Original optimized enumeration (no permutation edges) ── */
        for (uint64_t cfg = 0; cfg < n_configs; cfg++) {
            uint32_t partner_vals[128];
            uint64_t tmp = cfg;
            for (uint64_t c = 0; c < n_connected; c++) {
                uint32_t idx_in_active = tmp % partner_active_count[c];
                partner_vals[c] = partner_active[c][idx_in_active];
                tmp /= partner_active_count[c];
            }

            TrialityQuhit *q_site = (TrialityQuhit*)&g->locals[site];
            triality_ensure_view(q_site, VIEW_EDGE);
            double amp_re = q_site->edge_re[value];
            double amp_im = q_site->edge_im[value];

            for (uint64_t c = 0; c < n_connected; c++) {
                TrialityQuhit *q_p = (TrialityQuhit*)&g->locals[connected[c]];
                triality_ensure_view(q_p, VIEW_EDGE);
                uint32_t pv = partner_vals[c];
                double p_re = q_p->edge_re[pv], p_im = q_p->edge_im[pv];
                double new_re = amp_re * p_re - amp_im * p_im;
                double new_im = amp_re * p_im + amp_im * p_re;
                amp_re = new_re;
                amp_im = new_im;
            }

            for (uint64_t ei = 0; ei < n_conn_edges; ei++) {
                const HPCEdge *edge = &g->edges[conn_edge_ids[ei]];
                uint64_t sa = edge->site_a, sb = edge->site_b;

                uint32_t va = 0, vb = 0;
                if (sa == site) {
                    va = value;
                    for (uint64_t c = 0; c < n_connected; c++)
                        if (connected[c] == sb) { vb = partner_vals[c]; break; }
                } else if (sb == site) {
                    vb = value;
                    for (uint64_t c = 0; c < n_connected; c++)
                        if (connected[c] == sa) { va = partner_vals[c]; break; }
                } else {
                    for (uint64_t c = 0; c < n_connected; c++) {
                        if (connected[c] == sa) va = partner_vals[c];
                        if (connected[c] == sb) vb = partner_vals[c];
                    }
                }

                double wr, wi;
                if (edge->type == HPC_EDGE_CZ) {
                    uint32_t m = edge->syntheme_id ? edge->syntheme_id : 1;
                    uint32_t phase_idx = (m * va * vb) % HPC_D;
                    wr = HPC_W6_RE[phase_idx];
                    wi = HPC_W6_IM[phase_idx];
                } else {
                    wr = edge->w_re[va][vb];
                    wi = edge->w_im[va][vb];
                }

                double new_re = amp_re * wr - amp_im * wi;
                double new_im = amp_re * wi + amp_im * wr;
                amp_re = new_re;
                amp_im = new_im;
            }

            total_prob += amp_re * amp_re + amp_im * amp_im;
        }
    }

    return total_prob;
}

/* ═══════════════════════════════════════════════════════════════════════
 * EDGE COMPACTION — Merge parallel CZ edges
 *
 * Multiple CZ edges between the same pair of sites can be merged:
 *   CZ × CZ = CZ with phase ω^(2·a·b) → equivalent to CZ^2
 *   n CZ edges → one edge with accumulated phase ω^(n·a·b)
 *
 * For n ≡ 0 mod 6: the edge cancels (ω^6 = 1) → remove entirely.
 * For n ≡ 1 mod 6: standard CZ.
 * For n ≡ 3 mod 6: anti-CZ (ω³ = -1).
 *
 * This preserves perfect phase coherence at any lattice scale.
 * Without compaction, d-wave pairing bleeds out as parallel edges
 * fragment the phase structure.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_compact_edges(HPCGraph *g)
{
    /* Count CZ edges between each pair, merge into accumulated phase.
     * For bounded-degree lattices, this is O(E × degree) ≈ O(E). */

    for (uint64_t e = 0; e < g->n_edges; ) {
        HPCEdge *edge = &g->edges[e];
        if (edge->type != HPC_EDGE_CZ) { e++; continue; }

        uint64_t sa = edge->site_a, sb = edge->site_b;

        /* Count and remove duplicate CZ edges for this pair */
        int cz_count = edge->syntheme_id ? edge->syntheme_id : 1;  /* This edge's multiplicity */
        for (uint64_t e2 = e + 1; e2 < g->n_edges; ) {
            HPCEdge *other = &g->edges[e2];
            if (other->type == HPC_EDGE_CZ &&
                ((other->site_a == sa && other->site_b == sb) ||
                 (other->site_a == sb && other->site_b == sa))) {
                cz_count += (other->syntheme_id ? other->syntheme_id : 1);

                /* Remove adjacency entries for the duplicate */
                hpc_adj_remove(g, other->site_a, e2);
                hpc_adj_remove(g, other->site_b, e2);

                /* Swap-remove the duplicate edge */
                uint64_t last = g->n_edges - 1;
                if (e2 != last) {
                    /* Update adjacency for the edge being swapped in */
                    hpc_adj_replace(g, g->edges[last].site_a, last, e2);
                    hpc_adj_replace(g, g->edges[last].site_b, last, e2);
                    g->edges[e2] = g->edges[last];
                }
                g->n_edges--;
                g->cz_edges--;
            } else {
                e2++;
            }
        }

        /* Reduce cz_count mod 6 */
        int reduced = cz_count % 6;

        if (reduced == 0) {
            /* Complete cancellation: ω^(6k) = 1 → remove edge entirely */
            hpc_adj_remove(g, sa, e);
            hpc_adj_remove(g, sb, e);

            uint64_t last = g->n_edges - 1;
            if (e != last) {
                hpc_adj_replace(g, g->edges[last].site_a, last, e);
                hpc_adj_replace(g, g->edges[last].site_b, last, e);
                g->edges[e] = g->edges[last];
            }
            g->n_edges--;
            g->cz_edges--;
        } else {
            /* Keep as exact CZ edge, just update multiplicity */
            edge->syntheme_id = reduced;
            e++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * BORN SAMPLING — Collapse site k
 *
 * Uses adjacency lists for O(degree) edge identification.
 * Absorbs CZ phases into partners, removes resolved edges.
 * This IS measurement-induced disentanglement.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline uint32_t hpc_measure(HPCGraph *g, uint64_t site,
                                    double random_01)
{
    /* Compute marginals */
    double probs[HPC_D];
    double total = 0.0;
    for (int v = 0; v < HPC_D; v++) {
        probs[v] = hpc_marginal(g, site, v);
        total += probs[v];
    }
    if (total > 0) {
        for (int v = 0; v < HPC_D; v++) probs[v] /= total;
    }

    /* Sample */
    double cumul = 0.0;
    uint32_t outcome = HPC_D - 1;
    for (int v = 0; v < HPC_D; v++) {
        cumul += probs[v];
        if (random_01 <= cumul) { outcome = v; break; }
    }

    /* Collapse local state to |outcome⟩ */
    for (int v = 0; v < HPC_D; v++) {
        g->locals[site].edge_re[v] = (v == (int)outcome) ? 1.0 : 0.0;
        g->locals[site].edge_im[v] = 0.0;
    }
    g->locals[site].primary = VIEW_EDGE;
    g->locals[site].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    g->locals[site].delta_valid = 0;
    triality_update_mask(&g->locals[site]);

    /* Collect edge IDs touching this site from adjacency list — O(degree) */
    uint64_t edges_to_remove[512];
    uint64_t n_remove = 0;
    const HPCAdjList *adj = &g->adj[site];
    for (uint64_t i = 0; i < adj->count && n_remove < 512; i++)
        edges_to_remove[n_remove++] = adj->edge_ids[i];

    /* Absorb phases and remove edges */
    for (uint64_t r = 0; r < n_remove; r++) {
        uint64_t eid = edges_to_remove[r];
        if (eid >= g->n_edges) continue;  /* Already removed by swap */

        HPCEdge *edge = &g->edges[eid];
        /* Verify this edge still touches our site (may have been swapped) */
        if (edge->site_a != site && edge->site_b != site) continue;

        uint64_t partner = (edge->site_a == site) ?
                            edge->site_b : edge->site_a;
        TrialityQuhit *p = &g->locals[partner];

        if (edge->type == HPC_EDGE_PERMUTE) {
            /* Permutation edges cannot be absorbed as a phase factor.
             * Removing the edge from adjacency is handled below, but the
             * partner's state stays as-is. A full resolution would require
             * applying the inverse permutation to project the partner
             * onto the subspace consistent with the measured outcome. */
            /* Track removal below */
        } else {
            /* Absorb the phase: partner[k] *= w(outcome,k) or w(k,outcome) */
            for (int k = 0; k < HPC_D; k++) {
                double wr, wi;
                if (edge->type == HPC_EDGE_CZ) {
                    uint32_t m = edge->syntheme_id ? edge->syntheme_id : 1;
                    uint32_t phase_idx = (m * outcome * k) % HPC_D;
                    wr = HPC_W6_RE[phase_idx];
                    wi = HPC_W6_IM[phase_idx];
                } else if (edge->site_a == site) {
                    wr = edge->w_re[outcome][k];
                    wi = edge->w_im[outcome][k];
                } else {
                    wr = edge->w_re[k][outcome];
                    wi = edge->w_im[k][outcome];
                }

                double old_re = p->edge_re[k], old_im = p->edge_im[k];
                p->edge_re[k] = old_re * wr - old_im * wi;
                p->edge_im[k] = old_re * wi + old_im * wr;
            }
            p->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
            p->delta_valid = 0;
        }

        /* Track edge type removal */
        if (edge->type == HPC_EDGE_CZ) g->cz_edges--;
        else if (edge->type == HPC_EDGE_PHASE) g->phase_edges--;
        else if (edge->type == HPC_EDGE_SYNTHEME) g->syntheme_edges--;
        else if (edge->type == HPC_EDGE_PERMUTE) g->perm_edges--;

        /* Remove from adjacency lists */
        hpc_adj_remove(g, site, eid);
        hpc_adj_remove(g, partner, eid);

        /* Swap-remove the edge */
        uint64_t last = g->n_edges - 1;
        if (eid != last) {
            /* Update adjacency for the swapped-in edge */
            hpc_adj_replace(g, g->edges[last].site_a, last, eid);
            hpc_adj_replace(g, g->edges[last].site_b, last, eid);
            g->edges[eid] = g->edges[last];

            /* Update remaining removal targets that pointed to 'last' */
            for (uint64_t r2 = r + 1; r2 < n_remove; r2++)
                if (edges_to_remove[r2] == last)
                    edges_to_remove[r2] = eid;
        }
        g->n_edges--;
    }

    g->measurements++;
    hpc_update_fidelity_stats(g);
    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════
 * NORMALIZATION CHECK — Σ |ψ|² over ALL indices
 *
 * Cost: O(D^N × (N+E)) — small N only!
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double hpc_norm_sq(const HPCGraph *g)
{
    if (g->n_sites > 8) {
        fprintf(stderr, "hpc_norm_sq: N=%lu too large for brute force\n",
                g->n_sites);
        return -1.0;
    }

    uint64_t total_configs = 1;
    for (uint64_t i = 0; i < g->n_sites; i++) total_configs *= HPC_D;

    double norm = 0.0;
    uint32_t indices[8];

    for (uint64_t cfg = 0; cfg < total_configs; cfg++) {
        uint64_t tmp = cfg;
        for (uint64_t i = 0; i < g->n_sites; i++) {
            indices[i] = tmp % HPC_D;
            tmp /= HPC_D;
        }
        norm += hpc_probability(g, indices);
    }
    return norm;
}

/* ═══════════════════════════════════════════════════════════════════════
 * EXOTIC INVARIANT — weighted Δ across all sites
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double hpc_exotic_invariant(HPCGraph *g)
{
    double total = 0.0;
    for (uint64_t i = 0; i < g->n_sites; i++) {
        triality_ensure_view(&g->locals[i], VIEW_EDGE);
        total += triality_exotic_invariant_cached(&g->locals[i]);
    }
    return total / g->n_sites;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENTROPY ESTIMATE — across a bipartition cut
 *
 * CZ edges contribute exactly log₂(D) bits per crossing edge.
 * General edges contribute fidelity-weighted log₂(D) bits.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double hpc_entropy_cut(const HPCGraph *g, uint64_t cut_after)
{
    double entropy = 0.0;
    for (uint64_t e = 0; e < g->n_edges; e++) {
        uint64_t sa = g->edges[e].site_a;
        uint64_t sb = g->edges[e].site_b;
        if ((sa <= cut_after && sb > cut_after) ||
            (sb <= cut_after && sa > cut_after)) {
            entropy += g->edges[e].fidelity * log2((double)HPC_D);
        }
    }
    return entropy;
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_print_stats(const HPCGraph *g)
{
    printf("╔═════════════════════════════════════════════════════╗\n");
    printf("║  Holographic Phase Graph Statistics                ║\n");
    printf("╠═════════════════════════════════════════════════════╣\n");
    printf("║  Sites:           %10lu                       ║\n", g->n_sites);
    printf("║  Total edges:     %10lu                       ║\n", g->n_edges);
    printf("║    CZ (exact):    %10lu                       ║\n", g->cz_edges);
    printf("║    Phase (lossy): %10lu                       ║\n", g->phase_edges);
    printf("║    Syntheme:      %10lu                       ║\n", g->syntheme_edges);
    printf("║    Permute:       %10lu                       ║\n", g->perm_edges);
    printf("║  Gate log:        %10lu                       ║\n", g->n_log);
    printf("║  Amp evals:       %10lu                       ║\n", g->amp_evals);
    printf("║  Measurements:    %10lu                       ║\n", g->measurements);
    printf("║  Min fidelity:    %10.6f                       ║\n", g->min_fidelity);
    printf("║  Avg fidelity:    %10.6f                       ║\n", g->avg_fidelity);

    uint64_t mem_bytes = g->n_sites * sizeof(TrialityQuhit) +
                         g->n_edges * sizeof(HPCEdge) +
                         g->n_log * sizeof(HPCGateEntry) +
                         sizeof(HPCGraph);
    printf("║  Memory:          %10lu bytes                ║\n", mem_bytes);

    double full_sv_log = g->n_sites * log10(6.0) + log10(16.0);
    printf("║  Full SV:         10^%.1f bytes (impossible)    ║\n", full_sv_log);
    printf("╚═════════════════════════════════════════════════════╝\n");
}

static inline void hpc_print_state(const HPCGraph *g, const char *label)
{
    printf("── %s ──\n", label);
    printf("  Sites: %lu, Edges: %lu (CZ:%lu Phase:%lu Synth:%lu Perm:%lu)\n",
           g->n_sites, g->n_edges, g->cz_edges, g->phase_edges,
           g->syntheme_edges, g->perm_edges);
    printf("  Fidelity: min=%.4f avg=%.4f\n", g->min_fidelity, g->avg_fidelity);
    for (uint64_t i = 0; i < g->n_sites && i < 8; i++) {
        printf("  Site %lu: [", i);
        for (int j = 0; j < HPC_D; j++) {
            printf("%.3f%+.3fi", g->locals[i].edge_re[j],
                                  g->locals[i].edge_im[j]);
            if (j < HPC_D - 1) printf(", ");
        }
        printf("]\n");
    }
    if (g->n_sites > 8) printf("  ... (%lu more sites)\n", g->n_sites - 8);
}

#endif /* HPC_GRAPH_H */
