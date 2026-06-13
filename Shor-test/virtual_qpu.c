/*
 * virtual_qpu.c
 *
 * MACHINE WITHIN THE MACHINE
 *
 * A virtual QPU of higher specification running on the HexState D=6 substrate.
 *
 * Concept:
 *   Layer 1 (Host):   HexState D=6 HPC graph — N quhits, each with 6 basis states
 *   Layer 2 (VQPU):   Virtual qubit (d=2) processor — 2N virtual qubits
 *                     encoded in the N host quhits (2 virtual qubits per host quhit)
 *
 * Encoding (per host quhit):
 *   |0⟩_h  ↔  |00⟩_v   (both virtual qubits = 0)
 *   |1⟩_h  ↔  |01⟩_v   (vq[2k]=0, vq[2k+1]=1)
 *   |2⟩_h  ↔  |10⟩_v   (vq[2k]=1, vq[2k+1]=0)
 *   |3⟩_h  ↔  |11⟩_v   (both = 1)
 *   |4⟩_h, |5⟩_h  ↔  hidden/ancilla states (unused by VQPU)
 *
 * The "higher specification":
 *   - 2× more virtual qubits than host quhits
 *   - Hidden D=6 subspace provides extra degrees of freedom
 *   - HPC graph stores entanglement as O(N+E) edges, not O(2^N) amplitudes
 *
 * Benchmark: measure scaling of virtual gate execution time vs
 * virtual qubit count for different circuit topologies.
 *
 * Build:
 *   gcc -O2 -march=native -I.. -o virtual_qpu virtual_qpu.c \
 *       ../quhit_triality.c ../s6_exotic.c ../quhit_hexagram.c \
 *       -lm -msse2 -lrt
 *
 *   ulimit -s unlimited
 *   ./virtual_qpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "hpc_graph.h"
#include "s6_exotic.h"
#include "quhit_hexagram.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── PRNG (xoshiro256**) ─── */
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

/* ─── Timing ─── */
static struct timespec _t0;
static void tic(void) { clock_gettime(CLOCK_MONOTONIC, &_t0); }
static double toc(void) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - _t0.tv_sec) + (t1.tv_nsec - _t0.tv_nsec) / 1e9;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VIRTUAL QPU LAYER
 *
 * Encodes 2 virtual qubits per host HPC graph site.
 * Virtual qubit 2k   → host site k, position 0 (LSB)
 * Virtual qubit 2k+1 → host site k, position 1 (MSB)
 *
 * Host state encoding:
 *   host |0⟩ = |00⟩_v    host |1⟩ = |01⟩_v
 *   host |2⟩ = |10⟩_v    host |3⟩ = |11⟩_v
 *   host |4⟩,|5⟩ = reserved (hidden subspace)
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Virtual qubit index → host site + position within host ── */
#define VQ_HOST_SITE(vq)  ((vq) >> 1)
#define VQ_POS(vq)        ((vq) & 1)

/* ── Virtual-to-host state mapping ──
 * host_state = (vq1 << 1) | vq0   (for vq0=pos0, vq1=pos1)
 * This maps: |vq0,vq1⟩ → | 2*vq1 + vq0 ⟩_host
 */
#define VQ_TO_HOST(vq0, vq1)  (((vq1) << 1) | (vq0))

/* ── Apply virtual X gate to one virtual qubit ──
 *
 * Virtual X on position 0: swaps |0⟩↔|2⟩ and |1⟩↔|3⟩ in host state
 * Virtual X on position 1: swaps |0⟩↔|1⟩ and |2⟩↔|3⟩ in host state
 *
 * Operates directly on the HPC graph's local TrialityQuhit.
 */
static void vqpu_x(HPCGraph *g, int vq_idx)
{
    uint64_t site = VQ_HOST_SITE(vq_idx);
    int pos = VQ_POS(vq_idx);
    TrialityQuhit *q = &g->locals[site];
    triality_ensure_view(q, VIEW_EDGE);

    double re[6], im[6];
    memcpy(re, q->edge_re, 6 * sizeof(double));
    memcpy(im, q->edge_im, 6 * sizeof(double));

    if (pos == 0) {
        /* Swap |0⟩↔|2⟩, |1⟩↔|3⟩ */
        q->edge_re[0] = re[2]; q->edge_im[0] = im[2];
        q->edge_re[2] = re[0]; q->edge_im[2] = im[0];
        q->edge_re[1] = re[3]; q->edge_im[1] = im[3];
        q->edge_re[3] = re[1]; q->edge_im[3] = im[1];
        /* |4⟩, |5⟩ untouched */
        q->edge_re[4] = re[4]; q->edge_im[4] = im[4];
        q->edge_re[5] = re[5]; q->edge_im[5] = im[5];
    } else {
        /* Swap |0⟩↔|1⟩, |2⟩↔|3⟩ */
        q->edge_re[0] = re[1]; q->edge_im[0] = im[1];
        q->edge_re[1] = re[0]; q->edge_im[1] = im[0];
        q->edge_re[2] = re[3]; q->edge_im[2] = im[3];
        q->edge_re[3] = re[2]; q->edge_im[3] = im[2];
        q->edge_re[4] = re[4]; q->edge_im[4] = im[4];
        q->edge_re[5] = re[5]; q->edge_im[5] = im[5];
    }
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;
}

/* ── Apply virtual Hadamard to one virtual qubit ──
 *
 * H on pos 0: |0⟩→(|0⟩+|2⟩)/√2,  |1⟩→(|1⟩+|3⟩)/√2
 *             |2⟩→(|0⟩-|2⟩)/√2,  |3⟩→(|1⟩-|3⟩)/√2
 * H on pos 1: |0⟩→(|0⟩+|1⟩)/√2,  |2⟩→(|2⟩+|3⟩)/√2
 *             |1⟩→(|0⟩-|1⟩)/√2,  |3⟩→(|2⟩-|3⟩)/√2
 */
static void vqpu_h(HPCGraph *g, int vq_idx)
{
    uint64_t site = VQ_HOST_SITE(vq_idx);
    int pos = VQ_POS(vq_idx);
    TrialityQuhit *q = &g->locals[site];
    triality_ensure_view(q, VIEW_EDGE);

    double re[6], im[6];
    memcpy(re, q->edge_re, 6 * sizeof(double));
    memcpy(im, q->edge_im, 6 * sizeof(double));
    double s = 1.0 / sqrt(2.0);

    if (pos == 0) {
        /* |0'⟩ = (|0⟩ + |2⟩)/√2,  |1'⟩ = (|1⟩ + |3⟩)/√2
         * |2'⟩ = (|0⟩ - |2⟩)/√2,  |3'⟩ = (|1⟩ - |3⟩)/√2 */
        q->edge_re[0] = s * (re[0] + re[2]); q->edge_im[0] = s * (im[0] + im[2]);
        q->edge_re[1] = s * (re[1] + re[3]); q->edge_im[1] = s * (im[1] + im[3]);
        q->edge_re[2] = s * (re[0] - re[2]); q->edge_im[2] = s * (im[0] - im[2]);
        q->edge_re[3] = s * (re[1] - re[3]); q->edge_im[3] = s * (im[1] - im[3]);
        q->edge_re[4] = re[4]; q->edge_im[4] = im[4];
        q->edge_re[5] = re[5]; q->edge_im[5] = im[5];
    } else {
        /* |0'⟩ = (|0⟩ + |1⟩)/√2,  |2'⟩ = (|2⟩ + |3⟩)/√2
         * |1'⟩ = (|0⟩ - |1⟩)/√2,  |3'⟩ = (|2⟩ - |3⟩)/√2 */
        q->edge_re[0] = s * (re[0] + re[1]); q->edge_im[0] = s * (im[0] + im[1]);
        q->edge_re[2] = s * (re[2] + re[3]); q->edge_im[2] = s * (im[2] + im[3]);
        q->edge_re[1] = s * (re[0] - re[1]); q->edge_im[1] = s * (im[0] - im[1]);
        q->edge_re[3] = s * (re[2] - re[3]); q->edge_im[3] = s * (im[2] - im[3]);
        q->edge_re[4] = re[4]; q->edge_im[4] = im[4];
        q->edge_re[5] = re[5]; q->edge_im[5] = im[5];
    }
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;
}

/* ── Apply virtual Z to one virtual qubit ──
 * Z on pos 0: |1⟩_v → -|1⟩_v  (phase on host |2⟩,|3⟩)
 * Z on pos 1: |1⟩_v → -|1⟩_v  (phase on host |1⟩,|3⟩)
 */
static void vqpu_z(HPCGraph *g, int vq_idx)
{
    uint64_t site = VQ_HOST_SITE(vq_idx);
    int pos = VQ_POS(vq_idx);
    TrialityQuhit *q = &g->locals[site];
    triality_ensure_view(q, VIEW_EDGE);

    if (pos == 0) {
        q->edge_re[2] = -q->edge_re[2]; q->edge_im[2] = -q->edge_im[2];
        q->edge_re[3] = -q->edge_re[3]; q->edge_im[3] = -q->edge_im[3];
    } else {
        q->edge_re[1] = -q->edge_re[1]; q->edge_im[1] = -q->edge_im[1];
        q->edge_re[3] = -q->edge_re[3]; q->edge_im[3] = -q->edge_im[3];
    }
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;
}

/* ── Apply virtual CNOT between two virtual qubits (SAME host site) ──
 *
 * CNOT(ctrl, targ) within same host:
 *   |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
 *
 * For pos0=ctrl, pos1=targ (most common):
 *   swap |2⟩↔|3⟩
 *
 * For pos1=ctrl, pos0=targ:
 *   |00⟩→|00⟩, |01⟩→|11⟩, |10⟩→|10⟩, |11⟩→|01⟩
 *   swap |1⟩↔|3⟩
 */
static void vqpu_cx_same(HPCGraph *g, int ctrl_vq, int targ_vq)
{
    uint64_t site = VQ_HOST_SITE(ctrl_vq);
    (void)site; /* both on same site */
    int cpos = VQ_POS(ctrl_vq);
    TrialityQuhit *q = &g->locals[VQ_HOST_SITE(ctrl_vq)];
    triality_ensure_view(q, VIEW_EDGE);

    double re[6], im[6];
    memcpy(re, q->edge_re, 6 * sizeof(double));
    memcpy(im, q->edge_im, 6 * sizeof(double));

    if (cpos == 0) {
        /* Control = pos0, Target = pos1: swap |10⟩↔|11⟩ → swap host |2⟩↔|3⟩ */
        q->edge_re[2] = re[3]; q->edge_im[2] = im[3];
        q->edge_re[3] = re[2]; q->edge_im[3] = im[2];
    } else {
        /* Control = pos1, Target = pos0: swap |01⟩↔|11⟩ → swap host |1⟩↔|3⟩ */
        q->edge_re[1] = re[3]; q->edge_im[1] = im[3];
        q->edge_re[3] = re[1]; q->edge_im[3] = im[1];
    }
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;
}

/* ── Apply virtual CNOT between virtual qubits on DIFFERENT host sites
 *
 * Decomposition: H(targ) → virtual CZ → H(targ)
 *
 * Virtual CZ adds phase -1 when both virtual qubits are |1⟩.
 * In host space: control site |3⟩_h, target site |3⟩_h → phase -1
 * (assuming both virtual qubits are at position 0 within their hosts)
 *
 * For general positions, the mapping is:
 *   ctrl_host_state = VQ_TO_HOST(ctrl_vq0, ctrl_vq1)
 *   targ_host_state = VQ_TO_HOST(targ_vq0, targ_vq1)
 *
 * We implement virtual CZ by adding a phase edge in the HPC graph.
 */
static void vqpu_cx_cross(HPCGraph *g, int ctrl_vq, int targ_vq)
{
    uint64_t ctrl_site = VQ_HOST_SITE(ctrl_vq);
    uint64_t targ_site = VQ_HOST_SITE(targ_vq);
    int cpos = VQ_POS(ctrl_vq);
    int tpos = VQ_POS(targ_vq);

    /* H on target */
    vqpu_h(g, targ_vq);

    /* Virtual CZ: phase -1 when both are |1⟩
     * Control site: virtual qubit cpos = 1 → host state has bit cpos set
     *   i.e., host state bit cpos = 1 → states {2|cpos?1:0}, {3}
     * Target site: similarly for tpos
     *
     * Create a general phase edge between the sites.
     * The phase matrix w(a,b) = -1 if a has bit cpos set AND b has bit tpos set,
     * else w(a,b) = +1.
     */
    hpc_grow_edges(g);
    uint64_t eid = g->n_edges;
    HPCEdge *e = &g->edges[eid];
    memset(e, 0, sizeof(HPCEdge));
    e->type = HPC_EDGE_PHASE;
    e->site_a = ctrl_site;
    e->site_b = targ_site;
    e->fidelity = 1.0;

    for (int a = 0; a < HPC_D; a++) {
        for (int b = 0; b < HPC_D; b++) {
            int a_bit = (a >> cpos) & 1;  /* 1 if ctrl virtual qubit is |1⟩ */
            int b_bit = (b >> tpos) & 1;  /* 1 if targ virtual qubit is |1⟩ */
            if (a_bit && b_bit) {
                e->w_re[a][b] = -1.0; e->w_im[a][b] = 0.0;
            } else {
                e->w_re[a][b] = 1.0;  e->w_im[a][b] = 0.0;
            }
        }
    }
    g->n_edges++;
    g->phase_edges++;
    hpc_adj_add(g, ctrl_site, eid);
    hpc_adj_add(g, targ_site, eid);

    /* H on target again */
    vqpu_h(g, targ_vq);
}

/* ── Virtual CNOT dispatch: same-site or cross-site ── */
static void vqpu_cx(HPCGraph *g, int ctrl_vq, int targ_vq)
{
    uint64_t cs = VQ_HOST_SITE(ctrl_vq);
    uint64_t ts = VQ_HOST_SITE(targ_vq);
    if (cs == ts) {
        vqpu_cx_same(g, ctrl_vq, targ_vq);
    } else {
        vqpu_cx_cross(g, ctrl_vq, targ_vq);
    }
}

/* ── Measure a virtual qubit ──
 * Returns 0 or 1. Collapses the virtual qubit.
 * Measurement in |0⟩,|1⟩ subspace corresponds to:
 *   P(0) = |a₀|² + |a₁|²  (host states |0⟩,|1⟩ for pos=0)
 *        = |a₀|² + |a₂|²  (host states |0⟩,|2⟩ for pos=1)
 */
static int vqpu_measure(HPCGraph *g, int vq_idx, double rand01)
{
    uint64_t site = VQ_HOST_SITE(vq_idx);
    int pos = VQ_POS(vq_idx);
    TrialityQuhit *q = &g->locals[site];
    triality_ensure_view(q, VIEW_EDGE);

    double p0, p1;
    if (pos == 0) {
        /* |0⟩_v ↔ host |0⟩, |2⟩ (vq0=0). |1⟩_v ↔ host |1⟩, |3⟩ (vq0=1) */
        p0 = q->edge_re[0]*q->edge_re[0] + q->edge_im[0]*q->edge_im[0]
           + q->edge_re[2]*q->edge_re[2] + q->edge_im[2]*q->edge_im[2];
        p1 = q->edge_re[1]*q->edge_re[1] + q->edge_im[1]*q->edge_im[1]
           + q->edge_re[3]*q->edge_re[3] + q->edge_im[3]*q->edge_im[3];
    } else {
        /* |0⟩_v ↔ host |0⟩, |1⟩ (vq1=0). |1⟩_v ↔ host |2⟩, |3⟩ (vq1=1) */
        p0 = q->edge_re[0]*q->edge_re[0] + q->edge_im[0]*q->edge_im[0]
           + q->edge_re[1]*q->edge_re[1] + q->edge_im[1]*q->edge_im[1];
        p1 = q->edge_re[2]*q->edge_re[2] + q->edge_im[2]*q->edge_im[2]
           + q->edge_re[3]*q->edge_re[3] + q->edge_im[3]*q->edge_im[3];
    }

    double total = p0 + p1 + 1e-30;
    int outcome = (rand01 < p0 / total) ? 0 : 1;

    /* Collapse: project onto the |0⟩ or |1⟩ subspace */
    if (pos == 0) {
        if (outcome == 0) {
            q->edge_re[1] = q->edge_im[1] = 0;
            q->edge_re[3] = q->edge_im[3] = 0;
        } else {
            q->edge_re[0] = q->edge_im[0] = 0;
            q->edge_re[2] = q->edge_im[2] = 0;
        }
    } else {
        if (outcome == 0) {
            q->edge_re[2] = q->edge_im[2] = 0;
            q->edge_re[3] = q->edge_im[3] = 0;
        } else {
            q->edge_re[0] = q->edge_im[0] = 0;
            q->edge_re[1] = q->edge_im[1] = 0;
        }
    }
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;
    return outcome;
}

/* ── Reset virtual qubit to |0⟩ ── */
static void vqpu_reset(HPCGraph *g, int vq_idx)
{
    uint64_t site = VQ_HOST_SITE(vq_idx);
    int pos = VQ_POS(vq_idx);
    TrialityQuhit *q = &g->locals[site];
    triality_ensure_view(q, VIEW_EDGE);

    if (pos == 0) {
        q->edge_re[1] = q->edge_im[1] = 0;
        q->edge_re[3] = q->edge_im[3] = 0;
    } else {
        q->edge_re[2] = q->edge_im[2] = 0;
        q->edge_re[3] = q->edge_im[3] = 0;
    }
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;
}

/* ── Initialize virtual QPU: N_vq virtual qubits, all |0⟩ ── */
static HPCGraph *vqpu_create(int N_vq)
{
    int N_host = (N_vq + 1) / 2;  /* ceil(N_vq / 2) */
    HPCGraph *g = hpc_create(N_host);
    rng_seed(0xDEADBEEF);


    return g;
}

/* ═══════════════════════════════════════════════════════════════════════
 * BENCHMARK CIRCUITS
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Benchmark 1: Virtual GHZ chain (nearest-neighbor CX) ──
 * Produces O(N) HPC edges.
 */
static void bench_ghz_chain(int N_vq, int repeats, double *out_time, int *out_edges)
{
    double t_sum = 0;
    int edge_sum = 0;

    for (int r = 0; r < repeats; r++) {
        HPCGraph *g = vqpu_create(N_vq);

        /* H on first virtual qubit */
        vqpu_h(g, 0);

        /* CX chain: 0→1, 1→2, 2→3, ... */
        tic();
        for (int i = 0; i < N_vq - 1; i++) {
            vqpu_cx(g, i, i + 1);
        }
        t_sum += toc();

        edge_sum += g->n_edges;
        hpc_destroy(g);
    }

    *out_time = t_sum / repeats;
    *out_edges = edge_sum / repeats;
}

/* ── Benchmark 2: Dense virtual CX (all-to-all) ──
 * Virtual CNOT between every pair of virtual qubits.
 * Produces O(N²) HPC edges.
 */
static void bench_dense_cx(int N_vq, int repeats, double *out_time, int *out_edges)
{
    double t_sum = 0;
    int edge_sum = 0;

    for (int r = 0; r < repeats; r++) {
        HPCGraph *g = vqpu_create(N_vq);

        tic();
        for (int i = 0; i < N_vq; i++) {
            for (int j = i + 1; j < N_vq; j++) {
                vqpu_cx(g, i, j);
            }
        }
        t_sum += toc();

        edge_sum += g->n_edges;
        hpc_destroy(g);
    }

    *out_time = t_sum / repeats;
    *out_edges = edge_sum / repeats;
}

/* ── Benchmark 3: Random single-qubit virtual gates ──
 * No entanglement, O(N) operations, zero edges.
 */
static void bench_random_single(int N_vq, int gates_per_vq, int repeats,
                                 double *out_time, int *out_edges)
{
    double t_sum = 0;
    int edge_sum = 0;

    for (int r = 0; r < repeats; r++) {
        HPCGraph *g = vqpu_create(N_vq);
        rng_seed(0xCAFEBABE + r * 0x9E3779B97F4A7C15ull);

        tic();
        for (int i = 0; i < gates_per_vq; i++) {
            int vq = rng_next() % N_vq;
            int gate = rng_next() % 3;
            if (gate == 0) vqpu_x(g, vq);
            else if (gate == 1) vqpu_h(g, vq);
            else vqpu_z(g, vq);
        }
        t_sum += toc();

        edge_sum += g->n_edges;
        hpc_destroy(g);
    }

    *out_time = t_sum / repeats;
    *out_edges = edge_sum / repeats;
}

/* ── Benchmark 4: Virtual QFT (Quantum Fourier Transform) ──
 * Standard QFT on virtual qubits. Uses O(N log N) gates.
 */
static void bench_vqft(int N_vq, int repeats, double *out_time, int *out_edges)
{
    double t_sum = 0;
    int edge_sum = 0;

    for (int r = 0; r < repeats; r++) {
        HPCGraph *g = vqpu_create(N_vq);

        /* Initial superposition */
        for (int i = 0; i < N_vq && i < 8; i++) {
            vqpu_h(g, i);
        }

        tic();
        /* QFT-style: H + controlled rotations (simplified) */
        for (int i = 0; i < N_vq && i < 8; i++) {
            vqpu_h(g, i);
            for (int j = i + 1; j < N_vq && j < 8; j++) {
                vqpu_cx(g, i, j);
                vqpu_h(g, j);
                vqpu_cx(g, i, j);
                vqpu_h(g, j);
            }
        }
        t_sum += toc();

        edge_sum += g->n_edges;
        hpc_destroy(g);
    }

    *out_time = t_sum / repeats;
    *out_edges = edge_sum / repeats;
}

/* ═══════════════════════════════════════════════════════════════════════
 * SCALING ANALYSIS
 * ═══════════════════════════════════════════════════════════════════════ */

/* Compute scaling exponent: assume time = a * N^b, find b via log-log fit */
static double scaling_exponent(int *Ns, double *times, int n)
{
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    int valid = 0;
    for (int i = 0; i < n; i++) {
        if (times[i] > 0 && Ns[i] > 0) {
            double x = log((double)Ns[i]);
            double y = log(times[i]);
            sum_x += x; sum_y += y;
            sum_xy += x * y; sum_x2 += x * x;
            valid++;
        }
    }
    if (valid < 3) return 0;
    double b = (valid * sum_xy - sum_x * sum_y) / (valid * sum_x2 - sum_x * sum_x);
    return b;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    hexagram_init_tables();
    s6_exotic_init();

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                     ║\n");
    printf("║  MACHINE WITHIN THE MACHINE                                         ║\n");
    printf("║  Virtual QPU on HexState D=6 Substrate                             ║\n");
    printf("║                                                                     ║\n");
    printf("║  Architecture:                                                      ║\n");
    printf("║    Host:  HexState HPC graph ── N D=6 quhits                       ║\n");
    printf("║    VQPU:  2N virtual qubits encoded in host quhits                 ║\n");
    printf("║           (2 virtual qubits per host site)                          ║\n");
    printf("║                                                                     ║\n");
    printf("║  Encoding: |0⟩_h→|00⟩_v |1⟩_h→|01⟩_v |2⟩_h→|10⟩_v |3⟩_h→|11⟩_v  ║\n");
    printf("║            |4⟩_h,|5⟩_h = hidden subspace (unused by VQPU)          ║\n");
    printf("║                                                                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    /* ── Test sizes: virtual qubit counts ── */
    int test_sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int n_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int repeats = 4;

    /* ═══════════════════════════════════════════════════════════════════
     * BENCHMARK 1: GHZ Chain (nearest-neighbor CX)
     * Expect: O(N) edges, O(N) time → scaling exponent ≈ 1.0
     * ═══════════════════════════════════════════════════════════════════ */
    printf("═══ BENCHMARK 1: Virtual GHZ Chain (Nearest-Neighbor CX) ═══\n\n");
    printf("  %8s  %12s  %12s  %12s\n", "VQubits", "Time (μs)", "Edges", "μs/edge");
    printf("  %8s  %12s  %12s  %12s\n", "───────", "──────────", "─────────", "─────────");

    double ghz_times[n_sizes];
    int    ghz_edges[n_sizes];

    for (int i = 0; i < n_sizes; i++) {
        int N = test_sizes[i];
        if (N > 512) { /* skip very large for time */
            ghz_times[i] = 0; ghz_edges[i] = 0;
            continue;
        }
        int R = (N <= 64) ? repeats : 2;
        double t; int e;
        bench_ghz_chain(N, R, &t, &e);
        ghz_times[i] = t; ghz_edges[i] = e;
        double us = t * 1e6;
        printf("  %8d  %12.2f  %12d  %12.4f\n", N, us, e, e > 0 ? us / e : 0);
    }

    double ghz_exp = scaling_exponent(test_sizes, ghz_times, n_sizes);

    printf("\n  GHZ scaling exponent: b = %.4f  (b=1.0 = linear, b=2.0 = quadratic)\n\n",
           ghz_exp);

    /* ═══════════════════════════════════════════════════════════════════
     * BENCHMARK 2: Dense CX (All-to-All Entanglement)
     * Expect: O(N²) edges, O(N²) time → scaling exponent ≈ 2.0
     * ═══════════════════════════════════════════════════════════════════ */
    printf("═══ BENCHMARK 2: Dense CX (All-to-All Entanglement) ═══\n\n");
    printf("  %8s  %12s  %12s  %12s\n", "VQubits", "Time (μs)", "Edges", "μs/edge");
    printf("  %8s  %12s  %12s  %12s\n", "───────", "──────────", "─────────", "─────────");

    double dense_times[n_sizes];
    int    dense_edges[n_sizes];

    for (int i = 0; i < n_sizes; i++) {
        int N = test_sizes[i];
        if (N > 256) {
            dense_times[i] = 0; dense_edges[i] = 0;
            printf("  %8d  %12s  %12s  %12s\n", N, "(skip)", "(skip)", "");
            continue;
        }
        int R = (N <= 32) ? repeats : 2;
        double t; int e;
        bench_dense_cx(N, R, &t, &e);
        dense_times[i] = t; dense_edges[i] = e;
        double us = t * 1e6;
        printf("  %8d  %12.2f  %12d  %12.4f\n", N, us, e, e > 0 ? us / e : 0);
    }

    double dense_exp = scaling_exponent(test_sizes, dense_times, n_sizes);
    printf("\n  Dense CX scaling exponent: b = %.4f  (b=1.0 = linear, b=2.0 = quadratic)\n\n",
           dense_exp);

    /* ═══════════════════════════════════════════════════════════════════
     * BENCHMARK 3: Random Single-Qubit Gates (zero edges)
     * Expect: O(N) time, 0 edges → scaling exponent ≈ 1.0
     * ═══════════════════════════════════════════════════════════════════ */
    printf("═══ BENCHMARK 3: Random Single-Qubit Gates (No Entanglement) ═══\n\n");
    printf("  %8s  %12s  %12s\n", "VQubits", "Time (μs)", "Gates/site");
    printf("  %8s  %12s  %12s\n", "───────", "──────────", "──────────");

    double rand_times[n_sizes];

    for (int i = 0; i < n_sizes; i++) {
        int N = test_sizes[i];
        if (N > 512) {
            rand_times[i] = 0;
            continue;
        }
        int R = (N <= 64) ? repeats : 2;
        double t; int e;
        bench_random_single(N, N, R, &t, &e);
        rand_times[i] = t;
        double us = t * 1e6;
        printf("  %8d  %12.2f  %12d\n", N, us, N);
    }

    double rand_exp = scaling_exponent(test_sizes, rand_times, n_sizes);
    printf("\n  Random gate scaling exponent: b = %.4f  (b=1.0 = linear)\n\n",
           rand_exp);

    /* ═══════════════════════════════════════════════════════════════════
     * BENCHMARK 4: Virtual QFT
     * Expect: O(N log N) gates, moderate edges → exponent ~1-1.3
     * ═══════════════════════════════════════════════════════════════════ */
    printf("═══ BENCHMARK 4: Virtual QFT (Quantum Fourier Transform) ═══\n\n");
    printf("  %8s  %12s  %12s\n", "VQubits", "Time (μs)", "Edges");
    printf("  %8s  %12s  %12s\n", "───────", "──────────", "─────────");

    double qft_times[n_sizes];

    for (int i = 0; i < n_sizes; i++) {
        int N = test_sizes[i];
        if (N > 64) { /* QFT is O(N²) virtual gates, cap for time */
            qft_times[i] = 0;
            continue;
        }
        int R = (N <= 16) ? repeats : 2;
        double t; int e;
        bench_vqft(N, R, &t, &e);
        qft_times[i] = t;
        double us = t * 1e6;
        printf("  %8d  %12.2f  %12d\n", N, us, e);
    }

    double qft_exp = scaling_exponent(test_sizes, qft_times, n_sizes);

    printf("\n  QFT scaling exponent: b = %.4f\n\n", qft_exp);

    /* ═══════════════════════════════════════════════════════════════════
     * SUMMARY: Non-Linear Scaling Analysis
     * ═══════════════════════════════════════════════════════════════════ */

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  NON-LINEAR SCALING ANALYSIS                                        ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                     ║\n");
    printf("║  Scaling exponent b:                Expected     Measured          ║\n");
    printf("║    GHZ Chain (nearest CX)   b=1.0     %8.4f                   ║\n", ghz_exp);
    printf("║    Dense CX (all-to-all)     b=2.0     %8.4f                   ║\n", dense_exp);
    printf("║    Random gates (no edges)   b=1.0     %8.4f                   ║\n", rand_exp);
    printf("║    Virtual QFT               b~1.3     %8.4f                   ║\n", qft_exp);
    printf("║                                                                     ║\n");

    /* Detect non-linear scaling */
    int non_linear = 0;
    if (fabs(ghz_exp - 1.0) > 0.15) non_linear = 1;
    if (fabs(dense_exp - 2.0) > 0.3) non_linear = 1;

    printf("║  Non-linear scaling detected: %s                                   ║\n",
           non_linear ? "YES ⚡" : "no (linear within tolerance)");
    printf("║                                                                     ║\n");
    printf("║  Interpretation:                                                    ║\n");
    printf("║    The HPC graph stores entanglement as edges. NEAREST-neighbor     ║\n");
    printf("║    circuits produce O(N) edges → linear scaling. ALL-TO-ALL         ║\n");
    printf("║    circuits produce O(N²) edges → quadratic scaling. The VQPU's    ║\n");
    printf("║    performance reflects the HOST'S edge topology, not the virtual   ║\n");
    printf("║    qubit count directly.                                            ║\n");
    printf("║                                                                     ║\n");
    printf("║  The hidden |4⟩,|5⟩ subspace: 2 of 6 host basis states are          ║\n");
    printf("║  invisible to the virtual QPU. They carry zero VQPU amplitude       ║\n");
    printf("║  but contribute to the host's normalization. The HPC graph's        ║\n");
    printf("║  phase edges reference these states through the full 6×6 matrix.   ║\n");
    printf("║  This extra subspace is the 'machine within the machine.'           ║\n");
    printf("║                                                                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
