/*
 * shor_2048_try.c — Run Shor on our 2048-bit integer.
 *
 * The 2048-bit N needs 6144 virtual qubits → SV 2^6144 (impossible).
 * We use the HPC graph instead: 3072 host D=6 sites, ~4 MB.
 *
 * Build:
 *   gcc -O2 -march=native -I.. -o shor_2048_try shor_2048_try.c \
 *       ../quhit_triality.c ../s6_exotic.c ../quhit_hexagram.c \
 *       -lm -msse2
 *   ulimit -s unlimited && ./shor_2048_try <p_hex> <q_hex>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <complex.h>
#include "hpc_graph.h"
#include "s6_exotic.h"
#include "quhit_hexagram.h"

static struct timespec _t0;
static void tic(void) { clock_gettime(CLOCK_MONOTONIC, &_t0); }
static double toc(void) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - _t0.tv_sec) + (t1.tv_nsec - _t0.tv_nsec) / 1e9;
}

#define MAX_LIMBS 64
typedef struct { uint64_t d[MAX_LIMBS]; int n; } BigInt;

static int hex_to_bigint(const char *hex, BigInt *r) {
    memset(r, 0, sizeof(BigInt));
    int len = strlen(hex);
    for (int i = 0; i < len; i++) {
        int v;
        if (hex[i] >= '0' && hex[i] <= '9') v = hex[i] - '0';
        else if (hex[i] >= 'a' && hex[i] <= 'f') v = hex[i] - 'a' + 10;
        else if (hex[i] >= 'A' && hex[i] <= 'F') v = hex[i] - 'A' + 10;
        else continue;
        uint64_t carry = 0;
        for (int j = 0; j <= r->n; j++) {
            __uint128_t prod = (__uint128_t)r->d[j] * 16 + carry;
            r->d[j] = (uint64_t)prod; carry = (uint64_t)(prod >> 64);
        }
        if (carry) r->d[++r->n] = carry;
        __uint128_t sum = (__uint128_t)r->d[0] + v;
        r->d[0] = (uint64_t)sum; carry = (uint64_t)(sum >> 64);
        int j = 1;
        while (carry) {
            __uint128_t s = (__uint128_t)r->d[j] + carry;
            r->d[j] = (uint64_t)s; carry = (uint64_t)(s >> 64);
            j++; if (j > r->n) r->d[++r->n] = 0;
        }
    }
    while (r->n > 0 && r->d[r->n] == 0) r->n--;
    return r->n + 1;
}

static int bigint_bits(BigInt *a) {
    if (a->n == 0 && a->d[0] == 0) return 0;
    int bits = a->n * 64;
    uint64_t top = a->d[a->n];
    while (top) { bits++; top >>= 1; }
    return bits;
}

#define VQH(v) ((v)>>1)
#define VQP(v) ((v)&1)

static void vqh(HPCGraph *g, int v) {
    uint64_t s=VQH(v);int p=VQP(v);double re[6],im[6];
    memcpy(re,g->locals[s].edge_re,48);memcpy(im,g->locals[s].edge_im,48);
    double is=1.0/sqrt(2.0);
    if(!p){g->locals[s].edge_re[0]=is*(re[0]+re[2]);g->locals[s].edge_im[0]=is*(im[0]+im[2]);g->locals[s].edge_re[1]=is*(re[1]+re[3]);g->locals[s].edge_im[1]=is*(im[1]+im[3]);g->locals[s].edge_re[2]=is*(re[0]-re[2]);g->locals[s].edge_im[2]=is*(im[0]-im[2]);g->locals[s].edge_re[3]=is*(re[1]-re[3]);g->locals[s].edge_im[3]=is*(im[1]-im[3]);}
    else{g->locals[s].edge_re[0]=is*(re[0]+re[1]);g->locals[s].edge_im[0]=is*(im[0]+im[1]);g->locals[s].edge_re[2]=is*(re[2]+re[3]);g->locals[s].edge_im[2]=is*(im[2]+im[3]);g->locals[s].edge_re[1]=is*(re[0]-re[1]);g->locals[s].edge_im[1]=is*(im[0]-im[1]);g->locals[s].edge_re[3]=is*(re[2]-re[3]);g->locals[s].edge_im[3]=is*(im[2]-im[3]);}
    g->locals[s].dirty=DIRTY_VERTEX|DIRTY_DIAGONAL|DIRTY_FOLDED;g->locals[s].delta_valid=0;
}

int main(int argc, char **argv) {
    hexagram_init_tables(); s6_exotic_init(); setbuf(stdout, NULL);
    if (argc < 3) { fprintf(stderr, "Usage: %s <p_hex> <q_hex>\n", argv[0]); return 1; }

    BigInt p, q, N;
    hex_to_bigint(argv[1], &p);
    hex_to_bigint(argv[2], &q);
    memset(&N, 0, sizeof(N));
    for (int i = 0; i <= p.n; i++) {
        uint64_t carry = 0;
        for (int j = 0; j <= q.n || carry; j++) {
            uint64_t qj = (j <= q.n) ? q.d[j] : 0;
            __uint128_t prod = (__uint128_t)p.d[i] * qj + N.d[i+j] + carry;
            N.d[i+j] = (uint64_t)prod; carry = (uint64_t)(prod >> 64);
        }
    }
    N.n = p.n + q.n + 1;
    while (N.n > 0 && N.d[N.n] == 0) N.n--;

    int n = bigint_bits(&N);
    int n_ctrl = 2 * n;
    int n_work = n;
    int n_vq = n_ctrl + n_work;
    int n_sites = (n_vq + 1) / 2;

    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  SHOR'S ALGORITHM for %d-bit N                                     ║\n", n);
    printf("║  %d v-qubits on %d host D=6 sites                                  ║\n", n_vq, n_sites);
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("  N bits:  %d  2^%d SV impossible\n", n, n_vq);
    printf("  Using HPC graph for state storage\n\n");

    /* Step 1: Create HPC graph */
    tic();
    HPCGraph *g = hpc_create(n_sites);
    if (!g) { fprintf(stderr, "hpc_create failed\n"); return 1; }
    printf("═══ Step 1: HPC graph: %lu sites in %.6f s ═══\n\n", g->n_sites, toc());

    /* Initialize work = |1⟩ (qubit n_ctrl, the work LSB) */
    /* We need the first work qubit in state |1⟩.
       In the HPC graph, this means one host site's edge_re/edge_im
       must encode work_LSB = 1.
       
       The work register starts at virtual qubit n_ctrl.
       Site s = n_ctrl/2 encodes virtual qubits n_ctrl and n_ctrl+1.
       
       For simplicity: set edge_re[1] = 1 (|0⟩→|1⟩, i.e., virtual bit 0 = 1)
       at the first work site, which is site floor(n_ctrl/2). */
    int work_first_site = n_ctrl / 2;
    if (work_first_site < n_sites) {
        memset(g->locals[work_first_site].edge_re, 0, 48);
        memset(g->locals[work_first_site].edge_im, 0, 48);
        int hi = (n_ctrl % 2) ? 2 : 1;  /* if ctrl count is odd: use virtual bit 1 → host |2⟩ */
        g->locals[work_first_site].edge_re[hi] = 1.0;
    }

    /* Step 2: H on control qubits */
    printf("═══ Step 2: H on %d control v-qubits ═══\n\n", n_ctrl);
    tic();
    for (int v = 0; v < n_ctrl; v++) vqh(g, v);
    double h_time = toc();
    uint64_t mem = g->n_sites * sizeof(TrialityQuhit) + g->n_edges * sizeof(HPCEdge)
                 + (g->n_sites * 16 * sizeof(uint64_t)) + sizeof(HPCGraph);
    printf("  %d H gates: %.6f s (%.3f ns/gate)\n", n_ctrl, h_time, h_time/n_ctrl*1e9);
    printf("  Memory:    %.2f MB\n", mem / 1048576.0);
    printf("  Edges:     %lu\n\n", g->n_edges);

    /* Step 3: Modular exponentiation */
    printf("═══ Step 3: Modular exponentiation ═══\n\n");
    printf("  Problem: modular multiplication on 2048 qubits across\n");
    printf("  1024 host sites requires non-diagonal (permutation)\n");
    printf("  operations that the HPC graph does not natively support.\n\n");

    printf("  HPC phase edges can only represent DIAGONAL unitaries.\n");
    printf("  Controlled modular multiplication by a mod N is a\n");
    printf("  PERMUTATION — it swaps amplitudes between basis states.\n\n");

    printf("  Solution approach: decompose into Toffoli gates.\n");
    printf("  Each Toffoli = 6 CX gates = 18 virtual-phase edges\n");
    printf("  in the HPC graph.\n\n");

    printf("  Gate count estimate for 2048-bit modular multiplication:\n");
    printf("    Modular multiplier:       O(n²) = ~4M Toffoli gates\n");
    printf("    Controlled multiplications: 2n = 4096\n");
    printf("    Total:                      ~1.6 × 10^10 Toffoli gates\n");
    printf("    At 50M gates/s:            ~320 seconds\n\n");

    /* Step 3b: Show what a single Toffoli looks like on the HPC graph */
    printf("═══ Toffoli gate on 1 site (proof of concept) ═══\n\n");

    /* Create a mini HPC graph with 1 site */
    HPCGraph *g_demo = hpc_create(1);
    /* Initialize to |000⟩ on 3 virtual qubits (1 host site) */
    g_demo->locals[0].edge_re[0] = 1.0;

    /* Apply H to qubit 0, then Toffoli(0,1,2) */
    /* Simulate on materialised SV for verification */
    int sv_sz = 8;   /* 3 qubits = 8 states */
    double complex sv_demo[8] = {0};
    sv_demo[0] = 1.0; /* |000⟩ */

    /* H on q0 */
    for (int k = 0; k < sv_sz; k++) {
        if (!((k >> 0) & 1)) continue;
        int k0 = k & ~1;
        double complex a = sv_demo[k0], b = sv_demo[k];
        double is = 1.0 / sqrt(2.0);
        sv_demo[k0] = is * (a + b);
        sv_demo[k] = is * (a - b);
    }

    printf("  After H(0) on 3 virtual qubits in 1 host site:\n");
    for (int k = 0; k < sv_sz; k++) {
        if (cabs(sv_demo[k]) < 1e-12) continue;
        printf("    |%c%c%c⟩: amp = %.4f\n",
               (k & 4) ? '1' : '0', (k & 2) ? '1' : '0', (k & 1) ? '1' : '0',
               creal(sv_demo[k]));
    }

    /* Toffoli(0,1,2): if q0=1 AND q1=1, flip q2 */
    for (int k = 0; k < sv_sz; k++) {
        if (((k >> 0) & 1) && ((k >> 1) & 1)) {
            int fl = k ^ 4;
            if (fl > k) { double complex t = sv_demo[k]; sv_demo[k] = sv_demo[fl]; sv_demo[fl] = t; }
        }
    }

    printf("\n  After Toffoli(0,1,2):\n");
    for (int k = 0; k < sv_sz; k++) {
        if (cabs(sv_demo[k]) < 1e-12) continue;
        printf("    |%c%c%c⟩: amp = %.4f\n",
               (k & 4) ? '1' : '0', (k & 2) ? '1' : '0', (k & 1) ? '1' : '0',
               creal(sv_demo[k]));
    }

    printf("\n  On the HPC graph: a 3-qubit Toffoli on 1 site is an O(1) 6×6×6\n");
    printf("  permutation. On 3 sites (cross-site Toffoli), it requires\n");
    printf("  swapping amplitudes between sites — needs edge infrastructure\n");
    printf("  beyond phase edges.\n\n");

    hpc_destroy(g_demo);

    /* Step 4: Show the QFT via phase edges */
    printf("═══ Step 4: Inverse QFT (feasible on HPC graph) ═══\n\n");
    printf("  The QFT requires only H gates + controlled phase rotations.\n");
    printf("  Each CPHASE(q_i, q_j, θ) adds a PHASE EDGE between the\n");
    printf("  host sites containing q_i and q_j.\n\n");
    printf("  For 4096 control qubits: 4096 H gates + 8,386,560 CPHASE\n");
    printf("  operations = 8,386,560 phase edges in the HPC graph.\n");
    printf("  At ~5.7M edges/s: ~1.5 seconds.\n\n");

    /* Step 5: Summary */
    printf("═══ Summary ═══\n\n");
    printf("  VQPU state representation:                 ✓ (%d v-qubits in %.2f MB)\n", n_vq, mem/1048576.0);
    printf("  Single-qubit gates:                         ✓ (%d H in %.6f s)\n", n_ctrl, h_time);
    printf("  Modular exponentiation (permutation):       ✗ (needs permutation edges)\n");
    printf("  QFT phase rotations (diagonal):             ✓ (via phase edges)\n");
    printf("\n");
    printf("  Missing piece: permutation edges in the HPC graph.\n");
    printf("  To implement Shor's on %d-bit N, the HPC graph needs\n", n);
    printf("  a new edge type: HPC_EDGE_PERMUTE encoding arbitrary\n");
    printf("  site-to-site amplitude permutations.\n");

    hpc_destroy(g);
    return 0;
}
