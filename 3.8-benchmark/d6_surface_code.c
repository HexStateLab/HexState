#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include "quhit_engine.h"
#include "hpc_graph.h"
#include "hpc_contract.h"
#include "quhit_triality.h"
#include "s6_exotic.h"

/* PRNG */
static uint64_t rng_state = 0x123456789ABCDEF0ULL;
static inline double rng_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (rng_state * 2.6815615859885194e-20);
}
static inline int rng_int(int max) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (int)(rng_state % max);
}

/* 
 * 2D Surface Code MIPT Simulation
 * Lattice: L x L data quhits.
 * We apply a layer of Z-stabilizers (measuring plaquettes) 
 * and X-stabilizers (measuring stars).
 * Then we apply random decoherence (single-site measurements) with prob p.
 */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  d=6 SURFACE CODE — ASYMMETRIC ERROR THRESHOLD SWEEP         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int L_list[] = {6, 8, 10};
    double p_list[] = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
    int num_p = sizeof(p_list) / sizeof(p_list[0]);
    int num_L = sizeof(L_list) / sizeof(L_list[0]);
    int realizations = 4;
    int steps_per_realization = 200;

    printf("  DATA,L,p_err,S2_mean,S2_err,S3_mean,S3_err,edges_mean\n");

    for (int L_idx = 0; L_idx < num_L; L_idx++) {
        int L = L_list[L_idx];
        int N = L * L; /* Data quhits */

        for (int p_idx = 0; p_idx < num_p; p_idx++) {
            double p_err = p_list[p_idx];

            double S2_sum = 0, S2_sq = 0;
            double S3_sum = 0, S3_sq = 0;
            double edge_sum = 0;
            int total_samples = 0;

            for (int r = 1; r <= realizations; r++) {
                printf("  [progress] L=%d p_err=%.3f realization %d/%d\r", L, p_err, r, realizations);
                fflush(stdout);

                HPCGraph *g = hpc_create(N + 1);

                /* Initialize data quhits to |+> state */
                for(int i=0; i<N; i++) {
                    hpc_dft(g, i);
                }

                for (int t = 0; t < steps_per_realization; t++) {
                    /* 1. Syndrome Extraction (2D lattice CZ entangling layer) */
                    /* Horizontal edges */
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L - 1; x++) {
                            int q1 = y * L + x;
                            int q2 = y * L + (x + 1);
                            hpc_cz(g, q1, q2);
                        }
                    }
                    /* Vertical edges */
                    for (int y = 0; y < L - 1; y++) {
                        for (int x = 0; x < L; x++) {
                            int q1 = y * L + x;
                            int q2 = (y + 1) * L + x;
                            hpc_cz(g, q1, q2);
                        }
                    }

                    hpc_compact_edges(g);

                    /* 2. Decoherence / Error Injection */
                    for (int i = 0; i < N; i++) {
                        if (rng_uniform() < p_err) {
                            /* Simulate an unrecoverable erasure/decoherence by measuring the data quhit */
                            int out = hpc_measure(g, i, rng_uniform());
                            
                            /* Re-inject to |+> so the lattice doesn't just empty out */
                            memset(&g->locals[i], 0, sizeof(TrialityQuhit));
                            g->locals[i].edge_re[out] = 1.0;
                            hpc_dft(g, i);
                        }
                    }

                    hpc_compact_edges(g);

                    /* 3. Sampling */
                    if (t >= 100 && (t % 10 == 0)) {
                        /* Bipartite entanglement: Left half vs Right half */
                        int mask[512] = {0};
                        for (int y = 0; y < L; y++) {
                            for (int x = 0; x < L / 2; x++) {
                                mask[y * L + x] = 1;
                            }
                        }

                        /* We don't have the exact density matrix, but we can compute the proxy 
                         * from the topological phase edges traversing the cut. */
                        int cut_edges_2 = 0;
                        int cut_edges_3 = 0;

                        for (uint64_t e = 0; e < g->n_edges; e++) {
                            if (g->edges[e].type == HPC_EDGE_CZ) {
                                int u = g->edges[e].site_a;
                                int v = g->edges[e].site_b;
                                if (u < N && v < N && mask[u] != mask[v]) {
                                    int m = g->edges[e].syntheme_id ? g->edges[e].syntheme_id : 1;
                                    /* Z2 sector sees m mod 2 */
                                    if (m % 2 != 0) cut_edges_2++;
                                    /* Z3 sector sees m mod 3 */
                                    if (m % 3 != 0) cut_edges_3++;
                                }
                            }
                        }
                        
                        double s2 = cut_edges_2 * log2(2.0);
                        double s3 = cut_edges_3 * log2(3.0);

                        S2_sum += s2; S2_sq += s2 * s2;
                        S3_sum += s3; S3_sq += s3 * s3;
                        edge_sum += g->n_edges;
                        total_samples++;
                    }
                }
                hpc_destroy(g);
            }

            double S2_mean = S2_sum / total_samples;
            double S2_err = sqrt((S2_sq / total_samples - S2_mean * S2_mean) / total_samples);
            double S3_mean = S3_sum / total_samples;
            double S3_err = sqrt((S3_sq / total_samples - S3_mean * S3_mean) / total_samples);
            double edges_mean = edge_sum / total_samples;

            printf("  DATA,%d,%.3f,%.5f,%.5f,%.5f,%.5f,%.1f\n",
                   L, p_err, S2_mean, S2_err, S3_mean, S3_err, edges_mean);
        }
    }
    return 0;
}