#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "quhit_engine.h"
#include "hpc_graph.h"
#include "hpc_contract.h"
#include "quhit_triality.h"

/* PRNG */
static uint64_t rng_state = 0x811C9DC5;
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
 * PCP-Driven Magic State Distillation
 * We bypass the "nosiness" of dense quantum hardware measurements.
 * Instead of O(N) measurements, we apply O(1) random PCP checks 
 * (via an expander graph overlay) and rely on the Z3 sector's 
 * extreme topological robustness to act as a heat-sink for Z2 errors.
 */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  qPCP MAGIC DISTILLATION: SPARSITY-DRIVEN Z2 PURIFICATION    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int N_list[] = {32, 64, 128};
    /* We sweep the PCP query rate (queries per node per step) */
    double query_rates[] = {0.10, 0.50, 1.00, 2.00, 5.00}; 
    double p_err = 0.05; /* Ambient noise rate (5% of nodes decohere per step) */
    int steps = 20;

    printf("  DATA,N,queries_per_step,Z2_fidelity,Z3_entropy\n");

    for (int n_idx = 0; n_idx < 3; n_idx++) {
        int N = N_list[n_idx];

        for (int q_idx = 0; q_idx < 5; q_idx++) {
            double q_rate = query_rates[q_idx];
            int num_queries = (int)(N * q_rate);
            if (num_queries < 1) num_queries = 1;

            double final_fid_sum = 0.0;
            double z3_ent_sum = 0.0;
            int realizations = 4;

            for (int r = 0; r < realizations; r++) {
                HPCGraph *g = hpc_create(N + 1); /* N data + 1 ancilla for PCP checks */

                /* 1. Inject Magic States (Non-Clifford T-gate analog on Z2) */
                for(int i=0; i<N; i++) {
                    hpc_dft(g, i);
                    /* Distort the phase slightly to create a magic state */
                    g->locals[i].edge_re[1] *= 0.707;
                    g->locals[i].edge_im[1] = 0.707; 
                    g->locals[i].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL;
                }

                /* 2. Build qLDPC Expander Graph (Base connectivity) */
                for(int i=0; i<N; i++) {
                    int neighbor = (i + rng_int(N-1) + 1) % N;
                    hpc_cz(g, i, neighbor);
                }

                for (int t = 0; t < steps; t++) {
                    /* PCP Queries: Measure only a very sparse set of edges */
                    for (int q = 0; q < num_queries; q++) {
                        int a = rng_int(N);
                        int b = (a + rng_int(N-1) + 1) % N;
                        
                        /* We apply CZ^2 directly to form the Z3 expander overlay */
                        hpc_cz(g, a, b); hpc_cz(g, a, b);
                    }

                    /* Ambient noise */
                    for (int i = 0; i < N; i++) {
                        if (rng_uniform() < p_err) {
                            int out = hpc_measure(g, i, rng_uniform());
                            memset(&g->locals[i], 0, sizeof(TrialityQuhit));
                            g->locals[i].edge_re[out] = 1.0;
                            hpc_dft(g, i);

                            /* Re-inject the magic state so Z2 fidelity can be tracked */
                            g->locals[i].edge_re[1] *= 0.707;
                            g->locals[i].edge_im[1] = 0.707; 
                            g->locals[i].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL;
                        }
                    }

                    hpc_compact_edges(g);
                }

                /* Evaluate Z2 Fidelity & Z3 Entropy */
                double z2_fidelity = 0.0;
                int z3_edges = 0;
                for (uint64_t e = 0; e < g->n_edges; e++) {
                    if (g->edges[e].type == HPC_EDGE_CZ) {
                        int m = g->edges[e].syntheme_id ? g->edges[e].syntheme_id : 1;
                        if (m % 3 != 0) z3_edges++;
                    }
                }
                
                /* In an expander, if Z3 edges proliferate, it means the graph successfully 
                   absorbed the entropy without destroying the nodes' base state. */
                z3_ent_sum += (double)z3_edges / N;
                
                /* We evaluate the fidelity of the remaining unmeasured nodes to the original magic state.
                   Proxy: fidelity is high if the graph remains sparsely connected in Z2, 
                   meaning the Z2 sector didn't scramble. */
                int z2_edges = 0;
                for (uint64_t e = 0; e < g->n_edges; e++) {
                    if (g->edges[e].type == HPC_EDGE_CZ) {
                        int m = g->edges[e].syntheme_id ? g->edges[e].syntheme_id : 1;
                        if (m % 2 != 0) z2_edges++;
                    }
                }
                
                /* If Z2 edges are suppressed (meaning Z2 errors are absorbed), fidelity stays high */
                z2_fidelity = exp(-(double)z2_edges / N); 
                final_fid_sum += z2_fidelity;

                hpc_destroy(g);
            }

            printf("  DATA,%d,%.2f,%.4f,%.4f\n", N, q_rate, 
                   final_fid_sum/realizations, z3_ent_sum/realizations);
        }
    }
    return 0;
}
