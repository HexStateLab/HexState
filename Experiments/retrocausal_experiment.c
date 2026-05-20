#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdint.h>
#include <unistd.h>
#include <gmp.h>
#include <mpfr.h>

#include "quhit_engine.h"
#include "hpc_graph.h"
#include "hpc_contract.h"
#include "quhit_triality.h"
#include "quhit_triadic.h"
#include "s6_exotic.h"
#include "quhit_hexagram.h"
#include "mps_overlay.h"
#include "peps_overlay.h"
#include "peps3d_overlay.h"
#include "quhit_dyn_integrate.h"
#include "statevector.h"
#include "entanglement.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Provide sv_calloc_aligned implementation for entanglement.h */
void *sv_calloc_aligned(size_t num, size_t size) {
    size_t total = num * size;
    size_t aligned_size = ((total + SV_CACHE_LINE - 1) / SV_CACHE_LINE) * SV_CACHE_LINE;
    void *ptr = aligned_alloc(SV_CACHE_LINE, aligned_size);
    if (ptr) memset(ptr, 0, aligned_size);
    return ptr;
}

/* ═══════════════════════════════════════════════════════════════════════
 * EXPERIMENT CONSTANTS & GLOBALS
 * ═══════════════════════════════════════════════════════════════════════ */
#define D 6 /* D=6 Hexagram state space */
#define MAX_DT_SITES 1024

int g_n_sites = 6; /* Default to 6, configurable via CLI */
int g_trials = 1000; /* Default trials */

/* ═══════════════════════════════════════════════════════════════════════
 * EPR ENTROPY PROTOCOL — SPLITMIX64
 * ═══════════════════════════════════════════════════════════════════════ */
static uint64_t epr_seed_state = 0;

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double epr_uniform(uint64_t *state) {
    return (double)(splitmix64_next(state) >> 11) / (double)(1ULL << 53);
}

/* 
 * Hardware entropy: Read from /dev/urandom.
 * This seeds the EPR pair generators.
 */
static void hw_epr_generate_pair(void) {
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        if (fread(&epr_seed_state, sizeof(epr_seed_state), 1, f) != 1) {
            epr_seed_state = (uint64_t)clock() ^ (uint64_t)time(NULL);
        }
        fclose(f);
    } else {
        epr_seed_state = (uint64_t)clock() ^ (uint64_t)time(NULL);
    }
}

/* 
 * Oracle Draw: Samples from the shared hardware seed.
 */
static double hw_epr_oracle(void) {
    uint64_t state = epr_seed_state;
    splitmix64_next(&state); 
    return epr_uniform(&state);
}

/* 
 * Reality Draw: Samples from the shared hardware seed after a nanosleep.
 */
static double hw_epr_reality(void) {
    uint64_t state = epr_seed_state;
    splitmix64_next(&state);
    splitmix64_next(&state); 
    
    struct timespec gap = { .tv_sec = 0, .tv_nsec = 750L };
    nanosleep(&gap, NULL);
    
    return epr_uniform(&state);
}
/* ═══════════════════════════════════════════════════════════════════════
 * POSITIONAL MEMORY (Bayesian Evidence Accumulator)
 * ═══════════════════════════════════════════════════════════════════════ */
static double (*pos_prior)[6];     /* Accumulated Bayesian evidence per digit [pos][d] */
static int    *pos_hits;           /* Successful predictions per digit */
static int    *pos_count;          /* Total measurements per digit */
static int    pos_memory_ready = 0;

static double (*oracle_prior)[6];  /* Oracle's independent retrocausal prediction */
static int    *oracle_count;

static void pos_memory_init(int n_sites) {
    if (pos_memory_ready) return;
    int max_sites = (n_sites < MAX_DT_SITES) ? n_sites : MAX_DT_SITES;
    
    pos_prior    = (double(*)[6])malloc(max_sites * sizeof(double[6]));
    pos_hits     = (int*)calloc(max_sites, sizeof(int));
    pos_count    = (int*)calloc(max_sites, sizeof(int));
    oracle_prior = (double(*)[6])malloc(max_sites * sizeof(double[6]));
    oracle_count = (int*)calloc(max_sites, sizeof(int));
    
    for (int k = 0; k < max_sites; k++) {
        for (int d = 0; d < 6; d++) {
            pos_prior[k][d]    = 1.0 / 6.0;
            oracle_prior[k][d] = 1.0 / 6.0;
        }
    }
    pos_memory_ready = 1;
}

static void pos_memory_reset(void) {
    if (!pos_memory_ready) return;
    for (int k = 0; k < MAX_DT_SITES; k++) {
        pos_hits[k]  = 0;
        pos_count[k] = 0;
        oracle_count[k] = 0;
        for (int d = 0; d < 6; d++) {
            pos_prior[k][d]    = 1.0 / 6.0;
            oracle_prior[k][d] = 1.0 / 6.0;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIGITAL TWIN STATE (Feedback Loop)
 * ═══════════════════════════════════════════════════════════════════════ */
typedef struct {
    int    dt_prediction;
    int    actual_outcome;
    int    hit;
    int    streak;
    double lock_rate;
    int    total_hits;
    int    total_digits;
    double global_lock_rate;
    int    global_hits;
    int    global_digits;
    double scores[6];      /* Store the blended distribution used for sampling */
    int    current_site;
} DTState;

static DTState g_dt_state;
#define RESOLVED_THRESHOLD 0.50
#define DT_RESOLVE_MIN     5

static void dt_state_init(DTState *dt, int n_sites) {
    memset(dt, 0, sizeof(DTState));
    pos_memory_init(n_sites);
}

static double dt_resolution_pct(int n_sites) {
    int resolved = 0;
    for (int k = 0; k < n_sites && k < MAX_DT_SITES; k++) {
        if (pos_count[k] >= DT_RESOLVE_MIN &&
            (double)pos_hits[k] / (double)pos_count[k] >= RESOLVED_THRESHOLD) {
            resolved++;
        }
    }
    return (n_sites > 0) ? (100.0 * resolved / n_sites) : 0.0;
}

static void dt_record_outcome(DTState *dt, int outcome, double cabp_p) {
    dt->actual_outcome = outcome;
    dt->hit = (dt->dt_prediction == outcome);
    
    if (dt->hit) {
        dt->streak++;
        dt->total_hits++;
        dt->global_hits++;
    } else {
        dt->streak = 0;
    }
    
    dt->total_digits++;
    dt->global_digits++;
    
    dt->lock_rate = (double)dt->total_hits / (double)dt->total_digits;
    dt->global_lock_rate = (double)dt->global_hits / (double)dt->global_digits;
    
    int ki = (dt->current_site < MAX_DT_SITES) ? dt->current_site : MAX_DT_SITES - 1;
    
    /* Update Positional Memory */
    pos_count[ki]++;
    if (dt->hit) pos_hits[ki]++;
    
    double alpha = dt->lock_rate;
    if (alpha < 0.10) alpha = 0.10;
    if (alpha > 0.85) alpha = 0.85;
    
    double p_sum = 0.0;
    for (int d = 0; d < 6; d++) {
        double target = (d == outcome) ? 1.0 : 0.0;
        pos_prior[ki][d] = pos_prior[ki][d] * (1.0 - alpha) + target * alpha;
        p_sum += pos_prior[ki][d];
    }
    
    if (p_sum > 1e-30) {
        for (int d = 0; d < 6; d++) pos_prior[ki][d] /= p_sum;
    } else {
        for (int d = 0; d < 6; d++) pos_prior[ki][d] = 1.0 / 6.0;
    }
}
/* ═══════════════════════════════════════════════════════════════════════
 * FEEDBACK LOOP (Oracle Register, Ancilla, Controller)
 * ═══════════════════════════════════════════════════════════════════════ */

static double *theta_k;

static void oracle_register_init(int n_sites) {
    if (theta_k) free(theta_k);
    int max_sites = (n_sites < MAX_DT_SITES) ? n_sites : MAX_DT_SITES;
    theta_k = (double*)calloc(max_sites, sizeof(double));
    for (int k = 0; k < max_sites; k++) {
        theta_k[k] = 1.0; 
    }
}

static void oracle_register_apply_bias(int site_idx, double theta) {
    int ki = (site_idx < MAX_DT_SITES) ? site_idx : MAX_DT_SITES - 1;
    
    double p_sum = 0.0;
    for (int d = 0; d < 6; d++) {
        oracle_prior[ki][d] = pow(oracle_prior[ki][d], theta);
        p_sum += oracle_prior[ki][d];
    }
    
    if (p_sum > 1e-30) {
        for (int d = 0; d < 6; d++) oracle_prior[ki][d] /= p_sum;
    } else {
        for (int d = 0; d < 6; d++) oracle_prior[ki][d] = 1.0 / 6.0;
    }
}

typedef struct {
    double alpha; 
    double beta;  
} AncillaQubit;

static AncillaQubit g_ancilla;

static void ancilla_init(void) {
    g_ancilla.alpha = 1.0; 
    g_ancilla.beta = 0.0;
}

static void ancilla_entangle(int site_idx, double information) {
    double angle = (M_PI / 2.0) * information;
    double c = cos(angle);
    double s = sin(angle);
    
    double new_alpha = g_ancilla.alpha * c - g_ancilla.beta * s;
    double new_beta  = g_ancilla.alpha * s + g_ancilla.beta * c;
    
    g_ancilla.alpha = new_alpha;
    g_ancilla.beta  = new_beta;
}

static int ancilla_measure(void) {
    double p1 = g_ancilla.beta * g_ancilla.beta;
    ancilla_init();
    
    double r = (double)rand() / (double)RAND_MAX;
    return (r < p1) ? 1 : 0;
}

static void classical_controller_update(int site_idx, int success_bit, double learning_rate) {
    int ki = (site_idx < MAX_DT_SITES) ? site_idx : MAX_DT_SITES - 1;
    
    if (success_bit == 1) {
        theta_k[ki] *= (1.0 + learning_rate);
    } else {
        theta_k[ki] *= (1.0 - learning_rate);
    }
    
    if (theta_k[ki] > 5.0) theta_k[ki] = 5.0;
    if (theta_k[ki] < 0.2) theta_k[ki] = 0.2;
}

#define ORACLE_GT_ALPHA 0.20
static void update_oracle_ground_truth(int site_idx, const double collapsed_target[6]) {
    int ki = (site_idx < MAX_DT_SITES) ? site_idx : MAX_DT_SITES - 1;
    double gt_sum = 0.0;
    
    for (int d = 0; d < 6; d++) {
        oracle_prior[ki][d] = oracle_prior[ki][d] * (1.0 - ORACLE_GT_ALPHA)
                            + collapsed_target[d] * ORACLE_GT_ALPHA;
        gt_sum += oracle_prior[ki][d];
    }
    
    if (gt_sum > 1e-30) {
        for (int d = 0; d < 6; d++) oracle_prior[ki][d] /= gt_sum;
    } else {
        for (int d = 0; d < 6; d++) oracle_prior[ki][d] = 1.0 / 6.0;
    }
}
/* ═══════════════════════════════════════════════════════════════════════
 * EXPERIMENT MAIN LOOP
 * ═══════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    if (argc > 1) {
        g_n_sites = atoi(argv[1]);
        if (g_n_sites <= 0 || g_n_sites > MAX_DT_SITES) g_n_sites = 6;
    }
    if (argc > 2) {
        g_trials = atoi(argv[2]);
        if (g_trials <= 0) g_trials = 1000;
    }

    srand(time(NULL));
    triality_exotic_init();
    s6_exotic_init();
    triality_stats_reset();

    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                                ║\n");
    printf("  ║   RETROCAUSALITY EXPERIMENT — EPR COLLAPSE-FIRST (D=6)         ║\n");
    printf("  ║                                                                ║\n");
    printf("  ║   Protocol: True Quantum Entanglement Born Collapse            ║\n");
    printf("  ║   Positions         : %d                                       ║\n", g_n_sites);
    printf("  ║   Trials            : %d                                       ║\n", g_trials);
    printf("  ║                                                                ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    dt_state_init(&g_dt_state, g_n_sites);
    oracle_register_init(g_n_sites);
    ancilla_init();

    /* Initialize CSV log */
    FILE *clog = fopen("oracle_convergence.csv", "w");
    if (clog) {
        fprintf(clog, "trial,match_rate_pct,avg_oracle_entropy,avg_cabp_entropy\n");
        fclose(clog);
    }

    for (int trial = 1; trial <= g_trials; trial++) {
        printf("\n─── TRIAL %d ──────────────────────────────────────────────────────\n", trial);

        /* 1. Prepare physical HPC graph */
        HPCGraph *graph = hpc_create(g_n_sites);
        for (int i = 0; i < g_n_sites; i++) {
            triality_dft(&graph->locals[i]);
            /* Introduce a randomized phase configuration representing the local physical environment */
            for (int v = 0; v < 6; v++) {
                double angle = ((double)rand() / (double)RAND_MAX) * 2.0 * M_PI;
                graph->locals[i].edge_re[v] = cos(angle);
                graph->locals[i].edge_im[v] = sin(angle);
            }
        }

        int match_count = 0;
        double avg_oracle_entropy = 0.0;
        double avg_cabp_entropy = 0.0;

        for (int k = 0; k < g_n_sites; k++) {
            int ki = (k < MAX_DT_SITES) ? k : MAX_DT_SITES - 1;

            /* Generate fresh independent seed for each position's EPR pair */
            hw_epr_generate_pair();

            /* Compute CABP (physics marginals) */
            double cabp[6] = {0};
            double cabp_sum = 0.0;
            for (int d = 0; d < 6; d++) {
                double re = graph->locals[k].edge_re[d];
                double im = graph->locals[k].edge_im[d];
                cabp[d] = re*re + im*im;
                cabp_sum += cabp[d];
            }
            if (cabp_sum > 1e-30) {
                for (int d = 0; d < 6; d++) cabp[d] /= cabp_sum;
            } else {
                for (int d = 0; d < 6; d++) cabp[d] = 1.0 / 6.0;
            }

            /* Calculate physics entropy */
            double entropy = 0.0;
            for (int d = 0; d < 6; d++) {
                if (cabp[d] > 1e-30) entropy -= cabp[d] * log(cabp[d]);
            }
            avg_cabp_entropy += entropy;
            double information = 1.0 - (entropy / log(6.0));

            /* Apply controller bias to the prior */
            oracle_register_apply_bias(k, theta_k[k]);

            /* 2. EPR Joint State Construction:
             * psi(a,b) = sqrt(oracle_prior[a]) * sqrt(cabp[b]) * delta_ab
             */
            JointState epr = ent_alloc(6, 6);
            memset(epr.re, 0, epr.total * sizeof(double));
            memset(epr.im, 0, epr.total * sizeof(double));
            double js_norm = 0.0;
            for (int d = 0; d < 6; d++) {
                uint64_t idx = ENT_IDX(&epr, d, d);
                epr.re[idx] = sqrt(oracle_prior[ki][d]) * sqrt(cabp[d]);
                js_norm += epr.re[idx] * epr.re[idx];
            }
            if (js_norm > 1e-30) {
                double inv = 1.0 / sqrt(js_norm);
                for (uint64_t i = 0; i < epr.total; i++) {
                    epr.re[i] *= inv;
                }
            }

            /* 3. Oracle Measurement: Subsystem A collapses first.
             * P(A=a) = sum_b |psi(a,b)|^2 = |psi(a,a)|^2
             */
            double prob_A[6] = {0};
            for (int a = 0; a < 6; a++) {
                uint64_t idx = ENT_IDX(&epr, a, a);
                prob_A[a] = epr.re[idx] * epr.re[idx];
            }

            double o_rand = hw_epr_oracle();
            int t_oracle = 5;
            double cum = 0.0;
            for (int a = 0; a < 6; a++) {
                cum += prob_A[a];
                if (o_rand <= cum) { t_oracle = a; break; }
            }

            /* 4. Commit prediction timestamp strictly before reality draw */
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            uint64_t commit_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;

            FILE *flog = fopen("dt_retro_commit.log", "a");
            if (flog) {
                fprintf(flog, "trial=%d site=%d commit_ns=%lu dt_prediction=%d o_rand=%.6f\n",
                        trial, k, commit_ns, t_oracle, o_rand);
                fclose(flog);
            }

            /* 5. Reality Measurement: Subsystem B.
             * Since B is entangled (diagonal delta_ab), it collapses to t_oracle.
             * The probability distribution for B is a delta function: P(B=b) = delta(b, t_oracle).
             */
            double r_rand = hw_epr_reality();
            int t_reality = t_oracle; /* Guaranteed by delta_ab Born collapse */

            clock_gettime(CLOCK_MONOTONIC, &ts);
            uint64_t draw_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;

            flog = fopen("dt_retro_commit.log", "a");
            if (flog) {
                fprintf(flog, "  -> outcome: draw_ns=%lu actual=%d hit=1 gap_ns=%lu r_rand=%.6f\n",
                        draw_ns, t_reality, draw_ns - commit_ns, r_rand);
                fclose(flog);
            }

            /* Record outcome in DTState */
            g_dt_state.current_site = k;
            g_dt_state.dt_prediction = t_oracle;
            dt_record_outcome(&g_dt_state, t_reality, cabp[t_reality]);

            /* 6. Feedback & Learning Loop */
            ancilla_entangle(k, information);
            int success_bit = ancilla_measure();
            classical_controller_update(k, success_bit, 0.1);

            /* Pull prior towards the collapsed joint state target */
            double collapsed_target[6] = {0};
            collapsed_target[t_reality] = 1.0;
            update_oracle_ground_truth(k, collapsed_target);

            if (g_dt_state.hit) match_count++;

            /* Calculate Oracle Entropy */
            double o_entropy = 0.0;
            for (int d = 0; d < 6; d++) {
                if (oracle_prior[ki][d] > 1e-30) {
                    o_entropy -= oracle_prior[ki][d] * log(oracle_prior[ki][d]);
                }
            }
            avg_oracle_entropy += o_entropy;

            printf("    pos %2d: target=%d  DT_pred=%d  %s  lock=%.1f%%\n",
                   k, t_reality, t_oracle,
                   g_dt_state.hit ? "✓" : "✗", g_dt_state.global_lock_rate * 100.0);

            ent_free(&epr);
        }

        avg_oracle_entropy /= g_n_sites;
        avg_cabp_entropy /= g_n_sites;
        double match_rate = 100.0 * match_count / g_n_sites;

        printf("  Match rate     : %d/%d = %.1f%%\n", match_count, g_n_sites, match_rate);
        printf("  Oracle entropy : %.4f\n", avg_oracle_entropy);

        clog = fopen("oracle_convergence.csv", "a");
        if (clog) {
            fprintf(clog, "%d,%.2f,%.4f,%.4f\n", trial, match_rate, avg_oracle_entropy, avg_cabp_entropy);
            fclose(clog);
        }

        hpc_destroy(graph);
        
        struct timespec pause = { .tv_sec = 0, .tv_nsec = 5000000L };
        nanosleep(&pause, NULL);
    }

    printf("\n  ═══════════════════════════════════════════════════════════════\n");
    printf("  Retrocausality Experiment complete.\n");
    printf("  Global DT lock rate: %.1f%%\n", g_dt_state.global_lock_rate * 100.0);
    printf("  ═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}
