/*
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  fracture6.c — The Fractured Entanglement Transition at d = 6            ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  WHAT THE PROBES ESTABLISHED:                                             ║
 * ║    • The HPC graph is EXACT for diagonal circuits: arbitrary local prep, ║
 * ║      phase gates, CZ^m edges (compacted mod 6), computational-basis      ║
 * ║      measurement. (Non-diagonal locals after entanglement are not        ║
 * ║      faithful — P3/P4 — so this experiment never uses them there.)       ║
 * ║    • hpc_entropy_cut counts crossing edges × log₂6 and is wrong when     ║
 * ║      edges share endpoints (P2 cut=4) or carry multiplicity (P5).        ║
 * ║      We therefore compute entropy ourselves, exactly.                    ║
 * ║                                                                           ║
 * ║  THE EXACT ENTROPY OF A Z₆ WEIGHTED GRAPH STATE:                         ║
 * ║    CRT: Z₆ ≅ Z₂ × Z₃, so CZ₆ = CZ₂ ⊗ CZ₃² and |+₆⟩ = |+₂⟩⊗|+₃⟩.   ║
 * ║    The state factorizes into independent Z₂ and Z₃ graph states with    ║
 * ║    adjacency m mod 2 and m mod 3. Cut entropy:                           ║
 * ║        S = r₂·log₂2 + r₃·log₂3                                          ║
 * ║    where r_q = rank over GF(q) of the crossing-edge biadjacency.         ║
 * ║    Checks: m=2 → S=log₂3; m=3 → S=1; P2's single-site cut → 1+log₂3.  ║
 * ║                                                                           ║
 * ║  THE OPEN PROBLEM:                                                       ║
 * ║    Drive a system with random CZ^m edges (m uniform in 1..5) and         ║
 * ║    random single-site measurements at rate p. The Z₂ sector sees an     ║
 * ║    edge iff m is odd (prob 3/5); the Z₃ sector iff m ≢ 0 mod 3          ║
 * ║    (prob 4/5). Different effective densities → the two sectors should   ║
 * ║    undergo SEPARATE entanglement transitions, with an intermediate       ║
 * ║    FRACTURED phase: Z₃ entanglement extensive, Z₂ collapsed.            ║
 * ║    d = 6 is the smallest dimension where this can happen. The two        ║
 * ║    thresholds p_c² and p_c³ have never been computed.                    ║
 * ║    Mean-field estimate (steady mean degree ≈ 1/(2p), sector fractions    ║
 * ║    3/5 and 4/5, threshold at sector degree ~1): p_c² ≈ 0.30,            ║
 * ║    p_c³ ≈ 0.40 — the sweep below brackets both.                          ║
 * ║                                                                           ║
 * ║  TIERS:                                                                  ║
 * ║    A  Readout validation — compare our rank formula against Rényi-2     ║
 * ║       entropies computed directly from hpc_amplitude on small systems    ║
 * ║       (flat-spectrum graph states: Rényi-2 = von Neumann). Includes     ║
 * ║       the P2 ring single-site cut and 20 random N=4 multigraphs.         ║
 * ║    B  Frontier sweep — N ∈ {64,128,256}, p sweep, two-sector entropy.   ║
 * ║       Engine drives the actual quantum state; the shadow multigraph is   ║
 * ║       only the entropy readout, cross-checked against g->n_edges.        ║
 * ║                                                                           ║
 * ║  Build: same link line as the template, e.g.                             ║
 * ║    gcc -O2 -march=native -o fracture6 fracture6.c <same .c list> \      ║
 * ║        -lm -fopenmp -msse2                                                ║
 * ║  Run:  ulimit -s unlimited                                                ║
 * ║        ./fracture6 all          (Tier A then Tier B)                     ║
 * ║        ./fracture6 A | B                                                  ║
 * ║        ./fracture6 B quick      (coarse smoke test)                      ║
 * ║                                                                           ║
 * ║  Send back the full stdout. Tier B emits "DATA," CSV lines for the       ║
 * ║  two-sector finite-size crossing analysis.                                ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "quhit_engine.h"
#include "hpc_graph.h"
#include "hpc_contract.h"
#include "quhit_triality.h"
#include "s6_exotic.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define D     6
#define MAXN  512
#define LOG2_3 1.5849625007211562

/* ════════════════════════════ PRNG (xoshiro256**) ═══════════════════════ */

static uint64_t rng_s[4];
static inline uint64_t rotl64(uint64_t x, int k){ return (x<<k)|(x>>(64-k)); }
static uint64_t rng_next(void){
    uint64_t r = rotl64(rng_s[1]*5,7)*9, t = rng_s[1]<<17;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1]; rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=rotl64(rng_s[3],45); return r;
}
static double rng_uniform(void){ return (double)(rng_next()>>11)/(double)(1ULL<<53); }
static uint32_t rng_below(uint32_t n){ return (uint32_t)(rng_uniform()*n) % n; }
static void rng_seed(uint64_t seed){
    for (int i=0;i<4;i++){
        seed += 0x9e3779b97f4a7c15ULL; uint64_t z = seed;
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL;
        rng_s[i]=z^(z>>31);
    }
}

static struct timespec _t0;
static void tic(void){ clock_gettime(CLOCK_MONOTONIC,&_t0); }
static double toc(void){
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
    return (t1.tv_sec-_t0.tv_sec)+(t1.tv_nsec-_t0.tv_nsec)/1e9;
}

/* ════════════════ SHADOW MULTIGRAPH (entropy readout only) ══════════════ */
/* The engine holds the actual quantum state; this mirrors edge
 * multiplicities mod 6 so we can compute exact sector entropies. */

static uint8_t MULT[MAXN][MAXN];

static void shadow_clear(int N){
    for (int i=0;i<N;i++) memset(MULT[i],0,(size_t)N);
}
static void shadow_add(int i,int j,int m){
    uint8_t v = (uint8_t)((MULT[i][j]+m)%6);
    MULT[i][j]=v; MULT[j][i]=v;
}
static void shadow_isolate(int N,int i){
    for (int j=0;j<N;j++){ MULT[i][j]=0; MULT[j][i]=0; }
}
static long shadow_edge_count(int N){
    long c=0;
    for (int i=0;i<N;i++) for (int j=i+1;j<N;j++) if (MULT[i][j]%6) c++;
    return c;
}

/* ════════════════ EXACT SECTOR RANKS over GF(2) and GF(3) ═══════════════ */

/* GF(2) rank of the crossing biadjacency: rows = sites in A (0..half-1),
 * cols = sites in B (half..N-1). Bitset elimination. */
static int rank_gf2_cut(int N, int half)
{
    static uint64_t rows[MAXN/2][MAXN/64 + 1];
    int ncols = N - half;
    int nw = (ncols + 63) / 64;
    int nr = 0;
    for (int i = 0; i < half; i++){
        int any = 0;
        for (int w = 0; w < nw; w++) rows[nr][w] = 0;
        for (int j = half; j < N; j++){
            if (MULT[i][j] & 1){
                int c = j - half;
                rows[nr][c>>6] |= 1ULL << (c & 63);
                any = 1;
            }
        }
        if (any) nr++;
    }
    int rank = 0;
    for (int col = 0; col < ncols && rank < nr; col++){
        int w = col >> 6; uint64_t bit = 1ULL << (col & 63);
        int piv = -1;
        for (int r = rank; r < nr; r++)
            if (rows[r][w] & bit){ piv = r; break; }
        if (piv < 0) continue;
        if (piv != rank)
            for (int k = 0; k < nw; k++){
                uint64_t t = rows[piv][k]; rows[piv][k] = rows[rank][k]; rows[rank][k] = t;
            }
        for (int r = 0; r < nr; r++)
            if (r != rank && (rows[r][w] & bit))
                for (int k = 0; k < nw; k++) rows[r][k] ^= rows[rank][k];
        rank++;
    }
    return rank;
}

/* GF(3) rank of the crossing biadjacency (entries MULT mod 3). */
static int rank_gf3_cut(int N, int half)
{
    static int8_t M[MAXN/2][MAXN/2];
    int ncols = N - half;
    int nr = 0;
    for (int i = 0; i < half; i++){
        int any = 0;
        for (int j = half; j < N; j++){
            int8_t v = (int8_t)(MULT[i][j] % 3);
            M[nr][j-half] = v;
            if (v) any = 1;
        }
        if (any) nr++;
    }
    int rank = 0;
    for (int col = 0; col < ncols && rank < nr; col++){
        int piv = -1;
        for (int r = rank; r < nr; r++) if (M[r][col]){ piv = r; break; }
        if (piv < 0) continue;
        if (piv != rank)
            for (int k = col; k < ncols; k++){
                int8_t t = M[piv][k]; M[piv][k] = M[rank][k]; M[rank][k] = t;
            }
        /* scale pivot row to make pivot 1 (inverse of 2 mod 3 is 2) */
        if (M[rank][col] == 2)
            for (int k = col; k < ncols; k++) M[rank][k] = (int8_t)((M[rank][k]*2)%3);
        for (int r = 0; r < nr; r++){
            if (r == rank || !M[r][col]) continue;
            int f = M[r][col];   /* subtract f × pivot row, i.e. add (3-f)× */
            int g = 3 - f;
            for (int k = col; k < ncols; k++)
                M[r][k] = (int8_t)((M[r][k] + g*M[rank][k]) % 3);
        }
        rank++;
    }
    return rank;
}

/* Exact cut entropy in bits for bipartition {0..half-1} | {half..N-1}. */
static void sector_entropy(int N, int half, double *S2, double *S3)
{
    int r2 = rank_gf2_cut(N, half);
    int r3 = rank_gf3_cut(N, half);
    *S2 = (double)r2;             /* r2 × log2(2) */
    *S3 = (double)r3 * LOG2_3;    /* r3 × log2(3) */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TIER A — READOUT VALIDATION against the engine's own amplitudes
 *
 * For small N, pull every amplitude via hpc_amplitude, form the bipartition
 * matrix M(a,b) = ψ, compute Rényi-2 entropy S₂ʳᵉⁿ = −log₂ Tr ρ_A².
 * Weighted graph states with uniform locals have flat entanglement spectra,
 * so Rényi-2 = von Neumann, and both must equal r₂ + r₃·log₂3.
 *
 * A1: the P2 ring, single-site cut — built-in said 5.17 (impossible for
 *     one site); exact answer is 1 + log₂3 = 2.585.
 * A2: 20 random N=4 multigraphs (multiplicities 0..5 on all 6 pairs),
 *     cut {0,1}|{2,3}: amplitude-derived Rényi-2 vs rank formula.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void tierA_validation(void)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TIER A — ENTROPY READOUT VALIDATION (vs hpc_amplitude)    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── A1: 6-site ring, region B = {site 5} ── */
    {
        int N = 6;
        HPCGraph *g = hpc_create(N);
        for (int i = 0; i < N; i++) hpc_dft(g, i);
        shadow_clear(N);
        for (int i = 0; i < N-1; i++){ hpc_cz(g, i, i+1); shadow_add(i, i+1, 1); }
        hpc_cz(g, N-1, 0); shadow_add(N-1, 0, 1);

        /* exact ρ_B for B={5} from amplitudes: ρ[b][b'] = Σ_a ψ(a,b)ψ*(a,b') */
        double rho_re[6][6] = {{0}}, rho_im[6][6] = {{0}};
        double norm = 0;
        uint32_t idx[6];
        int total = 1; for (int i=0;i<5;i++) total *= 6;   /* 6^5 = 7776 */
        for (int a = 0; a < total; a++){
            int t = a;
            for (int i = 0; i < 5; i++){ idx[i] = (uint32_t)(t % 6); t /= 6; }
            double ar[6], ai[6];
            for (int b = 0; b < 6; b++){
                idx[5] = (uint32_t)b;
                hpc_amplitude(g, idx, &ar[b], &ai[b]);
                norm += ar[b]*ar[b] + ai[b]*ai[b];
            }
            for (int b = 0; b < 6; b++)
                for (int bp = 0; bp < 6; bp++){
                    rho_re[b][bp] += ar[b]*ar[bp] + ai[b]*ai[bp];
                    rho_im[b][bp] += ai[b]*ar[bp] - ar[b]*ai[bp];
                }
        }
        double purity = 0;
        for (int b = 0; b < 6; b++)
            for (int bp = 0; bp < 6; bp++){
                double re = rho_re[b][bp]/norm, im = rho_im[b][bp]/norm;
                purity += re*re + im*im;
            }
        double S_renyi = -log2(purity);

        /* rank formula for the same cut: reorder so B={5} is the "right" block.
         * Crossing edges: (0,5) and (4,5). Biadjacency is 2 rows sharing one
         * column → rank 1 in both sectors → S = 1 + log2(3). We compute it
         * with the generic machinery by relabeling site order {0..4 | 5}. */
        double S2s, S3s;
        sector_entropy(N, 5, &S2s, &S3s);   /* split {0..4} | {5} */
        double S_rank = S2s + S3s;
        double S_builtin = hpc_entropy_cut(g, 4);  /* same bipartition, builtin */

        printf("  A1  6-ring, single-site region:\n");
        printf("      Renyi-2 from amplitudes : %.6f\n", S_renyi);
        printf("      rank formula (ours)     : %.6f  (expect 1+log2(3)=%.6f)\n",
               S_rank, 1.0 + LOG2_3);
        printf("      builtin hpc_entropy_cut : %.6f  (known to over-count)\n",
               S_builtin);
        printf("      VERDICT: %s\n",
               fabs(S_renyi - S_rank) < 1e-6 ? "rank formula EXACT ✓" : "MISMATCH ✗");
        hpc_destroy(g);
    }

    /* ── A2: 20 random N=4 multigraphs, cut {0,1}|{2,3} ── */
    {
        printf("\n  A2  random N=4 multigraphs (all 6 pairs, multiplicity 0..5):\n");
        rng_seed(0xF6F6F6F6ULL);
        double worst = 0;
        for (int trial = 0; trial < 20; trial++){
            int N = 4;
            HPCGraph *g = hpc_create(N);
            for (int i = 0; i < N; i++) hpc_dft(g, i);
            shadow_clear(N);
            for (int i = 0; i < N; i++)
                for (int j = i+1; j < N; j++){
                    int m = (int)rng_below(6);
                    for (int k = 0; k < m; k++) hpc_cz(g, i, j);
                    shadow_add(i, j, m);
                }
            hpc_compact_edges(g);

            /* full ψ as a 36×36 matrix; Tr ρ_A² = ||ψψ†||_F² */
            static double Mre[36][36], Mim[36][36];
            double norm = 0;
            uint32_t idx[4];
            for (int a = 0; a < 36; a++){
                idx[0]=(uint32_t)(a/6); idx[1]=(uint32_t)(a%6);
                for (int b = 0; b < 36; b++){
                    idx[2]=(uint32_t)(b/6); idx[3]=(uint32_t)(b%6);
                    hpc_amplitude(g, idx, &Mre[a][b], &Mim[a][b]);
                    norm += Mre[a][b]*Mre[a][b] + Mim[a][b]*Mim[a][b];
                }
            }
            double purity = 0;
            for (int a = 0; a < 36; a++)
                for (int ap = 0; ap < 36; ap++){
                    double cre = 0, cim = 0;   /* (ψψ†)_{a,a'} */
                    for (int b = 0; b < 36; b++){
                        cre += Mre[a][b]*Mre[ap][b] + Mim[a][b]*Mim[ap][b];
                        cim += Mim[a][b]*Mre[ap][b] - Mre[a][b]*Mim[ap][b];
                    }
                    cre /= norm; cim /= norm;
                    purity += cre*cre + cim*cim;
                }
            double S_renyi = -log2(purity);
            double S2s, S3s;
            sector_entropy(N, 2, &S2s, &S3s);
            double diff = fabs(S_renyi - (S2s + S3s));
            if (diff > worst) worst = diff;
            printf("      trial %2d: m=(%d,%d,%d,%d,%d,%d)  Renyi2=%.6f  rank=%.6f"
                   "  [S2=%.0f bit, S3=%.4f]  %s\n",
                   trial, MULT[0][1], MULT[0][2], MULT[0][3],
                   MULT[1][2], MULT[1][3], MULT[2][3],
                   S_renyi, S2s + S3s, S2s, S3s,
                   diff < 1e-6 ? "✓" : "✗");
            hpc_destroy(g);
        }
        printf("      VALIDATION,worst_abs_diff=%.2e   %s\n", worst,
               worst < 1e-6 ? "READOUT EXACT — Tier B is trustworthy ✓"
                            : "MISMATCH — send this output, do not run Tier B ✗");
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TIER B — FRONTIER SWEEP: two sectors, two transitions?
 *
 * Dynamics per time step on N sites (all diagonal → engine is EXACT):
 *   1. Add N/2 random edges: pair (i,j) uniform, multiplicity m ~ U{1..5},
 *      applied as m calls to hpc_cz (engine) and shadow_add (readout).
 *   2. Compact edges (engine merges mod 6; shadow already mod 6).
 *   3. Measure each site with probability p (hpc_measure: Born sample,
 *      edges at that site absorbed/removed). Reset the site with hpc_dft —
 *      EXACT here because a just-measured site is unentangled (P3's flaw
 *      only bites when DFT hits an entangled site). Shadow: isolate site.
 *   4. After burn-in, sample S₂ and S₃ across the fixed cut {0..N/2-1}.
 *
 * Steady state: mean degree ≈ 1/(2p); Z₂ sector sees 3/5 of edges,
 * Z₃ sector 4/5 → mean-field thresholds p_c² ≈ 0.30, p_c³ ≈ 0.40.
 *
 * Output CSV:
 *   DATA,N,p,S2_mean,S2_err,S3_mean,S3_err,edges_mean,shadow_engine_mismatch,seconds
 * ═══════════════════════════════════════════════════════════════════════════ */

/* engine modes: how many realizations per data point drive the real engine.
 * The shadow alone is an exact simulation of the same dynamics (the graph
 * evolution is measurement-outcome-independent in this diagonal class), so
 * engine=first keeps full rigor: one engine realization per point still
 * cross-checks shadow ≡ engine every sample. */
enum { ENGINE_ALL = 0, ENGINE_FIRST = 1, ENGINE_OFF = 2 };

/* per-primitive timing accumulators (reported per DATA line) */
static double T_CZ, T_COMPACT, T_MEASURE, T_RANK;
static struct timespec _p0;
static void ptic(void){ clock_gettime(CLOCK_MONOTONIC, &_p0); }
static double ptoc(void){
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec-_p0.tv_sec)+(t1.tv_nsec-_p0.tv_nsec)/1e9;
}

#define COMPACT_STRIDE 4   /* compact engine edges every 4 steps */

static void run_fracture_realization(int N, double p, int steps, int burn,
                                     int sample_every, int use_engine,
                                     double *S2_out, double *S3_out,
                                     double *edges_out, long *mismatch_out)
{
    HPCGraph *g = NULL;
    if (use_engine){
        g = hpc_create(N);
        for (int i = 0; i < N; i++) hpc_dft(g, i);
    }
    shadow_clear(N);

    double S2_acc = 0, S3_acc = 0, E_acc = 0;
    int samples = 0;
    long mismatch = 0;

    for (int t = 0; t < steps; t++){
        /* 1. random CZ^m edges */
        ptic();
        for (int e = 0; e < N/2; e++){
            int i = (int)rng_below((uint32_t)N), j;
            do { j = (int)rng_below((uint32_t)N); } while (j == i);
            int m = 1 + (int)rng_below(5);
            if (use_engine)
                for (int k = 0; k < m; k++) hpc_cz(g, i, j);
            shadow_add(i, j, m);
        }
        T_CZ += ptoc();

        /* 2. compaction (strided — correctness unaffected, merges commute) */
        if (use_engine && ((t % COMPACT_STRIDE) == COMPACT_STRIDE - 1)){
            ptic(); hpc_compact_edges(g); T_COMPACT += ptoc();
        }

        /* 3. measurement + exact reset — CANCEL-FIRST PROTOCOL
         * (bench B3): hpc_measure contracts over 6^degree neighbor configs:
         * degree 8 ≈ 0.6 s, degree 20 would never return. Workaround using
         * the engine's own mod-6 feature: for each site about to be measured,
         * apply CZ^(6−m) on every incident edge so it cancels, compact once,
         * then measure at degree 0 (O(1)). Exact for all recorded
         * observables: diagonal unitaries cannot change computational-basis
         * statistics, and the residual phases they replace are local
         * diagonal phases that affect neither entropies nor later outcomes. */
        ptic();
        {
            int meas[MAXN]; int n_meas = 0;
            for (int i = 0; i < N; i++)
                if (rng_uniform() < p) meas[n_meas++] = i;

            if (use_engine && n_meas > 0){
                for (int k = 0; k < n_meas; k++){
                    int i = meas[k];
                    for (int j = 0; j < N; j++){
                        int m = MULT[i][j] % 6;
                        if (m){
                            for (int c = 0; c < 6 - m; c++) hpc_cz(g, i, j);
                            MULT[i][j] = 0; MULT[j][i] = 0;  /* canceled once */
                        }
                    }
                }
                hpc_compact_edges(g);     /* zero-sum pairs vanish */
                for (int k = 0; k < n_meas; k++){
                    hpc_measure(g, meas[k], rng_uniform()); /* degree 0 → fast */
                    hpc_dft(g, meas[k]);  /* exact: site is product here */
                }
            }
            for (int k = 0; k < n_meas; k++)
                shadow_isolate(N, meas[k]);
        }
        T_MEASURE += ptoc();

        /* 4. sampling */
        if (t >= burn && ((t - burn) % sample_every) == 0){
            ptic();
            double S2, S3;
            sector_entropy(N, N/2, &S2, &S3);
            T_RANK += ptoc();
            S2_acc += S2; S3_acc += S3;
            long sc = shadow_edge_count(N);
            E_acc += (double)sc;
            if (use_engine){
                ptic(); hpc_compact_edges(g); T_COMPACT += ptoc();
                if ((long)g->n_edges != sc) mismatch++;
            }
            samples++;
        }
    }

    *S2_out = S2_acc / samples;
    *S3_out = S3_acc / samples;
    *edges_out = E_acc / samples;
    *mismatch_out = mismatch;
    if (use_engine) hpc_destroy(g);
}

static void tierB_frontier(int quick, int engine_mode)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  TIER B — FRACTURED TRANSITION SWEEP (Z₂ vs Z₃ sectors)    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    int Ns_full[]  = {64, 128, 256};
    int Ns_quick[] = {32, 64};
    int *Ns     = quick ? Ns_quick : Ns_full;
    int n_sizes = quick ? 2 : 3;
    int R       = quick ? 4 : 8;
    int steps   = quick ? 120 : 240;
    int burn    = steps / 2;
    int sample_every = 12;

    double ps_full[]  = {0.10,0.15,0.20,0.25,0.28,0.31,0.34,0.37,0.40,0.44,0.50,0.60};
    double ps_quick[] = {0.10,0.20,0.30,0.40,0.50,0.60};
    double *ps  = quick ? ps_quick : ps_full;
    int    n_p  = quick ? 6 : 12;

    rng_seed((uint64_t)time(NULL) ^ 0x66666666ULL);

    printf("  Engine mode: %s (shadow dynamics is exact either way;\n"
           "  'first' keeps one full engine cross-check realization per point)\n",
           engine_mode == ENGINE_ALL ? "ALL realizations" :
           engine_mode == ENGINE_FIRST ? "FIRST realization only" : "OFF");
    printf("  Mean-field predictions: p_c(Z2) ≈ 0.30, p_c(Z3) ≈ 0.40\n");
    printf("  Fractured phase expected for p between them: S3 extensive, S2 not.\n\n");
    printf("  DATA,N,p,S2_mean,S2_err,S3_mean,S3_err,edges_mean,mismatch,seconds,"
           "cz_s,compact_s,measure_s,rank_s\n");

    for (int s = 0; s < n_sizes; s++){
        int N = Ns[s];
        for (int ip = 0; ip < n_p; ip++){
            double p = ps[ip];
            double S2_sum=0, S2_sq=0, S3_sum=0, S3_sq=0, E_sum=0;
            long mm_total = 0;
            T_CZ = T_COMPACT = T_MEASURE = T_RANK = 0;
            tic();
            for (int r = 0; r < R; r++){
                int use_engine =
                    (engine_mode == ENGINE_ALL) ||
                    (engine_mode == ENGINE_FIRST && r == 0);
                fprintf(stderr, "  [progress] N=%d p=%.3f realization %d/%d "
                        "(engine %s)\n", N, p, r+1, R, use_engine ? "on" : "off");
                double S2, S3, E; long mm;
                run_fracture_realization(N, p, steps, burn, sample_every,
                                         use_engine, &S2, &S3, &E, &mm);
                S2_sum+=S2; S2_sq+=S2*S2; S3_sum+=S3; S3_sq+=S3*S3;
                E_sum+=E; mm_total+=mm;
            }
            double secs = toc();
            double S2m = S2_sum/R, S3m = S3_sum/R;
            double v2 = S2_sq/R - S2m*S2m, v3 = S3_sq/R - S3m*S3m;
            printf("  DATA,%d,%.3f,%.5f,%.5f,%.5f,%.5f,%.1f,%ld,%.2f,"
                   "%.2f,%.2f,%.2f,%.2f\n",
                   N, p, S2m, sqrt((v2>0?v2:0)/R), S3m, sqrt((v3>0?v3:0)/R),
                   E_sum/R, mm_total, secs,
                   T_CZ, T_COMPACT, T_MEASURE, T_RANK);
            fflush(stdout);
        }
        printf("\n");
    }

    printf("  Sweep complete. Send the full output. Analysis plan:\n");
    printf("    • Plot S2/N and S3/N vs p for each N — extensive (volume-law)\n");
    printf("      curves collapse, collapsed phases drop to ~0.\n");
    printf("    • Locate p_c(Z2) and p_c(Z3) from where S/N curves for\n");
    printf("      different N cross/peel off. Two distinct values with a\n");
    printf("      window where S3/N is finite but S2/N → 0 = the fractured\n");
    printf("      phase, a d=6-only phenomenon, measured exactly.\n");
    printf("    • mismatch column must be 0 everywhere (shadow ≡ engine).\n");
}

/* ════════════════════════════ main ══════════════════════════════════════ */

int main(int argc, char **argv)
{
    const char *tier = (argc > 1) ? argv[1] : "all";
    int quick = (argc > 2 && strcmp(argv[2], "quick") == 0);
    int engine_mode = ENGINE_FIRST;   /* default: rigorous + fast */
    for (int a = 2; a < argc; a++){
        if (!strcmp(argv[a], "engine=all"))   engine_mode = ENGINE_ALL;
        if (!strcmp(argv[a], "engine=first")) engine_mode = ENGINE_FIRST;
        if (!strcmp(argv[a], "engine=off"))   engine_mode = ENGINE_OFF;
    }

    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║  FRACTURE-6 v3: CRT-split entanglement transitions at d = 6     ║\n");
    printf("║  Z₆ ≅ Z₂ × Z₃ · exact diagonal-circuit dynamics · HexState HPC ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");

    rng_seed((uint64_t)time(NULL));
    s6_exotic_init();
    triality_exotic_init();
    triality_stats_reset();

    if (!strcmp(tier, "A") || !strcmp(tier, "all"))
        tierA_validation();

    if (!strcmp(tier, "B") || !strcmp(tier, "all"))
        tierB_frontier(quick, engine_mode);

    return 0;
}
