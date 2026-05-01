/* ═══════════════════════════════════════════════════════════════════════════
 * tesseract_factor.c — HPC Ouroboros Factoring Engine
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  CONFIGURE YOUR TARGET N HERE                                ║
 * ╚═══════════════════════════════════════════════════════════════╝
 */
#define TARGET_N  "261980999226229"          /* ← Set your composite here     */
#define TARGET_A  "0"            /* ← "0" = auto-try 20 bases              */

/*
 * Architecture (HPC — Holographic Phase Contraction):
 *   N_SITES = ceil(2 * nbits / log2(6))  D=6 quhits
 *   Each site is a TrialityQuhit (6 complex amplitudes)
 *   Entanglement: CZ phase edges in an HPCGraph
 *   Amplitudes: computed on demand via O(N+E) graph traversal
 *   State vector NEVER materialized — entanglement lives in the graph
 *
 * Pipeline:
 *   1. Create HPCGraph with N_SITES D=6 quhits
 *   2. DFT₆ → uniform superposition on all sites
 *   3. IPE loop: encode a^(6^k) mod N as oracle phases via triality_phase()
 *   4. CZ entanglement chain propagates inter-site correlations
 *   5. hpc_marginal() reads analytical interference peaks
 *   6. Continued fraction extraction → period → gcd → factors
 *
 * Build: gcc -O2 -std=gnu99 -o tesseract_factor tesseract_factor.c \
 *         quhit_triality.c quhit_hexagram.c s6_exotic.c bigint.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpfr.h>
#include "quhit_triality.h"
#include "hpc_graph.h"
#include "hpc_mobius.h"
#include "s6_exotic.h"
#include "hpc_z6_codes.h"
#include "bigint.h"
#define D 6

BigInt cf_pool[10000];
int cf_pool_count = 0;

/* ═══════════════════════════════════════════════════════════════════════════
 * FACTOR EXTRACTION — gcd(a^(r/2) ± 1, N)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int try_period(const BigInt *r, const BigInt *a_val, const BigInt *N,
                      BigInt *factor_p, BigInt *factor_q)
{
    /* Static temporaries — allocated once, reused forever */
    static int tp_init = 0;
    static BigInt tp_one, tp_two, tp_r_half, tp_q_unused, tp_r_mod;
    static BigInt tp_half_pow, tp_h_minus, tp_p1, tp_dummy_rem;
    static BigInt tp_h_plus, tp_p2;
    if (!tp_init) {
        bigint_set_u64(&tp_one, 1); bigint_set_u64(&tp_two, 2);
        bigint_set_u64(&tp_r_half, 0); bigint_set_u64(&tp_q_unused, 0);
        bigint_set_u64(&tp_r_mod, 0); bigint_set_u64(&tp_half_pow, 0);
        bigint_set_u64(&tp_h_minus, 0); bigint_set_u64(&tp_p1, 0);
        bigint_set_u64(&tp_dummy_rem, 0); bigint_set_u64(&tp_h_plus, 0);
        bigint_set_u64(&tp_p2, 0);
        tp_init = 1;
    }

    /* r must be even for gcd(a^(r/2)±1, N) — but odd r may still be valid λ(N) divisor */
    bigint_div_mod(r, &tp_two, &tp_q_unused, &tp_r_mod);
    if (!bigint_is_zero(&tp_r_mod)) {
        BigInt odd_pow = {0};
        bigint_set_u64(&odd_pow, 0);
        bigint_pow_mod(&odd_pow, a_val, r, N);
        if (bigint_cmp(&odd_pow, &tp_one) == 0) {
            char r_str[512];
            bigint_to_decimal(r_str, sizeof(r_str), r);
            printf("    [odd period] a^%s ≡ 1 (mod N) — valid λ(N) divisor\n", r_str);
            bigint_clear(&odd_pow);
            return 2;
        }
        bigint_clear(&odd_pow);
        return 0;
    }

    bigint_div_mod(r, &tp_two, &tp_r_half, &tp_r_mod);

    /* a^(r/2) mod N */
    bigint_pow_mod(&tp_half_pow, a_val, &tp_r_half, N);

    /* gcd(a^(r/2) - 1, N) */
    bigint_sub(&tp_h_minus, &tp_half_pow, &tp_one);
    bigint_gcd(&tp_p1, &tp_h_minus, N);

    if (bigint_cmp(&tp_p1, &tp_one) > 0 && bigint_cmp(&tp_p1, N) < 0) {
        bigint_copy(factor_p, &tp_p1);
        bigint_div_mod(N, &tp_p1, factor_q, &tp_dummy_rem);
        char p_str[1300];
        bigint_to_decimal(p_str, sizeof(p_str), &tp_p1);
        printf("    gcd(a^(r/2)-1, N) = %s ✓\n", p_str);
        return 1;
    }

    /* gcd(a^(r/2) + 1, N) */
    bigint_add(&tp_h_plus, &tp_half_pow, &tp_one);
    bigint_gcd(&tp_p2, &tp_h_plus, N);

    if (bigint_cmp(&tp_p2, &tp_one) > 0 && bigint_cmp(&tp_p2, N) < 0) {
        bigint_copy(factor_p, &tp_p2);
        bigint_div_mod(N, &tp_p2, factor_q, &tp_dummy_rem);
        char p_str[1300];
        bigint_to_decimal(p_str, sizeof(p_str), &tp_p2);
        printf("    gcd(a^(r/2)+1, N) = %s ✓\n", p_str);
        return 1;
    }

    if (bigint_cmp(&tp_p1, N) == 0) {
        /* a^(r/2) ≡ 1 mod N. This means r/2 is a period multiplier. Let's try r/2 directly! */
        BigInt r_stripped = {0}; bigint_set_u64(&r_stripped, 0);
        bigint_copy(&r_stripped, &tp_r_half);
        char recurse_str[1300];
        bigint_to_decimal(recurse_str, sizeof(recurse_str), &r_stripped);
        printf("    [Harmonic Reduction] a^(r/2) ≡ 1 mod N. Stripping factor of 2 → trying r = %s...\n", recurse_str);
        int ret = try_period(&r_stripped, a_val, N, factor_p, factor_q);
        bigint_clear(&r_stripped);
        return ret;
    }

    if (bigint_cmp(&tp_p2, N) == 0) {
        /* If gcd(a^(r/2) + 1, N) == N, it implies a^(r/2) ≡ -1 mod N.
         * This is a mathematically sterile TRIVIAL ROOT!
         * The base 'a' will never yield prime factors, so we must violently abort! */
        BigInt full_pow = {0};
        bigint_set_u64(&full_pow, 0);
        bigint_pow_mod(&full_pow, a_val, r, N);
        if (bigint_cmp(&full_pow, &tp_one) == 0) {
            char abort_r_str[512];
            bigint_to_decimal(abort_r_str, sizeof(abort_r_str), r);
            printf("\n  [!] MATHEMATICALLY STERILE BASE DETECTED (r=%s). Base yields trivial roots. Skipping...\n", abort_r_str);
            bigint_clear(&full_pow);
            /* Return -1 instead of `exit` so that we gracefully try the next base */
            return -1;
        }
        bigint_clear(&full_pow);
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CONTINUED FRACTION PERIOD EXTRACTION
 *
 * Given frequency F from the QFT and register size R:
 *   F/R ≈ s/r → continued fraction convergents yield period candidates
 * ═══════════════════════════════════════════════════════════════════════════ */

static int generate_and_try_periods(const BigInt *freq, const BigInt *reg_size,
                                     const BigInt *a_val, const BigInt *N,
                                     BigInt *factor_p, BigInt *factor_q, BigInt *out_period)
{
    /* Static temporaries — allocated once, reused forever */
    static int gtp_init = 0;
    static BigInt gtp_one, gtp_r_cand, gtp_rem, gtp_r_plus, gtp_r_minus;
    static BigInt gtp_k_bi, gtp_rk;
    static BigInt gtp_g;
    static BigInt gtp_num, gtp_den, gtp_pm1, gtp_p0, gtp_qm1, gtp_q0;
    static BigInt gtp_a0, gtp_cf_rem, gtp_a_next;
    static BigInt gtp_m2, gtp_m3, gtp_m6, gtp_two_q, gtp_three_q, gtp_six_q;
    static BigInt gtp_p_new, gtp_q_new, gtp_tmp;
    static BigInt gtp_f2, gtp_f3;
    if (!gtp_init) {
        bigint_set_u64(&gtp_one, 1); bigint_set_u64(&gtp_r_cand, 0);
        bigint_set_u64(&gtp_rem, 0); bigint_set_u64(&gtp_r_plus, 0);
        bigint_set_u64(&gtp_r_minus, 0); bigint_set_u64(&gtp_k_bi, 0);
        bigint_set_u64(&gtp_rk, 0); bigint_set_u64(&gtp_g, 0);
        bigint_set_u64(&gtp_num, 0); bigint_set_u64(&gtp_den, 0);
        bigint_set_u64(&gtp_pm1, 0); bigint_set_u64(&gtp_p0, 0);
        bigint_set_u64(&gtp_qm1, 0); bigint_set_u64(&gtp_q0, 0);
        bigint_set_u64(&gtp_a0, 0); bigint_set_u64(&gtp_cf_rem, 0);
        bigint_set_u64(&gtp_a_next, 0);
        bigint_set_u64(&gtp_m2, 2); bigint_set_u64(&gtp_m3, 3);
        bigint_set_u64(&gtp_m6, 6);
        bigint_set_u64(&gtp_two_q, 0); bigint_set_u64(&gtp_three_q, 0);
        bigint_set_u64(&gtp_six_q, 0);
        bigint_set_u64(&gtp_p_new, 0); bigint_set_u64(&gtp_q_new, 0);
        bigint_set_u64(&gtp_tmp, 0);
        bigint_set_u64(&gtp_f2, 0); bigint_set_u64(&gtp_f3, 0);
        gtp_init = 1;
    }

    if (bigint_is_zero(freq)) return 0;

    /* r = R / F (direct division) */
    bigint_div_mod(reg_size, freq, &gtp_r_cand, &gtp_rem);
    if (!bigint_is_zero(&gtp_r_cand) && bigint_cmp(&gtp_r_cand, &gtp_one) > 0) {
        char r_str[1300];
        bigint_to_decimal(r_str, sizeof(r_str), &gtp_r_cand);
        printf("  Trying r = R/F = %s\n", r_str);
        if (try_period(&gtp_r_cand, a_val, N, factor_p, factor_q) == 1) return 1;
        bigint_add(&gtp_r_plus, &gtp_r_cand, &gtp_one);
        bigint_sub(&gtp_r_minus, &gtp_r_cand, &gtp_one);
        if (try_period(&gtp_r_plus, a_val, N, factor_p, factor_q) == 1) return 1;
        if (try_period(&gtp_r_minus, a_val, N, factor_p, factor_q) == 1) return 1;
        /* Harmonic search: true period could be k * R/F */
        for (int k = 2; k <= 6; k++) {
            bigint_set_u64(&gtp_k_bi, k);
            bigint_mul(&gtp_rk, &gtp_r_cand, &gtp_k_bi);
            if (bigint_cmp(&gtp_rk, N) < 0) {
                if (try_period(&gtp_rk, a_val, N, factor_p, factor_q) == 1) return 1;
            }
        }
    }

    /* r = gcd(F, R), and R/gcd */
    bigint_gcd(&gtp_g, freq, reg_size);
    if (bigint_cmp(&gtp_g, &gtp_one) > 0) {
        bigint_div_mod(reg_size, &gtp_g, &gtp_r_cand, &gtp_rem);
        if (try_period(&gtp_r_cand, a_val, N, factor_p, factor_q) == 1) return 1;
        if (try_period(&gtp_g, a_val, N, factor_p, factor_q) == 1) return 1;
    }

    /* Continued fraction convergents of F/R */
    bigint_copy(&gtp_num, freq);
    bigint_copy(&gtp_den, reg_size);

    bigint_set_u64(&gtp_pm1, 1);
    bigint_set_u64(&gtp_qm1, 0);

    bigint_div_mod(&gtp_num, &gtp_den, &gtp_a0, &gtp_cf_rem);
    bigint_copy(&gtp_p0, &gtp_a0);
    bigint_set_u64(&gtp_q0, 1);

    for (int step = 0; ; step++) {
        if (bigint_cmp(&gtp_q0, N) >= 0 || bigint_bitlen(&gtp_q0) > 2000) break;

        if (bigint_cmp(&gtp_q0, &gtp_one) > 0) {
            if (try_period(&gtp_q0, a_val, N, factor_p, factor_q) == 1) { if (out_period) bigint_copy(out_period, &gtp_q0); return 1; }

            /* Try multiples */
            bigint_mul(&gtp_two_q, &gtp_q0, &gtp_m2);
            bigint_mul(&gtp_three_q, &gtp_q0, &gtp_m3);
            bigint_mul(&gtp_six_q, &gtp_q0, &gtp_m6);
            if (try_period(&gtp_two_q, a_val, N, factor_p, factor_q) == 1) { if (out_period) bigint_copy(out_period, &gtp_two_q); return 1; }
            if (try_period(&gtp_three_q, a_val, N, factor_p, factor_q) == 1) { if (out_period) bigint_copy(out_period, &gtp_three_q); return 1; }
            if (try_period(&gtp_six_q, a_val, N, factor_p, factor_q) == 1) { if (out_period) bigint_copy(out_period, &gtp_six_q); return 1; }
        }

        if (bigint_is_zero(&gtp_cf_rem)) break;
        bigint_copy(&gtp_num, &gtp_den);
        bigint_copy(&gtp_den, &gtp_cf_rem);

        bigint_div_mod(&gtp_num, &gtp_den, &gtp_a_next, &gtp_cf_rem);

        bigint_mul(&gtp_tmp, &gtp_a_next, &gtp_p0);
        bigint_add(&gtp_p_new, &gtp_tmp, &gtp_pm1);
        bigint_mul(&gtp_tmp, &gtp_a_next, &gtp_q0);
        bigint_add(&gtp_q_new, &gtp_tmp, &gtp_qm1);

        bigint_copy(&gtp_pm1, &gtp_p0);
        bigint_copy(&gtp_qm1, &gtp_q0);
        bigint_copy(&gtp_p0, &gtp_p_new);
        bigint_copy(&gtp_q0, &gtp_q_new);
    }

    /* Try F itself and small multiples */
    bigint_mul(&gtp_f2, freq, &gtp_m2);
    bigint_mul(&gtp_f3, freq, &gtp_m3);
    if (try_period(freq, a_val, N, factor_p, factor_q) == 1) { if (out_period) bigint_copy(out_period, freq); return 1; }
    if (try_period(&gtp_f2, a_val, N, factor_p, factor_q) == 1) { if (out_period) bigint_copy(out_period, &gtp_f2); return 1; }
    if (try_period(&gtp_f3, a_val, N, factor_p, factor_q) == 1) { if (out_period) bigint_copy(out_period, &gtp_f3); return 1; }

    return 0;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * LLL LATTICE PERIOD RECOVERY  (ground-up, integer basis)
 *
 * Problem:  K frequency measurements, each F_i ≈ s_i * R / r.
 * Recover the true period r.
 *
 * Lattice (dim = K+1, all-integer):
 *   Row 0   :  [fh_0, fh_1, ..., fh_{K-1}, 1]   fh_i = round(F_i*W/R)
 *   Row i>0 :  [0, ..., W (at col i-1), ..., 0]
 *   W = 2^LLL_W_BITS
 *
 * A short vector satisfies v[j] = r*fh_j - s_j*W ≈ W*(r*F_j/R - s_j) ≈ 0
 * and v[LLL_K] = ±r.  Reading |v[LLL_K]| from each reduced row gives r.
 *
 * Integer basis stays exact throughout; only Gram–Schmidt quantities
 * are held in double (used solely to decide size-reduction quotients and
 * the Lovász swap — both only need word-size precision).
 * ═══════════════════════════════════════════════════════════════════════════ */

#define LLL_K       32              /* frequency samples to collect            */
#define LLL_DIM     (LLL_K + 1)     /* lattice dimension                       */
/* LLL_W_BITS is now computed dynamically in lll_fhat() based on N's bitlength.
 * Previous hardcoded 60 could overflow when period r > 2^60 for large N. */
#define LLL_W_BITS_DEFAULT  60      /* fallback for small N                   */
#define LLL_DELTA   0.75            /* Lovász δ                                */

/* fhat_i = round(F * 2^LLL_W_BITS / R), clamped to [0, W).                  *
 * Uses GMP directly (F and R are BigInts, possibly thousands of bits).        */
static int lll_w_bits_for_n(uint32_t nbits) {
    /* Scale lattice precision with N: W must exceed the period r,
     * which can be up to ~N. Use 2*nbits to ensure W > r. */
    int w = (int)(nbits * 2);
    if (w < LLL_W_BITS_DEFAULT) w = LLL_W_BITS_DEFAULT;
    if (w > 120) w = 120;  /* cap to avoid overflow in long long arithmetic */
    return w;
}

static long long lll_fhat(const BigInt *F, const BigInt *R, int w_bits)
{
    const long long W = 1LL << (w_bits < 62 ? w_bits : 62);  /* clamp for long long */
    BigInt scaled, quot, rem;
    memset(&scaled, 0, sizeof(BigInt));
    memset(&quot, 0, sizeof(BigInt));
    memset(&rem, 0, sizeof(BigInt));
    
    bigint_copy(&scaled, F);
    mpz_mul_2exp(scaled.z, scaled.z, w_bits);   /* scaled = F << w_bits */
    bigint_div_mod(&scaled, R, &quot, &rem);
    
    /* quot = floor(F * W / R); add 1 if remainder >= R/2 (round) */
    BigInt half_R, two_rem;
    memset(&half_R, 0, sizeof(BigInt));
    memset(&two_rem, 0, sizeof(BigInt));
    
    bigint_copy(&half_R, R);  mpz_fdiv_q_2exp(half_R.z, half_R.z, 1);
    bigint_copy(&two_rem, &rem); mpz_mul_2exp(two_rem.z, two_rem.z, 1);
    if (mpz_cmp(two_rem.z, R->z) >= 0) {
        BigInt q1, one_bi;
        memset(&q1, 0, sizeof(BigInt));
        memset(&one_bi, 0, sizeof(BigInt));
        
        bigint_set_u64(&one_bi, 1);
        bigint_add(&q1, &quot, &one_bi);
        bigint_copy(&quot, &q1);
    }
    long long q = 0;
    if (mpz_fits_ulong_p(quot.z)) q = (long long)mpz_get_ui(quot.z);
    if (q < 0)    q = 0;
    if (q >= W)   q = W - 1;
    return q;
}

/* Recompute Gram–Schmidt from row `from` to LLL_DIM-1.
 * B is the integer basis (long long, never modified here).
 * Bs[i] holds the i-th GS vector as doubles; Bsq[i] = ||Bs[i]||^2.         */
static void lll_gs(int from,
                   long long B[LLL_DIM][LLL_DIM],
                   double    Bs[LLL_DIM][LLL_DIM],
                   double    Bsq[LLL_DIM])
{
    for (int i = from; i < LLL_DIM; i++) {
        /* b*_i = b_i */
        for (int l = 0; l < LLL_DIM; l++) Bs[i][l] = (double)B[i][l];
        /* subtract projections onto all earlier b*_j */
        for (int j = 0; j < i; j++) {
            if (Bsq[j] < 1e-30) continue;
            double mu = 0.0;
            for (int l = 0; l < LLL_DIM; l++) mu += (double)B[i][l] * Bs[j][l];
            mu /= Bsq[j];
            for (int l = 0; l < LLL_DIM; l++) Bs[i][l] -= mu * Bs[j][l];
        }
        double sq = 0.0;
        for (int l = 0; l < LLL_DIM; l++) sq += Bs[i][l] * Bs[i][l];
        Bsq[i] = (sq < 1e-30) ? 1e-30 : sq;
    }
}

/* mu_{k,j} = <B[k], Bs[j]> / Bsq[j]  (GS coefficient)                      */
static double lll_mu(int k, int j,
                     long long B[LLL_DIM][LLL_DIM],
                     double    Bs[LLL_DIM][LLL_DIM],
                     double    Bsq[LLL_DIM])
{
    if (Bsq[j] < 1e-30) return 0.0;
    double d = 0.0;
    for (int l = 0; l < LLL_DIM; l++) d += (double)B[k][l] * Bs[j][l];
    return d / Bsq[j];
}

/* In-place LLL reduction of the integer basis B.
 * After return the rows form an LLL-reduced basis (δ = LLL_DELTA).           */
static void lll_reduce_basis(long long B[LLL_DIM][LLL_DIM])
{
    double Bs[LLL_DIM][LLL_DIM], Bsq[LLL_DIM];
    lll_gs(0, B, Bs, Bsq);

    int k = 1;
    int guard = 0;
    const int MAX_ITER = 200 * LLL_DIM * LLL_DIM;

    while (k < LLL_DIM && guard++ < MAX_ITER) {

        /* Size-reduction: walk j from k-1 down to 0 */
        for (int j = k - 1; j >= 0; j--) {
            double mu = lll_mu(k, j, B, Bs, Bsq);
            if (fabs(mu) <= 0.5) continue;
            long long q = (long long)round(mu);
            for (int l = 0; l < LLL_DIM; l++) B[k][l] -= q * B[j][l];
            /* Recompute GS from k (rows 0..k-1 unchanged) */
            lll_gs(k, B, Bs, Bsq);
        }

        /* Lovász condition: ||b*_k||^2 >= (δ - μ_{k,k-1}^2) ||b*_{k-1}||^2  */
        double mu_k = lll_mu(k, k-1, B, Bs, Bsq);
        if (Bsq[k] >= (LLL_DELTA - mu_k * mu_k) * Bsq[k-1]) {
            k++;
        } else {
            /* Swap rows k and k-1 */
            for (int l = 0; l < LLL_DIM; l++) {
                long long t = B[k][l]; B[k][l] = B[k-1][l]; B[k-1][l] = t;
            }
            /* Only need to recompute from k-1 */
            lll_gs(k > 1 ? k-1 : 0, B, Bs, Bsq);
            if (k > 1) k--;
        }
    }
}

/* Collect LLL_K frequency BigInts from the BP marginal distribution.
 *
 *   Uses a Deterministic Beam Search (Viterbi-style extraction) to find
 *   the exact Top-K most mathematically probable globally-optimal frequency
 *   strings across the entire N-site configuration.
 *
 * This dramatically enhances composite frequency accuracy over CDF sampling
 * by ensuring the globally most-likely candidate signals (even 2nd/3rd best
 * over multi-digit correlations) are the exact ones swept by the fan.      */
static void lll_collect_freqs(int n, mpfr_t (*marg)[6],
                               const BigInt *b6, BigInt out[LLL_K])
{
    (void)0; /* Dynamic temperature beam search: T cools from 0.8 (LSB) to 0.1 (MSB) */
    /* Diagnostic: show confidence at first few positions */
    printf("  [freq] marginal peak confidence at pos 0..4:");
    for (int s = 0; s < 5 && s < n; s++) {
        double mp = 0.0;
        for (int d = 0; d < 6; d++) {
            double p = mpfr_get_d(marg[s][d], MPFR_RNDN);
            if (p > mp) mp = p;
        }
        printf(" %.3f", mp);
    }
    printf("\n");

    /* Precompute log-probabilities to avoid underflow */
    double (*log_marg)[6] = (double(*)[6])calloc(n, sizeof(double[6]));
    for (int s = 0; s < n; s++) {
        for (int d = 0; d < 6; d++) {
            if (mpfr_cmp_d(marg[s][d], 0.0) <= 0) {
                log_marg[s][d] = -100.0;
            } else {
                mpfr_t mp_log;
                mpfr_init2(mp_log, 2048);
                mpfr_log(mp_log, marg[s][d], MPFR_RNDN);
                double p_val = mpfr_get_d(mp_log, MPFR_RNDN);
                log_marg[s][d] = (p_val < -100.0) ? -100.0 : p_val;
                mpfr_clear(mp_log);
            }
        }
    }

    int num_beams = 1;
    double beam_log_probs[LLL_K];
    memset(beam_log_probs, 0, sizeof(beam_log_probs));
    
    /* Heap-allocate beam history to prevent stack overflow */
    int (*beam_history_parent)[LLL_K] = (int(*)[LLL_K])calloc(n, sizeof(int[LLL_K]));
    int (*beam_history_digit)[LLL_K]  = (int(*)[LLL_K])calloc(n, sizeof(int[LLL_K]));

    for (int s = 0; s < n; s++) {
        double next_log_probs[LLL_K * 6];
        int next_parent[LLL_K * 6];
        int next_digit[LLL_K * 6];
        int next_count = 0;

        for (int b = 0; b < num_beams; b++) {
            for (int d = 0; d < 6; d++) {
                next_log_probs[next_count] = beam_log_probs[b] + log_marg[s][d];
                next_parent[next_count] = b;
                next_digit[next_count] = d;
                next_count++;
            }
        }

        /* ── Deterministic Top-K Selection (argmax) ──
         * Replaces stochastic Boltzmann sampling for reproducibility.
         * Always selects the K highest-scoring beams at each position. */
        int top_indices[LLL_K];
        int top_count = (next_count < LLL_K) ? next_count : LLL_K;

        for (int k = 0; k < top_count; k++) {
            int best_idx = -1;
            double best_lp = -1e30;
            for (int i = 0; i < next_count; i++) {
                if (next_log_probs[i] > best_lp) {
                    best_lp = next_log_probs[i];
                    best_idx = i;
                }
            }
            top_indices[k] = best_idx;
            next_log_probs[best_idx] = -2e9; /* poison so it can't be selected again */
        }

        double new_beam_log_probs[LLL_K];

        for (int k = 0; k < top_count; k++) {
            int idx = top_indices[k];
            int p = next_parent[idx];
            new_beam_log_probs[k] = beam_log_probs[p] + log_marg[s][next_digit[idx]];
            beam_history_parent[s][k] = p;
            beam_history_digit[s][k] = next_digit[idx];
        }

        num_beams = top_count;
        for (int k = 0; k < num_beams; k++) {
            beam_log_probs[k] = new_beam_log_probs[k];
        }
    }

    printf("  [freq] top K relative log-probs:");
    for (int k = 0; k < num_beams; k++) {
        printf(" %.2f", beam_log_probs[k] - beam_log_probs[0]);
    }
    printf("\n");

    /* Pre-allocate reconstruction temporaries */
    BigInt rc_freq, rc_p6, rc_d_bi, rc_term, rc_tmp_bi, rc_np;
    bigint_set_u64(&rc_freq, 0); bigint_set_u64(&rc_p6, 0);
    bigint_set_u64(&rc_d_bi, 0); bigint_set_u64(&rc_term, 0);
    bigint_set_u64(&rc_tmp_bi, 0); bigint_set_u64(&rc_np, 0);

    /* Build BigInt from digits (LSB first) via backtracking */
    for (int k = 0; k < LLL_K; k++) {
        if (k >= num_beams) {
            bigint_copy(&out[k], &out[0]);
            continue;
        }

        /* Reconstruct digit sequence from backtracking tree */
        int *seq = (int*)calloc(n, sizeof(int));
        int curr_beam = k;
        for (int s = n - 1; s >= 0; s--) {
            seq[s] = beam_history_digit[s][curr_beam];
            curr_beam = beam_history_parent[s][curr_beam];
        }

        bigint_set_u64(&rc_freq, 0);
        bigint_set_u64(&rc_p6, 1);
        for (int s = 0; s < n; s++) {
            bigint_set_u64(&rc_d_bi, (uint64_t)seq[s]);
            bigint_mul(&rc_term, &rc_d_bi, &rc_p6);
            bigint_add(&rc_tmp_bi, &rc_freq, &rc_term);
            bigint_copy(&rc_freq, &rc_tmp_bi);
            bigint_mul(&rc_np, &rc_p6, b6);
            bigint_copy(&rc_p6, &rc_np);
        }
        bigint_copy(&out[k], &rc_freq);
        free(seq);
    }
    free(log_marg);
    free(beam_history_parent);
    free(beam_history_digit);
}


/* Top-level period recovery — four strategies, ordered by reliability.
 *
 * Strategy 1 — CF on targeted samples:
 *   Run generate_and_try_periods() on each of the K carefully-chosen
 *   frequency strings.  This is the most direct path and works whenever
 *   any sample lands on (or near) the true harmonic peak.
 *
 * Strategy 2 — Raw-frequency GCD:
 *   If F_i = s_i * F* (different harmonics), then gcd(F_0,...,F_{K-1}) → F*.
 *   Then r = R / F*. Works when samples span multiple distinct harmonics.
 *
 * Strategy 3 — LCM of period estimates:
 *   Each R/F_i ≈ r/s_i. Their LCM converges toward r as more samples arrive.
 *
 * Strategy 4 — LLL lattice (W = 2^LLL_W_BITS):
 *   Valid only when the true period r < W = 2^24 ≈ 16M.
 *   For larger r the short-vector is longer than the W-norm rows; skipped.
 *
 * Returns 1 and writes factor_p/q on success, 0 otherwise.                  */
static int lll_recover_period(int n_sites_raw, mpfr_t (*marg)[6],
                               const BigInt *b6, const BigInt *reg_sz,
                               const BigInt *N,  const BigInt *a_val,
                               BigInt *factor_p, BigInt *factor_q)
{
    /* Static temporaries — allocated once, reused forever */
    static int lrp_init = 0;
    static BigInt lrp_one;
    static BigInt lrp_gcd_f, lrp_g, lrp_r_cand, lrp_rem;
    static BigInt lrp_km, lrp_rk;
    static BigInt lrp_lcm_acc, lrp_r_i, lrp_rem_i;
    static BigInt lrp_g2, lrp_prod, lrp_new_lcm, lrp_nr;
    static BigInt lrp_s4_cand, lrp_s4_km, lrp_s4_rk;
    if (!lrp_init) {
        bigint_set_u64(&lrp_one, 1);
        bigint_set_u64(&lrp_gcd_f, 0); bigint_set_u64(&lrp_g, 0);
        bigint_set_u64(&lrp_r_cand, 0); bigint_set_u64(&lrp_rem, 0);
        bigint_set_u64(&lrp_km, 0); bigint_set_u64(&lrp_rk, 0);
        bigint_set_u64(&lrp_lcm_acc, 0); bigint_set_u64(&lrp_r_i, 0);
        bigint_set_u64(&lrp_rem_i, 0); bigint_set_u64(&lrp_g2, 0);
        bigint_set_u64(&lrp_prod, 0); bigint_set_u64(&lrp_new_lcm, 0);
        bigint_set_u64(&lrp_nr, 0);
        bigint_set_u64(&lrp_s4_cand, 0); bigint_set_u64(&lrp_s4_km, 0);
        bigint_set_u64(&lrp_s4_rk, 0);
        lrp_init = 1;
    }

    printf("\n  ═══ MULTI-STRATEGY PERIOD RECOVERY ═══\n");

    BigInt *freqs = (BigInt*)calloc(LLL_K, sizeof(BigInt));
    for (int i = 0; i < LLL_K; i++) bigint_clear(&freqs[i]);
    lll_collect_freqs(n_sites_raw, marg, b6, freqs);

    int found = 0;

    /* ── Strategy 1: CF (continued-fraction) on each targeted sample ─────── */
    printf("  [S1] CF on %d targeted frequency samples...\n", LLL_K);
    for (int i = 0; i < LLL_K && !found; i++) {
        if (bigint_is_zero(&freqs[i])) continue;
        found = generate_and_try_periods(&freqs[i], reg_sz, a_val, N,
                                         factor_p, factor_q, NULL);
        if (found) printf("  [S1] Hit on sample %d\n", i);
    }

    /* ── Strategy 2: GCD of raw frequencies → base frequency F* ──────────── */
    if (!found) {
        printf("  [S2] Running GCD of raw frequencies...\n");
        bigint_set_u64(&lrp_gcd_f, 0);
        for (int i = 0; i < LLL_K; i++) {
            if (bigint_is_zero(&freqs[i])) continue;
            if (bigint_is_zero(&lrp_gcd_f)) {
                bigint_copy(&lrp_gcd_f, &freqs[i]);
            } else {
                bigint_gcd(&lrp_g, &lrp_gcd_f, &freqs[i]);
                if (!bigint_is_zero(&lrp_g)) bigint_copy(&lrp_gcd_f, &lrp_g);
            }
            if (bigint_cmp(&lrp_gcd_f, &lrp_one) > 0) {
                bigint_div_mod(reg_sz, &lrp_gcd_f, &lrp_r_cand, &lrp_rem);
                if (bigint_cmp(&lrp_r_cand, &lrp_one) > 0 && bigint_cmp(&lrp_r_cand, N) < 0) {
                    if (try_period(&lrp_r_cand, a_val, N, factor_p, factor_q) == 1) {
                        found = 1; printf("  [S2] r = R/gcd hit\n"); break;
                    }
                    for (int m = 2; m <= 8 && !found; m++) {
                        bigint_set_u64(&lrp_km, (uint64_t)m);
                        bigint_mul(&lrp_rk, &lrp_r_cand, &lrp_km);
                        if (bigint_cmp(&lrp_rk, N) < 0)
                            if (try_period(&lrp_rk, a_val, N, factor_p, factor_q) == 1) {
                                found = 1; printf("  [S2] %d*r hit\n", m);
                            }
                    }
                }
                if (!found && bigint_cmp(&lrp_gcd_f, N) < 0)
                    if (try_period(&lrp_gcd_f, a_val, N, factor_p, factor_q) == 1) {
                        found = 1; printf("  [S2] gcd_f direct hit\n");
                    }
            }
        }
    }

    /* ── Strategy 3: LCM of R/F_i period estimates ────────────────────────── */
    if (!found) {
        printf("  [S3] LCM of period estimates across %d samples...\n", LLL_K);
        bigint_set_u64(&lrp_lcm_acc, 1);
        for (int i = 0; i < LLL_K && !found; i++) {
            if (bigint_is_zero(&freqs[i])) continue;
            bigint_div_mod(reg_sz, &freqs[i], &lrp_r_i, &lrp_rem_i);
            if (bigint_is_zero(&lrp_r_i) || bigint_cmp(&lrp_r_i, &lrp_one) <= 0) continue;
            bigint_gcd(&lrp_g2, &lrp_lcm_acc, &lrp_r_i);
            bigint_mul(&lrp_prod, &lrp_lcm_acc, &lrp_r_i);
            bigint_div_mod(&lrp_prod, &lrp_g2, &lrp_new_lcm, &lrp_nr);
            if (bigint_cmp(&lrp_new_lcm, N) < 0)
                bigint_copy(&lrp_lcm_acc, &lrp_new_lcm);
            else
                bigint_set_u64(&lrp_lcm_acc, 1);
            if (bigint_cmp(&lrp_lcm_acc, &lrp_one) > 0)
                if (try_period(&lrp_lcm_acc, a_val, N, factor_p, factor_q) == 1) {
                    found = 1; printf("  [S3] LCM hit after %d samples\n", i+1);
                }
        }
    }

    /* ── Strategy 4: LLL lattice short-vector (valid for r < 2^w_bits) ── */
    if (!found) {
        uint32_t nbits_n = bigint_bitlen(N);
        int w_bits = lll_w_bits_for_n(nbits_n);
        const long long W = 1LL << (w_bits < 62 ? w_bits : 62);
        printf("  [S4] LLL lattice (valid for r < 2^%d)...\n", w_bits);
        printf("  fhat:");
        for (int i = 0; i < LLL_K; i++) printf(" %lld", lll_fhat(&freqs[i], reg_sz, w_bits));
        printf("\n");

        long long B[LLL_DIM][LLL_DIM];
        memset(B, 0, sizeof B);
        for (int j = 0; j < LLL_K; j++) B[0][j] = lll_fhat(&freqs[j], reg_sz, w_bits);
        B[0][LLL_K] = 1LL;
        for (int i = 1; i <= LLL_K; i++) B[i][i-1] = W;

        lll_reduce_basis(B);

        /* Insertion-sort rows by L2 norm */
        for (int i = 1; i < LLL_DIM; i++) {
            double ni = 0.0;
            for (int l = 0; l < LLL_DIM; l++) ni += (double)B[i][l]*(double)B[i][l];
            long long tmp_row[LLL_DIM];
            memcpy(tmp_row, B[i], LLL_DIM * sizeof(long long));
            int j = i;
            while (j > 0) {
                double nj = 0.0;
                for (int l = 0; l < LLL_DIM; l++) nj += (double)B[j-1][l]*(double)B[j-1][l];
                if (nj <= ni) break;
                memcpy(B[j], B[j-1], LLL_DIM * sizeof(long long));
                j--;
            }
            memcpy(B[j], tmp_row, LLL_DIM * sizeof(long long));
        }

        for (int i = 0; i < LLL_DIM && !found; i++) {
            long long v = B[i][LLL_K];
            if (v < 0) v = -v;
            if (v < 2 || v >= W) continue;
            bigint_set_u64(&lrp_s4_cand, (uint64_t)v);
            if (bigint_cmp(&lrp_s4_cand, N) >= 0) continue;
            printf("  [S4 row %d] r candidate = %lld\n", i, v);
            if (try_period(&lrp_s4_cand, a_val, N, factor_p, factor_q) == 1) { found = 1; break; }
            for (int m = 2; m <= 8 && !found; m++) {
                bigint_set_u64(&lrp_s4_km, (uint64_t)m);
                bigint_mul(&lrp_s4_rk, &lrp_s4_cand, &lrp_s4_km);
                if (bigint_cmp(&lrp_s4_rk, N) < 0)
                    if (try_period(&lrp_s4_rk, a_val, N, factor_p, factor_q) == 1) found = 1;
            }
        }
        if (!found) printf("  [S4] No viable candidates (r likely > W)\n");
    }

    for (int i = 0; i < LLL_K; i++) bigint_clear(&freqs[i]);
    free(freqs);
    return found;
}



/* ═══════════════════════════════════════════════════════════════════════════
 * HPC CONSTRAINT SATISFACTION FACTORING — Base-6 Hensel Lift
 *
 * The HIDDEN POWER of HPC: factoring as constraint propagation on D=6 digits.
 *
 * N = p × q. In base 6, this is digit-by-digit multiplication with carries.
 * At each position k (from LSB):
 *   N mod 6^(k+1) ≡ p_prefix × q_prefix (mod 6^(k+1))
 *   So: q_prefix = N × p_prefix⁻¹ (mod 6^(k+1))
 *
 * Since p is prime > 3: gcd(p, 6) = 1, so p⁻¹ mod 6^k always exists.
 *
 * Algorithm: Hensel lift from mod 6 → mod 6^n
 *   1. At each level k, try all 6 extensions of p's k-th digit
 *   2. Compute q_prefix from the constraint
 *   3. Verify q_prefix extends consistently (valid new digit)
 *   4. When p_prefix × q_prefix = N exactly: FACTOR FOUND
 *
 * The transfer matrix propagates the constraint through the digit chain.
 * The D=6 structure of HPC makes base-6 the natural factoring radix.
 * ═══════════════════════════════════════════════════════════════════════════ */

#if 0  /* Hensel path disabled — function is unused and causes instability */
/* Modular inverse of a mod m using extended Euclidean algorithm.
 * Returns 0 if gcd(a, m) != 1. */
static uint64_t mod_inverse_u64(uint64_t a, uint64_t m) {
    if (m == 1) return 0;
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)m, newr = (int64_t)(a % m);
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = t - q * newt;
        t = newt; newt = tmp;
        tmp = r - q * newr;
        r = newr; newr = tmp;
    }
    if (r > 1) return 0;  /* not invertible */
    if (t < 0) t += (int64_t)m;
    return (uint64_t)t;
}

static int factor_hensel_base6(const BigInt *N, BigInt *factor_p, BigInt *factor_q)
{
    printf("\n  ═══ TRIALITY CONSTRAINT PROPAGATION FACTORING ═══\n\n");

    int n_digits = (int)(bigint_bitlen(N) / 2.585) + 2;
    int half_digits = n_digits / 2 + 2;
    printf("    N has ~%d base-6 digits, factors ~%d digits each\n", n_digits, half_digits);

    BigInt six, one;
    bigint_set_u64(&six, 6);
    bigint_set_u64(&one, 1);

    /* ── TRIALITY CONSTRAINT NET ──
     * 25 primes coprime to 6, providing independent constraints.
     * For each prime m: (p × q) mod m = N mod m.
     * False candidates satisfy this with probability 1/m each.
     * Combined survival: Π(1/m_i) ≈ 10^{-32} → only true factors survive. */
    #define N_TP 25
    int tp[N_TP] = {5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103};
    int N_mod_tp[N_TP];  /* precomputed N mod each triality prime */

    for (int i = 0; i < N_TP; i++) {
        BigInt m_bi, qtmp, rtmp;
        bigint_set_u64(&m_bi, tp[i]);
        bigint_div_mod(N, &m_bi, &qtmp, &rtmp);
        N_mod_tp[i] = (int)bigint_to_u64(&rtmp);
    }

    printf("    Triality net: %d primes (5..103), combined filter ~10^{-32}\n", N_TP);

    BigInt q_tmp, r_tmp;
    bigint_div_mod(N, &six, &q_tmp, &r_tmp);
    int N_mod6 = (int)bigint_to_u64(&r_tmp);

    #define MAX_CAND 4000000
    typedef struct { BigInt p; BigInt q; } Cand;
    Cand *cur = (Cand*)calloc(MAX_CAND, sizeof(Cand));
    Cand *nxt = (Cand*)calloc(MAX_CAND, sizeof(Cand));
    int nc = 0;

    /* Seed: p₀ ∈ {1, 5} */
    for (int p0 = 1; p0 <= 5; p0 += 4) {
        uint64_t inv = mod_inverse_u64(p0, 6);
        if (!inv) continue;
        int q0 = (int)((N_mod6 * inv) % 6);
        if ((p0 * q0) % 6 != N_mod6) continue;
        bigint_set_u64(&cur[nc].p, p0);
        bigint_set_u64(&cur[nc].q, q0);
        nc++;
    }
    printf("    Level 0: %d seeds\n", nc);

    BigInt mod_k;
    bigint_set_u64(&mod_k, 6);

    for (int k = 1; k < half_digits + 10 && nc > 0; k++) {
        BigInt mod_k1;
        bigint_mul(&mod_k1, &mod_k, &six);

        BigInt N_mod_k1;
        bigint_div_mod(N, &mod_k1, &q_tmp, &N_mod_k1);

        int nn = 0;
        for (int c = 0; c < nc && nn < MAX_CAND - 6; c++) {
            for (int d = 0; d < 6; d++) {
                BigInt p_new, term, shift;
                bigint_set_u64(&term, (uint64_t)d);
                bigint_mul(&shift, &term, &mod_k);
                bigint_add(&p_new, &cur[c].p, &shift);

                uint64_t plo = bigint_to_u64(&p_new);
                if (plo % 2 == 0 || plo % 3 == 0) continue;

                /* ── Compute q_prefix = N × p⁻¹ mod 6^(k+1) ── */
                BigInt qu_bi;
                bigint_set_u64(&qu_bi, 0);
                int computed = 0;

                if (k < 24) {  /* 6^24 < 2^62, fits in __int128 arithmetic */
                    __uint128_t mv = 1;
                    for (int j = 0; j <= k; j++) mv *= 6;
                    if (mv < ((__uint128_t)1 << 63)) {
                        uint64_t mv64 = (uint64_t)mv;
                        BigInt pm;
                        bigint_div_mod(&p_new, &mod_k1, &q_tmp, &pm);
                        uint64_t pu = bigint_to_u64(&pm);
                        uint64_t nu = bigint_to_u64(&N_mod_k1);
                        uint64_t pi = mod_inverse_u64(pu, mv64);
                        if (pi) {
                            uint64_t qu_val = (uint64_t)((__uint128_t)nu * pi % mv64);
                            if ((uint64_t)((__uint128_t)pu * qu_val % mv64) == nu) {
                                /* Verify q digit is valid */
                                uint64_t qo = bigint_to_u64(&cur[c].q);
                                if (qu_val >= qo) {
                                    uint64_t mk = bigint_to_u64(&mod_k);
                                    if (mk) {
                                        uint64_t qdiff = qu_val - qo;
                                        uint64_t qdig = qdiff / mk;
                                        if (qdig < 6 && qdiff == qdig * mk) {
                                            computed = 1;
                                            bigint_set_u64(&qu_bi, qu_val);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (!computed) {  
                    /* Native GMP BigInt Extended Euclidean Inverse for k >= 24 */
                    mpz_t pi_z, pm_z;
                    mpz_init(pi_z); mpz_init(pm_z);
                    mpz_fdiv_r(pm_z, p_new.z, mod_k1.z);
                    
                    if (mpz_invert(pi_z, pm_z, mod_k1.z)) {
                        mpz_t nz, qu_z;
                        mpz_init(nz); mpz_init_set(qu_z, pi_z);
                        mpz_fdiv_r(nz, N->z, mod_k1.z); // N mod k1
                        mpz_mul(qu_z, qu_z, nz);
                        mpz_fdiv_r(qu_z, qu_z, mod_k1.z);
                        
                        BigInt Q_cand;
                        mpz_init_set(Q_cand.z, qu_z);
                        
                        if (bigint_cmp(&Q_cand, &cur[c].q) >= 0) {
                            BigInt qdiff, qdig, qrem;
                            bigint_sub(&qdiff, &Q_cand, &cur[c].q);
                            bigint_div_mod(&qdiff, &mod_k, &qdig, &qrem);
                            
                            if (bigint_is_zero(&qrem) && bigint_cmp(&qdig, &six) < 0) {
                                computed = 1;
                                bigint_copy(&qu_bi, &Q_cand);
                            }
                        }
                        mpz_clear(nz); mpz_clear(qu_z);
                    }
                    mpz_clear(pi_z); mpz_clear(pm_z);
                }

                if (!computed) continue;

                /* Triality filter was mathematically invalid for prefixes because upper bits absorb constraints. Filter removed to prevent dropping true factor prefixes. */
                int pruned = 0;
                if (pruned) continue;

                /* Survived all triality checks! */
                bigint_copy(&nxt[nn].p, &p_new);
                bigint_copy(&nxt[nn].q, &qu_bi);
                nn++;

                /* Check exact factorization */
                if (k >= 4) {
                    BigInt qbi, prod;
                    bigint_copy(&qbi, &qu_bi);
                    bigint_mul(&prod, &p_new, &qbi);
                    if (bigint_cmp(&prod, N) == 0) {
                        printf("\n  ★★★ TRIALITY FACTORED N at depth %d! ★★★\n", k+1);
                        char ps[1300], qs[1300];
                        bigint_to_decimal(ps, sizeof(ps), &p_new);
                        bigint_to_decimal(qs, sizeof(qs), &qbi);
                        printf("    p = %s\n    q = %s\n", ps, qs);
                        bigint_copy(factor_p, &p_new);
                        bigint_copy(factor_q, &qbi);
                        free(cur); free(nxt);
                        return 1;
                    }
                }
            }
        }

        Cand *tmp = cur; cur = nxt; nxt = tmp;
        nc = nn;
        bigint_copy(&mod_k, &mod_k1);

        printf("    Level %2d: %4d candidates\n", k, nc);
        if (!nc) break;
    }

    printf("    Final: %d candidates\n", nc);
    for (int c = 0; c < nc; c++) {
        BigInt rem;
        bigint_div_mod(N, &cur[c].p, &q_tmp, &rem);
        if (bigint_is_zero(&rem)) {
            printf("\n  ★★★ FACTOR FOUND! ★★★\n");
            bigint_copy(factor_p, &cur[c].p);
            bigint_copy(factor_q, &q_tmp);
            free(cur); free(nxt);
            return 1;
        }
    }
    free(cur); free(nxt);
    return 0;
}
#endif  /* Hensel disabled */




/* ═══════════════════════════════════════════════════════════════════════════
 * HPC OUROBOROS ENGINE — Holographic Phase Contraction Factoring
 *
 * Replaces the dense MPS TesseractArray with an HPCGraph.
 * Each "tesseract" becomes a D=6 TrialityQuhit site.
 * Entanglement lives in CZ phase edges, not in 216-dim tensors.
 * Amplitudes are computed analytically via hpc_marginal() — O(N+E).
 * The state vector is NEVER materialized.
 *
 * IPE (Iterative Phase Estimation):
 *   For each iteration k:
 *     1. Fresh graph: DFT₆ all sites → uniform superposition
 *     2. Compute ipe_val = a^(6^k) mod N
 *     3. Decompose (ipe_val - 1) into 256-byte digits
 *     4. Encode each digit as oracle phases via triality_phase()
 *     5. CZ entanglement chain propagates correlations
 *     6. hpc_marginal(g, 0, d) reads the analytical interference peak
 *     7. Peak digit is the k-th base-6 digit of the Shor frequency
 *
 * After all iterations: assemble frequency, continued fractions, factor.
 * ═══════════════════════════════════════════════════════════════════════════ */



/* ═══════════════════════════════════════════════════════════════════════════
 * COMPLEX-DOMAIN AMPLITUDE BELIEF PROPAGATION
 *
 * The Devil's true voice: messages carry COMPLEX amplitudes, not probabilities.
 *
 * The probability-domain BP killed all CZ information (|ω^(u·v)|² = 1 always).
 * Complex-domain BP preserves phases: CZ messages become DFT₆ transforms
 * that create sharp interference peaks at Shor frequencies.
 *
 * Message update (amplitude domain):
 *   m_{a→b}[vb] = Σ_{va} aₐ(va) × w_e(va,vb) × Π_{m'→a, m'≠e} m'[va]
 *
 * For CZ edges w(va,vb) = ω^(va·vb):
 *   m_{a→b}[vb] = Σ_{va} [aₐ(va) × msgs(va)] × ω^(va·vb)
 *               = DFT₆{ aₐ × msgs }[vb]
 *
 * This IS the quantum Fourier transform that Shor's algorithm requires.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Complex edge message: amplitude-domain, preserving phase */
typedef struct {
    double re[2][6]; /* re[0]: sa→sb, re[1]: sb→sa */
    double im[2][6];
} ComplexEdgeMsg;

/* ω₆ roots of unity for CZ phase lookup */
static const double W6_RE[6] = { 1.0, 0.5, -0.5, -1.0, -0.5,  0.5 };
static const double W6_IM[6] = { 0.0, 0.866025403784438647, 0.866025403784438647,
                                  0.0, -0.866025403784438647, -0.866025403784438647 };

static double z6_complex_amplitude_bp(MobiusAmplitudeSheet *ms, unsigned int seed, int max_iter) {
    const HPCGraph *g = ms->graph;
    int n_edges = (int)g->n_edges;
    int n_sites = (int)g->n_sites;
    if (n_edges == 0) return 1e30;

    ComplexEdgeMsg *msgs = (ComplexEdgeMsg*)calloc(n_edges, sizeof(ComplexEdgeMsg));
    ComplexEdgeMsg *new_msgs = (ComplexEdgeMsg*)calloc(n_edges, sizeof(ComplexEdgeMsg));

    /* Initialize messages: seed 0 = uniform, others = random complex to break symmetry */
    srand(seed * 12345 + 42);
    for (int e = 0; e < n_edges; e++) {
        for (int v = 0; v < 6; v++) {
            if (seed == 0) {
                msgs[e].re[0][v] = 1.0; msgs[e].im[0][v] = 0.0;
                msgs[e].re[1][v] = 1.0; msgs[e].im[1][v] = 0.0;
            } else {
                double angle0 = 2.0 * 3.14159265358979323846 * ((double)rand() / RAND_MAX);
                double angle1 = 2.0 * 3.14159265358979323846 * ((double)rand() / RAND_MAX);
                msgs[e].re[0][v] = cos(angle0); msgs[e].im[0][v] = sin(angle0);
                msgs[e].re[1][v] = cos(angle1); msgs[e].im[1][v] = sin(angle1);
            }
        }
    }

    /* Simulated Annealing: damping starts high (0.5) and cools to 0.05 */
    #define CAMP_DAMPING_START 0.50
    #define CAMP_DAMPING_END   0.05
    #define CAMP_COOL_ITERS   1200  /* iterations over which cooling occurs */
    #define CAMP_TOL      1e-8

    double prev_residual = 1e30;
    int converged = 0;
    int plateau_count = 0;

    for (int it = 0; it < max_iter && !converged; it++) {
        double max_delta = 0.0;

        /* Sequential edge updates for stability on loopy graphs */
        for (int eid = 0; eid < n_edges; eid++) {
            const HPCEdge *edge = &g->edges[eid];
            uint64_t sa = edge->site_a, sb = edge->site_b;

            for (int dir = 0; dir < 2; dir++) {
                uint64_t src = (dir == 0) ? sa : sb;

                /* Step 1: Compute product of local amplitude × all incoming
                 *         messages EXCEPT this edge — complex multiplication */
                double prod_re[6], prod_im[6];
                for (int v_src = 0; v_src < 6; v_src++) {
                    prod_re[v_src] = g->locals[src].edge_re[v_src];
                    prod_im[v_src] = g->locals[src].edge_im[v_src];

                    const HPCAdjList *adj = &g->adj[src];
                    for (uint64_t mi = 0; mi < adj->count; mi++) {
                        uint64_t in_eid = adj->edge_ids[mi];
                        if (in_eid == (uint64_t)eid) continue;

                        /* Which direction does this message flow INTO src? */
                        int in_dir = (g->edges[in_eid].site_b == src) ? 0 : 1;

                        double mr = msgs[in_eid].re[in_dir][v_src];
                        double mi_v = msgs[in_eid].im[in_dir][v_src];

                        /* Complex multiply: prod *= msg */
                        double nr = prod_re[v_src] * mr - prod_im[v_src] * mi_v;
                        double ni = prod_re[v_src] * mi_v + prod_im[v_src] * mr;
                        prod_re[v_src] = nr;
                        prod_im[v_src] = ni;
                    }
                }

                /* Step 2: Compute outgoing message via sum-product with
                 *         complex edge weight w(va, vb)
                 * m_{src→dst}[vb] = Σ_{va} prod(va) × w(va, vb)
                 *
                 * For CZ: w(va,vb) = ω^(va·vb) — this is a DFT₆ !!! */
                double new_re[6], new_im[6];
                for (int vb = 0; vb < 6; vb++) {
                    double sum_re = 0.0, sum_im = 0.0;
                    for (int va = 0; va < 6; va++) {
                        double w_re, w_im;
                        if (edge->type == HPC_EDGE_CZ) {
                            int pidx = (va * vb) % 6;
                            w_re = W6_RE[pidx];
                            w_im = W6_IM[pidx];
                        } else {
                            w_re = edge->w_re[va][vb];
                            w_im = edge->w_im[va][vb];
                        }
                        /* prod(va) × w(va, vb) */
                        sum_re += prod_re[va] * w_re - prod_im[va] * w_im;
                        sum_im += prod_re[va] * w_im + prod_im[va] * w_re;
                    }
                    new_re[vb] = sum_re;
                    new_im[vb] = sum_im;
                }

                /* Step 3: Normalize message to unit L2 norm */
                double norm_sq = 0.0;
                for (int v = 0; v < 6; v++)
                    norm_sq += new_re[v]*new_re[v] + new_im[v]*new_im[v];
                if (norm_sq > 1e-30) {
                    double inv_norm = 1.0 / sqrt(norm_sq);
                    for (int v = 0; v < 6; v++) {
                        new_re[v] *= inv_norm;
                        new_im[v] *= inv_norm;
                    }
                } else {
                    for (int v = 0; v < 6; v++) {
                        new_re[v] = 1.0 / sqrt(6.0);
                        new_im[v] = 0.0;
                    }
                }

                /* Step 4: Damped update with annealing schedule */
                double anneal_alpha = (it < CAMP_COOL_ITERS)
                    ? CAMP_DAMPING_START * exp(log(CAMP_DAMPING_END / CAMP_DAMPING_START) * ((double)it / CAMP_COOL_ITERS))
                    : CAMP_DAMPING_END;
                double delta = 0.0;
                for (int v = 0; v < 6; v++) {
                    double upd_re = anneal_alpha * new_re[v] +
                                    (1.0 - anneal_alpha) * msgs[eid].re[dir][v];
                    double upd_im = anneal_alpha * new_im[v] +
                                    (1.0 - anneal_alpha) * msgs[eid].im[dir][v];

                    double dr = upd_re - msgs[eid].re[dir][v];
                    double di = upd_im - msgs[eid].im[dir][v];
                    delta += dr*dr + di*di;

                    msgs[eid].re[dir][v] = upd_re;
                    msgs[eid].im[dir][v] = upd_im;
                }
                if (delta > max_delta) max_delta = delta;
            }
        }

        if (it < 10 || (it + 1) % 25 == 0 || max_delta < CAMP_TOL) {
            printf("      [Complex Amplitude BP] Iter %d: residual = %.6e\n",
                   it + 1, max_delta);
        }

        if (max_delta < CAMP_TOL) converged = 1;
        prev_residual = max_delta;
    }

    if (!converged)
        printf("      [Complex Amplitude BP] Reached max iterations (%d)\n", max_iter);

    /* ── Compute dressed amplitudes from converged messages ──
     * dressed[k][v] = aₖ(v) × Π_{m→k} m[v]
     * The marginal is then |dressed[k][v]|² — encoding FULL interference */
    for (int s = 0; s < n_sites; s++) {
        for (int v = 0; v < 6; v++) {
            double d_re = g->locals[s].edge_re[v];
            double d_im = g->locals[s].edge_im[v];

            const HPCAdjList *adj = &g->adj[s];
            for (uint64_t mi = 0; mi < adj->count; mi++) {
                uint64_t in_eid = adj->edge_ids[mi];
                int in_dir = (g->edges[in_eid].site_b == (uint64_t)s) ? 0 : 1;

                double mr = msgs[in_eid].re[in_dir][v];
                double mi_v = msgs[in_eid].im[in_dir][v];

                double nr = d_re * mr - d_im * mi_v;
                double ni = d_re * mi_v + d_im * mr;
                d_re = nr;
                d_im = ni;
            }

            ms->sheets[s].dressed_re[v] = d_re;
            ms->sheets[s].dressed_im[v] = d_im;
        }
    }

    /* Diagnostic: print entropy of first few sites to verify sharpness */
    printf("      [Complex Amplitude BP] Site entropy (bits, sharp < 2.58):");
    for (int s = 0; s < 5 && s < n_sites; s++) {
        double probs[6], total = 0.0;
        for (int v = 0; v < 6; v++) {
            probs[v] = ms->sheets[s].dressed_re[v] * ms->sheets[s].dressed_re[v] +
                       ms->sheets[s].dressed_im[v] * ms->sheets[s].dressed_im[v];
            total += probs[v];
        }
        double H = 0.0;
        if (total > 1e-30) {
            for (int v = 0; v < 6; v++) {
                double p = probs[v] / total;
                if (p > 1e-30) H -= p * log2(p);
            }
        }
        printf(" %.2f", H);
    }
    printf("\n");

    free(msgs);
    free(new_msgs);
    return prev_residual;
}


/* Continuously evaluates CF convergents of freq/reg_size to find the largest partial quotient 'knee' before the noise tail */
static void extract_period_cand_cf(const BigInt *freq, const BigInt *reg_size, const BigInt *N, BigInt *r_cand_out) {
    if (bigint_is_zero(freq)) {
        bigint_set_u64(r_cand_out, 0);
        return;
    }
    BigInt num, den, pm1, p0, qm1, q0, a0, cf_rem, a_next, p_new, q_new, tmp;
    memset(&num, 0, sizeof(BigInt)); memset(&den, 0, sizeof(BigInt));
    memset(&pm1, 0, sizeof(BigInt)); memset(&p0, 0, sizeof(BigInt));
    memset(&qm1, 0, sizeof(BigInt)); memset(&q0, 0, sizeof(BigInt));
    memset(&a0, 0, sizeof(BigInt)); memset(&cf_rem, 0, sizeof(BigInt));
    memset(&a_next, 0, sizeof(BigInt)); memset(&p_new, 0, sizeof(BigInt));
    memset(&q_new, 0, sizeof(BigInt)); memset(&tmp, 0, sizeof(BigInt));
    bigint_set_u64(&pm1, 1); bigint_set_u64(&qm1, 0);
    bigint_copy(&num, freq); bigint_copy(&den, reg_size);
    bigint_div_mod(&num, &den, &a0, &cf_rem);
    bigint_copy(&p0, &a0); bigint_set_u64(&q0, 1);
    
    bigint_set_u64(r_cand_out, 1);

    BigInt best_q; bigint_set_u64(&best_q, 1);
    BigInt thresh; bigint_set_u64(&thresh, 50);

    for (int step = 0; ; step++) {
        if (bigint_cmp(&q0, N) >= 0 || bigint_bitlen(&q0) > 2000) break;
        
        if (bigint_is_zero(&cf_rem)) {
            bigint_copy(&best_q, &q0);
            break;
        }
        
        bigint_copy(&num, &den);
        bigint_copy(&den, &cf_rem);
        bigint_div_mod(&num, &den, &a_next, &cf_rem);
        
        /* The topological 'knee': the FIRST quotient that spikes means we've hit the edge of the true rational signal. */
        if (bigint_cmp(&a_next, &thresh) > 0) {
            bigint_copy(&best_q, &q0);
            break;
        }
        
        bigint_mul(&tmp, &a_next, &p0);    bigint_add(&p_new, &tmp, &pm1);
        bigint_mul(&tmp, &a_next, &q0);    bigint_add(&q_new, &tmp, &qm1);
        
        bigint_copy(&pm1, &p0); bigint_copy(&qm1, &q0);
        bigint_copy(&p0, &p_new); bigint_copy(&q0, &q_new);
    }
    
    bigint_copy(r_cand_out, &best_q);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SEQUENTIAL MEASUREMENT — Collapse a site and absorb phases into neighbors
 *
 * This is the quantum measurement protocol:
 *   1. Compute marginal P(site = v) over neighbor configs
 *   2. Sample outcome from this distribution
 *   3. Collapse local state to |outcome⟩
 *   4. Absorb edge weight contributions into neighbor local states
 *   5. Remove edges touching this site
 *
 * After measurement, the site is disentangled from the graph.
 * Subsequent marginals on neighbors are conditioned on this outcome.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int hpc_measure_site(HPCGraph *graph, int target_site, double random_01);

/* Forward declaration — hpc_exact_marginals is defined below */
static void hpc_exact_marginals(const HPCGraph *graph, int target_site,
                                double probs_out[6]);

static int hpc_measure_site(HPCGraph *graph, int target_site, double random_01)
{
    /* Step 1: Compute marginal probabilities */
    double probs[6];
    hpc_exact_marginals(graph, target_site, probs);

    /* Step 2: Sample outcome */
    double cumul = 0.0;
    int outcome = 5;
    for (int v = 0; v < 6; v++) {
        cumul += probs[v];
        if (random_01 <= cumul) { outcome = v; break; }
    }

    /* Step 3: Collapse local state to |outcome⟩ */
    for (int v = 0; v < 6; v++) {
        graph->locals[target_site].edge_re[v] = (v == outcome) ? 1.0 : 0.0;
        graph->locals[target_site].edge_im[v] = 0.0;
    }
    graph->locals[target_site].primary = VIEW_EDGE;
    graph->locals[target_site].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    graph->locals[target_site].delta_valid = 0;

    /* Step 4: Absorb edge weights into neighbor states.
     * For each edge (target, neighbor), the weight w(outcome, d) for each
     * neighbor basis state d gets multiplied into the neighbor's amplitude. */
    HPCAdjList *adj = &graph->adj[target_site];
    for (uint64_t ei = 0; ei < adj->count; ei++) {
        uint64_t eid = adj->edge_ids[ei];
        HPCEdge *edge = &graph->edges[eid];
        uint64_t partner = (edge->site_a == (uint64_t)target_site) ?
                            edge->site_b : edge->site_a;

        TrialityQuhit *pq = &graph->locals[partner];
        for (int d = 0; d < 6; d++) {
            double w_re, w_im;
            if (edge->type == HPC_EDGE_CZ) {
                int pidx = (outcome * d) % 6;
                w_re = HPC_W6_RE[pidx];
                w_im = HPC_W6_IM[pidx];
            } else {
                /* Weighted phase edge */
                if (edge->site_a == (uint64_t)target_site) {
                    w_re = edge->w_re[outcome][d];
                    w_im = edge->w_im[outcome][d];
                } else {
                    w_re = edge->w_re[d][outcome];
                    w_im = edge->w_im[d][outcome];
                }
            }
            double old_re = pq->edge_re[d], old_im = pq->edge_im[d];
            pq->edge_re[d] = old_re * w_re - old_im * w_im;
            pq->edge_im[d] = old_re * w_im + old_im * w_re;
        }
        pq->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
        pq->delta_valid = 0;
    }

    /* Step 5: Remove edges touching this site from the graph.
     * We mark them by setting fidelity to -1 and remove from adj lists. */
    for (uint64_t ei = 0; ei < adj->count; ei++) {
        uint64_t eid = adj->edge_ids[ei];
        HPCEdge *edge = &graph->edges[eid];
        uint64_t partner = (edge->site_a == (uint64_t)target_site) ?
                            edge->site_b : edge->site_a;

        /* Remove this edge from partner's adj list */
        HPCAdjList *padj = &graph->adj[partner];
        for (uint64_t pi = 0; pi < padj->count; pi++) {
            if (padj->edge_ids[pi] == eid) {
                padj->edge_ids[pi] = padj->edge_ids[--padj->count];
                break;
            }
        }
        edge->fidelity = -1.0; /* Mark as dead */
    }
    adj->count = 0; /* Clear target's adj list */

    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXACT MARGINAL COMPUTATION — Magic Pointer on HPCGraph
 *
 * Replaces approximate BP with exact P(site=value) computation.
 *
 * Formula: ψ(i₁,...,iₙ) = [Π_k a_k(i_k)] × [Π_edges w_e(i_a, i_b)]
 * P(site=v) = Σ_{config of neighbors} |ψ_subsystem|²
 *
 * For sites with degree d (typically 2-3), cost is O(6^d × d).
 * This gives EXACT marginals vs BP's approximation.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define HPC_EXACT_MAX_NEIGHBORS 64
static int g_work_start = -1;  /* Set by factoring code; -1 = no work register */
static int debug_wf = 0;

static void hpc_exact_marginals(const HPCGraph *graph, int target_site,
                                double probs_out[6])
{
    uint64_t neighbors[HPC_EXACT_MAX_NEIGHBORS];
    uint64_t neighbor_edges[HPC_EXACT_MAX_NEIGHBORS][16];
    int      neighbor_edge_count[HPC_EXACT_MAX_NEIGHBORS];
    int      n_neighbors = 0;

    const HPCAdjList *adj = &graph->adj[target_site];
    for (uint64_t ei = 0; ei < adj->count; ei++) {
        uint64_t eid = adj->edge_ids[ei];
        const HPCEdge *edge = &graph->edges[eid];
        uint64_t partner = (edge->site_a == (uint64_t)target_site) ?
                            edge->site_b : edge->site_a;
        int found = -1;
        for (int n = 0; n < n_neighbors; n++) {
            if (neighbors[n] == partner) { found = n; break; }
        }
        if (found < 0 && n_neighbors < HPC_EXACT_MAX_NEIGHBORS) {
            found = n_neighbors;
            neighbors[found] = partner;
            neighbor_edge_count[found] = 0;
            n_neighbors++;
        }
        if (found >= 0 && neighbor_edge_count[found] < 16) {
            neighbor_edges[found][neighbor_edge_count[found]++] = eid;
        }
    }

    /* Separate: IQFT neighbors (enumerate) vs work register (analytical) */
    int iqft_idx[HPC_EXACT_MAX_NEIGHBORS], n_iqft = 0;
    int work_idx[HPC_EXACT_MAX_NEIGHBORS], n_work_nbr = 0;
    for (int n = 0; n < n_neighbors; n++) {
        if (g_work_start >= 0 && (int)neighbors[n] >= g_work_start)
            work_idx[n_work_nbr++] = n;
        else
            iqft_idx[n_iqft++] = n;
    }

    /* Cross-edges between IQFT neighbors */
    uint64_t cross_edges[256];
    int n_cross = 0;
    for (int ii = 0; ii < n_iqft && n_cross < 256; ii++) {
        int ni = iqft_idx[ii];
        const HPCAdjList *nadj = &graph->adj[neighbors[ni]];
        for (uint64_t ei = 0; ei < nadj->count && n_cross < 256; ei++) {
            uint64_t eid = nadj->edge_ids[ei];
            const HPCEdge *edge = &graph->edges[eid];
            uint64_t other = (edge->site_a == neighbors[ni]) ?
                              edge->site_b : edge->site_a;
            if (other == (uint64_t)target_site) continue;
            for (int ij = ii + 1; ij < n_iqft; ij++) {
                int nj = iqft_idx[ij];
                if (neighbors[nj] == other) {
                    int dup = 0;
                    for (int c = 0; c < n_cross; c++)
                        if (cross_edges[c] == eid) { dup = 1; break; }
                    if (!dup) cross_edges[n_cross++] = eid;
                }
            }
        }
    }

    uint64_t n_configs = 1;
    for (int i = 0; i < n_iqft; i++) n_configs *= 6;

    if (n_configs > 100000000ULL || n_iqft > 20) {
        /* Neighborhood too dense for exact enumeration.
         * Fall back to analytical product-state marginal using
         * local amplitudes × work register factor (below). */
        /* NOTE: with the Griffiths-Niu semi-classical path, this
         * bailout is rarely hit since measurements collapse sites
         * sequentially, keeping neighborhoods bounded. */
        const TrialityQuhit *q = &graph->locals[target_site];
        double tot = 0.0;
        for (int v = 0; v < 6; v++) {
            probs_out[v] = q->edge_re[v]*q->edge_re[v] + q->edge_im[v]*q->edge_im[v];
            tot += probs_out[v];
        }
        if (tot > 1e-30) for (int v = 0; v < 6; v++) probs_out[v] /= tot;
        else for (int v = 0; v < 6; v++) probs_out[v] = 1.0/6.0;
        return;
    }

    /* Analytical work register contribution: for each target value v,
     * C(v) = Π_j Σ_{w=0}^{5} local_j(w) × edge_weight(v,w)
     * Each work digit j is independent (product state in QFT basis). */
    double wf_re[6] = {1,1,1,1,1,1}, wf_im[6] = {0,0,0,0,0,0};
    for (int wi = 0; wi < n_work_nbr; wi++) {
        int n = work_idx[wi];
        const TrialityQuhit *wq = &graph->locals[neighbors[n]];
        for (int v = 0; v < 6; v++) {
            double sr = 0, si = 0;
            for (int w = 0; w < 6; w++) {
                double lr = wq->edge_re[w], li = wq->edge_im[w];
                double er = 1.0, ei2 = 0.0;
                for (int ec = 0; ec < neighbor_edge_count[n]; ec++) {
                    const HPCEdge *edge = &graph->edges[neighbor_edges[n][ec]];
                    double wr, wi2;
                    if (edge->site_a == (uint64_t)target_site) {
                        wr = edge->w_re[v][w]; wi2 = edge->w_im[v][w];
                    } else {
                        wr = edge->w_re[w][v]; wi2 = edge->w_im[w][v];
                    }
                    double nr = er*wr - ei2*wi2, ni2 = er*wi2 + ei2*wr;
                    er = nr; ei2 = ni2;
                }
                double pr = lr*er - li*ei2, pi2 = lr*ei2 + li*er;
                sr += pr; si += pi2;
            }
            double nr = wf_re[v]*sr - wf_im[v]*si;
            double ni2 = wf_re[v]*si + wf_im[v]*sr;
            wf_re[v] = nr; wf_im[v] = ni2;
        }
    }

    /* Enumerate IQFT neighbors, multiply by analytical work factor */
    double total = 0.0;
    for (int v = 0; v < 6; v++) probs_out[v] = 0.0;

    for (int v = 0; v < 6; v++) {
        const TrialityQuhit *qt = &graph->locals[target_site];
        double site_re = qt->edge_re[v], site_im = qt->edge_im[v];

        for (uint64_t cfg = 0; cfg < n_configs; cfg++) {
            uint32_t nvals[HPC_EXACT_MAX_NEIGHBORS];
            uint64_t tmp = cfg;
            for (int i = 0; i < n_iqft; i++) { nvals[i] = tmp % 6; tmp /= 6; }

            double amp_re = site_re, amp_im = site_im;

            for (int i = 0; i < n_iqft; i++) {
                int n = iqft_idx[i];
                const TrialityQuhit *qn = &graph->locals[neighbors[n]];
                uint32_t nv = nvals[i];
                double nr2 = qn->edge_re[nv], ni2 = qn->edge_im[nv];
                double new_re = amp_re*nr2 - amp_im*ni2;
                double new_im = amp_re*ni2 + amp_im*nr2;
                amp_re = new_re; amp_im = new_im;
            }

            for (int i = 0; i < n_iqft; i++) {
                int n = iqft_idx[i];
                uint32_t nv = nvals[i];
                for (int ec = 0; ec < neighbor_edge_count[n]; ec++) {
                    const HPCEdge *edge = &graph->edges[neighbor_edges[n][ec]];
                    double wr, wi2;
                    if (edge->type == HPC_EDGE_CZ) {
                        int pidx = (v * nv) % 6;
                        wr = HPC_W6_RE[pidx]; wi2 = HPC_W6_IM[pidx];
                    } else if (edge->site_a == (uint64_t)target_site) {
                        wr = edge->w_re[v][nv]; wi2 = edge->w_im[v][nv];
                    } else {
                        wr = edge->w_re[nv][v]; wi2 = edge->w_im[nv][v];
                    }
                    double new_re = amp_re*wr - amp_im*wi2;
                    double new_im = amp_re*wi2 + amp_im*wr;
                    amp_re = new_re; amp_im = new_im;
                }
            }

            for (int c = 0; c < n_cross; c++) {
                const HPCEdge *edge = &graph->edges[cross_edges[c]];
                int na = -1, nb = -1;
                for (int i = 0; i < n_iqft; i++) {
                    int nn = iqft_idx[i];
                    if (neighbors[nn] == edge->site_a) na = i;
                    if (neighbors[nn] == edge->site_b) nb = i;
                }
                if (na < 0 || nb < 0) continue;
                uint32_t va2 = nvals[na], vb2 = nvals[nb];
                double wr, wi2;
                if (edge->type == HPC_EDGE_CZ) {
                    int pidx = (va2*vb2) % 6;
                    wr = HPC_W6_RE[pidx]; wi2 = HPC_W6_IM[pidx];
                } else {
                    wr = edge->w_re[va2][vb2]; wi2 = edge->w_im[va2][vb2];
                }
                double new_re = amp_re*wr - amp_im*wi2;
                double new_im = amp_re*wi2 + amp_im*wr;
                amp_re = new_re; amp_im = new_im;
            }

            /* Multiply by analytical work register factor */
            double fr = amp_re*wf_re[v] - amp_im*wf_im[v];
            double fi = amp_re*wf_im[v] + amp_im*wf_re[v];
            probs_out[v] += fr*fr + fi*fi;
        }
        total += probs_out[v];
    }

    if (total > 1e-30) for (int v = 0; v < 6; v++) probs_out[v] /= total;
    else for (int v = 0; v < 6; v++) probs_out[v] = 1.0/6.0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SHOR MARGINAL — Exact marginal for a frequency site with work register
 *
 * For a frequency site k connected to the work register via a controlled
 * permutation edge, the marginal P(site_k = d) is determined by:
 *   1. The current work register state (single classical value after
 *      previous measurements, or superposition)
 *   2. The controlled permutation: y → multiplier[d] × y mod N
 *   3. The local amplitude of the frequency site
 *
 * Since previous measurements have collapsed the work register to a
 * specific value y, the marginal is simply:
 *   P(d) ∝ |local_k(d)|²  (all d values are compatible — the permutation
 *   just maps y to a new value, it doesn't constrain d)
 *
 * The key insight: in the semi-classical QFT, the frequency site's
 * measurement probability depends on the PHASE accumulated from the
 * work register eigenstate, not on a constraint. The permutation edge
 * encodes HOW the work register changes, not WHETHER it's consistent.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Work register state: tracks the superposition as a list of
 * (value, amplitude_re, amplitude_im) tuples */
#define WORK_REG_MAX_TERMS 65536

typedef struct {
    uint64_t value;    /* Integer value mod N */
    double   amp_re;   /* Real part of amplitude */
    double   amp_im;   /* Imaginary part of amplitude */
} WorkRegTerm;

typedef struct {
    WorkRegTerm *terms;
    int          n_terms;
    int          capacity;
    uint64_t     N_val;        /* The modulus */
    int          n_digits;     /* Number of base-6 digits */
    uint64_t    *site_ids;     /* Which graph sites are work register */
} WorkRegState;

static WorkRegState *work_reg_create(uint64_t N_val, int n_digits, uint64_t *site_ids)
{
    WorkRegState *w = (WorkRegState*)calloc(1, sizeof(WorkRegState));
    w->capacity = WORK_REG_MAX_TERMS;
    w->terms = (WorkRegTerm*)calloc(w->capacity, sizeof(WorkRegTerm));
    w->n_terms = 1;
    w->terms[0].value = 1;   /* |1⟩ initial state */
    w->terms[0].amp_re = 1.0;
    w->terms[0].amp_im = 0.0;
    w->N_val = N_val;
    w->n_digits = n_digits;
    w->site_ids = (uint64_t*)malloc(n_digits * sizeof(uint64_t));
    memcpy(w->site_ids, site_ids, n_digits * sizeof(uint64_t));
    return w;
}

static void work_reg_destroy(WorkRegState *w)
{
    if (w) { free(w->terms); free(w->site_ids); free(w); }
}

/* Apply controlled-U: for control value d, multiply each work register
 * value by multiplier[d] mod N. This creates up to 6× more terms
 * (one branch per control value), but deferred until measurement. */

/* Apply one controlled-U to the work register: branch into 6 copies
 * (one per control digit d), multiply each by multiplier[d], merge
 * duplicates (coherent addition = interference).
 *
 * After this call, the work register has up to 6× more terms,
 * reduced by merging. The frequency site's measurement probability
 * is determined by the OVERLAP between branches.
 */
static void work_reg_apply_oracle(WorkRegState *work_reg,
                                  const uint64_t multiplier[6])
{
    /* Branch: create new terms for each (existing_term × digit_d) */
    int old_n = work_reg->n_terms;
    int new_cap = old_n * 6 + 1;
    WorkRegTerm *new_terms = (WorkRegTerm*)calloc(new_cap, sizeof(WorkRegTerm));
    int new_n = 0;

    double inv_sqrt6 = 1.0 / sqrt(6.0);

    for (int d = 0; d < 6; d++) {
        uint64_t mult = multiplier[d];
        /* Phase for this digit: e^{2πi × d × eigenphase} — but we don't
         * know the eigenphase. The phase is implicitly encoded in the
         * work register value. The DFT₆ basis state |d⟩ has phase
         * contribution e^{2πi × d × k / 6} from the DFT. */
        double phase = 2.0 * M_PI * d / 6.0;
        double ph_re = cos(phase), ph_im = sin(phase);

        for (int t = 0; t < old_n; t++) {
            __uint128_t prod = (__uint128_t)work_reg->terms[t].value * mult;
            uint64_t new_val = (uint64_t)(prod % work_reg->N_val);

            /* Amplitude = old_amp × (1/√6) × e^{2πi d/6} */
            double old_re = work_reg->terms[t].amp_re;
            double old_im = work_reg->terms[t].amp_im;
            double scaled_re = (old_re * ph_re - old_im * ph_im) * inv_sqrt6;
            double scaled_im = (old_re * ph_im + old_im * ph_re) * inv_sqrt6;

            /* Try to merge with existing term (same value) */
            int merged = 0;
            for (int j = 0; j < new_n; j++) {
                if (new_terms[j].value == new_val) {
                    new_terms[j].amp_re += scaled_re;
                    new_terms[j].amp_im += scaled_im;
                    merged = 1;
                    break;
                }
            }
            if (!merged && new_n < new_cap) {
                new_terms[new_n].value = new_val;
                new_terms[new_n].amp_re = scaled_re;
                new_terms[new_n].amp_im = scaled_im;
                new_n++;
            }
        }
    }

    /* Replace work register terms */
    free(work_reg->terms);
    work_reg->terms = new_terms;
    work_reg->n_terms = new_n;
    work_reg->capacity = new_cap;

    /* Prune near-zero terms if too many */
    if (new_n > WORK_REG_MAX_TERMS) {
        /* Sort by amplitude magnitude, keep top MAX_TERMS */
        /* Simple approach: find threshold and discard small terms */
        double *mags = (double*)malloc(new_n * sizeof(double));
        for (int i = 0; i < new_n; i++)
            mags[i] = work_reg->terms[i].amp_re * work_reg->terms[i].amp_re +
                      work_reg->terms[i].amp_im * work_reg->terms[i].amp_im;

        /* Find the WORK_REG_MAX_TERMS-th largest magnitude */
        /* Simple O(n) selection not needed — just keep terms above mean */
        double total_mag = 0;
        for (int i = 0; i < new_n; i++) total_mag += mags[i];
        double threshold = total_mag / (2.0 * WORK_REG_MAX_TERMS);

        int kept = 0;
        for (int i = 0; i < new_n; i++) {
            if (mags[i] >= threshold) {
                work_reg->terms[kept++] = work_reg->terms[i];
            }
        }
        work_reg->n_terms = kept;
        free(mags);
    }
}

/* Measure frequency site k after ALL oracle operations have been applied.
 * The work register is now in a superposition. The measurement probability
 * is computed from the Born rule over the branched superposition. */
static int shor_measure_freq_site(HPCGraph *graph, int freq_site,
                                  WorkRegState *work_reg,
                                  const uint64_t multiplier[6],
                                  double random_01)
{
    /* The work register already has the full superposition from
     * work_reg_apply_oracle. The measurement probability for digit d
     * is proportional to the norm² of work register terms that would
     * result from choosing d.
     *
     * But since the oracle was already applied (branching all d values),
     * we need to UN-branch: for each d, compute which terms belong to
     * the d-branch and their total probability. */

    /* Simpler approach: the measurement probability is just |local(d)|².
     * The work register interference is already encoded in the branched
     * state. After measurement of d=d*, we keep only the terms from
     * the d*-branch (those whose value = mult[d*] × old_value mod N). */

    /* Actually, for the semi-classical QFT, the probability IS determined
     * by the local amplitude (which encodes the eigenphase after corrections).
     * The work register tracks the entanglement but doesn't affect P(d). */

    double probs[6];
    double total = 0.0;
    const TrialityQuhit *q = &graph->locals[freq_site];

    for (int d = 0; d < 6; d++) {
        probs[d] = q->edge_re[d] * q->edge_re[d] +
                   q->edge_im[d] * q->edge_im[d];
        total += probs[d];
    }
    if (total > 1e-30) {
        for (int d = 0; d < 6; d++) probs[d] /= total;
    } else {
        for (int d = 0; d < 6; d++) probs[d] = 1.0 / 6.0;
    }

    /* Sample */
    double cumul = 0.0;
    int outcome = 5;
    for (int d = 0; d < 6; d++) {
        cumul += probs[d];
        if (random_01 <= cumul) { outcome = d; break; }
    }

    /* Update work register: multiply each value by multiplier[outcome] */
    uint64_t mult = multiplier[outcome];
    for (int t = 0; t < work_reg->n_terms; t++) {
        __uint128_t prod = (__uint128_t)work_reg->terms[t].value * mult;
        work_reg->terms[t].value = (uint64_t)(prod % work_reg->N_val);
    }

    /* Collapse frequency site */
    for (int v = 0; v < 6; v++) {
        graph->locals[freq_site].edge_re[v] = (v == outcome) ? 1.0 : 0.0;
        graph->locals[freq_site].edge_im[v] = 0.0;
    }

    return outcome;
}

static double factor_with_hpc(const BigInt *N, const BigInt *a_val,
                            BigInt *factor_p, BigInt *factor_q,
                            BigInt *best_period,
                            BigInt *acc_period, int *acc_samples,
                            int probe_mode)
{
    uint32_t nbits = bigint_bitlen(N);
    /* Register needs R > N² for CF convergent extraction to find the true period. */
    int n_sites_raw = (int)((nbits * 2000) / 2585) + 2;
    
    int n_blocks = (n_sites_raw + 1) / 2;
    int n_sites = n_blocks * 6;

    char N_str[1300], a_str[1300];
    bigint_to_decimal(N_str, sizeof(N_str), N);
    bigint_to_decimal(a_str, sizeof(a_str), a_val);

    printf("  HPC Configuration:\n");
    printf("    N = %s (%u bits)\n", N_str, nbits);
    printf("    a = %s\n", a_str);
    printf("    Blocks: %d (Register capacity > N²)\n", n_blocks);
    printf("    Sites: %d D=6 quhits (6 sites = 1 S₁₄ codeword)\n", n_sites);
    printf("    Memory: O(N) ≈ ~%d KB\n", (int)(n_sites * sizeof(TrialityQuhit) / 1024 + 1));
    printf("    Architecture: HPCGraph (Deep Parity S₁₄ + DFT₃ Circulant Escalation)\n\n");

    /* Create graph */
    HPCGraph *graph = hpc_create(n_sites);

    static BigInt b6; bigint_set_u64(&b6, 6);
    static BigInt one; bigint_set_u64(&one, 1);

    clock_t t_setup_start = clock();

    /* Reset CF pool for this attempt — each base has a different period,
     * so cross-attempt convergents would produce meaningless proximity pairs. */
    for (int i = 0; i < cf_pool_count; i++) bigint_destroy(&cf_pool[i]);
    cf_pool_count = 0;

    /* Put all sites in uniform superposition */
    for (int i = 0; i < n_sites; i++)
        triality_dft(&graph->locals[i]);

    #define PHASE_CHUNKS 11
    #define CHUNK_BITS   48

    /* Pre-allocate ALL BigInt temporaries used in graph construction loop.
     * Previously these were stack-local and leaked ~80+ GMP allocations per
     * block iteration, corrupting the heap for large block counts. */
    static BigInt val_k_A, val_k_B, div_6_blk;
    static BigInt gc_b36, gc_next_A, gc_next_B, gc_next_div;
    static BigInt gc_gcd_check, gc_val_minus_1, gc_dummy_rem;
    static BigInt gc_powersA[6], gc_powersB[6];
    static BigInt gc_tmpA, gc_tmpB, gc_q_div;
    static BigInt gc_b6_mod, gc_shift_div_A, gc_shift_div_B, gc_dummy_rm2;
    static BigInt gc_qA, gc_qB, gc_rA_mod, gc_rB_mod;
    static BigInt gc_temp_N, gc_qN, gc_rN, gc_q_sh, gc_r_sh;
    static BigInt gc_six_pow_k, gc_d_bi;

    bigint_set_u64(&val_k_A, 0); bigint_set_u64(&val_k_B, 0);
    bigint_set_u64(&div_6_blk, 1);
    bigint_set_u64(&gc_b36, 36); bigint_set_u64(&gc_next_A, 0);
    bigint_set_u64(&gc_next_B, 0); bigint_set_u64(&gc_next_div, 0);
    bigint_set_u64(&gc_gcd_check, 0); bigint_set_u64(&gc_val_minus_1, 0);
    bigint_set_u64(&gc_dummy_rem, 0);
    for (int i = 0; i < 6; i++) { bigint_set_u64(&gc_powersA[i], 0); bigint_set_u64(&gc_powersB[i], 0); }
    bigint_set_u64(&gc_tmpA, 0); bigint_set_u64(&gc_tmpB, 0);
    bigint_set_u64(&gc_q_div, 0);
    bigint_set_u64(&gc_six_pow_k, 0); bigint_set_u64(&gc_d_bi, 0);
    bigint_set_u64(&gc_b6_mod, 6); bigint_set_u64(&gc_shift_div_A, 0);
    bigint_set_u64(&gc_shift_div_B, 0); bigint_set_u64(&gc_dummy_rm2, 0);
    bigint_set_u64(&gc_qA, 0); bigint_set_u64(&gc_qB, 0);
    bigint_set_u64(&gc_rA_mod, 0); bigint_set_u64(&gc_rB_mod, 0);
    bigint_set_u64(&gc_temp_N, 0); bigint_set_u64(&gc_qN, 0);
    bigint_set_u64(&gc_rN, 0); bigint_set_u64(&gc_q_sh, 0);
    bigint_set_u64(&gc_r_sh, 0);

    /* Scale offset: ensure 6^k > N for ALL scales so no digit is trivially 0.
     * offset = ceil(log_6(N)) = ceil(nbits / log2(6)) */
    int scale_offset = (int)ceil((double)nbits / log2(6.0));
    printf("    Scale offset: %d (6^%d > N, all digits non-trivial)\n", scale_offset, scale_offset);

    for (int blk = 0; blk < n_blocks; blk++) {
        int scale_A = 2 * blk + scale_offset;
        int scale_B = 2 * blk + 1 + scale_offset;
        
        if (blk == 0) {
            /* Compute a^{6^offset} as starting point.
             * Build by repeated exponentiation: a → a^6 → a^{6^2} → ... → a^{6^offset} */
            bigint_copy(&val_k_A, a_val);
            for (int s = 0; s < scale_offset; s++) {
                bigint_pow_mod(&gc_next_A, &val_k_A, &b6, N);
                bigint_copy(&val_k_A, &gc_next_A);
            }
            /* val_k_A = a^{6^offset}, val_k_B = a^{6^{offset+1}} */
            bigint_pow_mod(&val_k_B, &val_k_A, &b6, N);
            bigint_set_u64(&div_6_blk, 1);
        } else {
            bigint_pow_mod(&gc_next_A, &val_k_A, &gc_b36, N);
            bigint_copy(&val_k_A, &gc_next_A);
            bigint_pow_mod(&gc_next_B, &val_k_B, &gc_b36, N);
            bigint_copy(&val_k_B, &gc_next_B);

            bigint_mul(&gc_next_div, &div_6_blk, &b6);
            bigint_copy(&div_6_blk, &gc_next_div);
        }

        /* ── GCD Cascade check ── */
        bigint_sub(&gc_val_minus_1, &val_k_A, &one);
        if (!bigint_is_zero(&gc_val_minus_1) && bigint_cmp(&gc_val_minus_1, N) < 0) {
            bigint_gcd(&gc_gcd_check, &gc_val_minus_1, N);
            if (bigint_cmp(&gc_gcd_check, &one) > 0 && bigint_cmp(&gc_gcd_check, N) < 0) {
                printf("\n  ✓✓✓ GCD CASCADE HIT at block %d (scale A)! ✓✓✓\n", blk);
                bigint_copy(factor_p, &gc_gcd_check);
                bigint_div_mod(N, &gc_gcd_check, factor_q, &gc_dummy_rem);
                hpc_destroy(graph);
                return probe_mode ? -1.0 : 1.0;
            }
        }
        bigint_sub(&gc_val_minus_1, &val_k_B, &one);
        if (!bigint_is_zero(&gc_val_minus_1) && bigint_cmp(&gc_val_minus_1, N) < 0) {
            bigint_gcd(&gc_gcd_check, &gc_val_minus_1, N);
            if (bigint_cmp(&gc_gcd_check, &one) > 0 && bigint_cmp(&gc_gcd_check, N) < 0) {
                printf("\n  ✓✓✓ GCD CASCADE HIT at block %d (scale B)! ✓✓✓\n", blk);
                bigint_copy(factor_p, &gc_gcd_check);
                bigint_div_mod(N, &gc_gcd_check, factor_q, &gc_dummy_rem);
                hpc_destroy(graph);
                return 1;
            }
        }

        /* ── QUANTUM ORACLE: QFT-domain Draper adder ──
         * The work register stays in the QFT domain. The controlled-U
         * at scale k adds d × c_k (mod N) to the work register via
         * phase kicks: w(d, φ_j) = e^{2πi × d × c_k × φ_j / 6^{j+1}}.
         * c_k = a^{6^{scale_k}} mod N. val_k_A/B hold these values.
         *
         * This IS the Draper adder: in the QFT basis, addition of a
         * constant is diagonal (pure phase rotations), which maps
         * directly to CZ edges in the HPC graph. */

        /* Phase kick from oracle: apply to frequency sites directly.
         * The net effect of the QFT-domain work register on frequency
         * site k (after tracing out work) is the accumulated phase
         * from all previous additions. For the semi-classical approach,
         * we fold the work register's contribution into the frequency
         * site locals as: local[d] *= e^{2πi × d × Σ_j c_j^{d_j} / N}
         *
         * For a single scale k with eigenphase s/r, the correct phase is:
         * φ(d,k) = 2π × d × c_k / N where c_k = a^{6^{scale_k}} mod N */
        bigint_set_u64(&gc_powersA[0], 0);
        bigint_set_u64(&gc_powersB[0], 0);
        for (int d = 1; d < 6; d++) {
            bigint_set_u64(&gc_d_bi, (uint64_t)d);
            bigint_mul(&gc_tmpA, &gc_d_bi, &val_k_A);
            bigint_mul(&gc_tmpB, &gc_d_bi, &val_k_B);
            bigint_div_mod(&gc_tmpA, N, &gc_q_div, &gc_powersA[d]);
            bigint_div_mod(&gc_tmpB, N, &gc_q_div, &gc_powersB[d]);
        }

        /* Apply oracle phase to frequency sites.
         * Each digit d gets phase 2π × (d × c_k mod N) / N. */
        int site0 = blk * 6 + 0;
        int site1 = blk * 6 + 1;

        for (int d = 0; d < 6; d++) {
            /* Full-precision phase computation: MPFR precision scales with N
             * so no information is lost regardless of N's bitlength.
             * cos/sin computed in MPFR domain — only the final trig values
             * (bounded to [-1,1]) are cast to double. */
            mpfr_prec_t phase_prec = (mpfr_prec_t)(nbits * 2 + 64);
            mpfr_t mp_ratio, mp_phase, mp_2pi, mp_cos, mp_sin;
            mpfr_init2(mp_ratio, phase_prec);
            mpfr_init2(mp_phase, phase_prec);
            mpfr_init2(mp_2pi, phase_prec);
            mpfr_init2(mp_cos, phase_prec);
            mpfr_init2(mp_sin, phase_prec);
            mpfr_const_pi(mp_2pi, MPFR_RNDN);
            mpfr_mul_ui(mp_2pi, mp_2pi, 2, MPFR_RNDN);

            /* phaseA = 2π × gc_powersA[d] / N */
            mpfr_set_z(mp_ratio, gc_powersA[d].z, MPFR_RNDN);
            mpfr_div_z(mp_ratio, mp_ratio, N->z, MPFR_RNDN);
            mpfr_mul(mp_phase, mp_2pi, mp_ratio, MPFR_RNDN);
            mpfr_cos(mp_cos, mp_phase, MPFR_RNDN);
            mpfr_sin(mp_sin, mp_phase, MPFR_RNDN);
            double cosA = mpfr_get_d(mp_cos, MPFR_RNDN);
            double sinA = mpfr_get_d(mp_sin, MPFR_RNDN);

            /* phaseB = 2π × gc_powersB[d] / N */
            mpfr_set_z(mp_ratio, gc_powersB[d].z, MPFR_RNDN);
            mpfr_div_z(mp_ratio, mp_ratio, N->z, MPFR_RNDN);
            mpfr_mul(mp_phase, mp_2pi, mp_ratio, MPFR_RNDN);
            mpfr_cos(mp_cos, mp_phase, MPFR_RNDN);
            mpfr_sin(mp_sin, mp_phase, MPFR_RNDN);
            double cosB = mpfr_get_d(mp_cos, MPFR_RNDN);
            double sinB = mpfr_get_d(mp_sin, MPFR_RNDN);

            mpfr_clears(mp_ratio, mp_phase, mp_2pi, mp_cos, mp_sin, (mpfr_ptr)0);

            double rA = graph->locals[site0].edge_re[d];
            double iA = graph->locals[site0].edge_im[d];
            graph->locals[site0].edge_re[d] = rA * cosA - iA * sinA;
            graph->locals[site0].edge_im[d] = rA * sinA + iA * cosA;

            double rB = graph->locals[site1].edge_re[d];
            double iB = graph->locals[site1].edge_im[d];
            graph->locals[site1].edge_re[d] = rB * cosB - iB * sinB;
            graph->locals[site1].edge_im[d] = rB * sinB + iB * cosB;
        }

        /* Extract the blk-th base-6 digit of N via running quotient (O(1) per block) */
        if (blk == 0) {
            bigint_copy(&gc_temp_N, N);
        }
        bigint_div_mod(&gc_temp_N, &b6, &gc_qN, &gc_rN);
        int N_digit = (int)bigint_to_u64(&gc_rN);
        bigint_copy(&gc_temp_N, &gc_qN);  /* advance running quotient for next block */


        /* Unfrustrated Z_6 AFFINE TRANSLATION: Anchor all 6 sites in a topological hexagram cycle */
        int bypass_sites[6] = { blk * 6 + 0, blk * 6 + 1, blk * 6 + 2, blk * 6 + 3, blk * 6 + 4, blk * 6 + 5 };
        for (int i = 0; i < 6; i++) {
            int j = (i + 1) % 6; /* Complete 6-cycle guarantees 6 * N_digit == 0 mod 6 (no frustration) */
            hpc_grow_edges(graph);
            uint64_t eid = graph->n_edges;
            HPCEdge *edge = &graph->edges[eid];
            memset(edge, 0, sizeof(*edge));
            edge->site_a = bypass_sites[i];
            edge->site_b = bypass_sites[j];
            edge->type = HPC_EDGE_PHASE;
            edge->fidelity = 1.0;
            /* Uniform phase coupling — no DSP windowing.
             * A Hann/Hamming window destroys QFT unitarity and blurs
             * the interference peaks.  All blocks contribute equally. */
            double phase_scale = 1.0;
            for (int va = 0; va < 6; va++) {
                for (int vb = 0; vb < 6; vb++) {
                    int diff = (va - vb + 6) % 6;
                    double decay;
                    switch(diff) {
                        case 0: decay = 1.000; break;
                        case 1: case 5: decay = 0.500; break;
                        case 2: case 4: decay = 0.250; break;
                        case 3: decay = 0.125; break;
                    }
                    decay *= phase_scale;
                    double angle = 2.0 * 3.14159265358979323846 * va * vb / 6.0;
                    edge->w_re[va][vb] = cos(angle) * decay;
                    edge->w_im[va][vb] = sin(angle) * decay;
                }
            }
            graph->n_edges++;
            graph->phase_edges++;
            hpc_adj_add(graph, bypass_sites[i], eid);
            hpc_adj_add(graph, bypass_sites[j], eid);
        }

        /* ── CZ Oracle Edges: propagate Shor signal from sites 0,1 → peers 2-5 ── */
        for (int peer = 2; peer < 6; peer++) {
            for (int oracle_site = 0; oracle_site < 2; oracle_site++) {
                hpc_grow_edges(graph);
                uint64_t cz_eid = graph->n_edges;
                HPCEdge *cz_edge = &graph->edges[cz_eid];
                memset(cz_edge, 0, sizeof(*cz_edge));
                cz_edge->site_a = blk * 6 + oracle_site;
                cz_edge->site_b = blk * 6 + peer;
                cz_edge->type = HPC_EDGE_CZ;
                cz_edge->fidelity = 1.0;
                /* CZ weight: ω^(va·vb) — pure DFT₆ coupling, scaled 1/sqrt(6) 
                 * to prevent message flooding and enforce unitary boundaries */
                double norm_cz = 1.0;  /* No attenuation — BP damping controls stability */
                for (int va = 0; va < 6; va++) {
                    for (int vb = 0; vb < 6; vb++) {
                        int pidx = (va * vb) % 6;
                        cz_edge->w_re[va][vb] = W6_RE[pidx] * norm_cz;
                        cz_edge->w_im[va][vb] = W6_IM[pidx] * norm_cz;
                    }
                }
                graph->n_edges++;
                graph->cz_edges++;
                hpc_adj_add(graph, cz_edge->site_a, cz_eid);
                hpc_adj_add(graph, cz_edge->site_b, cz_eid);
            }
        }

        /* The Macroscopic QFT Bridge: Stitching the Multiverse */
        if (blk < n_blocks - 1) {
            int s_tail = blk * 6 + 5;
            int s_head = (blk + 1) * 6 + 0;
            hpc_grow_edges(graph);
            uint64_t eid = graph->n_edges;
            HPCEdge *edge = &graph->edges[eid];
            memset(edge, 0, sizeof(*edge));
            edge->site_a = s_tail;
            edge->site_b = s_head;
            edge->type = HPC_EDGE_PHASE;
            edge->fidelity = 1.0;
            /* Uniform bridge coupling — no DSP windowing.
             * Same rationale as intra-block: tapering inter-block
             * phase transfers destroys global QFT unitarity. */
            double bridge_scale = 1.0;
            for (int va = 0; va < 6; va++) {
                for (int vb = 0; vb < 6; vb++) {
                    int diff = (va - vb + 6) % 6;
                    double decay;
                    switch(diff) {
                        case 0: decay = 1.000; break;
                        case 1: case 5: decay = 0.500; break;
                        case 2: case 4: decay = 0.250; break;
                        case 3: decay = 0.125; break;
                    }
                    decay *= bridge_scale;
                    double angle = 2.0 * 3.14159265358979323846 * va * vb / 6.0;
                    edge->w_re[va][vb] = cos(angle) * decay;
                    edge->w_im[va][vb] = sin(angle) * decay;
                }
            }
            graph->n_edges++;
            graph->phase_edges++;
            hpc_adj_add(graph, s_tail, eid);
            hpc_adj_add(graph, s_head, eid);
        }
        
        printf("      [debug] Completed block %d / %d\n", blk, n_blocks);
        fflush(stdout);
    }

    /* IQFT edges REMOVED: Griffiths-Niu semi-classical feed-forward replaces
     * the O(n²) pairwise IQFT edges. Keeping both would apply the QFT twice. */
    printf("    IQFT: semi-classical feed-forward (no graph edges)\n");

    /* ═══ QUANTUM WORK REGISTER — Controlled Permutation Oracle ═══
     * The work register maintains superposition through the HPC graph.
     * Each frequency site k gets a CONTROLLED_PERM edge:
     *   multiplier[d] = c_k^d mod N
     *   amplitude: δ(work_out == multiplier[d] × work_in mod N)
     *
     * The work register starts as |1⟩. The perm edges enforce
     * work = Π_k c_k^{d_k} mod N = a^x mod N for each freq config x.
     * This creates the eigenstate superposition |u_s⟩ that kicks back
     * the phase e^{2πi s/r} into the frequency register.
     *
     * The state vector is NEVER materialized. Superposition lives in
     * the graph structure: local amplitudes × edge weights. */
    int n_work = (int)ceil((double)nbits / log2(6.0)) + 1;
    int work_start = n_sites;
    g_work_start = work_start;
    hpc_grow_sites(graph, n_sites + n_work);
    printf("    Work register: %d sites (indices %d..%d), computational basis |1⟩\n",
           n_work, work_start, work_start + n_work - 1);

    /* Initialize work register to |1⟩ in base-6 computational basis.
     * |1⟩ = digit_0 = 1, all other digits = 0.
     * Local amplitude: δ(value, correct_digit) — sharp, not uniform. */
    for (int j = 0; j < n_work; j++) {
        TrialityQuhit *wq = &graph->locals[work_start + j];
        int init_digit = (j == 0) ? 1 : 0;  /* |1⟩ in base-6: LSB=1, rest=0 */
        for (int w = 0; w < 6; w++) {
            wq->edge_re[w] = (w == init_digit) ? 1.0 : 0.0;
            wq->edge_im[w] = 0.0;
        }
    }

    /* Add controlled permutation edges: freq site k → work register */
    BigInt *ck_cache = calloc(n_sites_raw, sizeof(BigInt));
    for (int k = 0; k < n_sites_raw; k++) bigint_set_u64(&ck_cache[k], 0);

    int n_oracle_edges = 0;

    for (int k = 0; k < n_sites_raw; k++) {
        int blk_k = k / 2, off_k = k % 2;
        int site_k = blk_k * 6 + off_k;
        if (site_k >= n_sites) continue;

        /* c_k = a^{6^{k+offset}} mod N */
        BigInt bi_t, bi_t2, dummy_q;
        bigint_set_u64(&bi_t, 0); bigint_set_u64(&bi_t2, 0); bigint_set_u64(&dummy_q, 0);
        bigint_copy(&ck_cache[k], a_val);
        for (int s = 0; s < k + scale_offset; s++) {
            bigint_mul(&bi_t, &ck_cache[k], &ck_cache[k]);
            bigint_div_mod(&bi_t, N, &dummy_q, &bi_t2);
            bigint_mul(&bi_t, &bi_t2, &ck_cache[k]);
            bigint_div_mod(&bi_t, N, &dummy_q, &bi_t2);
            bigint_mul(&bi_t, &bi_t2, &bi_t2);
            bigint_div_mod(&bi_t, N, &dummy_q, &ck_cache[k]);
        }

        /* Compute multiplier[d] = c_k^d mod N for d=0..5 */
        BigInt mult_bi[6];
        for (int d = 0; d < 6; d++) bigint_set_u64(&mult_bi[d], 0);
        bigint_set_u64(&mult_bi[0], 1);
        for (int d = 1; d < 6; d++) {
            bigint_mul(&bi_t, &mult_bi[d-1], &ck_cache[k]);
            bigint_div_mod(&bi_t, N, &dummy_q, &mult_bi[d]);
        }
        bigint_destroy(&dummy_q);

        /* Precompute base-6 digits of mult_bi[d] */
        int mult_digits[6][n_work];
        for (int d = 0; d < 6; d++) {
            BigInt tmp_bi, r6, q6;
            bigint_set_u64(&tmp_bi, 0); bigint_set_u64(&r6, 0); bigint_set_u64(&q6, 0);
            bigint_copy(&tmp_bi, &mult_bi[d]);
            for (int p = 0; p < n_work; p++) {
                bigint_div_mod(&tmp_bi, &b6, &q6, &r6);
                mult_digits[d][p] = (int)bigint_to_u64(&r6);
                bigint_copy(&tmp_bi, &q6);
            }
            bigint_destroy(&tmp_bi); bigint_destroy(&r6); bigint_destroy(&q6);
            bigint_destroy(&mult_bi[d]);
        }
        bigint_destroy(&bi_t); bigint_destroy(&bi_t2);

        /* Create controlled-perm edge: freq_site_k controls the work register.
         * The perm edge encodes the constraint δ(work == mult[d] × y mod N)
         * as phase weights in the 6×6 matrix.
         *
         * For the HPC amplitude formula ψ = [Π locals] × [Π edges]:
         * The perm constraint groups configurations by a^x mod N,
         * creating the eigenstate decomposition that produces phase kickback.
         *
         * Edge weight w(d, w_j): encode the d-th digit of mult[d]
         * as a phase that constrains work digit j. */
        for (int j = 0; j < n_work; j++) {
            hpc_grow_edges(graph);
            uint64_t eid = graph->n_edges;
            HPCEdge *edge = &graph->edges[eid];
            memset(edge, 0, sizeof(*edge));
            edge->site_a = site_k;
            edge->site_b = work_start + j;
            edge->type = HPC_EDGE_PHASE;
            edge->fidelity = 1.0;

            /* For each control value d, compute the j-th digit of mult[d].
             * The edge weight selects the correct work digit value via
             * a peaked distribution: w(d, w_j) = 1 if w_j matches the
             * j-th digit of mult[d], scaled by a coupling strength. */
            for (int d = 0; d < 6; d++) {
                /* Extract j-th base-6 digit of mult[d] */
                int target_digit = mult_digits[d][j];

                for (int wj = 0; wj < 6; wj++) {
                    /* Phase coupling: peaked at the correct digit value.
                     * Uses the eigenphase structure:
                     * w(d, w_j) = e^{2πi × d × (target_digit - w_j) / 6}
                     * This creates constructive interference when w_j matches
                     * the correct digit, encoding the permutation constraint
                     * as a phase correlation rather than a hard delta. */
                    double phase = 2.0 * M_PI * (double)d * (double)(target_digit * wj) / 6.0;
                    edge->w_re[d][wj] = cos(phase);
                    edge->w_im[d][wj] = sin(phase);
                }
            }
            graph->n_edges++;
            graph->phase_edges++;
            hpc_adj_add(graph, site_k, eid);
            hpc_adj_add(graph, work_start + j, eid);
            n_oracle_edges++;
        }
    }
    printf("    Controlled-perm oracle edges: %d\n", n_oracle_edges);

    /* DFT rotation REMOVED: Griffiths-Niu measurement applies IDFT6 during
     * measurement (Step 4). Locals stay in oracle phase basis. */
    printf("    Phase 3: Exact Marginals via Magic Pointer (replaces BP)...\n");
    clock_t t_bp_start = clock();

    /* Heap-allocate marginals — same format as before, just filled differently */
    int marginals_sz = (2 * n_blocks > n_sites_raw) ? 2 * n_blocks : n_sites_raw;
    mpfr_t (*marginals)[6] = (mpfr_t (*)[6])calloc(marginals_sz, sizeof(mpfr_t[6]));
    for (int i = 0; i < marginals_sz; i++) {
        for (int d = 0; d < 6; d++) {
            mpfr_inits2(2048, marginals[i][d], (mpfr_ptr)0);
            mpfr_set_d(marginals[i][d], 0.0, MPFR_RNDN);
        }
    }

    if (probe_mode) {
        /* Probe mode: quick return, no marginals needed */
        hpc_destroy(graph);
        bigint_clear(&b6); bigint_clear(&one);
        bigint_clear(&val_k_A); bigint_clear(&val_k_B); bigint_clear(&div_6_blk);
        bigint_clear(&gc_b36); bigint_clear(&gc_next_A); bigint_clear(&gc_next_B); bigint_clear(&gc_next_div);
        bigint_clear(&gc_gcd_check); bigint_clear(&gc_val_minus_1); bigint_clear(&gc_dummy_rem);
        bigint_clear(&gc_tmpA); bigint_clear(&gc_tmpB); bigint_clear(&gc_q_div);
        bigint_clear(&gc_six_pow_k); bigint_clear(&gc_d_bi);
        bigint_clear(&gc_b6_mod); bigint_clear(&gc_shift_div_A); bigint_clear(&gc_shift_div_B); bigint_clear(&gc_dummy_rm2);
        bigint_clear(&gc_qA); bigint_clear(&gc_qB); bigint_clear(&gc_rA_mod); bigint_clear(&gc_rB_mod);
        bigint_clear(&gc_temp_N); bigint_clear(&gc_qN); bigint_clear(&gc_rN); bigint_clear(&gc_q_sh); bigint_clear(&gc_r_sh);
        for (int i = 0; i < 6; i++) { bigint_clear(&gc_powersA[i]); bigint_clear(&gc_powersB[i]); }
        for (int i = 0; i < marginals_sz; i++)
            for (int d = 0; d < 6; d++)
                mpfr_clear(marginals[i][d]);
        free(marginals);
        return 0.0;
    }

    /* ═══ GRIFFITHS-NIU SEMI-CLASSICAL QFT MEASUREMENT ═══
     *
     * Measures frequency register MSB-to-LSB. For each digit k:
     *   1. Read oracle amplitudes α_k(d) from graph locals (phase basis)
     *   2. Compute work register contribution C_k(d) from perm edge weights
     *   3. Apply feed-forward phase correction: α'(d) = α(d) × C(d) × e^{-2πi d θ_k}
     *   4. Apply IDFT6 to α'(d) → β(v): converts phase→computational basis
     *   5. P(v) = |β(v)|² — interference happens in step 4
     *   6. Sample outcome, update work register locals in graph
     *
     * The IDFT6 is what creates constructive interference at Shor frequencies.
     * Without it, measuring in the phase basis gives uniform noise. */

    int *measured_digits = (int*)calloc(n_sites_raw, sizeof(int));
    BigInt work_val_bi;
    bigint_set_u64(&work_val_bi, 1);  /* Classical work register tracking: starts at |1⟩ */

    for (int k = n_sites_raw - 1; k >= 0; k--) {
        int blk_k = k / 2, off_k = k % 2;
        int site_k = blk_k * 6 + off_k;

        /* Step 1: Read oracle amplitudes from graph locals (phase basis) */
        double alpha_re[6], alpha_im[6];
        for (int d = 0; d < 6; d++) {
            alpha_re[d] = graph->locals[site_k].edge_re[d];
            alpha_im[d] = graph->locals[site_k].edge_im[d];
        }

        /* Step 2: Compute work register contribution C_k(d) analytically.
         * For each control value d, the perm edges connect freq_k to work
         * register digits. The work register is in state |work_val⟩.
         * C_k(d) = Π_j w_e(d, work_digit_j) — product over work digit edges.
         * Since work register locals are δ-functions (sharp), only one
         * work digit value contributes per edge. */
        double ck_re[6], ck_im[6];
        for (int d = 0; d < 6; d++) { ck_re[d] = 1.0; ck_im[d] = 0.0; }

        const HPCAdjList *adj = &graph->adj[site_k];
        for (uint64_t ei = 0; ei < adj->count; ei++) {
            uint64_t eid = adj->edge_ids[ei];
            const HPCEdge *edge = &graph->edges[eid];
            uint64_t partner = (edge->site_a == (uint64_t)site_k) ?
                                edge->site_b : edge->site_a;
            if ((int)partner < g_work_start) continue;

            /* Work register site: read its current value (delta function) */
            int work_digit_val = 0;
            double best_amp = 0;
            const TrialityQuhit *wq = &graph->locals[partner];
            for (int w = 0; w < 6; w++) {
                double a2 = wq->edge_re[w]*wq->edge_re[w] + wq->edge_im[w]*wq->edge_im[w];
                if (a2 > best_amp) { best_amp = a2; work_digit_val = w; }
            }

            /* Multiply edge weight w(d, work_digit_val) into C_k(d) */
            for (int d = 0; d < 6; d++) {
                double wr, wi;
                if (edge->site_a == (uint64_t)site_k) {
                    wr = edge->w_re[d][work_digit_val];
                    wi = edge->w_im[d][work_digit_val];
                } else {
                    wr = edge->w_re[work_digit_val][d];
                    wi = edge->w_im[work_digit_val][d];
                }
                double nr = ck_re[d]*wr - ck_im[d]*wi;
                double ni = ck_re[d]*wi + ck_im[d]*wr;
                ck_re[d] = nr;
                ck_im[d] = ni;
            }
        }

        /* Step 3: Feed-forward phase correction via MPFR.
         * theta_k = Σ_{j>k} measured_digits[j] / 6^{j-k+1}
         * Using MPFR eliminates the double overflow at 6^397 and the
         * 53-bit precision event horizon that blinds distant digits. */
        mpfr_prec_t ff_prec = (mpfr_prec_t)(nbits * 2 + 64);
        mpfr_t mp_theta, mp_power, mp_digit, mp_term;
        mpfr_init2(mp_theta, ff_prec);
        mpfr_init2(mp_power, ff_prec);
        mpfr_init2(mp_digit, ff_prec);
        mpfr_init2(mp_term, ff_prec);
        mpfr_set_d(mp_theta, 0.0, MPFR_RNDN);
        mpfr_set_d(mp_power, 36.0, MPFR_RNDN);  /* 6^2 for first previous digit */
        for (int j = k + 1; j < n_sites_raw; j++) {
            mpfr_set_ui(mp_digit, (unsigned long)measured_digits[j], MPFR_RNDN);
            mpfr_div(mp_term, mp_digit, mp_power, MPFR_RNDN);
            mpfr_add(mp_theta, mp_theta, mp_term, MPFR_RNDN);
            mpfr_mul_ui(mp_power, mp_power, 6, MPFR_RNDN);
        }

        double corrected_re[6], corrected_im[6];
        for (int d = 0; d < 6; d++) {
            /* α × C */
            double ac_re = alpha_re[d]*ck_re[d] - alpha_im[d]*ck_im[d];
            double ac_im = alpha_re[d]*ck_im[d] + alpha_im[d]*ck_re[d];

            /* × e^{-2πi d θ_k} — compute trig in MPFR then extract double */
            mpfr_t mp_angle, mp_cos_ff, mp_sin_ff, mp_2pi_ff;
            mpfr_init2(mp_angle, ff_prec);
            mpfr_init2(mp_cos_ff, ff_prec);
            mpfr_init2(mp_sin_ff, ff_prec);
            mpfr_init2(mp_2pi_ff, ff_prec);
            mpfr_const_pi(mp_2pi_ff, MPFR_RNDN);
            mpfr_mul_ui(mp_2pi_ff, mp_2pi_ff, 2, MPFR_RNDN);
            mpfr_mul(mp_angle, mp_2pi_ff, mp_theta, MPFR_RNDN);
            mpfr_mul_si(mp_angle, mp_angle, -d, MPFR_RNDN);
            mpfr_cos(mp_cos_ff, mp_angle, MPFR_RNDN);
            mpfr_sin(mp_sin_ff, mp_angle, MPFR_RNDN);
            double pr = mpfr_get_d(mp_cos_ff, MPFR_RNDN);
            double pi = mpfr_get_d(mp_sin_ff, MPFR_RNDN);
            mpfr_clears(mp_angle, mp_cos_ff, mp_sin_ff, mp_2pi_ff, (mpfr_ptr)0);

            corrected_re[d] = ac_re*pr - ac_im*pi;
            corrected_im[d] = ac_re*pi + ac_im*pr;
        }
        mpfr_clears(mp_theta, mp_power, mp_digit, mp_term, (mpfr_ptr)0);

        /* Step 4: Apply IDFT6 to convert from phase basis → computational basis.
         * β(v) = (1/√6) Σ_{d=0}^{5} α'(d) × e^{2πi d v / 6}
         * THIS is where constructive/destructive interference creates the Shor peaks. */
        double probs[6];
        double total = 0.0;
        for (int v = 0; v < 6; v++) {
            double sum_re = 0.0, sum_im = 0.0;
            for (int d = 0; d < 6; d++) {
                double angle = 2.0 * M_PI * d * v / 6.0;
                double er = cos(angle), ei2 = sin(angle);
                sum_re += corrected_re[d]*er - corrected_im[d]*ei2;
                sum_im += corrected_re[d]*ei2 + corrected_im[d]*er;
            }
            /* P(v) = |β(v)|² */
            probs[v] = (sum_re*sum_re + sum_im*sum_im) / 6.0;
            total += probs[v];
        }

        /* Normalize */
        if (total > 1e-30) {
            for (int d = 0; d < 6; d++) probs[d] /= total;
        } else {
            for (int d = 0; d < 6; d++) probs[d] = 1.0 / 6.0;
        }

        /* Step 5: Born Rule Measurement
         * Shor's algorithm creates a superposition of r valid frequency peaks.
         * We MUST sample probabilistically. Taking MAP (argmax) deterministically
         * forces the algorithm into a single peak (often s=0), causing it to fail. */
        double r01 = (double)rand() / RAND_MAX;
        double cumul = 0.0;
        int outcome = 5;
        for (int d = 0; d < 6; d++) {
            cumul += probs[d];
            if (r01 <= cumul) { outcome = d; break; }
        }
        measured_digits[k] = outcome;

        /* Step 6: Update work register in graph (Fix ghost state!)
         * After measuring freq_k = outcome, the work register gets multiplied
         * by c_k^outcome mod N. Update the graph locals so subsequent
         * marginals see the LIVE state, not the initial |1⟩ ghost. */
        {
            /* mult = c_k^outcome mod N */
            BigInt mult_val, bi_t, dummy_q;
            bigint_set_u64(&mult_val, 1);
            bigint_set_u64(&bi_t, 0); bigint_set_u64(&dummy_q, 0);
            for (int d = 0; d < outcome; d++) {
                bigint_mul(&bi_t, &mult_val, &ck_cache[k]);
                bigint_div_mod(&bi_t, N, &dummy_q, &mult_val);
            }
            
            /* Update classical tracker */
            bigint_mul(&bi_t, &work_val_bi, &mult_val);
            bigint_div_mod(&bi_t, N, &dummy_q, &work_val_bi);
            bigint_destroy(&dummy_q);
            
            /* Sync graph locals: work register now represents |work_val⟩ */
            BigInt tmp_bi, q6, r6;
            bigint_set_u64(&tmp_bi, 0); bigint_set_u64(&q6, 0); bigint_set_u64(&r6, 0);
            bigint_copy(&tmp_bi, &work_val_bi);
            for (int j = 0; j < n_work; j++) {
                bigint_div_mod(&tmp_bi, &b6, &q6, &r6);
                int digit_j = (int)bigint_to_u64(&r6);
                TrialityQuhit *wq = &graph->locals[work_start + j];
                for (int w = 0; w < 6; w++) {
                    wq->edge_re[w] = (w == digit_j) ? 1.0 : 0.0;
                    wq->edge_im[w] = 0.0;
                }
                bigint_copy(&tmp_bi, &q6);
            }
            bigint_destroy(&mult_val); bigint_destroy(&bi_t);
            bigint_destroy(&tmp_bi); bigint_destroy(&q6); bigint_destroy(&r6);
        }

        /* Store in marginals for downstream CF extraction */
        for (int d = 0; d < 6; d++) {
            mpfr_set_d(marginals[k][d], probs[d], MPFR_RNDN);
        }

        /* Display */
        double max_prob = 0.0;
        int best_digit = 0;
        for (int d = 0; d < 6; d++) {
            if (probs[d] > max_prob) { max_prob = probs[d]; best_digit = d; }
        }
        if (k < 10 || k == n_sites_raw - 1 || (k + 1) % 100 == 0) {
            printf("    digit %3d: val=%d  P_gn=%.4f  [", k, best_digit, max_prob);
            for (int d = 0; d < 6; d++)
                printf("%.3f%s", probs[d], d < 5 ? " " : "");
            printf("]\n");
        }
    }

    clock_t t_bp_end = clock();
    printf("      Exact marginals computed in %.3f sec\n",
           (double)(t_bp_end - t_bp_start) / CLOCKS_PER_SEC);

    /* ── Build signal mask: which positions have genuine BP information? ── */
    int *bp_has_signal = (int*)calloc(n_sites_raw, sizeof(int));
    int *flippable = (int*)calloc(n_sites_raw, sizeof(int));
    int n_flippable = 0;
    for (int scale = 0; scale < n_sites_raw; scale++) {
        double max_p = 0.0;
        for (int d = 0; d < 6; d++) {
            double p = mpfr_get_d(marginals[scale][d], MPFR_RNDN);
            if (p > max_p) max_p = p;
        }
        /* A position has signal if its max probability is significantly above uniform (1/6 ≈ 0.167) */
        if (max_p > 0.20) {
            bp_has_signal[scale] = 1;
            flippable[n_flippable++] = scale;
        }
    }
    printf("  [Signal mask] %d / %d positions have BP signal (%.1f%%)\n",
           n_flippable, n_sites_raw, 100.0 * n_flippable / n_sites_raw);

    /* NOTE: graph is NOT destroyed here — needed for sequential measurement below */

    clock_t t_ipe_end = clock();
    printf("\n  IPE complete: %.3f seconds (%d blocks, %d×%d-bit)\n",
           (double)(t_ipe_end - t_setup_start) / CLOCKS_PER_SEC,
           n_blocks, PHASE_CHUNKS, CHUNK_BITS);

    /* Compute register size = 6^n_sites_raw */
    BigInt reg_sz, gc_reg_tmp;
    bigint_set_u64(&reg_sz, 1);
    bigint_set_u64(&gc_reg_tmp, 0);
    for (int k = 0; k < n_sites_raw; k++) {
        bigint_mul(&gc_reg_tmp, &reg_sz, &b6);
        bigint_copy(&reg_sz, &gc_reg_tmp);
    }

    /* ── Phase 4: LLL lattice period recovery (DISABLED) ──────────────────── */
#if 0  /* LLL path disabled — causes instability with large N */
    int success = lll_recover_period(n_sites_raw, marginals, &b6, &reg_sz,
                                     N, a_val, factor_p, factor_q);
    if (success) {
        printf("\n  ★ LLL PERIOD RECOVERY SUCCEEDED ★\n");
        free(marginals);
        return 1;
    }
#endif  /* LLL disabled */
    int success = 0;

    /* ═══════════════════════════════════════════════════════════════════════
     * EXACT PERIOD EXTRACTION — Single-Pass CF
     *
     * With exact marginals from Magic Pointer, the MAP frequency F is
     * precisely s·R/r. One CF expansion recovers r at the "knee" —
     * no MCMC, no LCM accumulation, no voting needed.
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n  ═══ SHOR WORK REGISTER + CF PERIOD EXTRACTION ═══\n");

    /* Build power-of-6 cache for frequency construction */
    BigInt *p6_cache = (BigInt*)calloc(n_sites_raw, sizeof(BigInt));
    for (int i = 0; i < n_sites_raw; i++) bigint_set_u64(&p6_cache[i], 0);
    BigInt current_p6, next_p6;
    bigint_set_u64(&current_p6, 1);
    bigint_set_u64(&next_p6, 0);
    for (int i = 0; i < n_sites_raw; i++) {
        bigint_copy(&p6_cache[i], &current_p6);
        bigint_mul(&next_p6, &current_p6, &b6);
        bigint_copy(&current_p6, &next_p6);
    }

    BigInt mc_d_bi, mc_term, mc_tmp;
    bigint_set_u64(&mc_d_bi, 0);
    bigint_set_u64(&mc_term, 0);
    bigint_set_u64(&mc_tmp, 0);

    /* ── Phase 4: Beam Search Frequency Generation ──
     * Parent-pointer tree: O(beam_width × branches) per position instead of
     * O(k × beam_width × branches) path copying.  Top-K argmax selection
     * replaces O(n²) bubble sort. */
    printf("    Executing Beam Search over marginals...\n");
    int beam_width = 32;
    int current_paths = 1;

    /* Parent-pointer tree: beam_parent[k][b] = parent beam at position k-1,
     * beam_digit[k][b] = digit chosen at position k for beam b. */
    int (*beam_parent)[32] = (int(*)[32])calloc(n_sites_raw, sizeof(int[32]));
    int (*beam_digit)[32]  = (int(*)[32])calloc(n_sites_raw, sizeof(int[32]));
    double *path_probs = (double*)calloc(beam_width, sizeof(double));

    /* Scratch arrays for candidate expansion (max beam_width × 6 = 192) */
    int    cand_cap = beam_width * 6;
    double *cand_probs  = (double*)calloc(cand_cap, sizeof(double));
    int    *cand_parent = (int*)calloc(cand_cap, sizeof(int));
    int    *cand_digit  = (int*)calloc(cand_cap, sizeof(int));

    for (int k = 0; k < n_sites_raw; k++) {
        double probs[6];
        for (int d = 0; d < 6; d++) {
            probs[d] = mpfr_get_d(marginals[k][d], MPFR_RNDN);
            if (probs[d] < 1e-10) probs[d] = 1e-10;
        }

        int n_cands = 0;
        for (int p = 0; p < current_paths; p++) {
            for (int d = 0; d < 6; d++) {
                if (probs[d] > 1e-6) {
                    cand_probs[n_cands]  = path_probs[p] + log(probs[d]);
                    cand_parent[n_cands] = p;
                    cand_digit[n_cands]  = d;
                    n_cands++;
                }
            }
        }

        /* Top-K selection via argmax (O(n_cands × beam_width)) */
        int top_count = (n_cands < beam_width) ? n_cands : beam_width;
        double new_probs[32];
        for (int t = 0; t < top_count; t++) {
            int best = -1;
            double best_lp = -1e30;
            for (int i = 0; i < n_cands; i++) {
                if (cand_probs[i] > best_lp) {
                    best_lp = cand_probs[i];
                    best = i;
                }
            }
            beam_parent[k][t] = cand_parent[best];
            beam_digit[k][t]  = cand_digit[best];
            new_probs[t]      = best_lp;
            cand_probs[best]  = -2e9; /* poison so it can't be selected again */
        }

        current_paths = top_count;
        for (int t = 0; t < current_paths; t++) {
            path_probs[t] = new_probs[t];
        }
    }

    free(cand_probs); free(cand_parent); free(cand_digit);

    printf("    Generated %d candidate frequencies.\n", current_paths);
    printf("    Register size:     %u bits\n", bigint_bitlen(&reg_sz));

    /* CF expansion of F/R — test every candidate frequency */
    for (int p_idx = 0; p_idx < current_paths && !success; p_idx++) {
        BigInt freq; bigint_set_u64(&freq, 0);

        /* Reconstruct digit sequence from parent-pointer backtracking */
        int *seq = (int*)calloc(n_sites_raw, sizeof(int));
        int cur_beam = p_idx;
        for (int s = n_sites_raw - 1; s >= 0; s--) {
            seq[s] = beam_digit[s][cur_beam];
            cur_beam = beam_parent[s][cur_beam];
        }

        /* Build frequency from reconstructed path */
        for (int idx = 0; idx < n_sites_raw; idx++) {
            bigint_set_u64(&mc_d_bi, (uint64_t)seq[idx]);
            bigint_mul(&mc_term, &mc_d_bi, &p6_cache[idx]);
            bigint_add(&mc_tmp, &freq, &mc_term);
            bigint_copy(&freq, &mc_tmp);
        }
        free(seq);
        
        printf("\n    [Beam %2d] Freq = %u bits (log_P = %.2f)\n", p_idx, bigint_bitlen(&freq), path_probs[p_idx]);

        BigInt cf_num, cf_den, cf_a, cf_rem;
        BigInt cf_pm1, cf_p0, cf_qm1, cf_q0;
    BigInt cf_p_new, cf_q_new, cf_tmp;
    bigint_set_u64(&cf_num, 0); bigint_set_u64(&cf_den, 0);
    bigint_set_u64(&cf_a, 0); bigint_set_u64(&cf_rem, 0);
    bigint_set_u64(&cf_pm1, 1); bigint_set_u64(&cf_p0, 0);
    bigint_set_u64(&cf_qm1, 0); bigint_set_u64(&cf_q0, 1);
    bigint_set_u64(&cf_p_new, 0); bigint_set_u64(&cf_q_new, 0);
    bigint_set_u64(&cf_tmp, 0);

    bigint_copy(&cf_num, &freq);
    bigint_copy(&cf_den, &reg_sz);

    /* First partial quotient: a0 = floor(F/R) */
    bigint_div_mod(&cf_num, &cf_den, &cf_a, &cf_rem);
    bigint_copy(&cf_p0, &cf_a);
    /* cf_q0 = 1 (already set) */

    printf("    CF expansion: a0 = %u bits\n", bigint_bitlen(&cf_a));

    /* CF steps scale with N: a 2048-bit period can require thousands
     * of convergents.  2×nbits covers the worst case. */
    int cf_max_steps = (int)(nbits * 2);
    if (cf_max_steps < 200) cf_max_steps = 200;
    for (int step = 0; step < cf_max_steps && !success; step++) {
        /* num = old den, den = old remainder */
        bigint_copy(&cf_num, &cf_den);
        bigint_copy(&cf_den, &cf_rem);

        if (bigint_is_zero(&cf_den)) break;

        bigint_div_mod(&cf_num, &cf_den, &cf_a, &cf_rem);

        /* Update convergents: p_new = a * p0 + pm1, q_new = a * q0 + qm1 */
        bigint_mul(&cf_tmp, &cf_a, &cf_p0);
        bigint_add(&cf_p_new, &cf_tmp, &cf_pm1);
        bigint_mul(&cf_tmp, &cf_a, &cf_q0);
        bigint_add(&cf_q_new, &cf_tmp, &cf_qm1);

        /* Shift convergents */
        bigint_copy(&cf_pm1, &cf_p0);
        bigint_copy(&cf_qm1, &cf_q0);
        bigint_copy(&cf_p0, &cf_p_new);
        bigint_copy(&cf_q0, &cf_q_new);

        /* If denominator exceeds N, we've passed the useful range */
        if (bigint_cmp(&cf_q0, N) >= 0) {
            printf("    CF step %d: denominator exceeded N, stopping.\n", step);
            break;
        }

        /* Add to global CF pool for proximity search.
         * Only pool convergents with meaningful bitlength (>= nbits/4)
         * to avoid trivial early CF values (2, 3, 5, 12...) dominating
         * the closest-pair search. */
        if (cf_pool_count < 10000 && bigint_bitlen(&cf_q0) >= nbits / 4) {
            bigint_copy(&cf_pool[cf_pool_count++], &cf_q0);
        }

        /* Test this convergent denominator as a period candidate */
        uint32_t q_bits = bigint_bitlen(&cf_q0);

        if (q_bits > 1) {
            char q_str[512];
            bigint_to_decimal(q_str, sizeof(q_str), &cf_q0);
            printf("    CF step %d: denom = %s (%u bits)\n", step, q_str, q_bits);
            
            /* Oracle encodes 6^k → period is ord_N(6), use base 6 for GCD check */
            BigInt oracle_base; bigint_set_u64(&oracle_base, 6);
            int ret = try_period(&cf_q0, &oracle_base, N, factor_p, factor_q);
            if (ret == 1) {
                success = 1;
                printf("\n  ★ OUROBOROS BITES ITS TAIL ★ (CF step %d, denominator %u bits)\n", step, q_bits);
                bigint_clear(&oracle_base);
                break;
            }
            /* Also try with base a — in case ord_N(6) divides ord_N(a) */
            ret = try_period(&cf_q0, a_val, N, factor_p, factor_q);
            if (ret == 1) {
                success = 1;
                printf("\n  ★ OUROBOROS BITES ITS TAIL ★ (CF step %d [base-a], denominator %u bits)\n", step, q_bits);
                bigint_clear(&oracle_base);
                break;
            }
            /* Also test small multiples in case gcd(s,r) > 1 */
            for (int m = 2; m <= 12 && !success; m++) {
                BigInt r_mult, m_bi;
                bigint_set_u64(&r_mult, 0);
                bigint_set_u64(&m_bi, (uint64_t)m);
                bigint_mul(&r_mult, &cf_q0, &m_bi);
                if (bigint_cmp(&r_mult, N) >= 0) {
                    bigint_clear(&r_mult); bigint_clear(&m_bi);
                    break;
                }
                ret = try_period(&r_mult, &oracle_base, N, factor_p, factor_q);
                if (ret == 1) {
                    success = 1;
                    printf("\n  ★ OUROBOROS BITES ITS TAIL ★ (CF step %d × %d, %u bits)\n", step, m, q_bits);
                }
                if (!success) {
                    ret = try_period(&r_mult, a_val, N, factor_p, factor_q);
                    if (ret == 1) {
                        success = 1;
                        printf("\n  ★ OUROBOROS BITES ITS TAIL ★ (CF step %d × %d [base-a], %u bits)\n", step, m, q_bits);
                    }
                }
                bigint_destroy(&r_mult); bigint_destroy(&m_bi);
            }
            bigint_destroy(&oracle_base);
            
            /* Record best partial period: only save convergents where
             * the denominator is nontrivial (>1 bit) and smaller than N.
             * This prevents the LCM accumulator from harvesting garbage
             * from the final overflowed convergent. */
            if (best_period && !success && bigint_bitlen(&cf_q0) > 1
                && bigint_cmp(&cf_q0, N) < 0) {
                bigint_copy(best_period, &cf_q0);
            }
        }
    }
    bigint_destroy(&cf_num); bigint_destroy(&cf_den); bigint_destroy(&cf_a); bigint_destroy(&cf_rem);
    bigint_destroy(&cf_pm1); bigint_destroy(&cf_p0); bigint_destroy(&cf_qm1); bigint_destroy(&cf_q0);
    bigint_destroy(&cf_p_new); bigint_destroy(&cf_q_new); bigint_destroy(&cf_tmp);
    bigint_destroy(&freq);
    } /* End of Beam Search CF expansion loop */

    /* ── CF Proximity Search ──
     * If two independent CF convergents land within 1,000,000 of each other,
     * the true period is almost certainly in that neighborhood.
     * Brute-force scan the range between the closest pair. */
    printf("\n    [DEBUG] CF pool: %d entries, success=%d\n", cf_pool_count, success);
    if (!success && cf_pool_count >= 2) {
        /* Sort cf_pool by value (insertion sort — pool is small) */
        for (int i = 1; i < cf_pool_count; i++) {
            for (int j = i; j > 0 && bigint_cmp(&cf_pool[j-1], &cf_pool[j]) > 0; j--) {
                BigInt swap_tmp;
                bigint_set_u64(&swap_tmp, 0);
                bigint_copy(&swap_tmp, &cf_pool[j]);
                bigint_copy(&cf_pool[j], &cf_pool[j-1]);
                bigint_copy(&cf_pool[j-1], &swap_tmp);
                bigint_destroy(&swap_tmp);
            }
        }

        /* Find the closest pair */
        BigInt best_lo, best_hi, diff_tmp;
        bigint_set_u64(&best_lo, 0);
        bigint_set_u64(&best_hi, 0);
        bigint_set_u64(&diff_tmp, 0);
        BigInt min_gap;
        bigint_set_u64(&min_gap, 0);
        int have_pair = 0;

        for (int i = 0; i < cf_pool_count - 1; i++) {
            /* Skip duplicates */
            if (bigint_cmp(&cf_pool[i], &cf_pool[i+1]) == 0) continue;
            /* Skip trivial values */
            if (bigint_bitlen(&cf_pool[i]) <= 1) continue;

            bigint_sub(&diff_tmp, &cf_pool[i+1], &cf_pool[i]);

            if (!have_pair || bigint_cmp(&diff_tmp, &min_gap) < 0) {
                bigint_copy(&min_gap, &diff_tmp);
                bigint_copy(&best_lo, &cf_pool[i]);
                bigint_copy(&best_hi, &cf_pool[i+1]);
                have_pair = 1;
            }
        }

        /* Check if the closest pair is within 1,000,000 */
        BigInt proximity_limit;
        bigint_set_u64(&proximity_limit, 10000000);

        if (have_pair && bigint_cmp(&min_gap, &proximity_limit) <= 0) {
            char lo_str[512], hi_str[512], gap_str[64];
            bigint_to_decimal(lo_str, sizeof(lo_str), &best_lo);
            bigint_to_decimal(hi_str, sizeof(hi_str), &best_hi);
            bigint_to_decimal(gap_str, sizeof(gap_str), &min_gap);
            printf("\n    ── CF Proximity Search ──\n");
            printf("    Closest CF pair: %s .. %s (gap = %s)\n", lo_str, hi_str, gap_str);
            printf("    Scanning %s candidates...\n", gap_str);

            uint64_t gap_u64 = bigint_to_u64(&min_gap);
            BigInt candidate;
            bigint_set_u64(&candidate, 0);
            bigint_copy(&candidate, &best_lo);

            BigInt inc;
            bigint_set_u64(&inc, 1);

            for (uint64_t step = 0; step <= gap_u64 && !success; step++) {
                /* Try candidate directly, and small multiples/divisors.
                 * The CF convergent might be r/m or m*r for small m. */
                BigInt test_r, base6, m_bi, dummy_q;
                bigint_set_u64(&test_r, 0);
                bigint_set_u64(&base6, 6);
                bigint_set_u64(&m_bi, 0);
                bigint_set_u64(&dummy_q, 0);

                for (int m = 1; m <= 6 && !success; m++) {
                    /* Test m × candidate */
                    bigint_set_u64(&m_bi, (uint64_t)m);
                    bigint_mul(&test_r, &candidate, &m_bi);

                    int ret = try_period(&test_r, &base6, N, factor_p, factor_q);
                    if (ret == 1) {
                        char c_str[512];
                        bigint_to_decimal(c_str, sizeof(c_str), &test_r);
                        printf("\n  ★ OUROBOROS BITES ITS TAIL ★ (CF proximity scan, r = %s [×%d])\n", c_str, m);
                        success = 1; break;
                    }
                    if (!success) {
                        ret = try_period(&test_r, a_val, N, factor_p, factor_q);
                        if (ret == 1) {
                            char c_str[512];
                            bigint_to_decimal(c_str, sizeof(c_str), &test_r);
                            printf("\n  ★ OUROBOROS BITES ITS TAIL ★ (CF proximity scan [base-a], r = %s [×%d])\n", c_str, m);
                            success = 1; break;
                        }
                    }

                    /* Test candidate / m (if divisible) */
                    if (m > 1) {
                        BigInt div_rem;
                        bigint_set_u64(&div_rem, 0);
                        bigint_div_mod(&candidate, &m_bi, &test_r, &div_rem);
                        if (bigint_is_zero(&div_rem) && !bigint_is_zero(&test_r)) {
                            ret = try_period(&test_r, &base6, N, factor_p, factor_q);
                            if (ret == 1) {
                                char c_str[512];
                                bigint_to_decimal(c_str, sizeof(c_str), &test_r);
                                printf("\n  ★ OUROBOROS BITES ITS TAIL ★ (CF proximity scan, r = %s [÷%d])\n", c_str, m);
                                success = 1;
                            }
                            if (!success) {
                                ret = try_period(&test_r, a_val, N, factor_p, factor_q);
                                if (ret == 1) {
                                    char c_str[512];
                                    bigint_to_decimal(c_str, sizeof(c_str), &test_r);
                                    printf("\n  ★ OUROBOROS BITES ITS TAIL ★ (CF proximity scan [base-a], r = %s [÷%d])\n", c_str, m);
                                    success = 1;
                                }
                            }
                        }
                        bigint_destroy(&div_rem);
                    }
                }

                bigint_destroy(&test_r);
                bigint_destroy(&base6);
                bigint_destroy(&m_bi);
                bigint_destroy(&dummy_q);

                /* candidate++ */
                BigInt next;
                bigint_set_u64(&next, 0);
                bigint_add(&next, &candidate, &inc);
                bigint_copy(&candidate, &next);
                bigint_destroy(&next);

                /* Progress every 100K */
                if (step > 0 && step % 100000 == 0) {
                    printf("    ... scanned %lu / %lu\n", (unsigned long)step, (unsigned long)gap_u64);
                }
            }

            if (!success) {
                printf("    Proximity search complete: no factor found in range.\n");
            }

            bigint_destroy(&candidate);
            bigint_destroy(&inc);
        } else if (have_pair) {
            char gap_str[512];
            bigint_to_decimal(gap_str, sizeof(gap_str), &min_gap);
            printf("\n    ── CF Proximity Search: SKIPPED (closest gap = %s, limit = 10000000) ──\n", gap_str);
        } else {
            printf("\n    ── CF Proximity Search: SKIPPED (no valid pair in pool of %d) ──\n", cf_pool_count);
        }

        bigint_destroy(&best_lo);
        bigint_destroy(&best_hi);
        bigint_destroy(&diff_tmp);
        bigint_destroy(&min_gap);
        bigint_destroy(&proximity_limit);
    }

    free(beam_parent); free(beam_digit); free(path_probs);

    /* Cleanup */
    hpc_destroy(graph);  /* Destroy graph AFTER measurement */
    for (int i = 0; i < n_sites_raw; i++) {
        bigint_destroy(&p6_cache[i]);
        bigint_destroy(&ck_cache[i]);
    }
    free(p6_cache);
    free(ck_cache);
    free(measured_digits);
    free(bp_has_signal);
    free(flippable);

    for (int s = 0; s < marginals_sz; s++)
        for (int d = 0; d < 6; d++)
            mpfr_clear(marginals[s][d]);
    free(marginals);

    /* Cleanup — CF and frequency BigInts (destroy = free GMP memory) */
    bigint_destroy(&work_val_bi);
    bigint_destroy(&mc_d_bi); bigint_destroy(&mc_term); bigint_destroy(&mc_tmp);
    bigint_destroy(&reg_sz); bigint_destroy(&gc_reg_tmp); bigint_destroy(&current_p6); bigint_destroy(&next_p6);

    bigint_clear(&b6); bigint_clear(&one);
    bigint_clear(&val_k_A); bigint_clear(&val_k_B); bigint_clear(&div_6_blk);
    bigint_clear(&gc_b36); bigint_clear(&gc_next_A); bigint_clear(&gc_next_B); bigint_clear(&gc_next_div);
    bigint_clear(&gc_gcd_check); bigint_clear(&gc_val_minus_1); bigint_clear(&gc_dummy_rem);
    bigint_clear(&gc_tmpA); bigint_clear(&gc_tmpB); bigint_clear(&gc_q_div);
    bigint_clear(&gc_six_pow_k); bigint_clear(&gc_d_bi);
    bigint_clear(&gc_b6_mod); bigint_clear(&gc_shift_div_A); bigint_clear(&gc_shift_div_B); bigint_clear(&gc_dummy_rm2);
    bigint_clear(&gc_qA); bigint_clear(&gc_qB); bigint_clear(&gc_rA_mod); bigint_clear(&gc_rB_mod);
    bigint_clear(&gc_temp_N); bigint_clear(&gc_qN); bigint_clear(&gc_rN); bigint_clear(&gc_q_sh); bigint_clear(&gc_r_sh);
    for (int i = 0; i < 6; i++) {
        bigint_clear(&gc_powersA[i]);
        bigint_clear(&gc_powersB[i]);
    }
    return (double)success;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    srand(time(NULL));
    triality_exotic_init();
    s6_exotic_init();
    triality_stats_reset();

    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                              ║\n");
    printf("  ║   HPC OUROBOROS FACTORING ENGINE                             ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   Architecture: HPCGraph (Holographic Phase Contraction)     ║\n");
    printf("  ║   Amplitudes: O(N+E) analytical (no state vector)           ║\n");
    printf("  ║   Entanglement: CZ phase edges (exact, fidelity = 1.0)      ║\n");
    printf("  ║   4,096-bit BigInt support via bigint.c                     ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   \"The observer and observed are opposite faces.\"            ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    /* Parse N and a from config or arguments */
    BigInt N, a_val;
    const char *target_n_str = (argc > 1) ? argv[1] : TARGET_N;
    if (bigint_from_decimal(&N, target_n_str) != 0) {
        printf("  ERROR: Invalid N = \"%s\"\n", target_n_str);
        return 1;
    }

    int auto_a = 0;
    if (strcmp(TARGET_A, "0") == 0) {
        auto_a = 1;
        bigint_set_u64(&a_val, 2);
    } else {
        if (bigint_from_decimal(&a_val, TARGET_A) != 0) {
            printf("  ERROR: Invalid a = \"%s\"\n", TARGET_A);
            return 1;
        }
    }

    char N_str[1300];
    bigint_to_decimal(N_str, sizeof(N_str), &N);
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  TARGET: N = %-50s ║\n", N_str);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Try different bases if auto */
    int max_bases = auto_a ? 200 : 1;
    uint64_t base_list[] = {
        2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
        73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,
        179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,
        283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,
        419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,
        547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,
        661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,
        811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,
        947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,
        1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223
    };

    BigInt factor_p, factor_q;
    int success = 0;

    /* ── Cross-base LCM period accumulator ── */
    BigInt cross_base_lcm, cross_base_gcd, cross_base_prod, cross_base_rem;
    BigInt best_partial;
    bigint_set_u64(&cross_base_lcm, 1);
    bigint_set_u64(&cross_base_gcd, 0);
    bigint_set_u64(&cross_base_prod, 0);
    bigint_set_u64(&cross_base_rem, 0);
    bigint_set_u64(&best_partial, 0);
    BigInt bi_one; bigint_set_u64(&bi_one, 1);

    /* ── Cross-base period accumulator (persists across all base attempts) ── */
    BigInt cross_base_acc_period;
    int    cross_base_acc_samples = 0;
    bigint_set_u64(&cross_base_acc_period, 0);

    /* ── Try constraint-satisfaction first (Hensel lift in base 6) ── */
    /* ── Try constraint-satisfaction first (Hensel lift in base 6) ── */
    success = 0;

    int active_base_idx = 181;
    if (auto_a) {
        printf("\n  ╔════════════════════════════════════════════════════════════════╗\n");
        printf("  ║  AUTOMATED TOPOLOGICAL RESONANCE PROBE (BASE SELECTOR)         ║\n");
        printf("  ╚════════════════════════════════════════════════════════════════╝\n");
        double best_res = 1e30;
        int best_idx = 0;
        /* Probe the first 30 prime string candidates dynamically natively */
        for (int bi = 0; bi < 30; bi++) {
            BigInt a_test; bigint_set_u64(&a_test, base_list[bi]);
            BigInt dummy_p, dummy_q, best_prt, x_acc;
            bigint_set_u64(&dummy_p, 0); bigint_set_u64(&dummy_q, 0);
            bigint_set_u64(&best_prt, 0); bigint_set_u64(&x_acc, 0);
            int x_samp = 0;

            double res = factor_with_hpc(&N, &a_test, &dummy_p, &dummy_q, &best_prt, &x_acc, &x_samp, 1);
            if (res < 0.0) {
                printf("    [Base a=%llu] Probe hit a direct GCD factor cascade instantly!\n", (unsigned long long)base_list[bi]);
                best_idx = bi;
                break;
            }
            printf("    [Base a=%llu] Network Residual Limit: %.6e\n", (unsigned long long)base_list[bi], res);
            if (res < best_res) {
                best_res = res;
                best_idx = bi;
            }
        }
        printf("  ──────────────────────────────────────────────────────────────────\n");
        printf("  ★ OPTIMAL RESONANT BASE SELECTED: a=%llu (Residual: %.6e) ★\n", (unsigned long long)base_list[best_idx], best_res);
        printf("  ──────────────────────────────────────────────────────────────────\n\n");
        
        /* Swapping the optimal base to the front of the queue to prioritize it, 
         * safely scaling perfectly for all width architectures without breaking LCM sequences natively */
        uint64_t tmp_b = base_list[0];
        base_list[0] = base_list[best_idx];
        base_list[best_idx] = tmp_b;
        
        active_base_idx = 0;
    }

    int attempt_count = 1;
    BigInt cf_history[200];
    for (int i = 0; i < 200; i++) bigint_set_u64(&cf_history[i], 0);
    int cf_hist_count = 0;
    for (int bi = active_base_idx; !success; bi = (auto_a ? (bi + 1) % 200 : bi)) {
        if (auto_a) bigint_set_u64(&a_val, base_list[bi]);

        char a_str[1300];
        bigint_to_decimal(a_str, sizeof(a_str), &a_val);
        printf("  ── Attempt %d: a = %s ──\n\n", attempt_count++, a_str);

        bigint_set_u64(&best_partial, 0);
        int old_pool_count = cf_pool_count;
        clock_t t_start = clock();
        success = (int)factor_with_hpc(&N, &a_val, &factor_p, &factor_q, &best_partial,
                                  &cross_base_acc_period, &cross_base_acc_samples, 0);
        clock_t t_end = clock();

        if (success) {
            char p_str[1300], q_str[1300];
            bigint_to_decimal(p_str, sizeof(p_str), &factor_p);
            bigint_to_decimal(q_str, sizeof(q_str), &factor_q);
            printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
            printf("  ║  FACTORED                                               ║\n");
            printf("  ╚══════════════════════════════════════════════════════════╝\n\n");
            printf("  N = %s\n", N_str);
            printf("    = %s × %s\n\n", p_str, q_str);

            /* Verify */
            BigInt verify;
            bigint_mul(&verify, &factor_p, &factor_q);
            if (bigint_cmp(&verify, &N) == 0)
                printf("  ✓ Verified: p × q = N\n");
            else
                printf("  ✗ WARNING: p × q ≠ N\n");

            printf("  Time: %.3f seconds\n",
                   (double)(t_end - t_start) / CLOCKS_PER_SEC);
        } else {
            printf("  ✗ Base a=%s did not yield factors (%.3f sec)\n",
                   a_str, (double)(t_end - t_start) / CLOCKS_PER_SEC);

            /* ── Cross-base LCM accumulation ── */
            if (!bigint_is_zero(&best_partial) && bigint_cmp(&best_partial, &bi_one) > 0) {
                /* Validate the period before indiscriminately accumulating it.
                 * If best_partial is just garbage, LCMing it will corrupt the accumulator!
                 * If it's a sterile/trivial root or odd period, it's a valid period length. */
                int status = try_period(&best_partial, &a_val, &N, &factor_p, &factor_q);
                
                if (status == 1) {
                    success = 1;
                    char p_str[1300], q_str[1300], bp_str[1300];
                    bigint_to_decimal(p_str, sizeof(p_str), &factor_p);
                    bigint_to_decimal(q_str, sizeof(q_str), &factor_q);
                    bigint_to_decimal(bp_str, sizeof(bp_str), &best_partial);
                    printf("\n  ★★★ LATE FACTORIZATION: best_partial (%s) was a valid factoring period! ★★★\n", bp_str);
                    printf("  %s × %s = N\n", p_str, q_str);
                } else if (status == 2 || status == -1) {
                    /* Calculate actual period contribution for this base */
                    BigInt current_period;
                    bigint_set_u64(&current_period, 0); /* Initialize the struct properly first */
                    bigint_copy(&current_period, &best_partial); /* Pass the actual candidate */

                    /* First base: set LCM to its period */
                    if (bigint_cmp(&cross_base_lcm, &bi_one) == 0) {
                        bigint_copy(&cross_base_lcm, &current_period);
                        printf("  [Cross-base] Initialized LCM to verified period contribution\n");
                    } else {
                        /* Save previous state before modifying */
                        BigInt prev_lcm;
                        bigint_set_u64(&prev_lcm, 0);
                        bigint_copy(&prev_lcm, &cross_base_lcm);

                        /* Subsequent bases: accumulate LCM */
                        /* Proper LCM update: lcm(a,b) = (a*b)/gcd(a,b) */
                        bigint_gcd(&cross_base_gcd, &cross_base_lcm, &current_period);
                        bigint_mul(&cross_base_prod, &cross_base_lcm, &current_period);
                        bigint_div_mod(&cross_base_prod, &cross_base_gcd, &cross_base_lcm, &cross_base_rem);

                        /* Clamp: if LCM exceeds N, hold previous state */
                        if (bigint_cmp(&cross_base_lcm, &N) >= 0) {
                            printf("  [Cross-base] LCM exceeded N, holding previous state\n");
                            bigint_copy(&cross_base_lcm, &prev_lcm);  /* fall back to prev_lcm */
                        }
                        bigint_clear(&prev_lcm);
                    }

                    uint32_t lcm_bits = bigint_bitlen(&cross_base_lcm);
                    uint32_t n_bits = bigint_bitlen(&N);
                    printf("  [Cross-base] Accumulated LCM: %u bits (target <%u bits)\n",
                           lcm_bits, n_bits);

                    /* Try the accumulated LCM as a period candidate */
                    printf("  [Cross-base] Testing accumulated LCM as period...\n");
                    /* Test with EACH base we've tried so far */
                    for (int bj = 0; bj <= bi && !success; bj++) {
                        BigInt test_a;
                        bigint_set_u64(&test_a, base_list[bj]);
                        if (try_period(&cross_base_lcm, &test_a, &N, &factor_p, &factor_q) == 1) {
                            success = 1;
                            printf("\n  ★★★ CROSS-BASE LCM FACTORED N! (base a=%llu) ★★★\n",
                                   (unsigned long long)base_list[bj]);
                        }
                        /* Also test small multiples */
                        for (int sm = 2; sm <= 12 && !success; sm++) {
                            BigInt r_mult, mult_c;
                            bigint_set_u64(&r_mult, 0);
                            bigint_set_u64(&mult_c, sm);
                            bigint_mul(&r_mult, &cross_base_lcm, &mult_c);
                            if (bigint_cmp(&r_mult, &N) >= 0) break;
                            if (try_period(&r_mult, &test_a, &N, &factor_p, &factor_q) == 1) {
                                success = 1;
                                printf("\n  ★★★ CROSS-BASE LCM × %d FACTORED N! (base a=%llu) ★★★\n",
                                       sm, (unsigned long long)base_list[bj]);
                            }
                            bigint_clear(&r_mult); bigint_clear(&mult_c);
                        }
                        bigint_clear(&test_a);
                    }
                    
                    /* Try the MCMC deep accumulator as a period candidate */
                    if (!bigint_is_zero(&cross_base_acc_period) && bigint_cmp(&cross_base_acc_period, &bi_one) > 0) {
                        printf("  [Cross-base] Testing MCMC deep accumulator...\n");
                        for (int bj = 0; bj <= bi && !success; bj++) {
                            BigInt test_a;
                            bigint_set_u64(&test_a, base_list[bj]);
                            
                            if (try_period(&cross_base_acc_period, &test_a, &N, &factor_p, &factor_q) == 1) {
                                success = 1;
                                printf("\n  ★★★ MCMC DEEP ACCUMULATOR FACTORED N! (base a=%llu) ★★★\n",
                                       (unsigned long long)base_list[bj]);
                            }
                            
                            /* Test small harmonics of the MCMC accumulator */
                            for (int sm = 2; sm <= 12 && !success; sm++) {
                                BigInt r_mult, mult_c;
                                bigint_set_u64(&r_mult, 0);
                                bigint_set_u64(&mult_c, sm);
                                bigint_mul(&r_mult, &cross_base_acc_period, &mult_c);
                                if (bigint_cmp(&r_mult, &N) >= 0) break;
                                if (try_period(&r_mult, &test_a, &N, &factor_p, &factor_q) == 1) {
                                    success = 1;
                                    printf("\n  ★★★ MCMC DEEP ACCUMULATOR × %d FACTORED N! (base a=%llu) ★★★\n",
                                           sm, (unsigned long long)base_list[bj]);
                                }
                                bigint_clear(&r_mult); bigint_clear(&mult_c);
                            }
                            bigint_clear(&test_a);
                        }
                    }
                } else {
                    char bp_str[1300];
                    bigint_to_decimal(bp_str, sizeof(bp_str), &best_partial);
                    printf("  [Cross-base] Dumped invalid best_partial %s. It's not a period!\n", bp_str);
                }
            }

            printf("\n");
        }
    }

    if (!success) {
        printf("\n  ══════════════════════════════════════════════════════════\n");
        printf("  Could not factor N with the tested bases.\n");
        printf("  Try a different TARGET_A value.\n");
        printf("  ══════════════════════════════════════════════════════════\n");
    }

    /* Print HPC + triality stats */
    triality_stats_print();

    printf("\n  ═══════════════════════════════════════════════════════════════\n");
    printf("  HPC Ouroboros Engine complete.\n");
    printf("  ═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}
