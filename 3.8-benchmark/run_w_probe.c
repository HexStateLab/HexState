/*
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  run_w_probe.c — Blind W-Axis Localization on the HexState 4D Overlay     ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  EXPERIMENT                                                               ║
 * ║  A hidden, seeded generator places a prolate 4-ellipsoid somewhere in a   ║
 * ║  4D lattice (Lx×Ly×Lz×Lw), tilted by angle θ in the x–w plane — a         ║
 * ║  rotation that exists only because the lattice has a W axis. The object   ║
 * ║  is PHASE-encoded: every site's local density stays exactly uniform       ║
 * ║  (1/6 per level), so no density scan can find it.                         ║
 * ║                                                                           ║
 * ║  The probe recovers it interferometrically:                               ║
 * ║    prep:    |0⟩ → DFT₆|0⟩ on all sites (uniform superposition)            ║
 * ║    tag:     interior sites get diag(e^{2πik/6}) — a "momentum kick"       ║
 * ║    hide:    local density is flat 1/6 everywhere (verified)               ║
 * ║    read:    apply DFT₆⁻¹ everywhere — tagged sites collapse to |1⟩,       ║
 * ║             untagged to |0⟩; the object reappears as P(level=1)           ║
 * ║    locate:  4D centroid → ŵ₀;  4D inertia tensor → principal axis → θ̂    ║
 * ║                                                                           ║
 * ║  Scored blind: estimates are computed from measurement statistics only,   ║
 * ║  then unblinded against the generator at the end. Output is a CSV of      ║
 * ║  |ŵ₀−w₀| and |θ̂−θ| versus shots-per-site (plus an exact-density row).     ║
 * ║                                                                           ║
 * ║  Build (with the HexState sources, mirroring the template):               ║
 * ║    gcc -O2 -march=native -o run_w_probe run_w_probe.c \                   ║
 * ║        quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \      ║
 * ║        quhit_register.c quhit_substrate.c peps4d_overlay.c \              ║
 * ║        quhit_svd_gate.c -lm -fopenmp -msse2                               ║
 * ║    ulimit -s unlimited                                                    ║
 * ║                                                                           ║
 * ║  Run:                                                                     ║
 * ║    ./run_w_probe [seed] [Lx Ly Lz Lw]      (defaults: time-seed, 4 4 4 8) ║
 * ║                                                                           ║
 * ║  ── API ASSUMPTIONS (adjust the SHIM block below if names differ) ──      ║
 * ║  Per hexstate_template.c §8.3: "4D, 5D, 6D follow the exact same          ║
 * ║  pattern" as tns3d/peps. This driver therefore assumes:                   ║
 * ║    Peps4dGrid *peps4d_init(int Lx,int Ly,int Lz,int Lw);                  ║
 * ║    void peps4d_set_product_state(g,x,y,z,w,re,im);                        ║
 * ║    void peps4d_gate_1site(g,x,y,z,w,U_re,U_im);                           ║
 * ║    void peps4d_local_density(g,x,y,z,w,probs);                            ║
 * ║    int  peps4d_measure_site(g,x,y,z,w);   (unused in product-state mode)  ║
 * ║    void peps4d_free(g);                                                   ║
 * ║                                                                           ║
 * ║  NOTE ON SHOTS: the basic protocol applies only 1-site gates to a         ║
 * ║  product state, so sites are unentangled and repeated shots are i.i.d.    ║
 * ║  multinomial draws from each site's exact density. We therefore sample    ║
 * ║  shots from peps4d_local_density with the driver PRNG — statistically     ║
 * ║  identical to re-preparing and measuring, at 1/M the cost. The entangled  ║
 * ║  upgrade (CZ bonds inside the object via peps4d_gate_w) REQUIRES real     ║
 * ║  re-prepare-and-measure; see the stub at the bottom.                      ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "quhit_engine.h"
#include "peps4d_overlay.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define D 6

/* ── SHIM: rename here if your peps4d API differs from the template note ── */
#define P4D_GRID       Peps4dGrid
#define P4D_INIT       peps4d_init
#define P4D_SET_PROD   peps4d_set_product_state
#define P4D_GATE_1     peps4d_gate_1site
#define P4D_DENSITY    peps4d_local_density
#define P4D_FREE       peps4d_free

/* ═══════════════════════════════════════════════════════════════════════════
 * PRNG (xoshiro256**, same as template) — drives BOTH the hidden generator
 * and shot sampling, from separate streams so unblinding is honest.
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct { uint64_t s[4]; } Rng;

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(Rng *r) {
    uint64_t v = rotl64(r->s[1] * 5, 7) * 9;
    uint64_t t = r->s[1] << 17;
    r->s[2] ^= r->s[0]; r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2]; r->s[0] ^= r->s[3];
    r->s[2] ^= t; r->s[3] = rotl64(r->s[3], 45);
    return v;
}
static double rng_uniform(Rng *r) { return (double)(rng_next(r) >> 11) / (double)(1ULL << 53); }
static void rng_seed(Rng *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        r->s[i] = z ^ (z >> 31);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Gate builders
 * ═══════════════════════════════════════════════════════════════════════════ */
static void build_dft6(double *U_re, double *U_im, int inverse) {
    double s = 1.0 / sqrt(6.0), sgn = inverse ? -1.0 : 1.0;
    for (int j = 0; j < D; j++)
    for (int k = 0; k < D; k++) {
        double a = sgn * 2.0 * M_PI * j * k / (double)D;
        U_re[j*D+k] = s * cos(a);
        U_im[j*D+k] = s * sin(a);
    }
}

/* Momentum-kick tag: diag(e^{i·η·2πk/6}) with strength η ∈ (0,1].
 * At η=1, DFT₆|0⟩ --tag--> DFT₆|1⟩ exactly and the readout is noiseless,
 * which makes the shots sweep degenerate (P₁ ∈ {0,1}). A partial kick
 * (default η=0.35) gives tagged sites 0 < P₁ < 1, so the precision-vs-shots
 * curve measures something real. Untagged sites still read out exactly |0⟩. */
#ifndef KICK_STRENGTH
#define KICK_STRENGTH 0.35
#endif
static void build_kick_gate(double *U_re, double *U_im) {
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));
    for (int k = 0; k < D; k++) {
        double a = KICK_STRENGTH * 2.0 * M_PI * k / (double)D;
        U_re[k*D+k] = cos(a);
        U_im[k*D+k] = sin(a);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HIDDEN OBJECT — prolate 4-ellipsoid, long axis u = (cosθ, 0, 0, sinθ)
 * tilted in the x–w plane. Anisotropy is what makes θ recoverable: a 3-ball
 * has an isotropic inertia tensor and carries no orientation information.
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double cx, cy, cz, cw;   /* center                                   */
    double a, b;             /* semi-axes: longitudinal a > transverse b */
    double theta;            /* tilt in x–w plane, radians, [0, π/2]     */
} HiddenObject;

static int object_contains(const HiddenObject *o, int x, int y, int z, int w) {
    double rx = x - o->cx, ry = y - o->cy, rz = z - o->cz, rw = w - o->cw;
    double ux = cos(o->theta), uw = sin(o->theta);
    double l  = rx*ux + rw*uw;                          /* longitudinal     */
    double r2 = rx*rx + ry*ry + rz*rz + rw*rw;
    double t2 = r2 - l*l;                               /* transverse²      */
    return (l*l)/(o->a*o->a) + t2/(o->b*o->b) <= 1.0;
}

static void generate_hidden_object(HiddenObject *o, Rng *gen,
                                   int Lx, int Ly, int Lz, int Lw) {
    o->a = 2.5;
    o->b = 1.2;
    o->theta = 0.5 * M_PI * rng_uniform(gen);           /* [0, π/2)         */
    /* keep the object inside the lattice for any tilt                      */
    double m = o->a + 0.5;
    o->cx = m + rng_uniform(gen) * ((Lx - 1) - 2.0*m);
    o->cy = 1.0 + rng_uniform(gen) * (Ly - 3.0);
    o->cz = 1.0 + rng_uniform(gen) * (Lz - 3.0);
    o->cw = m + rng_uniform(gen) * ((Lw - 1) - 2.0*m);
    if (Lx - 1 < 2.0*m) o->cx = 0.5 * (Lx - 1);         /* tiny-lattice guard */
    if (Lw - 1 < 2.0*m) o->cw = 0.5 * (Lw - 1);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4×4 symmetric eigensolver (cyclic Jacobi) — for the inertia tensor
 * ═══════════════════════════════════════════════════════════════════════════ */
static void jacobi4(double A[16], double V[16]) {
    for (int i = 0; i < 16; i++) V[i] = (i % 5 == 0) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 64; sweep++) {
        double off = 0;
        for (int p = 0; p < 4; p++)
            for (int q = p + 1; q < 4; q++) off += A[p*4+q]*A[p*4+q];
        if (off < 1e-24) break;
        for (int p = 0; p < 4; p++)
        for (int q = p + 1; q < 4; q++) {
            double apq = A[p*4+q];
            if (fabs(apq) < 1e-18) continue;
            double phi = 0.5 * atan2(2.0*apq, A[q*4+q] - A[p*4+p]);
            double c = cos(phi), s = sin(phi);
            for (int k = 0; k < 4; k++) {
                double akp = A[k*4+p], akq = A[k*4+q];
                A[k*4+p] = c*akp - s*akq;
                A[k*4+q] = s*akp + c*akq;
            }
            for (int k = 0; k < 4; k++) {
                double apk = A[p*4+k], aqk = A[q*4+k];
                A[p*4+k] = c*apk - s*aqk;
                A[q*4+k] = s*apk + c*aqk;
            }
            for (int k = 0; k < 4; k++) {
                double vkp = V[k*4+p], vkq = V[k*4+q];
                V[k*4+p] = c*vkp - s*vkq;
                V[k*4+q] = s*vkp + c*vkq;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ESTIMATORS — centroid (→ ŵ₀) and inertia principal axis (→ θ̂)
 * intensity[] is P(level=1) per site, exact or shot-estimated.
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct { double w0_hat, theta_hat; } Estimate;

static Estimate estimate_from_intensity(const double *intensity,
                                        int Lx, int Ly, int Lz, int Lw) {
    Estimate e = { -1.0, -1.0 };
    double W = 0, mx = 0, my = 0, mz = 0, mw = 0;
    long idx = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++, idx++) {
        double I = intensity[idx];
        W += I; mx += I*x; my += I*y; mz += I*z; mw += I*w;
    }
    if (W < 1e-12) return e;                            /* found nothing    */
    mx /= W; my /= W; mz /= W; mw /= W;
    e.w0_hat = mw;

    double S[16] = {0};
    idx = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++, idx++) {
        double I = intensity[idx];
        if (I <= 0) continue;
        double r[4] = { x - mx, y - my, z - mz, w - mw };
        for (int p = 0; p < 4; p++)
            for (int q = 0; q < 4; q++) S[p*4+q] += I * r[p] * r[q];
    }
    for (int i = 0; i < 16; i++) S[i] /= W;

    double V[16];
    jacobi4(S, V);
    int top = 0;
    for (int k = 1; k < 4; k++) if (S[k*4+k] > S[top*4+top]) top = k;
    double ux = V[0*4+top], uw = V[3*4+top];
    /* axis is sign-free; fold into [0, π/2] to compare with θ */
    double th = atan2(fabs(uw), fabs(ux));
    e.theta_hat = th;
    return e;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    uint64_t seed = (argc > 1) ? strtoull(argv[1], NULL, 10) : (uint64_t)time(NULL);
    int Lx = 4, Ly = 4, Lz = 4, Lw = 8;
    if (argc > 5) { Lx = atoi(argv[2]); Ly = atoi(argv[3]);
                    Lz = atoi(argv[4]); Lw = atoi(argv[5]); }
    long Nsites = (long)Lx * Ly * Lz * Lw;

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  W-AXIS PROBE — blind localization on the 4D overlay      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("  Lattice %d×%d×%d×%d = %ld sites, seed=%llu\n\n",
           Lx, Ly, Lz, Lw, Nsites, (unsigned long long)seed);

    Rng gen, shots;                 /* separate streams: generator vs sampling */
    rng_seed(&gen,   seed);
    rng_seed(&shots, seed ^ 0xD1CEB00CULL);

    /* ── 1. Hidden object (do not peek until unblinding) ── */
    HiddenObject obj;
    generate_hidden_object(&obj, &gen, Lx, Ly, Lz, Lw);

    /* ── 2. State prep: |0⟩ everywhere, then DFT₆ everywhere ── */
    double F_re[36],  F_im[36];   build_dft6(F_re,  F_im,  0);
    double Fi_re[36], Fi_im[36];  build_dft6(Fi_re, Fi_im, 1);
    double K_re[36],  K_im[36];   build_kick_gate(K_re, K_im);

    P4D_GRID *g = P4D_INIT(Lx, Ly, Lz, Lw);
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++) {
        double re[D] = {1, 0, 0, 0, 0, 0}, im[D] = {0};
        P4D_SET_PROD(g, x, y, z, w, re, im);
        P4D_GATE_1(g, x, y, z, w, F_re, F_im);
    }

    /* ── 3. Phase-tag the interior — and count it for the unblinding ── */
    long n_tagged = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++)
        if (object_contains(&obj, x, y, z, w)) {
            P4D_GATE_1(g, x, y, z, w, K_re, K_im);
            n_tagged++;
        }

    /* ── 4. HIDDENNESS CONTROL: density must be flat 1/6 everywhere ── */
    double probs[D], max_dev = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++) {
        P4D_DENSITY(g, x, y, z, w, probs);
        for (int k = 0; k < D; k++) {
            double d = fabs(probs[k] - 1.0/D);
            if (d > max_dev) max_dev = d;
        }
    }
    printf("  Hiddenness control: max |P(k) − 1/6| = %.3e  %s\n",
           max_dev, max_dev < 1e-9 ? "→ object is density-invisible ✓"
                                   : "→ WARNING: tag leaks into density");

    /* ── 5. Interferometric readout: DFT₆⁻¹ everywhere ── */
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++)
        P4D_GATE_1(g, x, y, z, w, Fi_re, Fi_im);

    /* exact intensities I = P(level=1), cached for shot sampling           */
    double *exactP1 = malloc(Nsites * sizeof(double));
    double *intens  = malloc(Nsites * sizeof(double));
    long idx = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++, idx++) {
        P4D_DENSITY(g, x, y, z, w, probs);
        exactP1[idx] = probs[1];
    }

    /* ── 6. Estimate vs shots-per-site, write CSV ── */
    FILE *csv = fopen("w_probe_results.csv", "w");
    fprintf(csv, "shots_per_site,w0_true,w0_hat,w0_abs_err,"
                 "theta_true_deg,theta_hat_deg,theta_abs_err_deg\n");

    int shot_sweep[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096 };
    int n_sweep = (int)(sizeof shot_sweep / sizeof *shot_sweep);
    printf("\n  %-10s %-22s %-22s\n", "shots", "ŵ₀ (true → est, err)", "θ̂ deg (true → est, err)");

    for (int s = 0; s <= n_sweep; s++) {
        const char *label;
        if (s == n_sweep) {                              /* exact-density row */
            memcpy(intens, exactP1, Nsites * sizeof(double));
            label = "exact";
        } else {
            int M = shot_sweep[s];
            for (long i = 0; i < Nsites; i++) {          /* binomial draws    */
                int hits = 0;
                for (int m = 0; m < M; m++)
                    if (rng_uniform(&shots) < exactP1[i]) hits++;
                intens[i] = (double)hits / M;
            }
            label = NULL;
        }
        Estimate e = estimate_from_intensity(intens, Lx, Ly, Lz, Lw);
        double w_err  = fabs(e.w0_hat   - obj.cw);
        double th_err = fabs(e.theta_hat - obj.theta) * 180.0 / M_PI;
        if (label) printf("  %-10s", label);
        else       printf("  %-10d", shot_sweep[s]);
        printf(" %5.2f → %5.2f (%.3f)     %6.1f → %6.1f (%.2f)\n",
               obj.cw, e.w0_hat, w_err,
               obj.theta * 180.0 / M_PI, e.theta_hat * 180.0 / M_PI, th_err);
        fprintf(csv, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                label ? 0 : shot_sweep[s],
                obj.cw, e.w0_hat, w_err,
                obj.theta * 180.0 / M_PI, e.theta_hat * 180.0 / M_PI, th_err);
    }
    fclose(csv);

    /* ── 7. Unblind ── */
    printf("\n  ── UNBLINDED ──\n");
    printf("  Object: prolate 4-ellipsoid, a=%.2f b=%.2f, %ld interior sites\n",
           obj.a, obj.b, n_tagged);
    printf("  Center (%.2f, %.2f, %.2f | w=%.2f), tilt θ=%.1f° in x–w plane\n",
           obj.cx, obj.cy, obj.cz, obj.cw, obj.theta * 180.0 / M_PI);
    printf("  CSV written: w_probe_results.csv\n");

    free(exactP1);
    free(intens);
    P4D_FREE(g);

    /* ──────────────────────────────────────────────────────────────────────
     * UPGRADE STUB — entangled object (strictly stronger hiding):
     *   for interior w-bonds: peps4d_gate_w(g, x, y, z, w, CZ_re, CZ_im);
     * The object then carries NO single-site signal even after DFT₆⁻¹ —
     * detection requires w-cut mutual information, and shot sampling must
     * re-prepare the lattice per shot (sites are no longer independent).
     * ────────────────────────────────────────────────────────────────────── */
    return 0;
}
