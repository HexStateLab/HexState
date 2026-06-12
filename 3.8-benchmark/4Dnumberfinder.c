/*
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║  run_w_probe.c (v2) — Hide a Number in 4D Space, Then Find It             ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  A user-specified number (up to MAX_BITS = 3000 bits) is encoded into a   ║
 * ║  4-ellipsoid hidden in the HexState 4D overlay:                           ║
 * ║                                                                           ║
 * ║   1. The number is converted to base-5 digits (one digit per site —       ║
 * ║      values 0..4 map to kick strengths g = 1..5, so EVERY digit site      ║
 * ║      reads nonzero and is distinguishable from empty space).              ║
 * ║   2. The lattice and ellipsoid are auto-sized so the ellipsoid interior   ║
 * ║      holds a 12-digit length header + all digits (3000 bits ≈ 1293       ║
 * ║      digits ≈ 1305 sites).                                                ║
 * ║   3. The ellipsoid is placed at a random center and tilted by a random    ║
 * ║      angle θ in the x–w plane. Digits are written in lexicographic        ║
 * ║      (x,y,z,w) scan order over the interior; leftover interior sites      ║
 * ║      are padded with digit 0 (ignored via the length header).             ║
 * ║   4. Each digit is a PHASE tag, diag(e^{i·g·2πk/6}), on the uniform       ║
 * ║      superposition: local density stays exactly 1/6 per level — the      ║
 * ║      number is invisible to any density scan (verified at runtime).       ║
 * ║                                                                           ║
 * ║  Recovery (blind — uses measurement statistics only):                     ║
 * ║   5. Apply DFT₆⁻¹ to every site: a g-kick on DFT₆|0⟩ collapses to |g⟩,    ║
 * ║      empty space to |0⟩. The number reappears as a cloud of levels 1..5.  ║
 * ║   6. Geometry: centroid of the cloud → ŵ₀; 4D inertia tensor → tilt θ̂.   ║
 * ║      That is the "find it in 4D space" half: WHERE along W, and HOW the   ║
 * ║      object's own long axis is rotated into the W direction.             ║
 * ║   7. Decoding: read nonzero sites in scan order → header → digits →      ║
 * ║      rebuild the integer → compare limb-for-limb with the original.      ║
 * ║      Success criterion: bit-exact recovery.                               ║
 * ║                                                                           ║
 * ║  Build (with the HexState sources, mirroring the template):               ║
 * ║    gcc -O2 -march=native -o run_w_probe run_w_probe.c \                   ║
 * ║        quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \      ║
 * ║        quhit_register.c quhit_substrate.c peps4d_overlay.c \              ║
 * ║        quhit_svd_gate.c -lm -fopenmp -msse2                               ║
 * ║    ulimit -s unlimited                                                    ║
 * ║                                                                           ║
 * ║  Run:                                                                     ║
 * ║    ./run_w_probe 123456789012345678901234567890        (decimal)         ║
 * ║    ./run_w_probe 0xDEADBEEFCAFE...                      (hex)            ║
 * ║    ./run_w_probe rand:3000 [seed]                       (random N-bit)   ║
 * ║                                                                           ║
 * ║  ── API ASSUMPTIONS (adjust the SHIM block if your names differ) ──       ║
 * ║  Per hexstate_template.c §8.3, peps4d follows the tns3d/peps pattern:     ║
 * ║    Peps4dGrid *peps4d_init(int Lx,int Ly,int Lz,int Lw);                  ║
 * ║    void peps4d_set_product_state(g,x,y,z,w,re,im);                        ║
 * ║    void peps4d_gate_1site(g,x,y,z,w,U_re,U_im);                           ║
 * ║    void peps4d_local_density(g,x,y,z,w,probs);                            ║
 * ║    void peps4d_free(g);                                                   ║
 * ║                                                                           ║
 * ║  NOTE: full-strength digit kicks make the readout deterministic (P=1 at   ║
 * ║  level g), so recovery needs only exact local densities — appropriate,    ║
 * ║  since the claim being tested is bit-exact retrieval, not shot scaling.   ║
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
#define MAX_BITS     3000
#define HEADER_DIGITS  12          /* base-5 length header: 5^12 ≈ 2.4e8     */
#define MAX_SITES   2000000L       /* lattice safety cap                      */

/* ── SHIM: rename here if your peps4d API differs from the template note ── */
#define P4D_GRID       Peps4dGrid
#define P4D_INIT       peps4d_init
#define P4D_SET_PROD   peps4d_set_product_state
#define P4D_GATE_1     peps4d_gate_1site
#define P4D_DENSITY    peps4d_local_density
#define P4D_FREE       peps4d_free

/* ═══════════════════════════════════════════════════════════════════════════
 * PRNG (xoshiro256**, same as the template)
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
 * BIGNUM — little-endian uint32 limbs, enough for MAX_BITS + headroom.
 * (Self-contained; swap in your bigint.c if you prefer.)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MAXLIMBS 128               /* 4096-bit headroom                       */
typedef struct { uint32_t l[MAXLIMBS]; int n; } Big;

static void big_zero(Big *b) { memset(b, 0, sizeof *b); b->n = 0; }
static int  big_is_zero(const Big *b) { return b->n == 0; }
static void big_norm(Big *b) { while (b->n > 0 && b->l[b->n-1] == 0) b->n--; }

static int big_mul_add_small(Big *b, uint32_t mul, uint32_t add) {
    uint64_t carry = add;
    for (int i = 0; i < b->n; i++) {
        uint64_t t = (uint64_t)b->l[i] * mul + carry;
        b->l[i] = (uint32_t)t;
        carry = t >> 32;
    }
    while (carry) {
        if (b->n >= MAXLIMBS) return -1;
        b->l[b->n++] = (uint32_t)carry;
        carry >>= 32;
    }
    return 0;
}

static uint32_t big_divmod_small(Big *b, uint32_t div) {
    uint64_t rem = 0;
    for (int i = b->n - 1; i >= 0; i--) {
        uint64_t cur = (rem << 32) | b->l[i];
        b->l[i] = (uint32_t)(cur / div);
        rem = cur % div;
    }
    big_norm(b);
    return (uint32_t)rem;
}

static int big_bits(const Big *b) {
    if (b->n == 0) return 0;
    uint32_t top = b->l[b->n-1];
    int k = 0;
    while (top) { top >>= 1; k++; }
    return (b->n - 1) * 32 + k;
}

static int big_eq(const Big *a, const Big *b) {
    if (a->n != b->n) return 0;
    return memcmp(a->l, b->l, (size_t)a->n * sizeof(uint32_t)) == 0;
}

static void big_print_hex(const Big *b, const char *label) {
    printf("  %s0x", label);
    if (b->n == 0) { printf("0\n"); return; }
    printf("%X", b->l[b->n-1]);
    for (int i = b->n - 2; i >= 0; i--) printf("%08X", b->l[i]);
    printf("\n");
}

/* parse "rand:NBITS", "0xHEX...", or decimal; returns 0 on success */
static int big_parse(Big *b, const char *s, Rng *gen) {
    big_zero(b);
    if (strncmp(s, "rand:", 5) == 0) {
        int nb = atoi(s + 5);
        if (nb < 1 || nb > MAX_BITS) return -2;
        int limbs = (nb + 31) / 32;
        for (int i = 0; i < limbs; i++) b->l[i] = (uint32_t)rng_next(gen);
        int topbits = nb - (limbs - 1) * 32;
        uint32_t mask = (topbits == 32) ? 0xFFFFFFFFu : ((1u << topbits) - 1);
        b->l[limbs-1] &= mask;
        b->l[limbs-1] |= 1u << (topbits - 1);          /* exactly nb bits     */
        b->n = limbs;
        return 0;
    }
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        for (const char *p = s + 2; *p; p++) {
            int v;
            if      (*p >= '0' && *p <= '9') v = *p - '0';
            else if (*p >= 'a' && *p <= 'f') v = *p - 'a' + 10;
            else if (*p >= 'A' && *p <= 'F') v = *p - 'A' + 10;
            else return -1;
            if (big_mul_add_small(b, 16, (uint32_t)v)) return -1;
        }
        return 0;
    }
    for (const char *p = s; *p; p++) {
        if (*p < '0' || *p > '9') return -1;
        if (big_mul_add_small(b, 10, (uint32_t)(*p - '0'))) return -1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Gate builders — DFT₆ and digit kicks diag(e^{i·g·2πk/6}), g = 1..5
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

static void build_digit_kick(double *U_re, double *U_im, int g) {
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));
    for (int k = 0; k < D; k++) {
        double a = 2.0 * M_PI * g * k / (double)D;
        U_re[k*D+k] = cos(a);
        U_im[k*D+k] = sin(a);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HIDDEN OBJECT — prolate 4-ellipsoid (long axis 2:1), tilted in x–w
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    double cx, cy, cz, cw, a, b, theta;
} HiddenObject;

static int object_contains(const HiddenObject *o, int x, int y, int z, int w) {
    double rx = x - o->cx, ry = y - o->cy, rz = z - o->cz, rw = w - o->cw;
    double ux = cos(o->theta), uw = sin(o->theta);
    double l  = rx*ux + rw*uw;
    double r2 = rx*rx + ry*ry + rz*rz + rw*rw;
    double t2 = r2 - l*l;
    return (l*l)/(o->a*o->a) + t2/(o->b*o->b) <= 1.0;
}

static long count_interior(const HiddenObject *o, int Lx, int Ly, int Lz, int Lw) {
    long c = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++)
        c += object_contains(o, x, y, z, w);
    return c;
}

/* Auto-size ellipsoid + lattice so the interior holds `need` sites.
 * 4-ellipsoid volume = (π²/2)·a·b³ with a = 2b → V = π²·b⁴.               */
static int size_experiment(HiddenObject *o, Rng *gen, long need,
                           int *Lx, int *Ly, int *Lz, int *Lw) {
    double b = pow((double)need / (M_PI * M_PI), 0.25);
    for (int attempt = 0; attempt < 40; attempt++, b *= 1.06) {
        o->b = b; o->a = 2.0 * b;
        o->theta = 0.5 * M_PI * rng_uniform(gen);
        int sx = (int)ceil(2.0 * o->a + 4.0);            /* x,w see the tilt  */
        int st = (int)ceil(2.0 * o->b + 4.0);
        *Lx = sx; *Ly = st; *Lz = st; *Lw = sx + sx / 2; /* W is the long axis*/
        if ((long)(*Lx) * (*Ly) * (*Lz) * (*Lw) > MAX_SITES) return -1;
        double m = o->a + 1.0;
        o->cx = m + rng_uniform(gen) * ((*Lx - 1) - 2.0*m);
        o->cy = o->b + 1.0 + rng_uniform(gen) * ((*Ly - 1) - 2.0*(o->b + 1.0));
        o->cz = o->b + 1.0 + rng_uniform(gen) * ((*Lz - 1) - 2.0*(o->b + 1.0));
        o->cw = m + rng_uniform(gen) * ((*Lw - 1) - 2.0*m);
        if (count_interior(o, *Lx, *Ly, *Lz, *Lw) >= need) return 0;
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4×4 symmetric Jacobi eigensolver — inertia tensor principal axis
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
                A[k*4+p] = c*akp - s*akq;  A[k*4+q] = s*akp + c*akq;
            }
            for (int k = 0; k < 4; k++) {
                double apk = A[p*4+k], aqk = A[q*4+k];
                A[p*4+k] = c*apk - s*aqk;  A[q*4+k] = s*apk + c*aqk;
            }
            for (int k = 0; k < 4; k++) {
                double vkp = V[k*4+p], vkq = V[k*4+q];
                V[k*4+p] = c*vkp - s*vkq;  V[k*4+q] = s*vkp + c*vkq;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <decimal | 0xHEX | rand:NBITS> [seed]\n", argv[0]);
        return 1;
    }
    uint64_t seed = (argc > 2) ? strtoull(argv[2], NULL, 10) : (uint64_t)time(NULL);
    Rng gen; rng_seed(&gen, seed);

    /* ── 0. Parse the number, enforce the 3000-bit cap ── */
    Big num;
    int prc = big_parse(&num, argv[1], &gen);
    if (prc == -2) { fprintf(stderr, "rand:NBITS must be 1..%d\n", MAX_BITS); return 1; }
    if (prc)       { fprintf(stderr, "could not parse number '%s'\n", argv[1]); return 1; }
    int nbits = big_bits(&num);
    if (nbits > MAX_BITS) {
        fprintf(stderr, "number is %d bits — cap is %d bits\n", nbits, MAX_BITS);
        return 1;
    }

    /* ── 1. Number → base-5 digits (LSD first) ── */
    static uint8_t digits[8192];
    long ndigits = 0;
    {
        Big tmp = num;
        if (big_is_zero(&tmp)) digits[ndigits++] = 0;
        while (!big_is_zero(&tmp)) digits[ndigits++] = (uint8_t)big_divmod_small(&tmp, 5);
    }
    long need = HEADER_DIGITS + ndigits;

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  W-AXIS PROBE v2 — hide a number in 4D, then find it      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("  Number: %d bits → %ld base-5 digits (+%d header) = %ld sites needed\n",
           nbits, ndigits, HEADER_DIGITS, need);

    /* ── 2. Auto-size lattice + hidden ellipsoid ── */
    HiddenObject obj;
    int Lx, Ly, Lz, Lw;
    if (size_experiment(&obj, &gen, need, &Lx, &Ly, &Lz, &Lw)) {
        fprintf(stderr, "could not fit %ld digit sites under the %ld-site cap\n",
                need, MAX_SITES);
        return 1;
    }
    long Nsites = (long)Lx * Ly * Lz * Lw;
    long interior = count_interior(&obj, Lx, Ly, Lz, Lw);
    printf("  Lattice %d×%d×%d×%d = %ld sites; ellipsoid interior = %ld sites\n",
           Lx, Ly, Lz, Lw, Nsites, interior);
    printf("  Seed = %llu\n\n", (unsigned long long)seed);

    /* ── 3. Prep |0⟩ → DFT₆ everywhere, then write header+digits+padding ──
     *  Payload stream (LSD-first): HEADER_DIGITS base-5 digits of ndigits,
     *  then the number's digits, then 0-padding to fill the interior.
     *  Digit value v∈{0..4} → kick g = v+1 ∈ {1..5}: always nonzero readout. */
    double F_re[36], F_im[36], Fi_re[36], Fi_im[36];
    build_dft6(F_re, F_im, 0);
    build_dft6(Fi_re, Fi_im, 1);
    double Kg_re[6][36], Kg_im[6][36];
    for (int g = 1; g <= 5; g++) build_digit_kick(Kg_re[g], Kg_im[g], g);

    P4D_GRID *g4 = P4D_INIT(Lx, Ly, Lz, Lw);
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++) {
        double re[D] = {1, 0, 0, 0, 0, 0}, im[D] = {0};
        P4D_SET_PROD(g4, x, y, z, w, re, im);
        P4D_GATE_1(g4, x, y, z, w, F_re, F_im);
    }

    long len_tmp = ndigits, stream_pos = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++) {
        if (!object_contains(&obj, x, y, z, w)) continue;
        int v;
        if (stream_pos < HEADER_DIGITS) {                /* length header     */
            v = (int)(len_tmp % 5); len_tmp /= 5;
        } else if (stream_pos < HEADER_DIGITS + ndigits) {
            v = digits[stream_pos - HEADER_DIGITS];      /* payload           */
        } else {
            v = 0;                                       /* padding           */
        }
        P4D_GATE_1(g4, x, y, z, w, Kg_re[v+1], Kg_im[v+1]);
        stream_pos++;
    }

    /* ── 4. HIDDENNESS CONTROL: density must be flat 1/6 everywhere ── */
    double probs[D], max_dev = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++) {
        P4D_DENSITY(g4, x, y, z, w, probs);
        for (int k = 0; k < D; k++) {
            double d = fabs(probs[k] - 1.0/D);
            if (d > max_dev) max_dev = d;
        }
    }
    printf("  Hiddenness control: max |P(k) − 1/6| = %.3e  %s\n",
           max_dev, max_dev < 1e-9 ? "→ number is density-invisible ✓"
                                   : "→ WARNING: encoding leaks into density");

    /* ── 5. Interferometric readout: DFT₆⁻¹ everywhere, read levels ── */
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++)
        P4D_GATE_1(g4, x, y, z, w, Fi_re, Fi_im);

    /* ── 6. Blind geometry: centroid + inertia from the nonzero cloud ── */
    double W = 0, mx = 0, my = 0, mz = 0, mw = 0;
    uint8_t *level = malloc((size_t)Nsites);
    long idx = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++, idx++) {
        P4D_DENSITY(g4, x, y, z, w, probs);
        int best = 0;
        for (int k = 1; k < D; k++) if (probs[k] > probs[best]) best = k;
        level[idx] = (uint8_t)best;
        if (best != 0) { W += 1; mx += x; my += y; mz += z; mw += w; }
    }
    if (W < 1) { printf("  Found nothing in 4D space — aborting.\n"); return 1; }
    mx /= W; my /= W; mz /= W; mw /= W;

    double S[16] = {0}, V[16];
    idx = 0;
    for (int x = 0; x < Lx; x++) for (int y = 0; y < Ly; y++)
    for (int z = 0; z < Lz; z++) for (int w = 0; w < Lw; w++, idx++) {
        if (level[idx] == 0) continue;
        double r[4] = { x - mx, y - my, z - mz, w - mw };
        for (int p = 0; p < 4; p++)
            for (int q = 0; q < 4; q++) S[p*4+q] += r[p] * r[q];
    }
    for (int i = 0; i < 16; i++) S[i] /= W;
    jacobi4(S, V);
    int top = 0;
    for (int k = 1; k < 4; k++) if (S[k*4+k] > S[top*4+top]) top = k;
    double theta_hat = atan2(fabs(V[3*4+top]), fabs(V[0*4+top]));

    printf("\n  ── FOUND IN 4D SPACE (blind) ──\n");
    printf("  Cloud: %.0f sites, centroid (%.2f, %.2f, %.2f | ŵ₀=%.2f)\n",
           W, mx, my, mz, mw);
    printf("  W-axis tilt θ̂ = %.1f°\n", theta_hat * 180.0 / M_PI);

    /* ── 7. Blind decode: nonzero sites in scan order → header → number ── */
    long rec_len = -1, pos = 0, hdr_pow = 1, payload_read = 0;
    long hdr_val = 0;
    static uint8_t rec_digits[8192];
    idx = 0;
    for (int x = 0; x < Lx && (rec_len < 0 || payload_read < rec_len); x++)
    for (int y = 0; y < Ly && (rec_len < 0 || payload_read < rec_len); y++)
    for (int z = 0; z < Lz && (rec_len < 0 || payload_read < rec_len); z++)
    for (int w = 0; w < Lw && (rec_len < 0 || payload_read < rec_len); w++) {
        long i = ((((long)x*Ly + y)*Lz + z)*Lw + w);
        if (level[i] == 0) continue;
        int v = level[i] - 1;
        if (pos < HEADER_DIGITS) {
            hdr_val += (long)v * hdr_pow; hdr_pow *= 5;
            if (pos == HEADER_DIGITS - 1) rec_len = hdr_val;
        } else if (rec_len >= 0 && payload_read < rec_len) {
            rec_digits[payload_read++] = (uint8_t)v;
        }
        pos++;
    }

    Big rec; big_zero(&rec);
    for (long i = payload_read - 1; i >= 0; i--)         /* MSD-first rebuild */
        big_mul_add_small(&rec, 5, rec_digits[i]);

    /* ── 8. Unblind + verdict ── */
    printf("\n  ── DECODE ──\n");
    printf("  Header length: %ld digits (true: %ld)\n", rec_len, ndigits);
    big_print_hex(&num, "Original:  ");
    big_print_hex(&rec, "Recovered: ");
    int ok = big_eq(&num, &rec);
    printf("\n  VERDICT: %s\n", ok ? "BIT-EXACT RECOVERY ✓" : "MISMATCH ✗");
    printf("\n  ── UNBLINDED GEOMETRY ──\n");
    printf("  True center (%.2f, %.2f, %.2f | w₀=%.2f), tilt θ=%.1f°\n",
           obj.cx, obj.cy, obj.cz, obj.cw, obj.theta * 180.0 / M_PI);
    printf("  Errors: |ŵ₀−w₀| = %.3f,  |θ̂−θ| = %.2f°\n",
           fabs(mw - obj.cw), fabs(theta_hat - obj.theta) * 180.0 / M_PI);

    free(level);
    P4D_FREE(g4);
    return ok ? 0 : 2;
}
