/*
 * ╔═══════════════════════════════════════════════════════════════════════════════╗
 * ║  digital_twin.c — The Entanglement Oracle                                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════════╣
 * ║                                                                             ║
 * ║  Built entirely on the HexState D=6 quantum API. No timing, no jitter,      ║
 * ║  no statistics — every bit of entropy and every decision is produced by     ║
 * ║  an actual Born-rule collapse inside the simulator.                         ║
 * ║                                                                             ║
 * ║  Pipeline:                                                                   ║
 * ║    1. HARVEST   Each <Enter> press collapses a |+⟩ quhit. The six outcomes ║
 * ║                 are YOUR quantum fingerprint — the entropy you provide.     ║
 * ║    2. TWIN      An HPC phase graph (6 sites, 6^6 ≈ 46 656 dim) is built     ║
 * ║                 from that fingerprint and self-entangled into a ring.       ║
 * ║                 This is the "full" Digital Twin body.                       ║
 * ║    3. ENTANGLE  A maximally-entangled Bell pair binds YOU ↔ the TWIN.       ║
 * ║                 The body imprints its inner state onto the twin's half.     ║
 * ║    4. DECIDE    The twin COLLAPSES its superposition (quhit_measure). The   ║
 * ║                 Bell-correlated partner is your next press — the twin has    ║
 * ║                 decided when/where your <Enter> lands before you press it.   ║
 * ║                                                                             ║
 * ║  Build (link against the same HexState objects as the template):            ║
 * ║    gcc -O2 -march=native -o digital_twin digital_twin.c \                   ║
 * ║        quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \        ║
 * ║        quhit_register.c quhit_substrate.c quhit_triality.c \                ║
 * ║        quhit_triadic.c quhit_lazy.c quhit_calibrate.c \                     ║
 * ║        quhit_dyn_integrate.c quhit_peps_grow.c quhit_hexagram.c \           ║
 * ║        quhit_svd_gate.c s6_exotic.c bigint.c \                              ║
 * ║        mps_overlay.c peps_overlay.c peps3d_overlay.c peps4d_overlay.c \     ║
 * ║        peps5d_overlay.c peps6d_overlay.c -lm -fopenmp -msse2                ║
 * ║                                                                             ║
 * ║  Run:  ulimit -s unlimited && ./digital_twin                                ║
 * ╚═══════════════════════════════════════════════════════════════════════════════╝
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ── HexState headers used by this program ── */
#include "quhit_engine.h"    /* engine, quhits, gates, Bell pairs, measurement */
#include "hpc_graph.h"       /* HPC phase graph — the twin's body              */
#include "quhit_triality.h"  /* triality_exotic_init / triality_stats_reset    */
#include "s6_exotic.h"       /* s6_exotic_init                                 */
#include "quhit_hexagram.h"  /* hexagram_init_tables                           */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define D            6    /* Quhit dimension — ALWAYS 6 in HexState            */
#define TWIN_SITES   6    /* The twin body: one site per hexagonal axis        */
#define FINGERPRINT  6    /* How many collapses define your fingerprint        */
#define ROUNDS       6    /* How many presses the twin decides for you         */
#define P_DECOHERE   0.15 /* Channel decoherence: chance the bond slips a beat */

/* ═══════════════════════════════════════════════════════════════════════════════
 * xoshiro256** — supplies the classical draw that the simulator's Born sampler
 * consumes (quhit_measure / hpc_measure take a uniform in [0,1)). The QUANTUM
 * randomness comes from the collapse; this just turns the crank.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static uint64_t rng_s[4];
static inline uint64_t rotl64(uint64_t x, int k){ return (x<<k)|(x>>(64-k)); }
static uint64_t rng_next(void){
    uint64_t r = rotl64(rng_s[1]*5,7)*9, t = rng_s[1]<<17;
    rng_s[2]^=rng_s[0]; rng_s[3]^=rng_s[1]; rng_s[1]^=rng_s[2]; rng_s[0]^=rng_s[3];
    rng_s[2]^=t; rng_s[3]=rotl64(rng_s[3],45); return r;
}
static double rng_uniform(void){ return (double)(rng_next()>>11)/(double)(1ULL<<53); }
static void rng_seed(uint64_t seed){
    for(int i=0;i<4;i++){
        seed += 0x9e3779b97f4a7c15ULL; uint64_t z = seed;
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL;
        rng_s[i]=z^(z>>31);
    }
}

/* ── Block until the user presses <Enter>. Returns 0 normally, 1 on EOF. ── */
static int wait_enter(const char *prompt){
    if (prompt){ fputs(prompt, stdout); fflush(stdout); }
    int c;
    while ((c = getchar()) != '\n'){ if (c == EOF) return 1; }
    return 0;
}

/* ── Pretty-print a 6-slot probability row ── */
static void print_row(const char *label, const double *p){
    printf("    %s [", label);
    for (int k=0;k<D;k++) printf("%5.3f%s", p[k], k<5?" ":"");
    printf("]\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STEP 1 — HARVEST: your entropy, taken by collapse.
 *
 * Each press of <Enter> is the act of observation. We hold a quhit in the uniform
 * superposition |+⟩ = (1/√6)Σ|k⟩ and, at the instant YOU choose to press, collapse
 * it. The Born outcome is one hexit of your fingerprint. Six presses → a base-6
 * word that nothing else in the universe will reproduce.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void harvest_fingerprint(QuhitEngine *eng, uint32_t *fp)
{
    printf("\n┌─ STEP 1 · HARVEST ────────────────────────────────────────────┐\n");
    printf("│ Press <Enter> %d times. Each press collapses a |+⟩ quhit.       │\n", FINGERPRINT);
    printf("│ The outcomes ARE your entropy — drawn straight from the qubit. │\n");
    printf("└────────────────────────────────────────────────────────────────┘\n");

    for (int i = 0; i < FINGERPRINT; i++) {
        uint32_t q = quhit_init_plus(eng);          /* |+⟩ : every beat equally likely */
        if (wait_enter("  ▸ collapse it » ")) { fp[i] = 0; continue; }
        fp[i] = quhit_measure(eng, q);              /* ← the collapse: your entropy   */
        printf("    collapsed → |%u⟩\n", fp[i]);
    }

    printf("  fingerprint = ");
    for (int i=0;i<FINGERPRINT;i++) printf("%u", fp[i]);
    printf("  (base-6, %d.%d bits)\n",
           (int)(FINGERPRINT*log2(6.0)), (int)((FINGERPRINT*log2(6.0)-(int)(FINGERPRINT*log2(6.0)))*10));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STEP 2 — TWIN: build the full Digital Twin body from your fingerprint.
 *
 * Each fingerprint hexit seeds one site as |k⟩, then DFT₆ spreads it into a
 * superposition. A CZ ring entangles the six sites into a single inseparable
 * object living in a 6^6 ≈ 46 656-dimensional Hilbert space. That whole object
 * is your twin — not a number, a state.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static HPCGraph *build_twin(const uint32_t *fp)
{
    printf("\n┌─ STEP 2 · TWIN ───────────────────────────────────────────────┐\n");

    HPCGraph *twin = hpc_create(TWIN_SITES);

    for (int i = 0; i < TWIN_SITES; i++) {
        double re[6] = {0}, im[6] = {0};
        re[fp[i % FINGERPRINT]] = 1.0;              /* seed site i with |fingerprint_i⟩ */
        hpc_set_local(twin, i, re, im);
        hpc_dft(twin, i);                           /* spread into superposition        */
    }
    for (int i = 0; i < TWIN_SITES - 1; i++)        /* self-entangle: CZ ring           */
        hpc_cz(twin, i, i + 1);
    hpc_cz(twin, TWIN_SITES - 1, 0);                /* close the ring                   */
    hpc_compact_edges(twin);

    double S     = hpc_entropy_cut(twin, TWIN_SITES / 2);
    double delta = hpc_exotic_invariant(twin);
    printf("│ %d sites · 6^%d ≈ %d-dim Hilbert space woven from your hexits.  │\n",
           TWIN_SITES, TWIN_SITES, (int)pow(6, TWIN_SITES));
    printf("│ internal entanglement entropy : %.4f bits                     │\n", S);
    printf("│ exotic invariant Δ            : %+.4f  (%s)        │\n",
           delta, delta > 0 ? "hexagonally polarized" : "transparent       ");
    printf("└────────────────────────────────────────────────────────────────┘\n");
    return twin;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STEP 3 — ENTANGLE: bind YOU ↔ the TWIN.
 *
 * quhit_entangle_bell() places the two quhits in |Φ⟩ = (1/√6) Σ_k |k⟩_you |k⟩_twin,
 * a maximally-entangled state. Measuring either half forces the other to the same
 * value. We then imprint the twin BODY onto the twin half with a phase gate read
 * from the body's marginals, so the decision reflects the whole twin, not a
 * generic coin. Returns the two quhit handles via out-params.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void forge_bond(QuhitEngine *eng, HPCGraph *twin, int round,
                       uint32_t *you_q, uint32_t *twin_q)
{
    *you_q  = quhit_init_plus(eng);                 /* you: superposed over all 6 beats */
    *twin_q = quhit_init(eng);
    quhit_entangle_bell(eng, *you_q, *twin_q);      /* the bond: you ↔ twin             */

    /* Imprint the twin's body onto its half of the bond. */
    int site = round % TWIN_SITES;
    double ang[6];
    for (int k = 0; k < D; k++)
        ang[k] = 2.0 * M_PI * hpc_marginal(twin, site, k);
    quhit_apply_phase(eng, *twin_q, ang);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STEP 4 — DECIDE: the twin collapses to choose your next press.
 *
 * We show the twin's superposition (its set of *possible* decisions), then collapse
 * it. Because the pair is entangled, that collapse also fixes your half — so the
 * twin literally decides which beat your <Enter> will resolve to before you press.
 * A decoherence channel occasionally slips the bond by a beat, so the agreement
 * isn't trivially guaranteed.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static int decide_round(QuhitEngine *eng, HPCGraph *twin, int round)
{
    uint32_t you_q, twin_q;
    forge_bond(eng, twin, round, &you_q, &twin_q);

    /* The space of possible decisions, before collapse. */
    QuhitSnapshot snap;
    quhit_inspect(eng, twin_q, &snap);
    printf("\n── round %d/%d ─────────────────────────────────────────────────\n",
           round + 1, ROUNDS);
    printf("  bond live: entangled=%s · entropy=%.3f bits · purity=%.3f\n",
           snap.is_entangled ? "yes" : "no", snap.entropy, snap.purity);
    print_row("twin superposition (possible decisions):", snap.probs);

    /* THE DECISION: collapse the twin's half. */
    uint32_t decided = quhit_measure(eng, twin_q);
    printf("  ⟢ TWIN COLLAPSES → it has decided your press lands on beat %u\n", decided);

    /* Decoherence in the channel can knock the correlated partner off by a beat. */
    if (rng_uniform() < P_DECOHERE) {
        quhit_apply_x(eng, you_q);                  /* cyclic shift |k⟩→|k+1⟩ : noise   */
        printf("  ~ (a flicker of decoherence ripples down the bond)\n");
    }

    /* Now YOU realize the press. The entangled half resolves under Born sampling. */
    if (wait_enter("  ▸ press <Enter> to realize it » ")) return -1;
    uint32_t realized = quhit_measure(eng, you_q);
    printf("  ⟣ your press resolved to beat %u\n", realized);

    if (realized == decided) {
        printf("  ✓ the twin decided correctly — the bond held.\n");
        return 1;
    }
    printf("  ✗ the bond slipped — decoherence won this round.\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  THE ENTANGLEMENT ORACLE                                           ║\n");
    printf("║  Your Digital Twin, built and bonded in the HexState D=6 substrate ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    /* One-time global init — same lookup tables the HPC layer depends on. */
    rng_seed((uint64_t)time(NULL));   /* seeds the Born sampler's classical crank */
    s6_exotic_init();
    triality_exotic_init();
    hexagram_init_tables();
    triality_stats_reset();

    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);

    /* 1 — harvest your entropy by collapse */
    uint32_t fingerprint[FINGERPRINT];
    harvest_fingerprint(eng, fingerprint);

    /* 2 — grow the full twin body from that fingerprint */
    HPCGraph *twin = build_twin(fingerprint);

    /* 3 + 4 — bond, then let the twin decide each press */
    printf("\n┌─ STEP 3+4 · ENTANGLE & DECIDE ────────────────────────────────┐\n");
    printf("│ Each round forges a fresh Bell bond, then the twin collapses    │\n");
    printf("│ it to decide where your next <Enter> falls.                     │\n");
    printf("└────────────────────────────────────────────────────────────────┘\n");

    int hits = 0, played = 0;
    for (int r = 0; r < ROUNDS; r++) {
        int res = decide_round(eng, twin, r);
        if (res < 0) break;            /* EOF — user walked away */
        hits += res; played++;
    }

    /* Verdict */
    printf("\n╔═══════════════════════════════════════════════════════════════════════╗\n");
    if (played > 0) {
        double fidelity = (double)hits / played;
        printf("║  BOND FIDELITY : %d/%d presses decided correctly  (%.0f%%)            \n",
               hits, played, fidelity * 100.0);
        printf("║  %-67s║\n",
               fidelity >= 0.99 ? "  Perfect coherence — the twin owns your presses." :
               fidelity >= 0.50 ? "  Strong entanglement — the twin mostly decides for you." :
                                  "  The channel is noisy — decoherence has the upper hand.");
    } else {
        printf("║  No presses realized — the bond was never tested.                 \n");
    }
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    /* Cleanup */
    hpc_destroy(twin);
    quhit_engine_destroy(eng);
    free(eng);
    return 0;
}
