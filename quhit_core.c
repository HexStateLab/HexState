/*
 * quhit_core.c — Engine Lifecycle, PRNG, Quhit Init
 *
 * The foundation. Every quhit starts here.
 * In SDR mode, quhits are physically sourced from the electromagnetic
 * field via an RTL-SDR dongle, and Born-rule randomness is physical
 * quantum noise instead of a deterministic LCG.
 */

#include "quhit_engine.h"
#include "quhit_sdr.h"

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENGINE LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_engine_init(QuhitEngine *eng)
{
    memset(eng, 0, sizeof(*eng));
    eng->prng_state = 0x5DEECE66DULL ^ 0xCAFEBABEULL;
    eng->sdr_mode   = 0;
    eng->sdr_state   = NULL;
}

void quhit_engine_destroy(QuhitEngine *eng)
{
    /* Deactivate all pairs (no heap allocs — QuhitJoint is inline) */
    for (uint32_t i = 0; i < eng->num_pairs; i++)
        eng->pairs[i].active = 0;

    eng->num_quhits    = 0;
    eng->num_pairs     = 0;
    eng->num_registers = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRNG — LCG (same constants as java.util.Random)
 *
 * In SDR mode, this is bypassed — quhit_sdr_random() provides physical
 * quantum noise from the RTL-SDR's RF front-end (or CPU timing jitter).
 * The deterministic LCG is the fallback for pure-simulation mode.
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint64_t quhit_prng(QuhitEngine *eng)
{
    if (eng->sdr_mode && eng->sdr_state) {
        /* Use physical noise — mix into LCG state for reproducibility tracking */
        double r = quhit_sdr_random((SdrState *)eng->sdr_state);
        uint64_t bits = (uint64_t)(r * (double)0x100000000ULL * (double)0x100000000ULL);
        eng->prng_state = (eng->prng_state ^ bits) * 6364136223846793005ULL
                        + 1442695040888963407ULL;
        return eng->prng_state;
    }

    eng->prng_state = eng->prng_state * 6364136223846793005ULL
                    + 1442695040888963407ULL;
    return eng->prng_state;
}

double quhit_prng_double(QuhitEngine *eng)
{
    return (double)(quhit_prng(eng) >> 11) / (double)(1ULL << 53);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * QUHIT INITIALIZATION
 *
 * Each quhit = 6 complex amplitudes = 96 bytes.
 * No heap allocation — everything is inline in the engine struct.
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint32_t quhit_init(QuhitEngine *eng)
{
    if (eng->num_quhits >= MAX_QUHITS) {
        fprintf(stderr, "[QUHIT] ERROR: max quhits (%d) reached\n", MAX_QUHITS);
        return UINT32_MAX;
    }

    uint32_t id = eng->num_quhits++;
    Quhit *q = &eng->quhits[id];

    q->id             = id;
    q->collapsed      = 0;
    q->collapse_value = 0;
    q->pair_id        = -1;
    q->pair_side      = 0;

    /* |0⟩ — uses quhit_management.h primitive */
    qm_init_zero(&q->state);

    return id;
}

uint32_t quhit_init_plus(QuhitEngine *eng)
{
    uint32_t id = quhit_init(eng);
    if (id == UINT32_MAX) return id;

    /* |+⟩ = (1/√6) Σ|k⟩ — uses quhit_management.h primitive */
    qm_init_plus(&eng->quhits[id].state);

    return id;
}

uint32_t quhit_init_basis(QuhitEngine *eng, uint32_t k)
{
    uint32_t id = quhit_init(eng);
    if (id == UINT32_MAX) return id;

    if (k < QUHIT_D) {
        QuhitState *s = &eng->quhits[id].state;
        memset(s, 0, sizeof(*s));
        s->re[k] = 1.0;
    }

    return id;
}

void quhit_reset(QuhitEngine *eng, uint32_t id)
{
    if (id >= eng->num_quhits) return;
    Quhit *q = &eng->quhits[id];

    /* If entangled, disentangle first */
    if (q->pair_id >= 0) {
        QuhitPair *p = &eng->pairs[q->pair_id];
        uint32_t partner = (q->pair_side == 0) ? p->id_b : p->id_a;
        quhit_disentangle(eng, id, partner);
    }

    q->collapsed      = 0;
    q->collapse_value = 0;
    q->pair_id        = -1;
    q->pair_side      = 0;
    qm_init_zero(&q->state);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SDR MODE — Physical EM field as quhit source and entropy
 *
 * When SDR mode is enabled, quhit_sdr_init() sources amplitudes from the
 * RTL-SDR's I/Q stream (partitioned into D=6 frequency bins), and all
 * Born-rule measurements use physical quantum noise instead of the
 * deterministic LCG.
 *
 * The engine's sdr_mode flag controls this behavior globally.
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_sdr_enable(QuhitEngine *eng, void *sdr_state)
{
    if (!sdr_state) return -1;
    SdrState *sdr = (SdrState *)sdr_state;
    if (sdr->mode == SDR_MODE_OFF) return -1;

    eng->sdr_mode  = 1;
    eng->sdr_state = sdr_state;

    fprintf(stderr, "[SDR] Engine SDR mode enabled: %s\n",
            quhit_sdr_mode_string(sdr));
    return 0;
}

void quhit_sdr_disable(QuhitEngine *eng)
{
    eng->sdr_mode  = 0;
    eng->sdr_state = NULL;
    fprintf(stderr, "[SDR] Engine SDR mode disabled — using deterministic PRNG\n");
}

uint32_t quhit_init_sdr(QuhitEngine *eng)
{
    if (!eng->sdr_mode || !eng->sdr_state) {
        fprintf(stderr, "[SDR] ERROR: SDR mode not enabled. "
                "Call quhit_sdr_enable() first.\n");
        return quhit_init_plus(eng);  /* Fall back to |+⟩ */
    }

    if (eng->num_quhits >= MAX_QUHITS) {
        fprintf(stderr, "[QUHIT] ERROR: max quhits (%d) reached\n", MAX_QUHITS);
        return UINT32_MAX;
    }

    return quhit_sdr_init_quhit((SdrState *)eng->sdr_state, eng);
}
