/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * quhit_engine.h — Quhit Engine: 6-State Quantum Processor
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Master header for the modular Quhit engine.
 *
 * Architecture:
 *   - Each quhit = 6 complex amplitudes (96 bytes)
 *   - Entangled pair = 36 complex amplitudes (576 bytes)
 *   - Total memory: O(N) for locals + O(P) for pairs
 *   - Never O(D^N) — polynomial, not exponential
 *
 * Side-channel primitives (header-only, already extracted):
 *   arithmetic.h      — IEEE-754 constants, magic numbers
 *   born_rule.h       — Born-rule sampling, fast inverse sqrt
 *   statevector.h     — Cache-aligned state vector storage
 *   superposition.h   — DFT₆, twiddle tables, superposition
 *   entanglement.h    — Joint states, Bell, partial trace
 *   quhit_management.h — Per-quhit state management
 *
 * Component files:
 *   quhit_core.c      — Engine lifecycle, PRNG, chunk init
 *   quhit_gates.c     — CZ, DFT, local unitaries
 *   quhit_measure.c   — Born-rule measurement, inspect, snapshot
 *   quhit_entangle.c  — Braid, product state, unbraid
 *   quhit_register.c  — 100T-scale quhit register API
 */

#ifndef QUHIT_ENGINE_H
#define QUHIT_ENGINE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ─── Side-channel primitives ─── */
#include "arithmetic.h"
#include "born_rule.h"
#include "quhit_gauss.h"
#include "statevector.h"
#include "superposition.h"
#include "entanglement.h"
#include "quhit_management.h"

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define QUHIT_D             6           /* Dimension per quhit              */
#define QUHIT_D2            36          /* D² = joint state dimension       */
#define QUHIT_LOCAL_BYTES   96          /* 6 × 16 bytes per quhit          */
#define QUHIT_JOINT_BYTES   576         /* 36 × 16 bytes per pair          */

#define MAX_QUHITS          262144      /* Max quhits in engine (256K)      */
#define MAX_PAIRS           262144      /* Max entangled pairs (256K)       */
#define MAX_REGISTERS       65536       /* Max quhit registers (expanded for large N) */
#define MAX_CZ_DEFERRED     4096        /* Max deferred CZ per pair         */

#define MAGIC_TAG           0xBEEF      /* Magic pointer tag                */
#define MAGIC_PTR(chunk, k) (((uint64_t)MAGIC_TAG << 48) | ((uint64_t)(chunk) << 16) | (k))

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ─── Single Quhit ─── */
typedef struct {
    QuhitState   state;             /* 6 complex amplitudes (96 bytes)      */
    uint32_t     id;                /* Quhit identifier                     */
    uint8_t      collapsed;         /* 1 = measured                         */
    uint32_t     collapse_value;    /* Measured outcome                     */
    int32_t      pair_id;           /* Index into pairs[] or -1             */
    uint8_t      pair_side;         /* 0 = A side, 1 = B side              */
} Quhit;

/* ─── Entangled Pair ─── */
typedef struct {
    QuhitJoint   joint;             /* 36 complex amplitudes (576 bytes)    */
    uint32_t     id_a;              /* Quhit A index                        */
    uint32_t     id_b;              /* Quhit B index                        */
    uint8_t      active;            /* 1 = pair is live                     */
    uint32_t     num_cz;            /* Deferred CZ count                    */
} QuhitPair;

/* ─── 128-bit basis state type (GCC/Clang __int128) ─── */
typedef unsigned __int128 basis_t;

/* ─── Quhit Register (100T-scale, Magic Pointer Hilbert Space) ─── */
typedef struct {
    uint64_t     chunk_id;          /* Parent chunk ID                      */
    uint64_t     n_quhits;          /* Number of quhits in register         */
    uint32_t     dim;               /* Dimension per quhit (D=6)            */
    uint8_t      bulk_rule;         /* 0=general, 1=GHZ, 2=Gauss circuit   */

    /* The Hilbert Space — self-describing sparse entries */
    uint32_t     num_nonzero;       /* Active basis entries                 */
    struct {
        basis_t  basis_state;       /* Packed basis: Σ q_k × D^k (128-bit) */
        double   amp_re;            /* Real amplitude                       */
        double   amp_im;            /* Imaginary amplitude                  */
    } entries[4096];                /* Sparse amplitude storage             */

    uint8_t      collapsed;         /* 1 = measurement has occurred         */
    uint32_t     collapse_outcome;  /* Determined value (all members)       */
    uint64_t     magic_base;        /* Base Magic Pointer for register      */

    /* ─── Gauss Circuit Log (bulk_rule=2) ─── */
    uint16_t     gauss_n_cz;        /* Number of CZ gates recorded          */
    uint16_t     gauss_cz_a[256];   /* CZ pair: qubit A index               */
    uint16_t     gauss_cz_b[256];   /* CZ pair: qubit B index               */
    uint16_t     gauss_n_dft;       /* Number of DFT sweeps applied         */
    uint8_t      gauss_ready;       /* 1 = full circuit applied, ready      */
} QuhitRegister;

/* ─── Inspection Snapshot (non-destructive readout) ─── */
typedef struct {
    uint32_t     quhit_id;          /* Which quhit was inspected            */
    uint32_t     dim;               /* Dimension (6)                        */
    double       probs[QUHIT_D];    /* Marginal probabilities P(k)          */
    double       total_prob;        /* Sum of probabilities (should be 1.0) */
    double       entropy;           /* Von Neumann entropy (bits)           */
    double       purity;            /* Tr(ρ²)                               */
    int          is_entangled;      /* 1 if part of a pair                  */
    int          schmidt_rank;      /* 1=product, >1=entangled              */

    /* Joint state entries (if entangled) */
    uint32_t     num_entries;
    struct {
        uint32_t idx_a, idx_b;      /* Basis indices |a,b⟩                  */
        double   amp_re, amp_im;    /* Complex amplitude                    */
        double   probability;       /* |amplitude|²                         */
        double   phase_rad;         /* Phase angle                          */
    } entries[QUHIT_D2];
} QuhitSnapshot;

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENGINE
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* ─── Quhit array ─── */
    Quhit        quhits[MAX_QUHITS];
    uint32_t     num_quhits;

    /* ─── Entangled pairs ─── */
    QuhitPair    pairs[MAX_PAIRS];
    uint32_t     num_pairs;

    /* ─── Quhit registers (100T scale) ─── */
    QuhitRegister registers[MAX_REGISTERS];
    uint32_t     num_registers;

    /* ─── PRNG state ─── */
    uint64_t     prng_state;

    /* ─── SDR mode (physically-sourced quhits and randomness) ─── */
    int          sdr_mode;        /* 0=deterministic PRNG, 1=physical SDR */
    void        *sdr_state;       /* Opaque SdrState pointer              */

    /* ─── Measurement log ─── */
    uint32_t     measured_values[MAX_QUHITS];
} QuhitEngine;

/* ─── Substrate opcodes (must come after QuhitEngine definition) ─── */
#include "substrate_opcodes.h"

/* ═══════════════════════════════════════════════════════════════════════════════
 * API — quhit_core.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

void     quhit_engine_init(QuhitEngine *eng);
void     quhit_engine_destroy(QuhitEngine *eng);
uint64_t quhit_prng(QuhitEngine *eng);
double   quhit_prng_double(QuhitEngine *eng);

uint32_t quhit_init(QuhitEngine *eng);           /* Init to |0⟩, returns id  */
uint32_t quhit_init_plus(QuhitEngine *eng);       /* Init to |+⟩             */
uint32_t quhit_init_basis(QuhitEngine *eng, uint32_t k); /* Init to |k⟩      */
void     quhit_reset(QuhitEngine *eng, uint32_t id);     /* Reset to |0⟩      */

/* SDR mode — use physical EM field as quhit source and Born-rule entropy.
 * Call quhit_sdr_enable() before any quhit_init_sdr() or measurement.
 * Requires quhit_sdr.h for SdrState configuration. */
int      quhit_sdr_enable(QuhitEngine *eng, void *sdr_state);
void     quhit_sdr_disable(QuhitEngine *eng);
uint32_t quhit_init_sdr(QuhitEngine *eng);        /* Init quhit from SDR EM spectrum */

/* ═══════════════════════════════════════════════════════════════════════════════
 * API — quhit_gates.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_apply_dft(QuhitEngine *eng, uint32_t id);
void quhit_apply_idft(QuhitEngine *eng, uint32_t id);
void quhit_apply_cz(QuhitEngine *eng, uint32_t id_a, uint32_t id_b);
void quhit_apply_unitary(QuhitEngine *eng, uint32_t id,
                         const double *U_re, const double *U_im);
void quhit_apply_phase(QuhitEngine *eng, uint32_t id,
                       const double *phases);
void quhit_apply_x(QuhitEngine *eng, uint32_t id); /* X = cyclic shift */
void quhit_apply_z(QuhitEngine *eng, uint32_t id); /* Z = phase gate    */

/* ═══════════════════════════════════════════════════════════════════════════════
 * API — quhit_measure.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint32_t quhit_measure(QuhitEngine *eng, uint32_t id);
void     quhit_inspect(QuhitEngine *eng, uint32_t id, QuhitSnapshot *snap);
double   quhit_prob(QuhitEngine *eng, uint32_t id, uint32_t outcome);

/* ═══════════════════════════════════════════════════════════════════════════════
 * API — quhit_entangle.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

int  quhit_entangle_bell(QuhitEngine *eng, uint32_t id_a, uint32_t id_b);
int  quhit_entangle_product(QuhitEngine *eng, uint32_t id_a, uint32_t id_b);
void quhit_disentangle(QuhitEngine *eng, uint32_t id_a, uint32_t id_b);

/* ═══════════════════════════════════════════════════════════════════════════════
 * API — quhit_register.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

int      quhit_reg_init(QuhitEngine *eng, uint64_t chunk_id,
                        uint64_t n_quhits, uint32_t dim);
void     quhit_reg_entangle_all(QuhitEngine *eng, int reg_idx);
void     quhit_reg_apply_dft(QuhitEngine *eng, int reg_idx, uint64_t quhit_idx);
void     quhit_reg_apply_cz(QuhitEngine *eng, int reg_idx,
                            uint64_t idx_a, uint64_t idx_b);
void     quhit_reg_apply_unitary_pos(QuhitEngine *eng, int reg_idx,
                                     uint64_t pos,
                                     const double *U_re, const double *U_im);
uint64_t quhit_reg_measure(QuhitEngine *eng, int reg_idx, uint64_t quhit_idx);

/* ─── Streaming State Vector (statevector.h compatible) ─── */
SV_Amplitude quhit_reg_sv_get(QuhitEngine *eng, int reg_idx,
                              basis_t basis_k);
void         quhit_reg_sv_set(QuhitEngine *eng, int reg_idx,
                              basis_t basis_k, double re, double im);

typedef void (*sv_stream_fn)(basis_t basis_state, SV_Amplitude amp,
                             void *user_data);
void         quhit_reg_sv_stream(QuhitEngine *eng, int reg_idx,
                                 sv_stream_fn callback, void *user_data);
double       quhit_reg_sv_total_prob(QuhitEngine *eng, int reg_idx);
SV_Amplitude quhit_reg_sv_inner(QuhitEngine *eng, int reg_a, int reg_b);
QuhitState   quhit_reg_local_sv(QuhitEngine *eng, int reg_idx,
                                uint64_t quhit_pos);

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-TEST
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_self_test(void);

#endif /* QUHIT_ENGINE_H */
