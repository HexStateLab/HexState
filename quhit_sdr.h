/*
 * quhit_sdr.h — RTL-SDR Physical Qubit Source
 *
 * Couples the HexState D=6 quantum simulator to the physical electromagnetic
 * field via an RTL-SDR dongle. The SDR's I/Q stream is a continuous heterodyne
 * measurement of the quantum EM vacuum — genuine field quadratures from a
 * real quantum system.
 *
 * Three modes of physical coupling:
 *
 *   1. QUHIT SOURCE    — Partition SDR bandwidth into D=6 frequency bins.
 *                         Complex amplitude in each bin = quhit amplitude.
 *                         The quhit IS a spectral snapshot of physical EM.
 *
 *   2. BORN NOISE      — Replace deterministic LCG PRNG with physical
 *                         quantum noise (LSBs of SDR I/Q samples) for
 *                         genuine Born-rule measurement.
 *
 *   3. FEEDBACK TX     — Write quhit amplitudes back into the EM field
 *                         by modulating CPU power draw at patterns the
 *                         SDR can detect. Closes the physical loop.
 *
 * Dependencies (optional):
 *   librtlsdr  — for hardware RTL-SDR dongle
 *   libfftw3   — for FFT (if available, else uses internal DFT6)
 *   Fallback:  — CPU timing jitter + /dev/urandom when no SDR detected
 */

#ifndef QUHIT_SDR_H
#define QUHIT_SDR_H

#include "quhit_engine.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * SDR CONFIGURATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define SDR_DEFAULT_FREQ      100000000   /* 100 MHz — quiet band for probing */
#define SDR_DEFAULT_RATE      2048000     /* 2.048 MSPS                       */
#define SDR_DEFAULT_GAIN      400         /* 40.0 dB (0=auto)                 */
#define SDR_FFT_SIZE          1024        /* FFT points for spectral binning  */
#define SDR_FEEDBACK_FREQ     100000000   /* Matches center freq — stays in-band */
#define SDR_MAX_SAMPLES       (1 << 20)   /* 1M I/Q sample buffer            */

/* Thermal safety — feedback TX is CPU modulation, not RF transmission.
 * The CPU's EM leakage is an unintentional radiator under FCC Part 15.
 * These limits prevent thermal runaway and power supply stress. */
#define SDR_TX_MAX_BURST_US   100         /* Max continuous burst (microseconds) */
#define SDR_TX_COOLDOWN_US    900         /* Mandatory cooldown between bursts   */
#define SDR_TX_DUTY_CYCLE     0.10        /* Max 10% duty cycle                  */
#define SDR_TX_THERMAL_LIMIT  85.0        /* °C — throttle if CPU temp exceeds   */

/* ═══════════════════════════════════════════════════════════════════════════════
 * SDR DEVICE STATE
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef enum {
    SDR_MODE_OFF       = 0,   /* No SDR — fall back to deterministic PRNG    */
    SDR_MODE_PHYSICAL  = 1,   /* Physical SDR dongle active                  */
    SDR_MODE_CPU_NOISE = 2,   /* No dongle — use CPU timing jitter as noise  */
    SDR_MODE_URANDOM   = 3,   /* Fallback to /dev/urandom                    */
    SDR_MODE_REPLAY    = 4,   /* Replay from pre-recorded I/Q file           */
} SdrMode;

typedef enum {
    SDR_COUPLING_NONE   = 0,  /* No physical coupling                        */
    SDR_COUPLING_READ   = 1,  /* Read-only: quhit source + Born noise        */
    SDR_COUPLING_WRITE  = 2,  /* Write-only: feedback transmit               */
    SDR_COUPLING_LOOP   = 3,  /* Full bidirectional loop                     */
} SdrCouplingMode;

typedef enum {
    SDR_BACKEND_NONE     = 0,  /* No SDR hardware                             */
    SDR_BACKEND_V4L2     = 1,  /* Kernel rtl2832_sdr via /dev/swradio0        */
    SDR_BACKEND_RTLSDR   = 2,  /* librtlsdr userspace driver                  */
} SdrBackend;

typedef struct {
    SdrBackend   backend;         /* Active hardware backend                   */
    int          dev_fd;          /* File descriptor (V4L2 / librtlsdr)        */
    void        *device;          /* Opaque device handle (librtlsdr only)     */
    SdrMode      mode;            /* Active SDR mode                          */
    SdrCouplingMode coupling;     /* Read/write coupling level                */
    uint32_t     center_freq;     /* Center frequency in Hz                   */
    uint32_t     sample_rate;     /* Sample rate in Hz                        */
    int          gain;            /* Tuner gain (0=auto, tenths of dB)        */
    int          device_index;    /* Which RTL-SDR device (0=first)           */
    int          ppm_error;       /* Frequency correction in ppm              */

    /* V4L2 buffer management */
    uint8_t    **v4l2_bufs;        /* Mmap'd buffer pointers                   */
    uint32_t     v4l2_buf_count;   /* Number of queued buffers                 */
    uint32_t     v4l2_buf_len[8];  /* Per-buffer length                        */
    int          v4l2_cur_buf;     /* Currently dequeued buffer index (-1=none)*/
    uint32_t     v4l2_cur_off;     /* Read offset within current buffer        */

    /* I/Q sample buffers */
    uint8_t     *iq_buffer;       /* Raw 8-bit unsigned I/Q interleaved      */
    double      *iq_complex;      /* Complex {re,im} interleaved              */
    int          iq_len;          /* Number of complex samples in buffer       */

    /* FFT state */
    double      *fft_re;          /* FFT real parts                           */
    double      *fft_im;          /* FFT imaginary parts                      */
    double      *fft_window;      /* Window function (Hann)                   */

    /* Feedback (transmit) state */
    double       tx_phase;        /* Accumulated TX phase                     */
    double       tx_amplitude;    /* Current TX amplitude scale               */
    uint64_t     tx_last_burst;   /* RDTSC at last TX burst start             */
    uint64_t     tx_total_ns;     /* Cumulative TX time                       */
    double       tx_thermal_c;    /* Estimated CPU temperature °C             */

    /* Statistics */
    uint64_t     samples_read;
    uint64_t     quhits_sourced;
    double       avg_noise_power;

    /* Record / Replay */
    FILE        *record_fp;       /* File for recording I/Q stream            */
    FILE        *replay_fp;       /* File for replaying I/Q stream            */
    uint64_t     replay_offset;   /* Byte offset in replay file               */
    uint64_t     replay_size;     /* Total size of replay file                */
} SdrState;

/* ═══════════════════════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Initialize SDR subsystem. Returns 0 on success. */
int  quhit_sdr_init(SdrState *sdr, uint32_t freq, uint32_t rate, int gain);

/* Initialize SDR in CPU-noise fallback mode (no dongle required). */
int  quhit_sdr_init_noise(SdrState *sdr);

/* Initialize SDR with urandom fallback. */
int  quhit_sdr_init_urandom(SdrState *sdr);

/* Shut down SDR subsystem. */
void quhit_sdr_close(SdrState *sdr);

/* Record: open a file for dumping raw I/Q bytes (CU8 format). */
int  quhit_sdr_record_open(SdrState *sdr, const char *path);

/* Replay: open a pre-recorded I/Q file, bypass all hardware. */
int  quhit_sdr_replay_open(SdrState *sdr, const char *path);

/* ─── Quhit Sourcing ─── */

/* Read physical EM spectrum and return a normalized D=6 quhit state.
 * The 6 amplitudes are complex values from 6 frequency bins of the
 * physical EM field, normalized to unit probability. */
QuhitState quhit_sdr_sample_state(SdrState *sdr);

/* Initialize a fresh quhit on the engine from physical EM. */
uint32_t quhit_sdr_init_quhit(SdrState *sdr, QuhitEngine *eng);

/* Read raw I/Q samples into the internal buffer. Returns sample count. */
int  quhit_sdr_read_iq(SdrState *sdr, int n_samples);

/* ─── Physical Random Numbers ─── */

/* Return a physical random double in [0,1) from SDR noise / CPU jitter.
 * This replaces the deterministic LCG for Born-rule measurement. */
double quhit_sdr_random(SdrState *sdr);

/* Fill a buffer with physical random bytes. */
void quhit_sdr_random_bytes(SdrState *sdr, uint8_t *buf, size_t len);

/* ─── Feedback Transmission ─── */

/* Write quhit amplitudes back into the physical EM environment.
 * Modulates CPU power draw patterns that the SDR can detect.
 * Closes the virtual↔physical loop. */
void quhit_sdr_feedback_tx(SdrState *sdr, const QuhitState *state);

/* Transmit a specific frequency tone for calibration. */
void quhit_sdr_tx_tone(SdrState *sdr, double freq_hz, double amplitude);

/* ─── Utilities ─── */

/* Return a human-readable description of the active SDR mode. */
const char* quhit_sdr_mode_string(const SdrState *sdr);

/* Print SDR status. */
void quhit_sdr_print_status(const SdrState *sdr);

/* Compute spectral power in each of D frequency bins from I/Q data. */
void quhit_sdr_spectral_power(SdrState *sdr, double *powers, int nbins);

/* Return 1 if a physical SDR dongle is detected on USB. */
int  quhit_sdr_detect(void);

#endif /* QUHIT_SDR_H */
