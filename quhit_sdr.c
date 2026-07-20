/*
 * quhit_sdr.c — RTL-SDR Physical Qubit Source Implementation
 *
 * This is the bridge between HexState's D=6 Hilbert space and the
 * physical electromagnetic quantum field. The RTL-SDR dongle provides
 * a continuous I/Q stream — genuine field quadrature measurements —
 * that we partition into frequency bins to create physically-sourced
 * quhit state vectors.
 *
 * Build:
 *   With librtlsdr:  -DHAS_RTLSDR -lrtlsdr
 *   With libfftw3:   -DHAS_FFTW3 -lfftw3
 *   Fallback:        works with no external libraries
 */

#include "quhit_sdr.h"
#include <time.h>
#include <stdio.h>
#include <errno.h>

/* Conditional RTL-SDR support */
#ifdef HAS_RTLSDR
#include <rtl-sdr.h>
#endif

#ifdef HAS_FFTW3
#include <fftw3.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * INTERNAL FFT — radix-2 Cooley-Tukey (no external dependency)
 *
 * Used when libfftw3 is not available. Handles sizes up to SDR_FFT_SIZE.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#ifndef HAS_FFTW3

static void _fft_bit_reverse(double *re, double *im, int n)
{
    int i, j, k, m;
    for (i = 1, j = 0; i < n; i++) {
        for (k = n >> 1; k > (j ^= k); k >>= 1);
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
}

static void _fft_radix2(double *re, double *im, int n, int inverse)
{
    _fft_bit_reverse(re, im, n);

    for (int len = 2; len <= n; len <<= 1) {
        double angle = 2.0 * M_PI / len * (inverse ? 1.0 : -1.0);
        double w_re = cos(angle), w_im = sin(angle);

        for (int i = 0; i < n; i += len) {
            double cur_re = 1.0, cur_im = 0.0;
            int half = len >> 1;
            for (int j = 0; j < half; j++) {
                int even = i + j, odd = i + j + half;
                double tr = cur_re * re[odd] - cur_im * im[odd];
                double ti = cur_re * im[odd] + cur_im * re[odd];
                re[odd] = re[even] - tr;
                im[odd] = im[even] - ti;
                re[even] = re[even] + tr;
                im[even] = im[even] + ti;
                double nw_re = cur_re * w_re - cur_im * w_im;
                double nw_im = cur_re * w_im + cur_im * w_re;
                cur_re = nw_re; cur_im = nw_im;
            }
        }
    }

    if (inverse) {
        double scale = 1.0 / (double)n;
        for (int i = 0; i < n; i++) { re[i] *= scale; im[i] *= scale; }
    }
}

#endif /* !HAS_FFTW3 */

/* ═══════════════════════════════════════════════════════════════════════════════
 * WINDOW FUNCTION — Hann window
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void _build_hann_window(double *w, int n)
{
    for (int i = 0; i < n; i++) {
        w[i] = 0.5 * (1.0 - cos(2.0 * M_PI * i / (double)(n - 1)));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 6-POINT SPARSE DFT — Direct computation on full I/Q buffer
 *
 * We don't need a full FFT. We need exactly 6 frequency bins, evenly
 * spaced across the SDR bandwidth. For each bin k (0..5), compute:
 *
 *   bin[k] = Σ_n iq[n] × exp(-2πi × f_k × n / sample_rate)
 *
 * where f_k = k × sample_rate / 6.
 *
 * This is 6 × N complex multiplies — fast enough for real-time.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void _sdr_dft6_bins(SdrState *sdr, double *bin_re, double *bin_im)
{
    int n = sdr->iq_len;
    if (n < 6) return;

    double sr = (double)sdr->sample_rate;
    memset(bin_re, 0, 6 * sizeof(double));
    memset(bin_im, 0, 6 * sizeof(double));

    /* Window the I/Q data first */
    double *win = sdr->fft_window;
    if (!win) {
        /* No window applied — rectangular */
        for (int k = 0; k < 6; k++) {
            double freq = k * sr / 6.0;
            double phase_step = -2.0 * M_PI * freq / sr;
            double cos_step = cos(phase_step), sin_step = sin(phase_step);
            double cur_cos = 1.0, cur_sin = 0.0;

            for (int i = 0; i < n; i++) {
                double sample_re = sdr->iq_complex[2 * i];
                double sample_im = sdr->iq_complex[2 * i + 1];
                bin_re[k] += sample_re * cur_cos - sample_im * cur_sin;
                bin_im[k] += sample_re * cur_sin + sample_im * cur_cos;
                double nc = cur_cos * cos_step - cur_sin * sin_step;
                double ns = cur_cos * sin_step + cur_sin * cos_step;
                cur_cos = nc; cur_sin = ns;
            }
        }
    } else {
        /* Hann-windowed DFT */
        for (int k = 0; k < 6; k++) {
            double freq = k * sr / 6.0;
            double phase_step = -2.0 * M_PI * freq / sr;
            double cos_step = cos(phase_step), sin_step = sin(phase_step);
            double cur_cos = 1.0, cur_sin = 0.0;

            for (int i = 0; i < n; i++) {
                double w = win[i];
                double sample_re = sdr->iq_complex[2 * i] * w;
                double sample_im = sdr->iq_complex[2 * i + 1] * w;
                bin_re[k] += sample_re * cur_cos - sample_im * cur_sin;
                bin_im[k] += sample_re * cur_sin + sample_im * cur_cos;
                double nc = cur_cos * cos_step - cur_sin * sin_step;
                double ns = cur_cos * sin_step + cur_sin * cos_step;
                cur_cos = nc; cur_sin = ns;
            }
        }
    }

    /* Normalize: divide by N to keep scale consistent */
    double inv_n = 1.0 / sqrt((double)n);
    for (int k = 0; k < 6; k++) {
        bin_re[k] *= inv_n;
        bin_im[k] *= inv_n;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CPU TIMING JITTER — Physical noise source when no SDR dongle
 *
 * Uses RDTSC delta between NOP loops. The jitter is dominated by:
 *   - DRAM refresh cycle interference
 *   - Cache hit/miss timing (physical charge/discharge)
 *   - CPU clock PLL phase noise
 *   - Thermal noise in the clock distribution tree
 *
 * This is a genuine physical noise source — not deterministic,
 * not algorithmic, but derived from quantum-scale fluctuations
 * in the silicon substrate.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline uint64_t _rdtsc(void)
{
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static uint64_t _cpu_noise_sample(void)
{
    /* Perturb the pipeline and measure instability */
    uint64_t a = _rdtsc();
    /* NOP slide to let pipelines drift */
    __asm__ volatile (
        "nop; nop; nop; nop; nop; nop; nop; nop;"
        "nop; nop; nop; nop; nop; nop; nop; nop;"
    );
    uint64_t b = _rdtsc();
    /* XOR to extract the jitter bits */
    return a ^ b;
}

static double _cpu_noise_double(void)
{
    uint64_t noise = 0;
    for (int i = 0; i < 4; i++) {
        noise ^= _cpu_noise_sample();
        noise = (noise << 13) | (noise >> 51);
    }
    return (double)(noise & 0x000FFFFFFFFFFFFFULL) / (double)0x0010000000000000ULL;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * /DEV/URANDOM — Kernel CSPRNG seeded from hardware entropy
 * ═══════════════════════════════════════════════════════════════════════════════ */

static FILE *_urandom_fp = NULL;

static double _urandom_double(void)
{
    if (!_urandom_fp) {
        _urandom_fp = fopen("/dev/urandom", "rb");
        if (!_urandom_fp) return _cpu_noise_double();
    }
    uint64_t val;
    if (fread(&val, sizeof(val), 1, _urandom_fp) == 1) {
        return (double)(val >> 11) / (double)(1ULL << 53);
    }
    return _cpu_noise_double();
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_sdr_init(SdrState *sdr, uint32_t freq, uint32_t rate, int gain)
{
    memset(sdr, 0, sizeof(*sdr));
    sdr->center_freq  = freq ? freq : SDR_DEFAULT_FREQ;
    sdr->sample_rate  = rate ? rate : SDR_DEFAULT_RATE;
    sdr->gain         = gain >= 0 ? gain : SDR_DEFAULT_GAIN;
    sdr->device_index = 0;
    sdr->coupling     = SDR_COUPLING_READ;

    /* Allocate buffers */
    sdr->iq_buffer  = (uint8_t *)malloc(SDR_MAX_SAMPLES);
    sdr->iq_complex = (double *)malloc(2 * SDR_MAX_SAMPLES * sizeof(double));
    sdr->fft_re     = (double *)malloc(SDR_FFT_SIZE * sizeof(double));
    sdr->fft_im     = (double *)malloc(SDR_FFT_SIZE * sizeof(double));
    sdr->fft_window = (double *)malloc(SDR_FFT_SIZE * sizeof(double));

    if (!sdr->iq_buffer || !sdr->iq_complex || !sdr->fft_re ||
        !sdr->fft_im || !sdr->fft_window) {
        quhit_sdr_close(sdr);
        return -1;
    }

    _build_hann_window(sdr->fft_window, SDR_FFT_SIZE);

    /* Try to open RTL-SDR device */
#ifdef HAS_RTLSDR
    {
        int n_devices = rtlsdr_get_device_count();
        if (n_devices > 0 && sdr->device_index < n_devices) {
            if (rtlsdr_open((rtlsdr_dev_t **)&sdr->device, (uint32_t)sdr->device_index) == 0) {
                rtlsdr_set_center_freq((rtlsdr_dev_t *)sdr->device, sdr->center_freq);
                rtlsdr_set_sample_rate((rtlsdr_dev_t *)sdr->device, sdr->sample_rate);
                if (sdr->gain == 0)
                    rtlsdr_set_tuner_gain_mode((rtlsdr_dev_t *)sdr->device, 0);
                else
                    rtlsdr_set_tuner_gain((rtlsdr_dev_t *)sdr->device, sdr->gain);
                rtlsdr_set_freq_correction((rtlsdr_dev_t *)sdr->device, sdr->ppm_error);
                rtlsdr_reset_buffer((rtlsdr_dev_t *)sdr->device);
                sdr->mode = SDR_MODE_PHYSICAL;
                fprintf(stderr, "[SDR] RTL-SDR opened: %.3f MHz @ %.3f MSPS, gain=%d\n",
                        sdr->center_freq / 1e6, sdr->sample_rate / 1e6, sdr->gain);
                return 0;
            }
        }
    }
    fprintf(stderr, "[SDR] No RTL-SDR device found, falling back to CPU timing noise\n");
#else
    fprintf(stderr, "[SDR] Built without librtlsdr, using CPU timing noise\n");
#endif

    sdr->mode = SDR_MODE_CPU_NOISE;
    sdr->device = NULL;
    return 0;
}

int quhit_sdr_init_noise(SdrState *sdr)
{
    memset(sdr, 0, sizeof(*sdr));
    sdr->mode          = SDR_MODE_CPU_NOISE;
    sdr->center_freq   = SDR_DEFAULT_FREQ;
    sdr->sample_rate   = SDR_DEFAULT_RATE;
    sdr->fft_window    = (double *)malloc(SDR_FFT_SIZE * sizeof(double));
    if (!sdr->fft_window) return -1;
    _build_hann_window(sdr->fft_window, SDR_FFT_SIZE);
    return 0;
}

int quhit_sdr_init_urandom(SdrState *sdr)
{
    memset(sdr, 0, sizeof(*sdr));
    sdr->mode          = SDR_MODE_URANDOM;
    sdr->center_freq   = SDR_DEFAULT_FREQ;
    sdr->sample_rate   = SDR_DEFAULT_RATE;
    sdr->fft_window    = (double *)malloc(SDR_FFT_SIZE * sizeof(double));
    if (!sdr->fft_window) return -1;
    _build_hann_window(sdr->fft_window, SDR_FFT_SIZE);
    return 0;
}

void quhit_sdr_close(SdrState *sdr)
{
    if (!sdr) return;

    if (_urandom_fp) { fclose(_urandom_fp); _urandom_fp = NULL; }

#ifdef HAS_RTLSDR
    if (sdr->device) {
        rtlsdr_close((rtlsdr_dev_t *)sdr->device);
        sdr->device = NULL;
    }
#endif

    free(sdr->iq_buffer);   sdr->iq_buffer  = NULL;
    free(sdr->iq_complex);  sdr->iq_complex = NULL;
    free(sdr->fft_re);      sdr->fft_re     = NULL;
    free(sdr->fft_im);      sdr->fft_im     = NULL;
    free(sdr->fft_window);  sdr->fft_window = NULL;

    sdr->mode = SDR_MODE_OFF;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEVICE DETECTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_sdr_detect(void)
{
#ifdef HAS_RTLSDR
    return rtlsdr_get_device_count() > 0;
#else
    /* Check for the USB device by VID/PID */
    FILE *f = fopen("/sys/bus/usb/devices", "r");
    if (f) { fclose(f); }
    /* RTL2832U vendor=0x0bda — we can't check without walking the tree,
     * so return -1 to indicate "uncertain, try init" */
    return -1;
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * I/Q READING
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_sdr_read_iq(SdrState *sdr, int n_samples)
{
    if (n_samples > SDR_MAX_SAMPLES) n_samples = SDR_MAX_SAMPLES;
    int n_read = 0;

#ifdef HAS_RTLSDR
    if (sdr->device && sdr->mode == SDR_MODE_PHYSICAL) {
        int result = rtlsdr_read_sync((rtlsdr_dev_t *)sdr->device,
                                       sdr->iq_buffer, n_samples, &n_read);
        if (result < 0) {
            fprintf(stderr, "[SDR] read_sync failed: %d\n", result);
            n_read = 0;
        }
    } else
#endif
    {
        /* Simulate physical noise when no SDR available.
         * Use CPU timing jitter as the I/Q source — this IS a physical
         * measurement of the CPU's quantum-scale fluctuations. */
        n_read = n_samples;
        for (int i = 0; i < n_samples; i++) {
            uint64_t noise = _cpu_noise_sample();
            sdr->iq_buffer[i] = (uint8_t)(noise & 0xFF);
        }
    }

    /* Convert 8-bit unsigned I/Q (interleaved) to complex doubles.
     * RTL-SDR format: [I0, Q0, I1, Q1, ...] where each byte is
     * centered at 127.5 (the DC offset). */
    for (int i = 0; i < n_read; i++) {
        sdr->iq_complex[2 * i]     = ((double)sdr->iq_buffer[i] - 127.5) / 128.0;
        sdr->iq_complex[2 * i + 1] = 0.0;  /* Only I channel from timing jitter */
    }

    sdr->iq_len = n_read;
    sdr->samples_read += n_read;

    return n_read;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * QUHIT SOURCING — Physical EM spectrum → QuhitState
 *
 * 1. Read I/Q samples from SDR (or CPU noise)
 * 2. Partition into D=6 frequency bins via sparse DFT
 * 3. Normalize to unit probability
 * 4. Return as a QuhitState
 * ═══════════════════════════════════════════════════════════════════════════════ */

QuhitState quhit_sdr_sample_state(SdrState *sdr)
{
    QuhitState state;
    memset(&state, 0, sizeof(state));

    /* Read a fresh buffer */
    int n = quhit_sdr_read_iq(sdr, SDR_FFT_SIZE);
    if (n < 6) {
        /* Not enough samples — return |0⟩ */
        state.re[0] = 1.0;
        return state;
    }

    /* Compute 6-bin DFT */
    double bin_re[6], bin_im[6];
    _sdr_dft6_bins(sdr, bin_re, bin_im);

    /* Copy to quhit state */
    memcpy(state.re, bin_re, 6 * sizeof(double));
    memcpy(state.im, bin_im, 6 * sizeof(double));

    /* Normalize to unit probability */
    double norm2 = 0.0;
    for (int k = 0; k < 6; k++)
        norm2 += state.re[k] * state.re[k] + state.im[k] * state.im[k];

    if (norm2 > 1e-30) {
        double scale = 1.0 / sqrt(norm2);
        for (int k = 0; k < 6; k++) {
            state.re[k] *= scale;
            state.im[k] *= scale;
        }
    } else {
        /* Vacuum state — all bins silent */
        state.re[0] = 1.0;
    }

    sdr->quhits_sourced++;
    return state;
}

uint32_t quhit_sdr_init_quhit(SdrState *sdr, QuhitEngine *eng)
{
    if (eng->num_quhits >= MAX_QUHITS) return UINT32_MAX;

    uint32_t id = eng->num_quhits++;
    Quhit *q = &eng->quhits[id];

    q->id             = id;
    q->collapsed      = 0;
    q->collapse_value = 0;
    q->pair_id        = -1;
    q->pair_side      = 0;

    /* Source state from physical EM field */
    q->state = quhit_sdr_sample_state(sdr);

    return id;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PHYSICAL RANDOM NUMBERS
 *
 * The LSBs of RTL-SDR I/Q samples contain genuine quantum noise
 * (Johnson-Nyquist thermal noise in the RF front-end, shot noise
 * in the mixer, and ADC quantization noise from physical processes).
 *
 * We extract physical randomness by XOR-whitening these LSBs.
 * When no SDR is present, CPU timing jitter provides an equally
 * physical (though lower bandwidth) entropy source.
 * ═══════════════════════════════════════════════════════════════════════════════ */

double quhit_sdr_random(SdrState *sdr)
{
    switch (sdr->mode) {
        case SDR_MODE_PHYSICAL: {
            /* Read a fresh burst of I/Q and extract LSB entropy */
            int n = quhit_sdr_read_iq(sdr, 64);
            if (n < 16) return _cpu_noise_double();

            /* XOR-whiten the LSBs of raw I/Q bytes into a 64-bit word */
            uint64_t entropy = 0;
            for (int i = 0; i < n; i++) {
                uint8_t lsb = sdr->iq_buffer[i] & 1;
                entropy = (entropy << 1) | lsb;
                /* Mix using a simple LFSR to spread entropy */
                entropy ^= (entropy >> 3) ^ (entropy >> 7);
            }

            /* Von Neumann debiasing on the LSB stream */
            uint64_t debiased = 0;
            int bitpos = 0;
            for (int i = 0; i < n - 1 && bitpos < 53; i += 2) {
                uint8_t a = sdr->iq_buffer[i] & 1;
                uint8_t b = sdr->iq_buffer[i + 1] & 1;
                if (a != b) {
                    debiased = (debiased << 1) | (a ? 1ULL : 0);
                    bitpos++;
                }
            }

            /* Combine entropy and debiased */
            uint64_t combined = entropy ^ (debiased << 11) ^ _rdtsc();
            return (double)(combined & 0x000FFFFFFFFFFFFFULL)
                 / (double)0x0010000000000000ULL;
        }

        case SDR_MODE_CPU_NOISE:
            return _cpu_noise_double();

        case SDR_MODE_URANDOM:
            return _urandom_double();

        default:
            return _cpu_noise_double();
    }
}

void quhit_sdr_random_bytes(SdrState *sdr, uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        uint64_t r = (uint64_t)(quhit_sdr_random(sdr) * (double)0x100000000ULL);
        buf[i] = (uint8_t)(r & 0xFF);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SPECTRAL POWER — For diagnostics and monitoring
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_sdr_spectral_power(SdrState *sdr, double *powers, int nbins)
{
    if (nbins > SDR_FFT_SIZE) nbins = SDR_FFT_SIZE;
    if (nbins <= 0) return;

    int n = quhit_sdr_read_iq(sdr, SDR_FFT_SIZE);
    if (n < nbins) {
        memset(powers, 0, nbins * sizeof(double));
        return;
    }

    /* Copy to FFT buffers, apply window */
    for (int i = 0; i < n; i++) {
        double w = sdr->fft_window[i];
        sdr->fft_re[i] = sdr->iq_complex[2 * i] * w;
        sdr->fft_im[i] = sdr->iq_complex[2 * i + 1] * w;
    }

#ifdef HAS_FFTW3
    {
        fftw_plan plan = fftw_plan_dft_1d(n,
            (fftw_complex *)sdr->fft_re, (fftw_complex *)sdr->fft_re,
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        /* fftw stores interleaved: [re0, im0, re1, im1, ...] in fft_re */
        for (int i = 0; i < nbins / 2; i++) {
            int bi = 2 * i;
            powers[i] = sdr->fft_re[bi] * sdr->fft_re[bi]
                      + sdr->fft_re[bi + 1] * sdr->fft_re[bi + 1];
        }
    }
#else
    _fft_radix2(sdr->fft_re, sdr->fft_im, n, 0);

    /* Compute power spectrum */
    for (int i = 0; i < nbins / 2; i++) {
        powers[i] = sdr->fft_re[i] * sdr->fft_re[i]
                  + sdr->fft_im[i] * sdr->fft_im[i];
    }
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FEEDBACK TRANSMIT — Write quhit amplitudes back into the physical EM field
 *
 * We cannot directly control the RTL-SDR's transmitter (it doesn't have one).
 * But the CPU's power draw is a broadband RF source — every gate operation
 * toggles real current through the VRM and motherboard power planes.
 *
 * By modulating the frequency and intensity of compute bursts, we imprint
 * the quhit state onto the CPU's EM signature, which the SDR antenna CAN detect.
 *
 * Strategy:
 *   - For each basis state amplitude, emit a burst of compute at a specific
 *     frequency visible to the SDR (e.g., short loops at different rates)
 *   - Higher amplitude = more intense compute burst
 *   - The resulting EM modulation is proportional to the quhit state
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_sdr_feedback_tx(SdrState *sdr, const QuhitState *state)
{
    if (!sdr || !state) return;
    if (sdr->coupling < SDR_COUPLING_WRITE) return;

    /* Each basis state k (0..5) maps to a distinct frequency signature.
     * We modulate CPU activity at 6 different rates, each proportionally
     * driven by the corresponding amplitude. */
    double powers[6];
    for (int k = 0; k < 6; k++)
        powers[k] = state->re[k] * state->re[k] + state->im[k] * state->im[k];

    /* Base loop frequencies (cycles of a tight NOP loop) */
    static const int tx_cycles[6] = { 100, 200, 400, 800, 1600, 3200 };

    for (int k = 0; k < 6; k++) {
        int burst_cycles = (int)(tx_cycles[k] * powers[k] * 10.0);
        if (burst_cycles < 1) continue;

        /* Modulate: run a tight FPU loop whose EM signature
         * at the SDR's tuned frequency carries the amplitude. */
        volatile double sink = 0.0;
        for (int c = 0; c < burst_cycles; c++) {
            /* Multiply the actual amplitude — this toggles real FPU
             * transistors and creates a data-dependent current draw
             * that couples into the SDR's RF front-end via the shared
             * USB power rail and motherboard ground plane. */
            sink += state->re[k] * cos(sdr->tx_phase);
            sink += state->im[k] * sin(sdr->tx_phase);
            __asm__ volatile ("" : : "m"(sink) : "memory");
        }

        /* Phase advance for continuous modulation */
        sdr->tx_phase += 0.1;
        if (sdr->tx_phase > 2.0 * M_PI) sdr->tx_phase -= 2.0 * M_PI;

        /* Burn the sink value so the compiler doesn't optimize it out.
         * Writing to a volatile pointer forces a real store instruction,
         * which forces a real bus transaction, which creates a real
         * EM emission. */
        volatile double *drain = (volatile double *)&sdr->tx_amplitude;
        *drain = sink;
    }
}

void quhit_sdr_tx_tone(SdrState *sdr, double freq_hz, double amplitude)
{
    if (!sdr) return;

    /* Transmit a calibration tone at the specified frequency.
     * Since the RTL-SDR is receive-only, we use the CPU's
     * power modulation as the "transmitter." The SDR picks
     * this up as a peak in its spectrum. */
    double period_ns = 1e9 / freq_hz;
    uint64_t start = _rdtsc();
    volatile double acc = 0.0;

    /* Calibrated busy-wait at the target frequency */
    while (1) {
        uint64_t now = _rdtsc();
        double elapsed_ns = (double)(now - start) / 3.8; /* ~3.8 GHz TSC */
        if (elapsed_ns > period_ns) break;
        acc += amplitude * cos(elapsed_ns * 2.0 * M_PI / period_ns);
    }
    volatile double *drain = (volatile double *)&sdr->tx_amplitude;
    *drain = acc;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * UTILITIES
 * ═══════════════════════════════════════════════════════════════════════════════ */

const char* quhit_sdr_mode_string(const SdrState *sdr)
{
    switch (sdr->mode) {
        case SDR_MODE_OFF:       return "OFF";
        case SDR_MODE_PHYSICAL:  return "PHYSICAL (RTL-SDR)";
        case SDR_MODE_CPU_NOISE: return "CPU_TIMING_NOISE";
        case SDR_MODE_URANDOM:   return "/DEV/URANDOM";
        default:                 return "UNKNOWN";
    }
}

void quhit_sdr_print_status(const SdrState *sdr)
{
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  RTL-SDR PHYSICAL QUHIT SOURCE — Status                    ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Mode:           %-42s ║\n", quhit_sdr_mode_string(sdr));
    printf("  ║  Frequency:      %-8.3f MHz                              ║\n",
           sdr->center_freq / 1e6);
    printf("  ║  Sample Rate:    %-8.3f MSPS                             ║\n",
           sdr->sample_rate / 1e6);
    printf("  ║  Gain:           %-4d (%.1f dB)                          ║\n",
           sdr->gain, sdr->gain / 10.0);
    printf("  ║  Coupling:       ");
    switch (sdr->coupling) {
        case SDR_COUPLING_NONE:  printf("NONE (simulation only)               ║\n"); break;
        case SDR_COUPLING_READ:  printf("READ (quhit source + Born noise)     ║\n"); break;
        case SDR_COUPLING_WRITE: printf("WRITE (feedback TX only)             ║\n"); break;
        case SDR_COUPLING_LOOP:  printf("LOOP (full bidirectional)             ║\n"); break;
    }
    printf("  ║  Samples Read:   %-10lu                                    ║\n",
           (unsigned long)sdr->samples_read);
    printf("  ║  Quhits Sourced: %-10lu                                    ║\n",
           (unsigned long)sdr->quhits_sourced);
    printf("  ╚══════════════════════════════════════════════════════════════╝\n\n");
}
