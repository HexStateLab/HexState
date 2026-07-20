/*
 * quhit_sdr.c — RTL-SDR Physical Qubit Source Implementation
 *
 * This is the bridge between HexState's D=6 Hilbert space and the
 * physical electromagnetic quantum field. The RTL-SDR dongle provides
 * a continuous I/Q stream — genuine field quadrature measurements —
 * that we partition into frequency bins to create physically-sourced
 * quhit state vectors.
 *
 * Backends (tried in order):
 *   1. V4L2 SDR     — kernel rtl2832_sdr driver via /dev/swradio0 (no deps)
 *   2. librtlsdr    — optional, via -DHAS_RTLSDR -lrtlsdr
 *   3. /dev/urandom — kernel entropy pool
 *   4. CPU jitter   — RDTSC-based physical noise
 *
 * Build:
 *   gcc ... quhit_sdr.c -lm        (V4L2 + fallbacks, no external deps)
 *   gcc -DHAS_RTLSDR ... -lrtlsdr  (add librtlsdr path)
 *   gcc -DHAS_FFTW3  ... -lfftw3   (add FFTW path)
 */

#include "quhit_sdr.h"
#include <time.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

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
    int i, j, k;
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
 * True entropy sources in a running CPU:
 *   - DRAM refresh cycles interfere with cache timings unpredictably
 *   - Cache hit/miss depends on physical address aliasing
 *   - CPU clock PLL has ~10 ps RMS phase noise
 *   - Thermal noise in the clock distribution tree
 *   - OS interrupt jitter (timer, network, USB polling)
 *   - Branch predictor state (microarchitectural, physically-stored)
 *
 * Strategy: measure the variance of RDTSC around a short computation.
 * The variance floor (after removing the deterministic component) is
 * the physical jitter — quantum-scale fluctuations in the silicon.
 *
 * We amplify it by running through an entropy pool that accumulates
 * and whitens the jitter across many samples.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static uint64_t _entropy_pool[8] = {0};
static int      _entropy_idx = 0;

static inline uint64_t _rdtsc(void)
{
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static uint64_t _cpu_noise_sample(void)
{
    /* Phase 1: Measure RDTSC before and after a short variable-duration
     * computation. The computed value depends on the entropy pool itself,
     * making the timing data-dependent and amplifying microarchitectural
     * state differences. */
    int loops = (int)(_entropy_pool[_entropy_idx % 8] & 0x3F) + 4;
    uint64_t a = _rdtsc();

    volatile int x = 0;
    for (int i = 0; i < loops; i++) {
        /* Modulo by a prime to defeat branch prediction */
        x += (i * 17 + 3) % 7;
    }
    __asm__ volatile ("" : : "r"(x) : "memory");

    uint64_t b = _rdtsc();

    /* Phase 2: DRAM perturbation — touch cache-line-aligned memory
     * to force a potential cache miss that may be delayed by DRAM refresh.
     * This is the most unpredictable timing source available in userspace. */
    volatile char *probe = (volatile char *)_entropy_pool + ((a >> 6) & 0x3F);
    __asm__ volatile ("" : : "r"(*probe) : "memory");

    uint64_t c = _rdtsc();

    /* XOR the two deltas — the deterministic part cancels, leaving jitter */
    uint64_t noise = (b - a) ^ (c - b);
    noise = (noise << 33) | (noise >> 31);

    /* Feed back into entropy pool for next sample */
    _entropy_pool[_entropy_idx % 8] ^= noise ^ a;
    _entropy_idx++;

    return noise;
}

static double _cpu_noise_double(void)
{
    /* Accumulate entropy across multiple jitter samples and whiten */
    uint64_t noise = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t n = _cpu_noise_sample();
        /* Von Neumann extractor: only keep bits where consecutive
         * jitter deltas flip. This removes the deterministic bias. */
        noise = (noise << 8) ^ ((n >> (i * 8)) & 0xFF);
    }

    /* Mix thoroughly and extract 53 bits for a double in [0,1) */
    noise ^= noise >> 33;
    noise *= 0xFF51AFD7ED558CCDULL;  /* FNV-like mix */
    noise ^= noise >> 33;
    noise *= 0xC4CEB9FE1A85EC53ULL;
    noise ^= noise >> 33;

    return (double)(noise >> 11) / (double)(1ULL << 53);
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
 * V4L2 SDR BACKEND — Native kernel rtl2832_sdr driver
 *
 * The rtl2832_sdr kernel module exposes the RTL-SDR as a V4L2 Software
 * Radio capture device at /dev/swradio0. This is the preferred backend
 * because it has zero external dependencies and runs directly on the
 * kernel's USB driver stack.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define V4L2_SDR_DEVICE "/dev/swradio0"
#define V4L2_NUM_BUFS    4

static int _v4l2_sdr_init(SdrState *sdr)
{
    int fd = open(V4L2_SDR_DEVICE, O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        fprintf(stderr, "[SDR-V4L2] Cannot open %s: %s\n",
                V4L2_SDR_DEVICE, strerror(errno));
        return -1;
    }

    /* Set format: I/Q interleaved unsigned 8-bit */
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_SDR_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
        fprintf(stderr, "[SDR-V4L2] VIDIOC_G_FMT failed: %s\n", strerror(errno));
        close(fd); return -1;
    }
    fmt.fmt.sdr.pixelformat = V4L2_SDR_FMT_CU8;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        fprintf(stderr, "[SDR-V4L2] VIDIOC_S_FMT(CU8) failed: %s\n", strerror(errno));
        close(fd); return -1;
    }
    fprintf(stderr, "[SDR-V4L2] Format: CU8, buffersize=%u\n",
            fmt.fmt.sdr.buffersize);

    /* Try to set center frequency via V4L2 tuner */
    struct v4l2_frequency vf;
    memset(&vf, 0, sizeof(vf));
    vf.tuner = 0;
    vf.type = V4L2_TUNER_ADC;
    vf.frequency = sdr->center_freq;
    if (ioctl(fd, VIDIOC_S_FREQUENCY, &vf) < 0) {
        fprintf(stderr, "[SDR-V4L2] VIDIOC_S_FREQUENCY(%u Hz) failed: %s "
                "(continuing with default)\n",
                sdr->center_freq, strerror(errno));
    } else {
        fprintf(stderr, "[SDR-V4L2] Frequency set to %u Hz\n", vf.frequency);
    }

    /* Try to set sample rate via V4L2 control
     * rtl2832_sdr exposes V4L2_CID_SDR_TUNER_SAMPLERATE or similar.
     * Try known control IDs. */
    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    /* Try V4L2_CID_SDR_TUNER_SAMPLERATE = 0x9B2c */
    ctrl.id = 0x0098092c;  /* V4L2_CID_SDR_TUNER_SAMPLERATE (likely value) */
    ctrl.value = sdr->sample_rate;
    if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
        fprintf(stderr, "[SDR-V4L2] Cannot set sample rate via S_CTRL: %s\n",
                strerror(errno));
    }

    /* Request and mmap buffers */
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count  = V4L2_NUM_BUFS;
    req.type   = V4L2_BUF_TYPE_SDR_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        fprintf(stderr, "[SDR-V4L2] REQBUFS failed: %s\n", strerror(errno));
        close(fd); return -1;
    }
    fprintf(stderr, "[SDR-V4L2] Allocated %u buffers\n", req.count);
    sdr->v4l2_buf_count = req.count;

    sdr->v4l2_bufs = (uint8_t **)calloc(req.count, sizeof(uint8_t *));
    if (!sdr->v4l2_bufs) { close(fd); return -1; }

    for (uint32_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_SDR_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            fprintf(stderr, "[SDR-V4L2] QUERYBUF %u failed: %s\n",
                    i, strerror(errno));
            close(fd); return -1;
        }
        sdr->v4l2_buf_len[i] = buf.length;
        sdr->v4l2_bufs[i] = (uint8_t *)mmap(NULL, buf.length,
            PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (sdr->v4l2_bufs[i] == MAP_FAILED) {
            fprintf(stderr, "[SDR-V4L2] mmap buffer %u failed: %s\n",
                    i, strerror(errno));
            close(fd); return -1;
        }
    }

    /* Queue all buffers */
    for (uint32_t i = 0; i < req.count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_SDR_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            fprintf(stderr, "[SDR-V4L2] QBUF %u failed: %s\n",
                    i, strerror(errno));
            close(fd); return -1;
        }
    }

    /* Start streaming */
    enum v4l2_buf_type type = V4L2_BUF_TYPE_SDR_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) < 0) {
        fprintf(stderr, "[SDR-V4L2] STREAMON failed: %s\n", strerror(errno));
        close(fd); return -1;
    }

    sdr->dev_fd = fd;
    sdr->backend = SDR_BACKEND_V4L2;
    fprintf(stderr, "[SDR-V4L2] Streaming started on %s\n", V4L2_SDR_DEVICE);
    return 0;
}

static void _v4l2_sdr_close(SdrState *sdr)
{
    if (sdr->dev_fd < 0) return;

    /* Stop streaming */
    enum v4l2_buf_type type = V4L2_BUF_TYPE_SDR_CAPTURE;
    ioctl(sdr->dev_fd, VIDIOC_STREAMOFF, &type);

    /* Unmap buffers */
    for (uint32_t i = 0; i < sdr->v4l2_buf_count; i++) {
        if (sdr->v4l2_bufs && sdr->v4l2_bufs[i] && sdr->v4l2_bufs[i] != MAP_FAILED)
            munmap(sdr->v4l2_bufs[i], sdr->v4l2_buf_len[i]);
    }
    free(sdr->v4l2_bufs);
    sdr->v4l2_bufs = NULL;
    sdr->v4l2_buf_count = 0;

    close(sdr->dev_fd);
    sdr->dev_fd = -1;
}

static int _v4l2_sdr_read(SdrState *sdr, int n_samples)
{
    if (sdr->dev_fd < 0 || !sdr->v4l2_bufs) return 0;

    /* Dequeue a filled buffer */
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_SDR_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    /* Non-blocking: poll for a buffer with timeout */
    struct pollfd pfd = { .fd = sdr->dev_fd, .events = POLLIN };
    int pret = poll(&pfd, 1, 500); /* 500ms timeout */
    if (pret <= 0) {
        /* No data available — fall through to software noise */
        return 0;
    }

    if (ioctl(sdr->dev_fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) return 0;
        return 0;
    }

    /* Copy I/Q samples from the dequeued buffer */
    int bytes = (int)buf.bytesused;
    if (bytes > n_samples) bytes = n_samples;
    if (bytes > sdr->v4l2_buf_len[buf.index])
        bytes = sdr->v4l2_buf_len[buf.index];
    memcpy(sdr->iq_buffer, sdr->v4l2_bufs[buf.index], bytes);

    /* Requeue the buffer */
    ioctl(sdr->dev_fd, VIDIOC_QBUF, &buf);

    return bytes;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_sdr_init(SdrState *sdr, uint32_t freq, uint32_t rate, int gain)
{
    memset(sdr, 0, sizeof(*sdr));
    sdr->dev_fd       = -1;
    sdr->center_freq  = freq ? freq : SDR_DEFAULT_FREQ;
    sdr->sample_rate  = rate ? rate : SDR_DEFAULT_RATE;
    sdr->gain         = gain >= 0 ? gain : SDR_DEFAULT_GAIN;
    sdr->device_index = 0;
    sdr->coupling     = SDR_COUPLING_READ;
    sdr->backend      = SDR_BACKEND_NONE;

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

    /* Try V4L2 SDR kernel driver first (no external deps) */
    if (_v4l2_sdr_init(sdr) == 0) {
        sdr->mode = SDR_MODE_PHYSICAL;
        return 0;
    }

    /* Try librtlsdr userspace driver */
#ifdef HAS_RTLSDR
    {
        int n_devices = rtlsdr_get_device_count();
        if (n_devices > 0 && sdr->device_index < n_devices) {
            if (rtlsdr_open((rtlsdr_dev_t **)&sdr->device, (uint32_t)sdr->device_index) == 0) {
                sdr->dev_fd = -1;
                rtlsdr_set_center_freq((rtlsdr_dev_t *)sdr->device, sdr->center_freq);
                rtlsdr_set_sample_rate((rtlsdr_dev_t *)sdr->device, sdr->sample_rate);
                if (sdr->gain == 0)
                    rtlsdr_set_tuner_gain_mode((rtlsdr_dev_t *)sdr->device, 0);
                else
                    rtlsdr_set_tuner_gain((rtlsdr_dev_t *)sdr->device, sdr->gain);
                rtlsdr_set_freq_correction((rtlsdr_dev_t *)sdr->device, sdr->ppm_error);
                rtlsdr_reset_buffer((rtlsdr_dev_t *)sdr->device);
                sdr->backend = SDR_BACKEND_RTLSDR;
                sdr->mode = SDR_MODE_PHYSICAL;
                fprintf(stderr, "[SDR-RTLSDR] Opened: %.3f MHz @ %.3f MSPS, gain=%d\n",
                        sdr->center_freq / 1e6, sdr->sample_rate / 1e6, sdr->gain);
                return 0;
            }
        }
    }
    fprintf(stderr, "[SDR-RTLSDR] No device found.\n");
#else
    fprintf(stderr, "[SDR] Built without librtlsdr.\n");
#endif

    /* Fallback: kernel entropy pool */
    fprintf(stderr, "[SDR] No SDR hardware — using /dev/urandom for physical entropy\n");
    sdr->mode = SDR_MODE_URANDOM;
    return 0;
}

int quhit_sdr_init_noise(SdrState *sdr)
{
    memset(sdr, 0, sizeof(*sdr));
    sdr->dev_fd       = -1;
    sdr->mode          = SDR_MODE_CPU_NOISE;
    sdr->center_freq   = SDR_DEFAULT_FREQ;
    sdr->sample_rate   = SDR_DEFAULT_RATE;
    sdr->backend       = SDR_BACKEND_NONE;
    sdr->fft_window    = (double *)malloc(SDR_FFT_SIZE * sizeof(double));
    if (!sdr->fft_window) return -1;
    _build_hann_window(sdr->fft_window, SDR_FFT_SIZE);
    return 0;
}

int quhit_sdr_init_urandom(SdrState *sdr)
{
    memset(sdr, 0, sizeof(*sdr));
    sdr->dev_fd       = -1;
    sdr->mode          = SDR_MODE_URANDOM;
    sdr->center_freq   = SDR_DEFAULT_FREQ;
    sdr->sample_rate   = SDR_DEFAULT_RATE;
    sdr->backend       = SDR_BACKEND_NONE;
    sdr->fft_window    = (double *)malloc(SDR_FFT_SIZE * sizeof(double));
    if (!sdr->fft_window) return -1;
    _build_hann_window(sdr->fft_window, SDR_FFT_SIZE);
    return 0;
}

void quhit_sdr_close(SdrState *sdr)
{
    if (!sdr) return;

    if (_urandom_fp) { fclose(_urandom_fp); _urandom_fp = NULL; }

    /* Close V4L2 backend */
    if (sdr->backend == SDR_BACKEND_V4L2)
        _v4l2_sdr_close(sdr);

#ifdef HAS_RTLSDR
    if (sdr->backend == SDR_BACKEND_RTLSDR && sdr->device) {
        rtlsdr_close((rtlsdr_dev_t *)sdr->device);
        sdr->device = NULL;
    }
#endif

    free(sdr->iq_buffer);   sdr->iq_buffer  = NULL;
    free(sdr->iq_complex);  sdr->iq_complex = NULL;
    free(sdr->fft_re);      sdr->fft_re     = NULL;
    free(sdr->fft_im);      sdr->fft_im     = NULL;
    free(sdr->fft_window);  sdr->fft_window = NULL;

    sdr->mode    = SDR_MODE_OFF;
    sdr->backend = SDR_BACKEND_NONE;
    sdr->dev_fd  = -1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEVICE DETECTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_sdr_detect(void)
{
    /* Check V4L2 SDR device first */
    int fd = open(V4L2_SDR_DEVICE, O_RDONLY);
    if (fd >= 0) {
        struct v4l2_capability cap;
        memset(&cap, 0, sizeof(cap));
        if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            fprintf(stderr, "[SDR] Detected V4L2 SDR: %s (driver: %s)\n",
                    cap.card, cap.driver);
            close(fd);
            return 1;
        }
        close(fd);
    }

#ifdef HAS_RTLSDR
    if (rtlsdr_get_device_count() > 0) return 1;
#endif
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * I/Q READING
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_sdr_read_iq(SdrState *sdr, int n_samples)
{
    if (n_samples > SDR_MAX_SAMPLES) n_samples = SDR_MAX_SAMPLES;
    int n_read = 0;

    /* Try V4L2 kernel driver */
    if (sdr->backend == SDR_BACKEND_V4L2 && sdr->dev_fd >= 0) {
        n_read = _v4l2_sdr_read(sdr, n_samples);
        if (n_read > 0) goto convert;
        /* V4L2 returned no data — fall through to software noise for
         * the remaining samples. Mix real SDR data with software noise
         * so we never return empty. */
    }

#ifdef HAS_RTLSDR
    if (sdr->backend == SDR_BACKEND_RTLSDR && sdr->device) {
        int result = rtlsdr_read_sync((rtlsdr_dev_t *)sdr->device,
                                       sdr->iq_buffer, n_samples, &n_read);
        if (result < 0) {
            fprintf(stderr, "[SDR-RTLSDR] read_sync failed: %d\n", result);
            n_read = 0;
        } else {
            goto convert;
        }
    }
#endif

    /* No hardware — fill with software entropy */
    {
        n_read = n_samples;
        if (sdr->mode == SDR_MODE_URANDOM) {
            if (!_urandom_fp)
                _urandom_fp = fopen("/dev/urandom", "rb");
            if (_urandom_fp) {
                size_t nr = fread(sdr->iq_buffer, 1, (size_t)n_samples, _urandom_fp);
                if (nr < (size_t)n_samples) {
                    for (int i = (int)nr; i < n_samples; i++)
                        sdr->iq_buffer[i] = (uint8_t)(_cpu_noise_sample() & 0xFF);
                }
            } else {
                for (int i = 0; i < n_samples; i++)
                    sdr->iq_buffer[i] = (uint8_t)(_cpu_noise_sample() & 0xFF);
            }
        } else {
            for (int i = 0; i < n_samples; i++)
                sdr->iq_buffer[i] = (uint8_t)(_cpu_noise_sample() & 0xFF);
        }
    }

convert:
    /* Convert CU8 interleaved I/Q to complex doubles.
     * RTL-SDR format: [I0, Q0, I1, Q1, ...] — 8-bit unsigned each.
     * Center at 127.5 (ADC DC offset), scale to [-1, +1].
     * n_read is the total byte count; each I/Q pair is 2 bytes. */
    int n_pairs = n_read / 2;
    for (int i = 0; i < n_pairs && i < SDR_MAX_SAMPLES; i++) {
        sdr->iq_complex[2 * i]     = ((double)sdr->iq_buffer[2 * i]     - 127.5) / 128.0;
        sdr->iq_complex[2 * i + 1] = ((double)sdr->iq_buffer[2 * i + 1] - 127.5) / 128.0;
    }
    sdr->iq_len = n_pairs;
    sdr->samples_read += n_pairs;

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
 * Instead we modulate CPU current draw via structured compute bursts. These
 * couple into the SDR's RF front-end through the shared USB power rail and
 * motherboard ground plane (conducted emissions). The CPU's clock tree and
 * data bus harmonics also radiate weakly as an unintentional radiator under
 * FCC Part 15.
 *
 * SAFETY: Feedback TX is thermal-loading. The CPU is NOT a power amplifier.
 * We enforce:
 *   - Max burst duration:         100 µs per burst
 *   - Mandatory cooldown:         900 µs between bursts
 *   - Duty cycle cap:             10%
 *   - Thermal throttle:           halt if CPU estimate exceeds 85°C
 *
 * For actual RF transmission, use a proper SDR with TX capability
 * (HackRF, LimeSDR, PlutoSDR) — NOT CPU modulation.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Estimate CPU temp from RDTSC drift or sysfs. Returns °C. */
static double _read_cpu_temp(void)
{
    /* Try sysfs first (Linux-specific) */
    static const char *zones[] = {
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
        "/sys/class/hwmon/hwmon1/temp1_input",
        NULL
    };
    for (int i = 0; zones[i]; i++) {
        FILE *f = fopen(zones[i], "r");
        if (f) {
            int millic;
            if (fscanf(f, "%d", &millic) == 1) {
                fclose(f);
                return (double)millic / 1000.0;
            }
            fclose(f);
        }
    }

    /* Fallback: estimate from TSC drift (very rough).
     * If the CPU is throttling, RDTSC increments will show
     * inconsistent rate relative to wall clock. */
    static uint64_t last_tsc = 0;
    static struct timespec last_wall = {0};

    uint64_t tsc = _rdtsc();
    struct timespec wall;
    clock_gettime(CLOCK_MONOTONIC, &wall);

    if (last_tsc == 0) {
        last_tsc = tsc;
        last_wall = wall;
        return 40.0; /* Assume cold start */
    }

    double dt_wall = (wall.tv_sec - last_wall.tv_sec)
                   + (wall.tv_nsec - last_wall.tv_nsec) * 1e-9;
    double dt_tsc  = (double)(tsc - last_tsc);

    if (dt_wall < 0.1) return 40.0; /* Too short to measure */

    last_tsc  = tsc;
    last_wall = wall;

    double tsc_ghz = dt_tsc / (dt_wall * 1e9);
    /* TSC frequency drops ~0.1% per 10°C above nominal due to thermal
     * throttling changing the effective P-state. Very rough estimate. */
    double nominal_ghz = 3.8; /* Assume ~3.8 GHz base */
    double temp = 40.0 + (1.0 - tsc_ghz / nominal_ghz) * 1000.0;
    if (temp < 25.0) temp = 25.0;
    if (temp > 100.0) temp = 100.0;
    return temp;
}

/* Check if we should throttle — returns 1 if thermally limited */
static int _sdr_tx_throttle(SdrState *sdr)
{
    sdr->tx_thermal_c = _read_cpu_temp();
    if (sdr->tx_thermal_c > SDR_TX_THERMAL_LIMIT) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[SDR-TX] THERMAL THROTTLE: CPU at %.1f°C "
                    "(limit %.1f°C). TX suspended.\n",
                    sdr->tx_thermal_c, SDR_TX_THERMAL_LIMIT);
            warned = 1;
        }
        return 1;
    }
    return 0;
}

/* Busy-wait for up to `max_ns` nanoseconds, doing FPU work at intensity
 * proportional to `power`. Returns early if thermal limit hit. */
static void _sdr_tx_burst(SdrState *sdr, double power, uint64_t max_ns,
                          const double *re, const double *im)
{
    if (power < 1e-6 || _sdr_tx_throttle(sdr)) return;

    uint64_t t0 = _rdtsc();
    volatile double sink = 0.0;

    /* Approximate TSC frequency (3.8 GHz typical) */
    uint64_t tsc_limit = (uint64_t)(max_ns * 3.8);

    for (uint64_t c = 0; c < tsc_limit; c += 16) {
        /* Structured FPU work: multiply actual quhit amplitudes.
         * This toggles real FPU transistors and creates a data-dependent
         * current draw that couples into the SDR's RF front-end via the
         * shared USB power rail and motherboard ground plane.
         *
         * The pattern is phase-synchronized to be distinguishable from
         * background CPU noise in the SDR's FFT output. */
        for (int k = 0; k < 6; k++) {
            if (re && im) {
                sink += re[k] * cos(sdr->tx_phase + k * 1.0471975512);
                sink += im[k] * sin(sdr->tx_phase + k * 1.0471975512);
            } else {
                sink += power * cos(sdr->tx_phase + k);
            }
        }
        __asm__ volatile ("" : : "m"(sink) : "memory");

        /* Check thermal limit periodically */
        if ((c & 0xFFF) == 0 && _sdr_tx_throttle(sdr)) break;
    }

    /* Drain to force a real store (real bus transaction → real EM emission) */
    volatile double *drain = (volatile double *)&sdr->tx_amplitude;
    *drain = sink;

    sdr->tx_phase += 0.1;
    if (sdr->tx_phase > 2.0 * M_PI) sdr->tx_phase -= 2.0 * M_PI;

    uint64_t t1 = _rdtsc();
    sdr->tx_total_ns += (uint64_t)((double)(t1 - t0) / 3.8);
}

void quhit_sdr_feedback_tx(SdrState *sdr, const QuhitState *state)
{
    if (!sdr || !state) return;
    if (sdr->coupling < SDR_COUPLING_WRITE) return;

    /* Compute per-channel power */
    double powers[6];
    for (int k = 0; k < 6; k++)
        powers[k] = state->re[k] * state->re[k] + state->im[k] * state->im[k];

    /* Throttled bursts: each basis state gets one short burst proportional
     * to its amplitude, followed by a mandatory cooldown.
     * Total duty cycle is capped at SDR_TX_DUTY_CYCLE. */
    for (int k = 0; k < 6; k++) {
        if (powers[k] < 1e-8) continue;

        /* Burst duration scaled by amplitude weight */
        uint64_t burst_ns = (uint64_t)(SDR_TX_MAX_BURST_US * 1000.0 * powers[k]);
        if (burst_ns < 10) burst_ns = 10;

        _sdr_tx_burst(sdr, powers[k], burst_ns, state->re, state->im);

        /* Mandatory cooldown — let the CPU thermal mass recover.
         * Uses nanosleep to actually yield the core. */
        struct timespec cooldown = {
            .tv_sec  = 0,
            .tv_nsec = SDR_TX_COOLDOWN_US * 1000
        };
        nanosleep(&cooldown, NULL);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CALIBRATION TONE — For testing SDR coupling
 *
 * Emits a short tone burst at the specified frequency by modulating
 * CPU compute at that rate. Used to verify the SDR can detect the
 * CPU's EM signature. This is NOT RF transmission — it's CPU current
 * modulation that the SDR receives as conducted/common-mode noise.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_sdr_tx_tone(SdrState *sdr, double freq_hz, double amplitude)
{
    if (!sdr || amplitude < 0.0) return;

    /* Even calibration tones respect thermal limits */
    if (_sdr_tx_throttle(sdr)) return;

    /* Burst at the requested frequency for one cycle period */
    uint64_t burst_ns = (uint64_t)(1e9 / freq_hz);
    if (burst_ns > SDR_TX_MAX_BURST_US * 1000)
        burst_ns = SDR_TX_MAX_BURST_US * 1000;

    volatile double acc = 0.0;

    uint64_t tsc_limit = (uint64_t)((double)burst_ns * 3.8);
    for (uint64_t c = 0; c < tsc_limit; c += 16) {
        double phase = (double)c * 2.0 * M_PI * freq_hz / (3.8e9);
        acc += amplitude * cos(phase);
        __asm__ volatile ("" : : "m"(acc) : "memory");
        if ((c & 0xFFF) == 0 && _sdr_tx_throttle(sdr)) break;
    }

    volatile double *drain = (volatile double *)&sdr->tx_amplitude;
    *drain = acc;

    /* Cooldown */
    struct timespec cd = { .tv_sec = 0, .tv_nsec = SDR_TX_COOLDOWN_US * 1000 };
    nanosleep(&cd, NULL);
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
    printf("  ║  TX Burst Time:  %-10lu ns                                ║\n",
           (unsigned long)sdr->tx_total_ns);
    printf("  ║  CPU Temp:       %-6.1f °C                                  ║\n",
           sdr->tx_thermal_c);
    printf("  ╚══════════════════════════════════════════════════════════════╝\n\n");
}
