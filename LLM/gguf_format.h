/*
 * gguf_format.h — GGUF v3 Binary Format Writer
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  HExState GGUF Output Module                                 ║
 * ║  Implements the GGUF v3 binary specification for writing     ║
 * ║  quantized LLM weight files compatible with llama.cpp        ║
 * ╚═══════════════════════════════════════════════════════════════╝
 *
 * File Layout:
 *   1. Header:    magic(4) + version(4) + tensor_count(8) + kv_count(8)
 *   2. Metadata:  Key-Value pairs (variable length)
 *   3. Tensor Info: Per-tensor descriptors (name, dims, type, offset)
 *   4. Padding:   Align to GGUF_DEFAULT_ALIGNMENT bytes
 *   5. Tensor Data: Raw quantized weight data
 *
 * All values are little-endian.
 */

#ifndef GGUF_FORMAT_H
#define GGUF_FORMAT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * GGUF CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define GGUF_MAGIC            0x46554747  /* "GGUF" in little-endian    */
#define GGUF_VERSION          3
#define GGUF_DEFAULT_ALIGNMENT 32

/* ═══════════════════════════════════════════════════════════════════════
 * GGML TENSOR TYPES
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    GGML_TYPE_F32   = 0,
    GGML_TYPE_F16   = 1,
    GGML_TYPE_Q4_0  = 2,
    GGML_TYPE_Q4_1  = 3,
    GGML_TYPE_Q5_0  = 6,
    GGML_TYPE_Q5_1  = 7,
    GGML_TYPE_Q8_0  = 8,
    GGML_TYPE_Q8_1  = 9,
    GGML_TYPE_Q2_K  = 10,
    GGML_TYPE_Q3_K  = 11,
    GGML_TYPE_Q4_K  = 12,
    GGML_TYPE_Q5_K  = 13,
    GGML_TYPE_Q6_K  = 14,
    GGML_TYPE_Q8_K  = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT
} GGMLType;

/* ═══════════════════════════════════════════════════════════════════════
 * GGUF METADATA VALUE TYPES
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
} GGUFValueType;

/* ═══════════════════════════════════════════════════════════════════════
 * Q8_0 BLOCK STRUCTURE
 *
 * The fundamental quantized unit: 32 weights + 1 fp16 scale.
 * Total: 34 bytes per block = 8.5 bits per weight.
 *
 * Dequantization: w_i = qs[i] * d
 * ═══════════════════════════════════════════════════════════════════════ */

#define QK8_0 32  /* Block size for Q8_0 */

typedef struct {
    uint16_t d;           /* fp16 scale (delta)                         */
    int8_t   qs[QK8_0];  /* quantized values [-127, 127]               */
} BlockQ8_0;

/* Verify: sizeof(BlockQ8_0) should be 34 bytes (2 + 32) */

/* ═══════════════════════════════════════════════════════════════════════
 * Q4_0 BLOCK STRUCTURE
 *
 * 32 weights per block with 4-bit quantization.
 * Layout: 1 fp16 scale + 16 bytes packed quants (2 weights per byte)
 * Total: 18 bytes per block = 4.5 bits per weight.
 *
 * Dequantization: w_i = (q_i - 8) * d
 *   where q_i in {0..15}, stored as nibbles
 * ═══════════════════════════════════════════════════════════════════════ */

#define QK4_0 32  /* Block size for Q4_0 */

typedef struct {
    uint16_t d;             /* fp16 scale (delta)                         */
    uint8_t  qs[QK4_0/2];  /* 16 bytes: packed 4-bit quants (2 per byte) */
} BlockQ4_0;

/* sizeof(BlockQ4_0) = 2 + 16 = 18 bytes for 32 weights */

/* ═══════════════════════════════════════════════════════════════════════
 * Q2_K BLOCK STRUCTURE (K-Quant, 2-bit)
 *
 * 256-weight superblock divided into 16 sub-blocks of 16 weights.
 *
 * Layout (must match ggml block_q2_K):
 *   d:          fp16 super-block scale for scales
 *   dmin:       fp16 super-block scale for mins
 *   scales[16]: Per-sub-block scale (low 4 bits) + min (high 4 bits)
 *   qs[64]:     Packed 2-bit quants (4 weights per byte)
 *
 * Dequantization: w_i = d * scale_j * q_i - dmin * min_j
 *   where j = sub-block index, q_i in {0, 1, 2, 3}
 *
 * Effective: 2.625 bits per weight (84 bytes / 256 weights)
 * ═══════════════════════════════════════════════════════════════════════ */

#define QK_K 256   /* K-quant superblock size */

typedef struct {
    uint8_t  scales[QK_K/16]; /* 16 bytes: scale(4bit) | min(4bit)       */
    uint8_t  qs[QK_K/4];     /* 64 bytes: packed 2-bit quants            */
    uint16_t d;              /* fp16 super-block scale                   */
    uint16_t dmin;           /* fp16 super-block min scale               */
} BlockQ2K;

/* sizeof(BlockQ2K) = 2 + 2 + 16 + 64 = 84 bytes for 256 weights */

/* ═══════════════════════════════════════════════════════════════════════
 * FP16 ←→ FP32 CONVERSION
 *
 * IEEE 754 half-precision (binary16):
 *   1 sign bit, 5 exponent bits, 10 mantissa bits
 * ═══════════════════════════════════════════════════════════════════════ */

static inline uint16_t gguf_fp32_to_fp16(float f)
{
    /* Use the union approach for bit manipulation */
    union { float f; uint32_t u; } fu;
    fu.f = f;
    uint32_t x = fu.u;

    uint16_t sign = (x >> 16) & 0x8000;
    int32_t exponent = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFF;

    if (exponent <= 0) {
        /* Subnormal or zero */
        if (exponent < -10) return sign;  /* too small → ±0 */
        mantissa = (mantissa | 0x800000) >> (1 - exponent);
        return sign | (uint16_t)(mantissa >> 13);
    } else if (exponent >= 0x1F) {
        /* Infinity or NaN */
        return sign | 0x7C00 | (uint16_t)(mantissa ? (mantissa >> 13) : 0);
    }

    /* Normalized */
    return sign | (uint16_t)(exponent << 10) | (uint16_t)(mantissa >> 13);
}

static inline float gguf_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    uint32_t result;

    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;  /* ±0 */
        } else {
            /* Subnormal → normalize */
            exponent = 1;
            while (!(mantissa & 0x0400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FF;
            result = sign | ((uint32_t)(exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        result = sign | 0x7F800000 | (mantissa << 13);  /* Inf/NaN */
    } else {
        result = sign | ((uint32_t)(exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    union { uint32_t u; float f; } uf;
    uf.u = result;
    return uf.f;
}

/* BFloat16 → Float32 (just shift left by 16, it IS the top 16 bits of fp32) */
static inline float gguf_bf16_to_fp32(uint16_t bf)
{
    union { uint32_t u; float f; } uf;
    uf.u = (uint32_t)bf << 16;
    return uf.f;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GGUF STRING — Length-prefixed UTF-8 (no null terminator in file)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void gguf_write_string(FILE *fp, const char *s)
{
    uint64_t len = strlen(s);
    fwrite(&len, sizeof(uint64_t), 1, fp);
    fwrite(s, 1, len, fp);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GGUF METADATA KEY-VALUE WRITERS
 *
 * Each KV entry: key_string + value_type(u32) + value_data
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void gguf_write_kv_string(FILE *fp, const char *key, const char *val)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_STRING;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    gguf_write_string(fp, val);
}

static inline void gguf_write_kv_uint32(FILE *fp, const char *key, uint32_t val)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_UINT32;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    fwrite(&val, sizeof(uint32_t), 1, fp);
}

static inline void gguf_write_kv_int32(FILE *fp, const char *key, int32_t val)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_INT32;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    fwrite(&val, sizeof(int32_t), 1, fp);
}

static inline void gguf_write_kv_uint64(FILE *fp, const char *key, uint64_t val)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_UINT64;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    fwrite(&val, sizeof(uint64_t), 1, fp);
}

static inline void gguf_write_kv_float32(FILE *fp, const char *key, float val)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_FLOAT32;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    fwrite(&val, sizeof(float), 1, fp);
}

static inline void gguf_write_kv_bool(FILE *fp, const char *key, int val)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_BOOL;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    uint8_t b = val ? 1 : 0;
    fwrite(&b, sizeof(uint8_t), 1, fp);
}

/* Write an array of float32 values */
static inline void gguf_write_kv_float32_array(FILE *fp, const char *key,
                                                 const float *vals, uint64_t count)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_ARRAY;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    uint32_t subtype = GGUF_TYPE_FLOAT32;
    fwrite(&subtype, sizeof(uint32_t), 1, fp);
    fwrite(&count, sizeof(uint64_t), 1, fp);
    fwrite(vals, sizeof(float), count, fp);
}

/* Write an array of int32 values */
static inline void gguf_write_kv_int32_array(FILE *fp, const char *key,
                                               const int32_t *vals, uint64_t count)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_ARRAY;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    uint32_t subtype = GGUF_TYPE_INT32;
    fwrite(&subtype, sizeof(uint32_t), 1, fp);
    fwrite(&count, sizeof(uint64_t), 1, fp);
    fwrite(vals, sizeof(int32_t), count, fp);
}

/* Write an array of string values */
static inline void gguf_write_kv_string_array(FILE *fp, const char *key,
                                                 const char **vals, uint64_t count)
{
    gguf_write_string(fp, key);
    uint32_t vtype = GGUF_TYPE_ARRAY;
    fwrite(&vtype, sizeof(uint32_t), 1, fp);
    uint32_t subtype = GGUF_TYPE_STRING;
    fwrite(&subtype, sizeof(uint32_t), 1, fp);
    fwrite(&count, sizeof(uint64_t), 1, fp);
    for (uint64_t i = 0; i < count; i++) {
        gguf_write_string(fp, vals[i] ? vals[i] : "");
    }
}
/* ═══════════════════════════════════════════════════════════════════════
 * GGUF TENSOR INFO WRITER
 *
 * Per-tensor descriptor in the file:
 *   name_string + n_dims(u32) + dims[n_dims](u64 each) +
 *   type(u32) + offset(u64)
 *
 * Offset is relative to the start of the tensor data section.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void gguf_write_tensor_info(FILE *fp, const char *name,
                                            uint32_t n_dims, const uint64_t *dims,
                                            GGMLType type, uint64_t offset)
{
    gguf_write_string(fp, name);
    fwrite(&n_dims, sizeof(uint32_t), 1, fp);
    for (uint32_t i = 0; i < n_dims; i++) {
        fwrite(&dims[i], sizeof(uint64_t), 1, fp);
    }
    uint32_t t = (uint32_t)type;
    fwrite(&t, sizeof(uint32_t), 1, fp);
    fwrite(&offset, sizeof(uint64_t), 1, fp);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GGUF HEADER WRITER
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void gguf_write_header(FILE *fp, uint64_t tensor_count,
                                       uint64_t metadata_kv_count)
{
    uint32_t magic = GGUF_MAGIC;
    uint32_t version = GGUF_VERSION;
    fwrite(&magic, sizeof(uint32_t), 1, fp);
    fwrite(&version, sizeof(uint32_t), 1, fp);
    fwrite(&tensor_count, sizeof(uint64_t), 1, fp);
    fwrite(&metadata_kv_count, sizeof(uint64_t), 1, fp);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ALIGNMENT PADDING
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void gguf_write_padding(FILE *fp, uint32_t alignment)
{
    long pos = ftell(fp);
    long pad = (alignment - (pos % alignment)) % alignment;
    if (pad > 0) {
        uint8_t zeros[64] = {0};
        while (pad > 0) {
            long write_n = (pad > 64) ? 64 : pad;
            fwrite(zeros, 1, write_n, fp);
            pad -= write_n;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Q8_0 QUANTIZATION — Reference Implementation
 *
 * For each block of 32 floats:
 *   1. Find amax = max(|x_i|)
 *   2. Scale d = amax / 127.0
 *   3. Quantize: qs[i] = round(x_i / d)
 *
 * This is the STANDARD brute-force approach.
 * The HExState MCMC optimizer replaces step 2 with intelligent
 * search for the optimal d that minimizes weighted error.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void gguf_quantize_q8_0_reference(const float *x,
                                                   BlockQ8_0 *y,
                                                   int64_t n_elements)
{
    int64_t n_blocks = n_elements / QK8_0;

    for (int64_t i = 0; i < n_blocks; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(x[i * QK8_0 + j]);
            if (v > amax) amax = v;
        }

        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        y[i].d = gguf_fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; j++) {
            float v = x[i * QK8_0 + j] * id;
            y[i].qs[j] = (int8_t)roundf(v);
        }
    }
}

/* Dequantize a single Q8_0 block back to float (for error measurement) */
static inline void gguf_dequantize_q8_0_block(const BlockQ8_0 *block,
                                                float *out)
{
    float d = gguf_fp16_to_fp32(block->d);
    for (int j = 0; j < QK8_0; j++) {
        out[j] = (float)block->qs[j] * d;
    }
}

/* Compute L2 reconstruction error for a Q8_0 quantized block */
static inline float gguf_q8_0_block_error(const float *original,
                                            const BlockQ8_0 *block)
{
    float deq[QK8_0];
    gguf_dequantize_q8_0_block(block, deq);
    float err = 0.0f;
    for (int j = 0; j < QK8_0; j++) {
        float diff = original[j] - deq[j];
        err += diff * diff;
    }
    return err;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Q2_K QUANTIZATION — Reference Implementation
 *
 * For each superblock of 256 floats:
 *   1. Divide into 16 sub-blocks of 16 weights
 *   2. For each sub-block: find optimal (scale, min) → w ≈ min + scale * q
 *   3. Quantize sub-block scales/mins to 4 bits each
 *   4. Re-quantize weights to 2 bits using final scales
 *   5. Pack 4 quants per byte
 *
 * The HExState MCMC optimizer replaces step 2's brute-force grid search
 * with intelligent Boltzmann-guided exploration.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Helper: find nearest integer (ggml-compatible) */
static inline int gguf_nearest_int(float fval)
{
    float val = fval + 12582912.f;  /* 2^23 + 2^22 */
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

/* Quantize a sub-block of 16 floats with scale+min scheme.
 * Returns scale; stores abs(min) in *the_min.
 * Outputs L[i] ∈ {0, 1, 2, 3} (nmax = 3). */
static inline float gguf_make_qkx_quants(int n, int nmax,
                                           const float *x, uint8_t *L,
                                           float *the_min)
{
    float min_val = x[0];
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min_val) min_val = x[i];
        if (x[i] > max_val) max_val = x[i];
    }
    if (max_val == min_val) {
        for (int i = 0; i < n; i++) L[i] = 0;
        *the_min = -min_val;
        return 0.0f;
    }
    if (min_val > 0) min_val = 0;

    float iscale = nmax / (max_val - min_val);
    float scale = 1.0f / iscale;

    /* Iterative refinement (matches ggml's make_qkx1_quants) */
    for (int itry = 0; itry < 5; itry++) {
        float sumlx = 0;
        int suml2 = 0;
        int did_change = 0;
        for (int i = 0; i < n; i++) {
            int l = gguf_nearest_int(iscale * (x[i] - min_val));
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            if (l != (int)L[i]) { L[i] = l; did_change = 1; }
            sumlx += (x[i] - min_val) * l;
            suml2 += l * l;
        }
        if (suml2 > 0) scale = sumlx / suml2;
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += x[i] - scale * L[i];
        }
        min_val = 0.7f * min_val + 0.3f * sum / n;
        if (min_val > 0) min_val = 0;
        if (scale > 1e-15f) iscale = 1.0f / scale;
        if (!did_change) break;
    }

    *the_min = -min_val;
    return scale;
}

static inline void gguf_quantize_q2_k_reference(const float *x,
                                                  BlockQ2K *y,
                                                  int64_t n_elements)
{
    int64_t n_blocks = n_elements / QK_K;
    const float q4scale = 15.0f;

    for (int64_t i = 0; i < n_blocks; i++) {
        const float *block_x = x + i * QK_K;
        uint8_t L[QK_K];
        float mins[QK_K / 16];
        float scales[QK_K / 16];

        float max_scale = 0.0f;
        float max_min = 0.0f;

        /* Step 1: Find scale and min for each of 16 sub-blocks */
        for (int j = 0; j < QK_K / 16; j++) {
            scales[j] = gguf_make_qkx_quants(16, 3,
                                               block_x + 16 * j,
                                               L + 16 * j, &mins[j]);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        /* Step 2: Quantize the 16 sub-block scales to 4 bits */
        if (max_scale > 0) {
            float iscale = q4scale / max_scale;
            for (int j = 0; j < QK_K / 16; j++) {
                int l = gguf_nearest_int(iscale * scales[j]);
                if (l < 0) l = 0;
                if (l > 15) l = 15;
                y[i].scales[j] = (uint8_t)l;
            }
            y[i].d = gguf_fp32_to_fp16(max_scale / q4scale);
        } else {
            for (int j = 0; j < QK_K / 16; j++) y[i].scales[j] = 0;
            y[i].d = gguf_fp32_to_fp16(0.0f);
        }

        /* Step 3: Quantize the 16 sub-block mins to 4 bits (packed in high nibble) */
        if (max_min > 0) {
            float iscale = q4scale / max_min;
            for (int j = 0; j < QK_K / 16; j++) {
                int l = gguf_nearest_int(iscale * mins[j]);
                if (l < 0) l = 0;
                if (l > 15) l = 15;
                y[i].scales[j] |= ((uint8_t)l << 4);
            }
            y[i].dmin = gguf_fp32_to_fp16(max_min / q4scale);
        } else {
            y[i].dmin = gguf_fp32_to_fp16(0.0f);
        }

        /* Step 4: Re-quantize weights to 2 bits using final rounded scales */
        for (int j = 0; j < QK_K / 16; j++) {
            float d = gguf_fp16_to_fp32(y[i].d) * (y[i].scales[j] & 0xF);
            if (d < 1e-15f) {
                for (int ii = 0; ii < 16; ii++) L[16 * j + ii] = 0;
                continue;
            }
            float dm = gguf_fp16_to_fp32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ii++) {
                int l = gguf_nearest_int((block_x[16 * j + ii] + dm) / d);
                if (l < 0) l = 0;
                if (l > 3) l = 3;
                L[16 * j + ii] = (uint8_t)l;
            }
        }

        /* Step 5: Pack 4 quants per byte (2 bits each)
         * Layout: 2 groups of 128, each packed as 32 bytes holding 4×32 quants */
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) {
                y[i].qs[j / 4 + l] = L[j + l]
                                    | (L[j + l + 32] << 2)
                                    | (L[j + l + 64] << 4)
                                    | (L[j + l + 96] << 6);
            }
        }
    }
}

/* Dequantize a single Q2_K superblock to float (for error measurement) */
static inline void gguf_dequantize_q2_k_block(const BlockQ2K *block,
                                                float *out)
{
    float d = gguf_fp16_to_fp32(block->d);
    float dmin = gguf_fp16_to_fp32(block->dmin);

    const uint8_t *q = block->qs;
    int is = 0;

    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc = block->scales[is++];
            float dl = d * (sc & 0xF);
            float ml = dmin * (sc >> 4);
            for (int l = 0; l < 16; l++) {
                *out++ = dl * ((float)((q[l] >> shift) & 3)) - ml;
            }

            sc = block->scales[is++];
            dl = d * (sc & 0xF);
            ml = dmin * (sc >> 4);
            for (int l = 0; l < 16; l++) {
                *out++ = dl * ((float)((q[l + 16] >> shift) & 3)) - ml;
            }

            shift += 2;
        }
        q += 32;
    }
}

/* Compute L2 error for a Q2_K quantized superblock */
static inline float gguf_q2_k_block_error(const float *original,
                                            const BlockQ2K *block)
{
    float deq[QK_K];
    gguf_dequantize_q2_k_block(block, deq);
    float err = 0.0f;
    for (int j = 0; j < QK_K; j++) {
        float diff = original[j] - deq[j];
        err += diff * diff;
    }
    return err;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GGML TYPE METADATA — Size calculations
 * ═══════════════════════════════════════════════════════════════════════ */

/* Block size for a given type */
static inline int64_t ggml_type_block_size(GGMLType type)
{
    switch (type) {
        case GGML_TYPE_F32:   return 1;
        case GGML_TYPE_F16:   return 1;
        case GGML_TYPE_Q8_0:  return QK8_0;
        case GGML_TYPE_Q2_K:  return QK_K;
        case GGML_TYPE_Q4_0:  return 32;
        case GGML_TYPE_Q4_1:  return 32;
        case GGML_TYPE_Q5_0:  return 32;
        case GGML_TYPE_Q5_1:  return 32;
        case GGML_TYPE_Q4_K:  return 256;
        case GGML_TYPE_Q5_K:  return 256;
        case GGML_TYPE_Q6_K:  return 256;
        default: return 1;
    }
}

/* Bytes per block for a given type */
static inline int64_t ggml_type_bytes_per_block(GGMLType type)
{
    switch (type) {
        case GGML_TYPE_F32:   return 4;
        case GGML_TYPE_F16:   return 2;
        case GGML_TYPE_Q8_0:  return sizeof(BlockQ8_0);  /* 34 */
        case GGML_TYPE_Q2_K:  return sizeof(BlockQ2K);   /* 84 */
        case GGML_TYPE_Q4_0:  return 18;   /* 2 + 16 */
        case GGML_TYPE_Q4_1:  return 20;   /* 2 + 2 + 16 */
        default: return 4;
    }
}

/* Total bytes for n_elements of a given type */
static inline int64_t ggml_type_size(GGMLType type, int64_t n_elements)
{
    int64_t block_size = ggml_type_block_size(type);
    int64_t bytes_per_block = ggml_type_bytes_per_block(type);
    int64_t n_blocks = (n_elements + block_size - 1) / block_size;
    return n_blocks * bytes_per_block;
}

#endif /* GGUF_FORMAT_H */
