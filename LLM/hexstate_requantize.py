#!/usr/bin/env python3
"""
HExState GGUF Re-Quantizer — GGUF-to-GGUF Q2_K quantization.

Reads a source GGUF (F16/BF16/F32), copies all metadata verbatim,
and re-quantizes eligible weight tensors to Q2_K using numpy.

This bypasses the tokenizer parsing problem entirely — the source GGUF
(from llama.cpp's convert_hf_to_gguf.py) has correct metadata.

Usage:
    python3 hexstate_requantize.py input.gguf output.gguf
"""

import struct
import sys
import time
import os
import io
import ctypes
import numpy as np

# ─── HExState C Library (HPC-optimized Q2_K quantization) ──────────────────
_HEXSTATE_LIB = None

def _load_hexstate_lib():
    """Try to load the HExState C shared library for HPC-optimized quantization."""
    global _HEXSTATE_LIB
    if _HEXSTATE_LIB is not None:
        return _HEXSTATE_LIB

    lib_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(lib_dir, "libhexstate_q2k.so")

    if not os.path.exists(lib_path):
        return None

    try:
        lib = ctypes.CDLL(lib_path)

        # void hexstate_init(void)
        lib.hexstate_init.restype = None
        lib.hexstate_init.argtypes = []

        # void hexstate_quantize_tensor_q2k(const float*, int64_t, void*, float*, int, int)
        lib.hexstate_quantize_tensor_q2k.restype = None
        lib.hexstate_quantize_tensor_q2k.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # weights
            ctypes.c_int64,                   # n_elements
            ctypes.c_void_p,                  # output
            ctypes.POINTER(ctypes.c_float),   # out_error
            ctypes.c_int,                     # opt_mode (0=HPC, 1=MSE, 2=Hybrid)
            ctypes.c_int,                     # verbose
        ]

        lib.hexstate_q2k_block_bytes.restype = ctypes.c_int
        lib.hexstate_q2k_block_bytes.argtypes = []
        lib.hexstate_q2k_block_elements.restype = ctypes.c_int
        lib.hexstate_q2k_block_elements.argtypes = []

        # imatrix-aware version
        lib.hexstate_quantize_tensor_q2k_imat.restype = None
        lib.hexstate_quantize_tensor_q2k_imat.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # weights
            ctypes.c_int64,                   # n_elements
            ctypes.c_void_p,                  # output
            ctypes.POINTER(ctypes.c_float),   # out_error
            ctypes.c_int,                     # opt_mode
            ctypes.POINTER(ctypes.c_float),   # imat_importance (can be NULL)
            ctypes.c_int,                     # verbose
        ]

        # Q4_0 HPC quantizer (for attention tensors)
        if hasattr(lib, 'hexstate_quantize_tensor_q4_0_hpc'):
            lib.hexstate_quantize_tensor_q4_0_hpc.restype = None
            lib.hexstate_quantize_tensor_q4_0_hpc.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # weights
                ctypes.c_int64,                   # n_elements
                ctypes.c_void_p,                  # output
                ctypes.POINTER(ctypes.c_float),   # out_error
                ctypes.POINTER(ctypes.c_float),   # imat_importance (can be NULL)
                ctypes.c_int,                     # verbose
            ]

        lib.hexstate_init()
        _HEXSTATE_LIB = lib
        return lib
    except Exception as e:
        print(f"  WARNING: Failed to load HexState library: {e}")
        return None


def _skip_gguf_kv_value(f, vtype):
    """Skip a GGUF KV value of the given type."""
    import struct as st
    size_map = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    if vtype == 8:  # string
        slen = st.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype == 9:  # array
        arr_type = st.unpack('<I', f.read(4))[0]
        arr_len = st.unpack('<Q', f.read(8))[0]
        if arr_type == 8:  # array of strings
            for _ in range(arr_len):
                slen = st.unpack('<Q', f.read(8))[0]
                f.read(slen)
        else:
            sz = size_map.get(arr_type, 4)
            f.read(arr_len * sz)
    else:
        sz = size_map.get(vtype, 4)
        f.read(sz)


def read_imatrix(path):
    """Read llama.cpp importance matrix file (GGUF or legacy .dat format).

    Returns dict: tensor_name -> normalized importance array (float32)
    """
    import struct as st
    imat = {}

    with open(path, 'rb') as f:
        magic = st.unpack('<I', f.read(4))[0]

        if magic == 0x46554747:  # GGUF format (modern llama.cpp)
            _ver = st.unpack('<I', f.read(4))[0]
            n_tensors = st.unpack('<Q', f.read(8))[0]
            n_kv = st.unpack('<Q', f.read(8))[0]

            # Skip KV pairs
            for _ in range(n_kv):
                slen = st.unpack('<Q', f.read(8))[0]
                f.read(slen)  # key
                vtype = st.unpack('<I', f.read(4))[0]
                _skip_gguf_kv_value(f, vtype)

            # Read tensor infos
            tensor_infos = []
            for _ in range(n_tensors):
                slen = st.unpack('<Q', f.read(8))[0]
                name = f.read(slen).decode('utf-8', errors='replace')
                n_dims = st.unpack('<I', f.read(4))[0]
                dims = [st.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
                ttype = st.unpack('<I', f.read(4))[0]
                offset = st.unpack('<Q', f.read(8))[0]
                n_el = 1
                for d in dims:
                    n_el *= d
                tensor_infos.append((name, n_el, offset))

            # Data section start (32-byte aligned)
            data_start = ((f.tell() + 31) // 32) * 32

            # Group by base tensor name: collect in_sum2 and counts
            sum2_data = {}
            counts_data = {}
            for name, n_el, offset in tensor_infos:
                f.seek(data_start + offset)
                data = np.frombuffer(f.read(n_el * 4), dtype=np.float32).copy()
                if name.endswith('.in_sum2'):
                    base = name[:-len('.in_sum2')]
                    sum2_data[base] = data
                elif name.endswith('.counts'):
                    base = name[:-len('.counts')]
                    counts_data[base] = data

            # Compute normalized importance: sqrt(in_sum2 / counts) / mean
            for base_name in sum2_data:
                in_sum2 = sum2_data[base_name]
                count = counts_data.get(base_name, np.array([1.0]))[0]
                if count > 0:
                    importance = np.sqrt(in_sum2 / count)
                else:
                    importance = np.ones_like(in_sum2)
                mean = importance.mean()
                if mean > 1e-30:
                    imat[base_name] = importance / mean
                else:
                    imat[base_name] = np.ones_like(importance)

        else:
            # Legacy format: first 4 bytes were n_entries
            f.seek(0)
            n_entries = st.unpack('<i', f.read(4))[0]
            for _ in range(n_entries):
                name_len = st.unpack('<i', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                n_values = st.unpack('<i', f.read(4))[0]
                n_samples = st.unpack('<i', f.read(4))[0]
                values = np.frombuffer(f.read(n_values * 4), dtype=np.float32).copy()
                mean = values.mean()
                if mean > 1e-30:
                    imat[name] = values / mean
                else:
                    imat[name] = np.ones_like(values)

    return imat


def quantize_tensor_q2k_hpc(f32_data, opt_mode=2, importance=None):
    """Quantize tensor using HexState HPC-optimized C implementation.

    opt_mode: 0=HPC (BP only), 1=MSE (grid search), 2=Hybrid (recommended)
    importance: optional per-element importance weights (from imatrix)
    Returns: (bytes, n_blocks) same as quantize_tensor_q2k()
    """
    lib = _load_hexstate_lib()
    if lib is None:
        raise RuntimeError("HexState library not available")

    n_elements = len(f32_data)
    if n_elements % QK_K != 0:
        pad_len = QK_K - (n_elements % QK_K)
        f32_data = np.concatenate([f32_data, np.zeros(pad_len, dtype=np.float32)])
        if importance is not None:
            importance = np.concatenate([importance, np.ones(pad_len, dtype=np.float32)])
        n_elements = len(f32_data)

    n_blocks = n_elements // QK_K
    block_bytes = lib.hexstate_q2k_block_bytes()  # 84

    # Allocate output buffer
    output = np.zeros(n_blocks * block_bytes, dtype=np.uint8)
    error = ctypes.c_float(0.0)

    # Call C quantizer with or without importance weights
    f32_contiguous = np.ascontiguousarray(f32_data, dtype=np.float32)

    if importance is not None:
        imat_contiguous = np.ascontiguousarray(importance, dtype=np.float32)
        imat_ptr = imat_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        imat_ptr = None

    lib.hexstate_quantize_tensor_q2k_imat(
        f32_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(n_elements),
        output.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(error),
        ctypes.c_int(opt_mode),
        imat_ptr,
        ctypes.c_int(0),  # not verbose
    )

    return output.tobytes(), n_blocks


# ─── Constants ──────────────────────────────────────────────────────────────
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGNMENT = 32
QK_K = 256

GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q4_0  = 2
GGML_TYPE_Q2_K  = 10
GGML_TYPE_BF16  = 30

TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 30: "BF16",
}

# Block sizes and byte sizes for each type
TYPE_BLOCK_SIZE = {
    0: 1, 1: 1, 2: 32, 3: 32, 6: 32, 7: 32,
    8: 32, 9: 32, 10: 256, 11: 256, 12: 256,
    13: 256, 14: 256, 15: 256, 30: 1,
}
TYPE_BLOCK_BYTES = {
    0: 4, 1: 2, 2: 18, 3: 20, 6: 20, 7: 22,
    8: 34, 9: 36, 10: 84, 11: 110, 12: 144,
    13: 176, 14: 210, 15: 292, 30: 2,
}


def align_offset(offset, alignment=ALIGNMENT):
    return (offset + alignment - 1) & ~(alignment - 1)


def read_string(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8', errors='replace')


def write_string(f, s):
    data = s.encode('utf-8')
    f.write(struct.pack('<Q', len(data)))
    f.write(data)


def read_kv_value(f, vtype):
    """Read a KV value and return (vtype, raw_bytes) for passthrough."""
    start = f.tell()
    if vtype == 0:   f.read(1)      # UINT8
    elif vtype == 1: f.read(1)      # INT8
    elif vtype == 2: f.read(2)      # UINT16
    elif vtype == 3: f.read(2)      # INT16
    elif vtype == 4: f.read(4)      # UINT32
    elif vtype == 5: f.read(4)      # INT32
    elif vtype == 6: f.read(4)      # FLOAT32
    elif vtype == 7: f.read(1)      # BOOL
    elif vtype == 8:                # STRING
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype == 9:                # ARRAY
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            read_kv_value(f, arr_type)
    elif vtype == 10: f.read(8)     # UINT64
    elif vtype == 11: f.read(8)     # INT64
    elif vtype == 12: f.read(8)     # FLOAT64
    else:
        raise ValueError(f"Unknown KV type {vtype}")
    end = f.tell()
    f.seek(start)
    raw = f.read(end - start)
    return raw


# ─── BF16 ↔ F32 conversion ─────────────────────────────────────────────────
def bf16_to_f32(data_bytes, n_elements):
    """Convert BF16 raw bytes to float32 numpy array."""
    bf16 = np.frombuffer(data_bytes, dtype=np.uint16)
    # BF16 → F32: shift left 16 bits
    f32_bits = bf16.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def f16_to_f32(data_bytes, n_elements):
    """Convert F16 raw bytes to float32 numpy array."""
    f16 = np.frombuffer(data_bytes, dtype=np.float16)
    return f16.astype(np.float32)


def f32_to_f16(f32_array):
    """Convert float32 array to F16 bytes."""
    return f32_array.astype(np.float16).tobytes()


def f32_to_bf16(f32_array):
    """Convert float32 array to BF16 bytes."""
    f32_bits = f32_array.view(np.uint32)
    bf16 = ((f32_bits + 0x8000) >> 16).astype(np.uint16)
    return bf16.tobytes()


# ─── Q2_K quantization — faithful port of ggml quantize_row_q2_K_ref ───────
# Vectorized with numpy for performance. Uses make_qkx2_quants algorithm:
# - Weighted MAD error with weights[i] = |x[i]|
# - Joint scale+min least-squares solve
# - 16-step grid search for initial iscale

def quantize_tensor_q2k(f32_data):
    """Quantize an entire tensor to Q2_K format.

    Faithful vectorized port of ggml quantize_row_q2_K_ref with
    make_qkx2_quants sub-block optimization.

    Q2_K block layout (84 bytes, must match ggml block_q2_K):
        d          : fp16 super-block scale
        dmin       : fp16 super-block min-scale
        scales[16] : packed 4-bit scale + 4-bit min per sub-block
        qs[64]     : interleaved 2-bit quants (4 weights 32-apart per byte)
    """
    n_elements = len(f32_data)
    nmax = 3
    q4scale = 15.0

    # Pad to QK_K (256) multiple
    if n_elements % QK_K != 0:
        pad_len = QK_K - (n_elements % QK_K)
        f32_data = np.concatenate([f32_data, np.zeros(pad_len, dtype=np.float32)])
        n_elements = len(f32_data)

    n_blocks = n_elements // QK_K

    # Reshape: [n_blocks, 16 sub-blocks, 16 weights]
    data = f32_data.reshape(n_blocks, 16, 16).astype(np.float64)

    # ── make_qkx2_quants vectorized over all sub-blocks ──
    # Shape key: S = [n_blocks, 16], V = [n_blocks, 16, 16]

    weights = np.abs(data)  # [n_blocks, 16, 16]

    sb_min = data.min(axis=2)  # [n_blocks, 16]
    sb_max = data.max(axis=2)  # [n_blocks, 16]
    sb_min = np.minimum(sb_min, 0.0)

    # Weighted sums (needed for least-squares solve)
    sum_w = weights.sum(axis=2)           # [n_blocks, 16]
    sum_x = (weights * data).sum(axis=2)  # [n_blocks, 16]

    sb_range = sb_max - sb_min
    degenerate = sb_range < 1e-30  # [n_blocks, 16]
    safe_range = np.maximum(sb_range, 1e-30)

    # Initial quantization
    iscale0 = nmax / safe_range
    scale0 = 1.0 / np.maximum(iscale0, 1e-30)

    shifted0 = data - sb_min[:, :, None]  # [n_blocks, 16, 16]
    L0 = np.clip(np.round(iscale0[:, :, None] * shifted0), 0, nmax).astype(np.float64)

    # Initial error (MAD): sum(w * |scale*L + min - x|)
    recon0 = scale0[:, :, None] * L0 + sb_min[:, :, None]
    best_error = (weights * np.abs(recon0 - data)).sum(axis=2)  # [n_blocks, 16]

    best_L = L0.copy()
    best_scale = scale0.copy()
    best_min = sb_min.copy()

    # Grid search: 16 steps (nstep=15, rmin=-0.5, rdelta=0.1)
    rmin, rdelta, nstep = -0.5, 0.1, 15
    for ist in range(nstep + 1):
        iscale_try = (rmin + rdelta * ist + nmax) / safe_range  # [n_blocks, 16]

        shifted = data - sb_min[:, :, None]  # use original min for quantization
        Laux = np.clip(np.round(iscale_try[:, :, None] * shifted), 0, nmax).astype(np.float64)

        # Weighted sums for least-squares solve
        wL = weights * Laux  # [n_blocks, 16, 16]
        sum_l = wL.sum(axis=2)            # [n_blocks, 16]
        sum_l2 = (wL * Laux).sum(axis=2)  # [n_blocks, 16]
        sum_xl = (wL * data).sum(axis=2)  # [n_blocks, 16]

        # Solve 2-var system: x[i] ≈ this_scale * L[i] + this_min
        D = sum_w * sum_l2 - sum_l * sum_l
        valid_D = D > 0

        this_scale = np.where(valid_D,
                              (sum_w * sum_xl - sum_x * sum_l) / np.maximum(D, 1e-30),
                              0.0)
        this_min = np.where(valid_D,
                            (sum_l2 * sum_x - sum_l * sum_xl) / np.maximum(D, 1e-30),
                            0.0)

        # If this_min > 0, clamp to 0 and recompute scale
        pos_min = this_min > 0
        this_min = np.where(pos_min, 0.0, this_min)
        this_scale = np.where(pos_min & (sum_l2 > 0),
                              sum_xl / np.maximum(sum_l2, 1e-30),
                              this_scale)

        # Compute error for this trial
        recon = this_scale[:, :, None] * Laux + this_min[:, :, None]
        cur_error = (weights * np.abs(recon - data)).sum(axis=2)

        # Update where this trial is better
        better = valid_D & (cur_error < best_error) & ~degenerate
        if better.any():
            # Expand mask to weight dimension for L update
            better3d = better[:, :, None]
            best_L = np.where(better3d, Laux, best_L)
            best_error = np.where(better, cur_error, best_error)
            best_scale = np.where(better, this_scale, best_scale)
            best_min = np.where(better, this_min, best_min)

    # the_min = -best_min (make positive)
    sb_scale = np.maximum(best_scale, 0.0).astype(np.float32)  # [n_blocks, 16]
    sb_the_min = np.maximum(-best_min, 0.0).astype(np.float32)  # [n_blocks, 16]

    # Handle degenerate sub-blocks
    sb_scale[degenerate] = 0.0
    sb_the_min[degenerate] = np.maximum(-sb_min[degenerate], 0.0).astype(np.float32)

    # ── Phase 2: quantize scales/mins to 4-bit ──
    max_scale = sb_scale.max(axis=1)     # [n_blocks]
    max_min = sb_the_min.max(axis=1)     # [n_blocks]

    # Quantize sub-block scales to 4-bit
    has_scale = max_scale > 0
    iscale_s = np.where(has_scale, q4scale / np.maximum(max_scale, 1e-30), 0.0)
    scales_q = np.where(has_scale[:, None],
                        np.clip(np.round(iscale_s[:, None] * sb_scale), 0, 15),
                        0.0).astype(np.uint8)

    # Quantize sub-block mins to 4-bit
    has_min = max_min > 0
    iscale_m = np.where(has_min, q4scale / np.maximum(max_min, 1e-30), 0.0)
    mins_q = np.where(has_min[:, None],
                      np.clip(np.round(iscale_m[:, None] * sb_the_min), 0, 15),
                      0.0).astype(np.uint8)

    d_fp16 = np.where(has_scale, max_scale / q4scale, 0.0).astype(np.float16)
    dmin_fp16 = np.where(has_min, max_min / q4scale, 0.0).astype(np.float16)

    # ── Phase 3: requantize using fp16-truncated d/dmin ──
    scales_packed = scales_q | (mins_q << 4)  # [n_blocks, 16]

    d_f32 = d_fp16.astype(np.float32)
    dmin_f32 = dmin_fp16.astype(np.float32)

    d_sub = d_f32[:, None] * (scales_packed & 0xF).astype(np.float32)
    dm_sub = dmin_f32[:, None] * (scales_packed >> 4).astype(np.float32)

    # l = nearest_int((x + dm) / d), clamp [0,3]
    valid_d = d_sub > 0
    inv_d = np.where(valid_d, 1.0 / np.maximum(d_sub, 1e-30), 0.0)
    q_vals = np.where(valid_d[:, :, None],
                      np.clip(np.round(
                          (f32_data.reshape(n_blocks, 16, 16) + dm_sub[:, :, None]) * inv_d[:, :, None]
                      ), 0, 3),
                      0).astype(np.uint8)

    # ── Phase 4: pack ──
    q_flat = q_vals.reshape(n_blocks, QK_K)
    q_groups = q_flat.reshape(n_blocks, 2, 4, 32)
    qs_packed = (q_groups[:, :, 0, :] |
                 (q_groups[:, :, 1, :] << 2) |
                 (q_groups[:, :, 2, :] << 4) |
                 (q_groups[:, :, 3, :] << 6)).astype(np.uint8)
    qs_packed = qs_packed.reshape(n_blocks, 64)

    # Build output: [n_blocks, 84] bytes
    # Layout matches ggml block_q2_K: scales[16] | qs[64] | d(fp16) | dmin(fp16)
    result = np.zeros((n_blocks, 84), dtype=np.uint8)
    result[:, 0:16] = scales_packed
    result[:, 16:80] = qs_packed
    result[:, 80:82] = d_fp16.view(np.uint8).reshape(n_blocks, 2)
    result[:, 82:84] = dmin_fp16.view(np.uint8).reshape(n_blocks, 2)

    return result.tobytes(), n_blocks


def is_attention_tensor(name):
    """Detect attention Q/K/V/O projection tensors.
    These are the most sensitive to quantization and get promoted to Q4_0."""
    attn_patterns = [
        'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
        'attn_qkv.weight',
        'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
        'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
    ]
    for pat in attn_patterns:
        if pat in name:
            return True
    return False


def should_quantize(name, n_dims, dims, tied_embeddings=False):
    """Should this tensor be quantized to Q2_K?

    With iMatrix importance weighting, Q2_K is applied to ALL eligible
    tensors including embeddings for maximum compression.

    Tensors kept as-is:
      - 1D tensors (norms, biases) — always kept
      - _norm, .bias — normalization layers
      - ffn_gate_inp — MoE routing gate
      - layer_output_scale — per-layer scaling factor (scalar)
      - altup, laurel — small Gemma-specific tensors
      - token_embd.weight / output.weight when embeddings are tied
        (the same tensor serves as both embedding lookup AND LM head;
         quantizing it to Q2_K destroys logit precision → garbage output)
    """
    n_elements = 1
    for d in dims:
        n_elements *= d
    if n_dims < 2:
        return False
    if 'norm' in name:
        return False
    if '.bias' in name:
        return False
    if 'ffn_gate_inp' in name:
        return False
    if 'altup' in name or 'laurel' in name:
        return False
    if 'layer_output_scale' in name:
        return False
    # When embeddings are tied, token_embd.weight doubles as the output
    # projection (LM head). It gets routed to Q4_0 in the quant plan
    # instead of Q2_K — handled in main(), not here.
    # Skip vision/audio encoder tensors
    if 'v.' in name and name.startswith('v.'):
        return False
    if name.startswith('mm.') or name.startswith('a.'):
        return False
    # Small tensors are not worth quantizing
    if n_elements < QK_K:
        return False
    # Must be divisible by QK_K
    if n_elements % QK_K != 0:
        return False
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 hexstate_requantize.py <input.gguf> <output.gguf> [--keep-metadata]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    keep_metadata = '--keep-metadata' in sys.argv
    quantize_none = '--quantize-none' in sys.argv

    # Check for imatrix
    imatrix_data = None
    for i, arg in enumerate(sys.argv):
        if arg == '--imatrix' and i + 1 < len(sys.argv):
            imat_path = sys.argv[i + 1]
            if os.path.exists(imat_path):
                imatrix_data = read_imatrix(imat_path)
                print(f"  Loaded imatrix: {len(imatrix_data)} tensors from {imat_path}")
            else:
                print(f"  WARNING: imatrix file not found: {imat_path}")
            break

    # Check for HPC C library
    use_hpc = _load_hexstate_lib() is not None

    print()
    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║  HExState GGUF Re-Quantizer                                  ║")
    print("  ║  GGUF → Q2_K GGUF with metadata passthrough                  ║")
    if use_hpc and imatrix_data:
        print("  ║  Engine: HPC + iMatrix (calibrated sensitivity propagation)  ║")
    elif use_hpc:
        print("  ║  Engine: HPC (BP + MSE Grid + Sensitivity Propagation)       ║")
    else:
        print("  ║  Engine: Python (numpy vectorized)                           ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()
    file_size = os.path.getsize(input_path)
    print(f"  Input:  {input_path}")
    print(f"  Size:   {file_size / 1024**3:.2f} GB")
    print(f"  Output: {output_path}")
    print()

    with open(input_path, 'rb') as fin:
        # ── Read Header ──
        magic = struct.unpack('<I', fin.read(4))[0]
        assert magic == GGUF_MAGIC, f"Bad GGUF magic: 0x{magic:08X}"
        version = struct.unpack('<I', fin.read(4))[0]
        n_tensors = struct.unpack('<Q', fin.read(8))[0]
        n_kv = struct.unpack('<Q', fin.read(8))[0]

        print(f"  GGUF v{version}: {n_tensors} tensors, {n_kv} KV pairs")
        print()

        # ── Read KV pairs (store as raw bytes for passthrough) ──
        kv_pairs = []
        for i in range(n_kv):
            key = read_string(fin)
            vtype = struct.unpack('<I', fin.read(4))[0]
            raw_value = read_kv_value(fin, vtype)
            kv_pairs.append((key, vtype, raw_value))

        # ── Read Tensor Info ──
        tensor_infos = []
        for i in range(n_tensors):
            name = read_string(fin)
            n_dims = struct.unpack('<I', fin.read(4))[0]
            dims = [struct.unpack('<Q', fin.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', fin.read(4))[0]
            offset = struct.unpack('<Q', fin.read(8))[0]

            n_elements = 1
            for d in dims:
                n_elements *= d

            blk_sz = TYPE_BLOCK_SIZE.get(ttype, 1)
            blk_bytes = TYPE_BLOCK_BYTES.get(ttype, 4)
            n_blocks = (n_elements + blk_sz - 1) // blk_sz
            data_size = n_blocks * blk_bytes

            tensor_infos.append({
                'name': name, 'n_dims': n_dims, 'dims': dims,
                'type': ttype, 'offset': offset,
                'n_elements': n_elements, 'data_size': data_size,
            })

        # Calculate data section start
        pos_after_info = fin.tell()
        data_section_start = align_offset(pos_after_info)

        print(f"  Data section starts at: {data_section_start:,}")
        print()

        # ── Detect tied embeddings ──
        # If no separate output.weight tensor exists, token_embd.weight
        # doubles as the LM head. Must preserve it at full precision.
        tensor_names = {ti['name'] for ti in tensor_infos}
        has_output_weight = 'output.weight' in tensor_names
        tied_embeddings = not has_output_weight and 'token_embd.weight' in tensor_names
        if tied_embeddings:
            print("  ⚠ Tied embeddings detected — token_embd.weight promoted to Q4_0 (serves as LM head)")
            print()

        # ── Determine output types ──
        quant_plan = []
        total_quant = 0
        total_attn = 0
        total_keep = 0
        for ti in tensor_infos:
            if quantize_none:
                will_quant = False
            elif should_quantize(ti['name'], ti['n_dims'], ti['dims'], tied_embeddings):
                if is_attention_tensor(ti['name']):
                    will_quant = 'ATTN_Q4'  # Promote attention to Q4_0 HPC
                    total_attn += 1
                elif tied_embeddings and ti['name'] in ('token_embd.weight', 'output.weight'):
                    will_quant = 'ATTN_Q4'  # Promote tied embedding to Q4_0
                    total_attn += 1
                else:
                    will_quant = True
                    total_quant += 1
            else:
                will_quant = False
                total_keep += 1
            quant_plan.append(will_quant)

        print(f"  Tensors to quantize (Q2_K):     {total_quant}")
        print(f"  Tensors to promote (Q4_0·HPC):  {total_attn}")
        print(f"  Tensors to keep as-is:          {total_keep}")
        print()

        # ── Compute output tensor sizes and offsets ──
        out_tensor_infos = []
        out_data_offset = 0

        for i, ti in enumerate(tensor_infos):
            if quant_plan[i]:
                out_dims = list(ti['dims'])
                dim0 = out_dims[0] if ti['n_dims'] >= 2 else ti['n_elements']

                if quant_plan[i] == 'ATTN_Q4':
                    # Attention tensor → Q4_0 HPC (4.5 bpw)
                    out_type = GGML_TYPE_Q4_0
                    n_blocks = (ti['n_elements'] + 31) // 32
                    out_size = n_blocks * 18
                    print(f"  [ATTN→Q4_0·HPC] {ti['name']} ({ti['n_elements']} elements)")
                elif dim0 % QK_K == 0:
                    # Q2_K (2.6 bpw, block_size=256)
                    out_type = GGML_TYPE_Q2_K
                    n_blocks = (ti['n_elements'] + QK_K - 1) // QK_K
                    out_size = n_blocks * 84
                elif dim0 % 32 == 0:
                    # Q4_0 fallback (4.5 bpw, block_size=32)
                    out_type = GGML_TYPE_Q4_0
                    n_blocks = ti['n_elements'] // 32
                    out_size = n_blocks * 18
                    quant_plan[i] = 'Q4_0'
                    print(f"  Q4_0: {ti['name']} (dims[0]={dim0})")
                else:
                    out_type = ti['type']
                    out_size = ti['data_size']
                    quant_plan[i] = False
                    print(f"  Keep: {ti['name']} (dims[0]={dim0})")
            else:
                out_type = ti['type']
                out_size = ti['data_size']
                out_dims = list(ti['dims'])

            out_tensor_infos.append({
                'name': ti['name'],
                'n_dims': ti['n_dims'],
                'dims': out_dims,
                'type': out_type,
                'offset': out_data_offset,
                'data_size': out_size,
            })
            out_data_offset += out_size
            out_data_offset = align_offset(out_data_offset)

        # ── Update KV pairs ──
        updated_kv = []
        if keep_metadata:
            print("  --keep-metadata: passing through ALL KV pairs unchanged")
            updated_kv = list(kv_pairs)
        else:
            for key, vtype, raw_value in kv_pairs:
                if key == 'general.file_type' and vtype == 4:  # UINT32
                    # file_type=10 means Q2_K in llama.cpp
                    updated_kv.append((key, vtype, struct.pack('<I', 10)))
                elif key == 'general.quantization_version' and vtype == 4:
                    updated_kv.append((key, vtype, struct.pack('<I', 2)))
                elif key == 'tokenizer.ggml.token_type' and vtype == 9:
                    # ── Fix Gemma 4 token types ──
                    # convert_hf_to_gguf.py incorrectly marks control tokens as
                    # NORMAL (1), causing llama.cpp to sample them (e.g. <unused24>
                    # spam). Fix: read the tokens array to find control-looking
                    # tokens, then patch their types to CONTROL (3).
                    # See: https://github.com/ggml-org/llama.cpp/issues/21321
                    tokens_kv = next((v for k, vt, v in kv_pairs
                                      if k == 'tokenizer.ggml.tokens' and vt == 9), None)
                    token_names = []
                    if tokens_kv:
                        bio = io.BytesIO(tokens_kv)
                        arr_type = struct.unpack('<I', bio.read(4))[0]
                        arr_len = struct.unpack('<Q', bio.read(8))[0]
                        for _ in range(arr_len):
                            slen = struct.unpack('<Q', bio.read(8))[0]
                            token_names.append(bio.read(slen).decode('utf-8', errors='replace'))

                    # Parse the token_type array
                    bio2 = io.BytesIO(raw_value)
                    arr_type2 = struct.unpack('<I', bio2.read(4))[0]
                    arr_len2 = struct.unpack('<Q', bio2.read(8))[0]
                    ttypes = list(struct.unpack(f'<{arr_len2}i', bio2.read(arr_len2 * 4)))

                    # Patch control-looking tokens
                    n_fixed = 0
                    CONTROL_TYPE = 3
                    import re
                    for i, tname in enumerate(token_names):
                        if ttypes[i] == CONTROL_TYPE:
                            continue  # already correct
                        if ttypes[i] == 6:
                            continue  # BYTE type — leave as-is
                        # Only fix tokens that are genuine control/special tokens:
                        # - <eos>, <bos>, <unk>, <mask>, </s> — sentence markers
                        # - <|turn>, <turn|>, <|tool_*|> etc — delimiters
                        # NOTE: do NOT mark <unused*> as CONTROL — Gemma 4 uses
                        # these tokens internally for thinking/channel markers
                        # (e.g. <unused24> = <|channel>). The llama.cpp parser
                        # handles them via the peg-gemma4 format instead.
                        is_control = False
                        if tname in ('<eos>', '<bos>', '<unk>', '<mask>', '</s>',
                                     '<pad>', '<s>'):
                            is_control = True
                        elif re.match(r'^<\|.*\|?>$', tname) or re.match(r'^<.*\|>$', tname):
                            is_control = True
                        if is_control and ttypes[i] != CONTROL_TYPE:
                            ttypes[i] = CONTROL_TYPE
                            n_fixed += 1

                    print(f"  Fixed {n_fixed} token types to CONTROL (Gemma 4 <unused> fix)")

                    # Rebuild the raw value
                    new_raw = struct.pack('<I', arr_type2)
                    new_raw += struct.pack('<Q', arr_len2)
                    new_raw += struct.pack(f'<{arr_len2}i', *ttypes)
                    updated_kv.append((key, vtype, new_raw))
                elif key == 'tokenizer.chat_template' and vtype == 8:
                    # ── Replace chat template with fixed Gemma 4 template ──
                    # The HF-exported template doesn't handle thinking mode, causing
                    # the model to emit <unused24> tokens. The fixed template from
                    # llama.cpp PR #21418 pre-fills an empty thought block when
                    # thinking is disabled: <|channel>thought\n<channel|>
                    # See: https://github.com/ggml-org/llama.cpp/pull/21418
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    workspace_dir = os.path.dirname(script_dir)
                    template_path = os.path.join(workspace_dir, 'llama-cpp-latest',
                        'models', 'templates', 'google-gemma-4-31B-it.jinja')
                    if os.path.exists(template_path):
                        with open(template_path, 'r') as tf:
                            new_template = tf.read()
                        new_raw = struct.pack('<Q', len(new_template.encode('utf-8')))
                        new_raw += new_template.encode('utf-8')
                        updated_kv.append((key, vtype, new_raw))
                        print(f"  Replaced chat template with fixed Gemma 4 template ({len(new_template)} chars)")
                    else:
                        print(f"  WARNING: Fixed template not found at {template_path}, keeping original")
                        updated_kv.append((key, vtype, raw_value))
                else:
                    updated_kv.append((key, vtype, raw_value))

        # ── Write output GGUF ──
        print("  Writing output GGUF...")
        with open(output_path, 'wb') as fout:
            # Header
            fout.write(struct.pack('<I', GGUF_MAGIC))
            fout.write(struct.pack('<I', GGUF_VERSION))
            fout.write(struct.pack('<Q', n_tensors))
            fout.write(struct.pack('<Q', n_kv))

            # KV pairs (passthrough)
            for key, vtype, raw_value in updated_kv:
                write_string(fout, key)
                fout.write(struct.pack('<I', vtype))
                fout.write(raw_value)

            # Tensor info
            for oti in out_tensor_infos:
                write_string(fout, oti['name'])
                fout.write(struct.pack('<I', oti['n_dims']))
                for d in oti['dims']:
                    fout.write(struct.pack('<Q', d))
                fout.write(struct.pack('<I', oti['type']))
                fout.write(struct.pack('<Q', oti['offset']))

            # Alignment padding before data
            pos = fout.tell()
            aligned = align_offset(pos)
            if aligned > pos:
                fout.write(b'\x00' * (aligned - pos))

            # ── Write tensor data ──
            quant_count = 0
            total_quant_bytes = 0
            total_keep_bytes = 0
            total_rmse = 0.0

            for i, ti in enumerate(tensor_infos):
                # Progress bar
                pct = (i + 1) / n_tensors * 100
                bar_width = 40
                filled = int(bar_width * (i + 1) / n_tensors)
                bar = '█' * filled + '░' * (bar_width - filled)
                elapsed = time.time() - start_time
                eta = elapsed / max(i + 1, 1) * (n_tensors - i - 1)
                sys.stdout.write(f"\r  [{bar}] {pct:5.1f}% ({i+1}/{n_tensors}) {elapsed:.0f}s ETA:{eta:.0f}s  {ti['name'][:50]}")
                sys.stdout.flush()

                # Read source tensor data
                abs_offset = data_section_start + ti['offset']
                fin.seek(abs_offset)
                raw_data = fin.read(ti['data_size'])

                if quant_plan[i] in ('Q4_0', 'ATTN_Q4'):
                    # ── Q4_0 quantization (fallback or attention HPC) ──
                    if ti['type'] == GGML_TYPE_BF16:
                        f32 = bf16_to_f32(raw_data, ti['n_elements'])
                    elif ti['type'] == GGML_TYPE_F16:
                        f32 = f16_to_f32(raw_data, ti['n_elements'])
                    elif ti['type'] == GGML_TYPE_F32:
                        f32 = np.frombuffer(raw_data, dtype=np.float32).copy()
                    else:
                        fout.write(raw_data)
                        pad = align_offset(fout.tell()) - fout.tell()
                        if pad > 0: fout.write(b'\x00' * pad)
                        continue

                    # Pad to 32-element boundary
                    n_el = len(f32)
                    pad_to = ((n_el + 31) // 32) * 32
                    if pad_to > n_el:
                        f32 = np.concatenate([f32, np.zeros(pad_to - n_el, dtype=np.float32)])
                        n_el = pad_to

                    n_blocks_q4 = n_el // 32

                    # Use HPC for attention tensors if available
                    if quant_plan[i] == 'ATTN_Q4' and use_hpc and hasattr(_HEXSTATE_LIB, 'hexstate_quantize_tensor_q4_0_hpc'):
                        output_buf = np.zeros(n_blocks_q4 * 18, dtype=np.uint8)
                        error = ctypes.c_float(0.0)
                        f32_c = np.ascontiguousarray(f32, dtype=np.float32)

                        # Look up imatrix importance
                        imat_ptr = None
                        if imatrix_data and ti['name'] in imatrix_data:
                            iw = imatrix_data[ti['name']]
                            n_cols = iw.shape[0]
                            n_rows = n_el // n_cols if n_cols > 0 else 1
                            imat_full = np.tile(iw, n_rows)[:n_el].astype(np.float32)
                            imat_c = np.ascontiguousarray(imat_full)
                            imat_ptr = imat_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

                        _HEXSTATE_LIB.hexstate_quantize_tensor_q4_0_hpc(
                            f32_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            ctypes.c_int64(n_el),
                            output_buf.ctypes.data_as(ctypes.c_void_p),
                            ctypes.byref(error),
                            imat_ptr,
                            ctypes.c_int(0),  # verbose
                        )
                        fout.write(output_buf.tobytes())
                        print(f"\n  [Q4_0·HPC] {ti['name']} RMSE={np.sqrt(error.value / ti['n_elements']):.6e}")
                    else:
                        # Vectorized Q4_0: process all blocks at once
                        blocks = f32.reshape(-1, 32)
                        amax = np.max(np.abs(blocks), axis=1)
                        d = amax / 7.0
                        d[d == 0] = 1.0  # avoid div by zero
                        qs = np.clip(np.round(blocks / d[:, None]) + 8, 0, 15).astype(np.uint8)
                        d_orig = amax / 7.0  # restore zeros
                        d_fp16 = d_orig.astype(np.float16)

                        out_buf = bytearray(n_blocks_q4 * 18)
                        for b in range(n_blocks_q4):
                            off = b * 18
                            struct.pack_into('<e', out_buf, off, float(d_fp16[b]))
                            for j in range(16):
                                out_buf[off + 2 + j] = int(qs[b, j]) | (int(qs[b, j + 16]) << 4)
                        fout.write(bytes(out_buf))

                    quant_count += 1
                    total_quant_bytes += n_blocks_q4 * 18

                elif quant_plan[i]:
                    # Convert to F32 for quantization
                    if ti['type'] == GGML_TYPE_BF16:
                        f32 = bf16_to_f32(raw_data, ti['n_elements'])
                    elif ti['type'] == GGML_TYPE_F16:
                        f32 = f16_to_f32(raw_data, ti['n_elements'])
                    elif ti['type'] == GGML_TYPE_F32:
                        f32 = np.frombuffer(raw_data, dtype=np.float32).copy()
                    else:
                        # Can't re-quantize from quantized format — keep as-is
                        fout.write(raw_data)
                        pad = align_offset(fout.tell()) - fout.tell()
                        if pad > 0:
                            fout.write(b'\x00' * pad)
                        continue

                    # Quantize to Q2_K — always use HPC with chunked processing
                    # Each chunk gets full HPC treatment (no size threshold)
                    HPC_CHUNK = 50_000_000  # 50M elements per HPC chunk
                    HPC_CHUNK = (HPC_CHUNK // QK_K) * QK_K  # align to QK_K

                    # Look up imatrix importance for this tensor
                    imat_full = None
                    if imatrix_data and ti['name'] in imatrix_data:
                        iw = imatrix_data[ti['name']]
                        n_cols = iw.shape[0]
                        n_rows = ti['n_elements'] // n_cols if n_cols > 0 else 1
                        imat_full = np.tile(iw, n_rows)[:ti['n_elements']]

                    n_el = ti['n_elements']
                    if use_hpc and n_el <= HPC_CHUNK:
                        # Small tensor — single HPC pass
                        q2k_data, n_blocks = quantize_tensor_q2k_hpc(f32, opt_mode=2, importance=imat_full)
                    elif use_hpc:
                        # Large tensor — chunked HPC (each chunk gets BP)
                        chunks = []
                        processed = 0
                        while processed < n_el:
                            end = min(processed + HPC_CHUNK, n_el)
                            chunk_f32 = f32[processed:end]
                            if len(chunk_f32) % QK_K != 0:
                                pad_len = QK_K - (len(chunk_f32) % QK_K)
                                chunk_f32 = np.concatenate([chunk_f32, np.zeros(pad_len, dtype=np.float32)])
                            chunk_imp = imat_full[processed:end] if imat_full is not None else None
                            if chunk_imp is not None and len(chunk_imp) < len(chunk_f32):
                                chunk_imp = np.concatenate([chunk_imp, np.ones(len(chunk_f32) - len(chunk_imp), dtype=np.float32)])
                            chunk_data, _ = quantize_tensor_q2k_hpc(chunk_f32, opt_mode=2, importance=chunk_imp)
                            actual_blocks = (end - processed + QK_K - 1) // QK_K
                            chunks.append(chunk_data[:actual_blocks * 84])
                            processed = end
                            pct = 100.0 * processed / n_el
                            print(f"\r    → {processed/1e6:.0f}M/{n_el/1e6:.0f}M ({pct:.0f}%)", end='', flush=True)
                        print()
                        q2k_data = b''.join(chunks)
                        n_blocks = n_el // QK_K
                    else:
                        # No HPC available — python fallback
                        CHUNK_SIZE = 10_000_000
                        CHUNK_SIZE = (CHUNK_SIZE // QK_K) * QK_K
                        chunks = []
                        processed = 0
                        while processed < n_el:
                            end = min(processed + CHUNK_SIZE, n_el)
                            chunk_data, _ = quantize_tensor_q2k(f32[processed:end])
                            chunks.append(chunk_data)
                            processed = end
                            pct = 100.0 * processed / n_el
                            print(f"\r    → {processed/1e6:.0f}M/{n_el/1e6:.0f}M ({pct:.0f}%)", end='', flush=True)
                        print()
                        q2k_data = b''.join(chunks)
                        n_blocks = n_el // QK_K
                    fout.write(q2k_data)

                    quant_count += 1
                    total_quant_bytes += len(q2k_data)
                else:
                    # Keep as-is (passthrough)
                    fout.write(raw_data)
                    total_keep_bytes += len(raw_data)

                # Alignment padding
                pad = align_offset(fout.tell()) - fout.tell()
                if pad > 0:
                    fout.write(b'\x00' * pad)

            final_size = fout.tell()

    elapsed = time.time() - start_time
    print(f"\r  {'█' * 40}  100.0% ({n_tensors}/{n_tensors}) {elapsed:.0f}s" + " " * 60)
    print()

    # ── Summary ──
    original_bytes = sum(ti['data_size'] for ti in tensor_infos)
    compression = original_bytes / max(final_size, 1)

    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║  RE-QUANTIZATION SUMMARY                                     ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")
    print(f"  ║  Tensors quantized (Q2_K): {quant_count:<33d} ║")
    print(f"  ║  Tensors kept as-is:       {total_keep:<33d} ║")
    print(f"  ║  Q2_K data:         {total_quant_bytes:>12,} bytes ({total_quant_bytes/1024**2:>7.1f} MB) ║")
    print(f"  ║  Kept data:         {total_keep_bytes:>12,} bytes ({total_keep_bytes/1024**2:>7.1f} MB) ║")
    print(f"  ║  Original size:     {file_size:>12,} bytes ({file_size/1024**3:>7.2f} GB) ║")
    print(f"  ║  Output size:       {final_size:>12,} bytes ({final_size/1024**3:>7.2f} GB) ║")
    print(f"  ║  Compression:       {compression:>42.1f}x ║")
    print(f"  ║  Total time:        {elapsed:>39.1f} sec ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Output: {output_path}")
    print()


if __name__ == '__main__':
    main()
