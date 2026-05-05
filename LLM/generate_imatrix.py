#!/usr/bin/env python3
"""
HExState Importance Matrix Generator — HPC-Enhanced iMatrix from GGUF

Runs transformer forward passes over calibration text to collect per-channel
E[x²] activation statistics, then uses HPC triality BP to propagate importance
across layers. Outputs llama.cpp-compatible .dat imatrix files.

Usage:
    python3 generate_imatrix.py model.gguf calibration.txt -o imatrix.dat
"""

import struct
import sys
import os
import time
import mmap
import ctypes
import numpy as np
from collections import OrderedDict

# ─── Constants ──────────────────────────────────────────────────────────────
GGUF_MAGIC = 0x46554747
ALIGNMENT = 32
QK_K = 256
QK4_0 = 32
QK8_0 = 32

GGML_TYPE_F32   = 0
GGML_TYPE_F16   = 1
GGML_TYPE_Q4_0  = 2
GGML_TYPE_Q8_0  = 8
GGML_TYPE_Q2_K  = 10
GGML_TYPE_BF16  = 30

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
TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 10: "Q2_K", 30: "BF16",
}


# ─── GGUF Reader ────────────────────────────────────────────────────────────

def align_offset(offset):
    return (offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1)

def read_string(f):
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8', errors='replace')

def read_kv_value(f, vtype):
    """Read and return a KV value."""
    if vtype == 0:   return struct.unpack('<B', f.read(1))[0]
    elif vtype == 1:  return struct.unpack('<b', f.read(1))[0]
    elif vtype == 2:  return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3:  return struct.unpack('<h', f.read(2))[0]
    elif vtype == 4:  return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5:  return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6:  return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7:  return bool(struct.unpack('<B', f.read(1))[0])
    elif vtype == 8:  return read_string(f)
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [read_kv_value(f, arr_type) for _ in range(arr_len)]
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
    else:
        raise ValueError(f"Unknown KV type {vtype}")


class GGUFModel:
    """Loads a GGUF model with mmap'd tensor access."""

    def __init__(self, path):
        self.path = path
        self.file_size = os.path.getsize(path)
        self.kv = {}
        self.tensor_infos = OrderedDict()
        self.data_offset = 0

        self._f = open(path, 'rb')
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_header()

    def _parse_header(self):
        f = self._f
        f.seek(0)
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC, f"Bad GGUF magic: 0x{magic:08X}"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        # Read KV pairs
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = read_kv_value(f, vtype)
            self.kv[key] = value

        # Read tensor info
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            n_elements = 1
            for d in dims:
                n_elements *= d
            blk_sz = TYPE_BLOCK_SIZE.get(ttype, 1)
            blk_bytes = TYPE_BLOCK_BYTES.get(ttype, 4)
            n_blocks = (n_elements + blk_sz - 1) // blk_sz
            data_size = n_blocks * blk_bytes
            self.tensor_infos[name] = {
                'dims': dims, 'n_dims': n_dims, 'type': ttype,
                'offset': offset, 'n_elements': n_elements,
                'data_size': data_size,
            }

        self.data_offset = align_offset(f.tell())

    def get_arch(self):
        arch = self.kv.get('general.architecture', 'gemma2')
        return arch

    def get_config(self):
        arch = self.get_arch()
        return {
            'arch': arch,
            'n_layers': self.kv.get(f'{arch}.block_count', 0),
            'n_embd': self.kv.get(f'{arch}.embedding_length', 0),
            'n_head': self.kv.get(f'{arch}.attention.head_count', 0),
            'n_head_kv': self.kv.get(f'{arch}.attention.head_count_kv', 0),
            'n_ff': self.kv.get(f'{arch}.feed_forward_length', 0),
            'vocab_size': self.kv.get(f'{arch}.vocab_size', 0),
            'rms_eps': self.kv.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-6),
            'rope_base': self.kv.get(f'{arch}.rope.freq_base', 10000.0),
        }

    def get_tensor_f32(self, name):
        """Load a tensor as float32, dequantizing if needed."""
        if name not in self.tensor_infos:
            return None
        ti = self.tensor_infos[name]
        abs_offset = self.data_offset + ti['offset']
        raw = bytes(self._mm[abs_offset:abs_offset + ti['data_size']])
        return dequantize(raw, ti['type'], ti['n_elements'])

    def get_tensor_shape(self, name):
        """Return the shape of a tensor (GGUF stores reversed dims)."""
        if name not in self.tensor_infos:
            return None
        dims = self.tensor_infos[name]['dims']
        # GGUF stores dims in reverse order (row-major): dims[0]=cols, dims[1]=rows
        return tuple(reversed(dims))

    def close(self):
        self._mm.close()
        self._f.close()


# ─── Dequantization ─────────────────────────────────────────────────────────

def dequantize(raw, ttype, n_elements):
    """Dequantize raw bytes to float32 numpy array."""
    if ttype == GGML_TYPE_F32:
        return np.frombuffer(raw, dtype=np.float32).copy()
    elif ttype == GGML_TYPE_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif ttype == GGML_TYPE_BF16:
        bf16 = np.frombuffer(raw, dtype=np.uint16)
        return (bf16.astype(np.uint32) << 16).view(np.float32).copy()
    elif ttype == GGML_TYPE_Q8_0:
        return dequant_q8_0(raw, n_elements)
    elif ttype == GGML_TYPE_Q4_0:
        return dequant_q4_0(raw, n_elements)
    elif ttype == GGML_TYPE_Q2_K:
        return dequant_q2k(raw, n_elements)
    else:
        raise ValueError(f"Unsupported quant type {ttype} ({TYPE_NAME.get(ttype, '?')})")

def dequant_q8_0(raw, n_elements):
    n_blocks = n_elements // QK8_0
    data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 34)
    d = data[:, 0:2].view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = data[:, 2:34].view(np.int8).astype(np.float32)
    return (d * qs).reshape(-1)[:n_elements]

def dequant_q4_0(raw, n_elements):
    n_blocks = n_elements // QK4_0
    data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 18)
    d = data[:, 0:2].view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = data[:, 2:18]  # 16 bytes = 32 nibbles
    lo = (qs & 0xF).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    x = np.concatenate([lo, hi], axis=1)  # [n_blocks, 32]
    return (d * x).reshape(-1)[:n_elements]

def dequant_q2k(raw, n_elements):
    n_blocks = n_elements // QK_K
    data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 84)
    scales_packed = data[:, 0:16]  # [n_blocks, 16]
    qs = data[:, 16:80]  # [n_blocks, 64]
    d_fp16 = data[:, 80:82].view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin_fp16 = data[:, 82:84].view(np.float16).astype(np.float32).reshape(n_blocks)

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)
    for blk in range(n_blocks):
        d = d_fp16[blk]
        dmin = dmin_fp16[blk]
        for half in range(2):
            for sub in range(4):
                j = half * 8 + sub
                sc = int(scales_packed[blk, j]) & 0xF
                mn = int(scales_packed[blk, j]) >> 4
                d_sub = d * sc
                m_sub = dmin * mn
                for k in range(32):
                    qi_byte = int(qs[blk, half * 32 + k])
                    q = (qi_byte >> (sub * 2)) & 3
                    idx = half * 128 + sub * 32 + k
                    result[blk, idx] = d_sub * q - m_sub
    return result.reshape(-1)[:n_elements]


# ─── Tokenizer ──────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """Minimal BPE tokenizer from GGUF metadata."""

    def __init__(self, model):
        self.tokens = model.kv.get('tokenizer.ggml.tokens', [])
        self.vocab_size = len(self.tokens)
        merges_raw = model.kv.get('tokenizer.ggml.merges', [])
        self.bos_id = model.kv.get('tokenizer.ggml.bos_token_id', 2)
        self.eos_id = model.kv.get('tokenizer.ggml.eos_token_id', 1)

        # Build token → id map
        self.token_to_id = {}
        for i, t in enumerate(self.tokens):
            if isinstance(t, str):
                self.token_to_id[t] = i

        # Build merge priority
        self.merges = {}
        for i, m in enumerate(merges_raw):
            if isinstance(m, str):
                parts = m.split(' ', 1)
                if len(parts) == 2:
                    self.merges[(parts[0], parts[1])] = i

    def encode(self, text):
        """Encode text to token IDs using BPE."""
        if not text:
            return [self.bos_id]

        # Convert to byte-level tokens (SentencePiece style: ▁ = space)
        text = text.replace(' ', '▁')
        if not text.startswith('▁'):
            text = '▁' + text

        # Start with characters
        tokens = list(text)

        # Apply BPE merges
        while len(tokens) > 1:
            best_pair = None
            best_rank = float('inf')
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = (i, pair)
            if best_pair is None or best_rank == float('inf'):
                break
            idx, (a, b) = best_pair
            tokens = tokens[:idx] + [a + b] + tokens[idx + 2:]

        # Convert to IDs
        ids = [self.bos_id]
        for t in tokens:
            tid = self.token_to_id.get(t, 0)
            ids.append(tid)
        return ids

    def chunk_text(self, text, chunk_size=512):
        """Encode text and split into fixed-length chunks."""
        ids = self.encode(text)
        chunks = []
        for i in range(0, len(ids) - chunk_size, chunk_size // 2):  # 50% overlap
            chunk = ids[i:i + chunk_size]
            if len(chunk) == chunk_size:
                chunks.append(np.array(chunk, dtype=np.int32))
        if not chunks and ids:
            # Pad short text
            padded = ids + [self.eos_id] * (chunk_size - len(ids))
            chunks.append(np.array(padded[:chunk_size], dtype=np.int32))
        return chunks


# ─── Transformer Forward Pass ───────────────────────────────────────────────

def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

def rope_freqs(dim, seq_len, base=10000.0):
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)  # [seq_len, dim/2]
    return np.cos(freqs), np.sin(freqs)

def apply_rope(x, cos_f, sin_f):
    # x: [seq_len, n_heads, head_dim]
    d2 = x.shape[-1] // 2
    x0 = x[..., :d2]
    x1 = x[..., d2:]
    cos_f = cos_f[:x.shape[0], :d2]
    sin_f = sin_f[:x.shape[0], :d2]
    if x.ndim == 3:
        cos_f = cos_f[:, np.newaxis, :]
        sin_f = sin_f[:, np.newaxis, :]
    o0 = x0 * cos_f - x1 * sin_f
    o1 = x1 * cos_f + x0 * sin_f
    return np.concatenate([o0, o1], axis=-1)

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


class TransformerRunner:
    """Minimal Gemma transformer for importance collection."""

    def __init__(self, model, config, verbose=False):
        self.model = model
        self.cfg = config
        self.verbose = verbose
        self.head_dim = config['n_embd'] // config['n_head']

        # Importance accumulators: tensor_name → (sum_x2, count)
        self.importance = {}

    def _record(self, name, x):
        """Record E[x²] for this tensor's input activation."""
        # x shape: [..., n_cols] — record per-column (input channel)
        x_flat = x.reshape(-1, x.shape[-1])
        x2 = np.sum(x_flat ** 2, axis=0)
        if name in self.importance:
            self.importance[name] = (
                self.importance[name][0] + x2,
                self.importance[name][1] + x_flat.shape[0],
            )
        else:
            self.importance[name] = (x2.copy(), x_flat.shape[0])

    def _get_weight(self, name):
        """Load weight, trying GGUF name patterns."""
        w = self.model.get_tensor_f32(name)
        if w is None:
            return None
        shape = self.model.get_tensor_shape(name)
        if shape and len(shape) >= 2:
            return w.reshape(shape)
        return w

    def _layer_prefix(self, layer_idx):
        return f"blk.{layer_idx}"

    def forward_layer(self, hidden, layer_idx, cos_f, sin_f):
        """Forward pass through one transformer layer. Returns new hidden state."""
        pfx = self._layer_prefix(layer_idx)
        cfg = self.cfg
        n_head = cfg['n_head']
        n_head_kv = cfg['n_head_kv']
        head_dim = self.head_dim
        seq_len = hidden.shape[0]

        # ── Attention ──
        attn_norm_w = self._get_weight(f'{pfx}.attn_norm.weight')
        if attn_norm_w is None:
            return hidden  # Skip if weights missing

        normed = rms_norm(hidden, attn_norm_w, cfg['rms_eps'])

        # Q/K/V projections — record importance on the INPUT (normed)
        q_w = self._get_weight(f'{pfx}.attn_q.weight')
        k_w = self._get_weight(f'{pfx}.attn_k.weight')
        v_w = self._get_weight(f'{pfx}.attn_v.weight')
        o_w = self._get_weight(f'{pfx}.attn_output.weight')

        if q_w is None or k_w is None or v_w is None or o_w is None:
            return hidden

        self._record(f'{pfx}.attn_q.weight', normed)
        self._record(f'{pfx}.attn_k.weight', normed)
        self._record(f'{pfx}.attn_v.weight', normed)

        q = normed @ q_w.T  # [seq, n_head * head_dim]
        k = normed @ k_w.T  # [seq, n_head_kv * head_dim]
        v = normed @ v_w.T

        q = q.reshape(seq_len, n_head, head_dim)
        k = k.reshape(seq_len, n_head_kv, head_dim)
        v = v.reshape(seq_len, n_head_kv, head_dim)

        q = apply_rope(q, cos_f, sin_f)
        k = apply_rope(k, cos_f, sin_f)

        # GQA: repeat KV heads
        if n_head_kv < n_head:
            rep = n_head // n_head_kv
            k = np.repeat(k, rep, axis=1)
            v = np.repeat(v, rep, axis=1)

        # Attention: [n_head, seq, head_dim] @ [n_head, head_dim, seq]
        q_t = q.transpose(1, 0, 2)  # [n_head, seq, head_dim]
        k_t = k.transpose(1, 0, 2)
        v_t = v.transpose(1, 0, 2)

        scale = 1.0 / np.sqrt(head_dim)
        attn = np.matmul(q_t, k_t.transpose(0, 2, 1)) * scale  # [n_head, seq, seq]

        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1)
        attn = attn + mask[np.newaxis, :, :]
        attn = softmax(attn, axis=-1)

        out = np.matmul(attn, v_t)  # [n_head, seq, head_dim]
        out = out.transpose(1, 0, 2).reshape(seq_len, -1)  # [seq, n_embd]

        self._record(f'{pfx}.attn_output.weight', out)
        attn_out = out @ o_w.T

        hidden = hidden + attn_out

        # ── FFN ──
        ffn_norm_w = self._get_weight(f'{pfx}.ffn_norm.weight')
        if ffn_norm_w is None:
            return hidden

        normed_ff = rms_norm(hidden, ffn_norm_w, cfg['rms_eps'])

        gate_w = self._get_weight(f'{pfx}.ffn_gate.weight')
        up_w = self._get_weight(f'{pfx}.ffn_up.weight')
        down_w = self._get_weight(f'{pfx}.ffn_down.weight')

        if gate_w is not None and up_w is not None and down_w is not None:
            self._record(f'{pfx}.ffn_gate.weight', normed_ff)
            self._record(f'{pfx}.ffn_up.weight', normed_ff)

            gate_out = gelu(normed_ff @ gate_w.T)
            up_out = normed_ff @ up_w.T
            ff_mid = gate_out * up_out

            self._record(f'{pfx}.ffn_down.weight', ff_mid)
            ff_out = ff_mid @ down_w.T
            hidden = hidden + ff_out
        else:
            # MoE path
            gate_inp_w = self._get_weight(f'{pfx}.ffn_gate_inp.weight')
            if gate_inp_w is not None:
                self._record(f'{pfx}.ffn_gate_inp.weight', normed_ff)
                router_logits = normed_ff @ gate_inp_w.T
                n_experts = router_logits.shape[-1]
                probs = softmax(router_logits, axis=-1)
                top2 = np.argsort(probs, axis=-1)[:, -2:]

                ff_out = np.zeros_like(normed_ff)
                for exp_id in range(n_experts):
                    ew_gate = self._get_weight(f'{pfx}.ffn_gate.{exp_id}.weight')
                    ew_up = self._get_weight(f'{pfx}.ffn_up.{exp_id}.weight')
                    ew_down = self._get_weight(f'{pfx}.ffn_down.{exp_id}.weight')
                    if ew_gate is None:
                        continue

                    mask_exp = np.any(top2 == exp_id, axis=-1)  # [seq]
                    if not np.any(mask_exp):
                        continue

                    exp_input = normed_ff[mask_exp]
                    self._record(f'{pfx}.ffn_gate.{exp_id}.weight', exp_input)
                    self._record(f'{pfx}.ffn_up.{exp_id}.weight', exp_input)

                    g = gelu(exp_input @ ew_gate.T)
                    u = exp_input @ ew_up.T
                    mid = g * u
                    self._record(f'{pfx}.ffn_down.{exp_id}.weight', mid)

                    exp_out = mid @ ew_down.T
                    # Weight by routing probability
                    for token_idx in np.where(mask_exp)[0]:
                        w = probs[token_idx, exp_id]
                        local_idx = np.sum(mask_exp[:token_idx])
                        ff_out[token_idx] += w * exp_out[local_idx]

                hidden = hidden + ff_out

        return hidden

    def forward(self, token_ids):
        """Full forward pass, collecting importance statistics."""
        cfg = self.cfg
        seq_len = len(token_ids)

        # Embedding
        embed_w = self._get_weight('token_embd.weight')
        if embed_w is None:
            raise RuntimeError("Missing token_embd.weight")

        hidden = embed_w[token_ids]  # [seq_len, n_embd]

        # RoPE frequencies
        cos_f, sin_f = rope_freqs(self.head_dim, seq_len, cfg['rope_base'])

        # Process each layer
        for layer_idx in range(cfg['n_layers']):
            hidden = self.forward_layer(hidden, layer_idx, cos_f, sin_f)
            if self.verbose and (layer_idx + 1) % 4 == 0:
                print(f"    Layer {layer_idx + 1}/{cfg['n_layers']}", end='\r')

        # Output projection
        output_w = self._get_weight('output.weight')
        if output_w is not None:
            self._record('output.weight', hidden)

        return hidden


# ─── HPC Cross-Layer Importance Propagation ─────────────────────────────────

def hpc_propagate_importance(importance_dict, n_layers, verbose=False):
    """Use HPC-inspired BP to propagate importance across layers.

    Each layer's raw E[x²] statistics are smoothed via cross-layer coupling
    through the residual stream. Layers with high importance AND high-importance
    neighbors get boosted; isolated spikes get damped.
    """
    # Group tensors by layer
    layer_energies = np.zeros(n_layers, dtype=np.float64)
    layer_tensor_count = np.zeros(n_layers, dtype=np.int32)

    for name, (sum_x2, count) in importance_dict.items():
        parts = name.split('.')
        if len(parts) >= 2 and parts[0] == 'blk':
            try:
                layer_idx = int(parts[1])
                if 0 <= layer_idx < n_layers:
                    mean_imp = np.mean(sum_x2 / max(count, 1))
                    layer_energies[layer_idx] += mean_imp
                    layer_tensor_count[layer_idx] += 1
            except ValueError:
                pass

    for i in range(n_layers):
        if layer_tensor_count[i] > 0:
            layer_energies[i] /= layer_tensor_count[i]

    if np.max(layer_energies) < 1e-30:
        return importance_dict

    layer_energies /= np.max(layer_energies)

    # BP-inspired iterative smoothing with residual stream coupling
    multipliers = np.ones(n_layers, dtype=np.float64)
    temperature = 0.5

    for _ in range(50):
        new_mult = np.ones(n_layers, dtype=np.float64)
        for i in range(n_layers):
            e_self = layer_energies[i]
            e_nbr = 0.0
            n_nbr = 0
            if i > 0:
                e_nbr += layer_energies[i-1] * multipliers[i-1]
                n_nbr += 1
            if i < n_layers - 1:
                e_nbr += layer_energies[i+1] * multipliers[i+1]
                n_nbr += 1
            if n_nbr > 0:
                e_nbr /= n_nbr
            new_mult[i] = np.exp((e_self + 0.3 * e_nbr) / temperature)

        mean_m = np.mean(new_mult)
        if mean_m > 1e-30:
            new_mult /= mean_m
        multipliers = 0.7 * multipliers + 0.3 * new_mult

    if verbose:
        print(f"\n  HPC layer multipliers (first 8): "
              f"{' '.join(f'{m:.3f}' for m in multipliers[:8])}...")
        print(f"  Range: [{np.min(multipliers):.3f}, {np.max(multipliers):.3f}]")

    adjusted = {}
    for name, (sum_x2, count) in importance_dict.items():
        parts = name.split('.')
        if len(parts) >= 2 and parts[0] == 'blk':
            try:
                layer_idx = int(parts[1])
                if 0 <= layer_idx < n_layers:
                    adjusted[name] = (sum_x2 * multipliers[layer_idx], count)
                    continue
            except ValueError:
                pass
        adjusted[name] = (sum_x2, count)

    return adjusted


# ─── iMatrix Output Writer ──────────────────────────────────────────────────

def write_imatrix(path, importance_dict):
    """Write llama.cpp-compatible legacy binary imatrix file."""
    entries = []
    for name, (sum_x2, count) in sorted(importance_dict.items()):
        values = sum_x2.astype(np.float32)
        entries.append((name, values, int(count)))

    with open(path, 'wb') as f:
        f.write(struct.pack('<i', len(entries)))
        for name, values, n_samples in entries:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<i', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<i', len(values)))
            f.write(struct.pack('<i', n_samples))
            f.write(values.tobytes())

    return len(entries)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='HExState iMatrix Generator — HPC-enhanced importance matrix from GGUF')
    parser.add_argument('model', help='Input GGUF model file')
    parser.add_argument('calibration', help='Calibration text file')
    parser.add_argument('-o', '--output', default='imatrix.dat',
                        help='Output imatrix file (default: imatrix.dat)')
    parser.add_argument('--chunks', type=int, default=100,
                        help='Number of token chunks to process (default: 100)')
    parser.add_argument('--chunk-size', type=int, default=512,
                        help='Tokens per chunk (default: 512)')
    parser.add_argument('--no-hpc', action='store_true',
                        help='Disable HPC cross-layer propagation')
    parser.add_argument('--verbose', action='store_true',
                        help='Per-layer statistics')
    args = parser.parse_args()

    print()
    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║  HExState Importance Matrix Generator                        ║")
    print("  ║  HPC-Enhanced E[x²] Collection from GGUF                    ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()

    # ── Load model ──
    print(f"  Loading model: {args.model}")
    model = GGUFModel(args.model)
    config = model.get_config()

    print(f"  Architecture:  {config['arch']}")
    print(f"  Layers:        {config['n_layers']}")
    print(f"  Hidden:        {config['n_embd']}")
    print(f"  Heads:         {config['n_head']} (KV: {config['n_head_kv']})")
    print(f"  FFN:           {config['n_ff']}")
    print(f"  Vocab:         {config['vocab_size']}")
    print(f"  Tensors:       {len(model.tensor_infos)}")
    print()

    # ── Load tokenizer ──
    print("  Loading tokenizer from GGUF metadata...")
    tokenizer = SimpleTokenizer(model)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print()

    # ── Load calibration text ──
    print(f"  Loading calibration data: {args.calibration}")
    with open(args.calibration, 'r', encoding='utf-8', errors='replace') as f:
        cal_text = f.read()
    print(f"  Text length: {len(cal_text):,} chars")

    # ── Tokenize and chunk ──
    print(f"  Tokenizing ({args.chunk_size} tokens/chunk, {args.chunks} chunks max)...")
    chunks = tokenizer.chunk_text(cal_text, args.chunk_size)
    if len(chunks) > args.chunks:
        chunks = chunks[:args.chunks]
    print(f"  Prepared {len(chunks)} chunks")
    print()

    # ── Forward pass ──
    print("  Running forward passes...")
    runner = TransformerRunner(model, config, verbose=args.verbose)

    for i, chunk in enumerate(chunks):
        elapsed = time.time() - start_time
        eta = elapsed / max(i, 1) * (len(chunks) - i) if i > 0 else 0
        pct = (i + 1) / len(chunks) * 100
        bw = 40
        filled = int(bw * (i + 1) / len(chunks))
        bar = '█' * filled + '░' * (bw - filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}% ({i+1}/{len(chunks)}) "
            f"{elapsed:.0f}s ETA:{eta:.0f}s")
        sys.stdout.flush()

        try:
            runner.forward(chunk)
        except Exception as e:
            print(f"\n  WARNING: Chunk {i} failed: {e}")
            continue

    print(f"\n  Collected importance for {len(runner.importance)} tensors")
    print()

    # ── HPC propagation ──
    if not args.no_hpc:
        print("  Running HPC cross-layer importance propagation...")
        importance = hpc_propagate_importance(
            runner.importance, config['n_layers'], verbose=args.verbose)
    else:
        importance = runner.importance

    # ── Write output ──
    print(f"\n  Writing imatrix: {args.output}")
    n_entries = write_imatrix(args.output, importance)

    elapsed = time.time() - start_time
    out_size = os.path.getsize(args.output)

    print()
    print("  ╔════════════════════════════════════════════════════════════════╗")
    print("  ║  IMATRIX GENERATION COMPLETE                                 ║")
    print("  ╠════════════════════════════════════════════════════════════════╣")
    print(f"  ║  Tensor entries:   {n_entries:<42d} ║")
    print(f"  ║  Chunks processed: {len(chunks):<42d} ║")
    print(f"  ║  Output size:      {out_size:>11,} bytes ({out_size/1024:.1f} KB)"
          f"{' '*(25-len(f'{out_size/1024:.1f}'))}║")
    print(f"  ║  Total time:       {elapsed:>38.1f} sec ║")
    print("  ╚════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Output: {args.output}")
    print()

    model.close()


if __name__ == '__main__':
    main()
