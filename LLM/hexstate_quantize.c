/* ═══════════════════════════════════════════════════════════════════════════
 * hexstate_quantize.c — HExState GGUF Quantizer
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  HPC-Optimized GGUF Quantization Engine                      ║
 * ║                                                               ║
 * ║  Architecture: HPCGraph Sensitivity Propagation               ║
 * ║  Optimization: Complex Amplitude BP + MCMC Scale Search       ║
 * ║  Enhancements: MSE Grid Search, Importance Matrix Weighting   ║
 * ║  Output: GGUF v3 (Q2_K)                                       ║
 * ║                                                               ║
 * ║  "The weight and the quantized are opposite faces."           ║
 * ╚═══════════════════════════════════════════════════════════════╝
 *
 * This tool adapts the HExState HPC Ouroboros factoring engine for
 * LLM weight quantization. The core mathematical machinery is reused:
 *
 *   Factoring Domain          →  Quantization Domain
 *   ─────────────────────────────────────────────────
 *   HPCGraph + CZ edges       →  Block sensitivity graph
 *   Complex Amplitude BP      →  Importance propagation
 *   MCMC period sampler       →  Optimal scale search
 *   try_period() validation   →  Error bound checking
 *   LLL lattice reduction     →  (future) Adaptive bit allocation
 *
 * Additional techniques ported from llm-compressor:
 *   MSE grid search           →  Optimal min/max range shrinking
 *   Importance matrix (imatrix) →  Per-channel error weighting
 *
 * Build:
 *   make -f Makefile.quantize
 *
 * Usage:
 *   ./hexstate_quantize <input> <output.gguf> [options]
 *
 * Input can be:
 *   - A single .safetensors file
 *   - A model directory containing sharded .safetensors files
 *
 * Options:
 *   --optimizer hpc|mse|hybrid   Scale optimization strategy (default: hybrid)
 *   --imatrix <file>             Importance matrix for weighted quantization
 *   --verbose                    Per-block diagnostics
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <mpfr.h>

/* HExState headers — reused from the factoring engine */
#include "quhit_triality.h"
#include "hpc_graph.h"
#include "hpc_mobius.h"
#include "s6_exotic.h"

/* Quantization-specific headers */
#include "gguf_format.h"
#include "safetensors_reader.h"
#include "tokenizer_reader.h"
#include "imatrix_reader.h"

#define D 6  /* Preserved from HExState — the triality dimension */

/* ═══════════════════════════════════════════════════════════════════════════
 * OPTIMIZER MODE
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    OPT_HPC,     /* HExState BP only          */
    OPT_MSE,     /* MSE grid search only      */
    OPT_HYBRID   /* HPC sensitivity + MSE     */
} OptimizerMode;

/* ═══════════════════════════════════════════════════════════════════════════
 * MODEL ARCHITECTURE AUTO-DETECTION
 *
 * Infers model architecture metadata from tensor names and shapes.
 * Supports: LLaMA, Mistral, Qwen2, Phi-3, Gemma, GPT-NeoX, Falcon, DeepSeek
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char     architecture[64];   /* "llama", "phi3", "gemma", etc.  */
    char     name[256];          /* Human-readable model name       */
    uint32_t block_count;        /* Number of transformer layers    */
    uint32_t embedding_length;   /* Hidden dimension                */
    uint32_t head_count;         /* Number of attention heads       */
    uint32_t head_count_kv;      /* Number of KV heads (GQA)        */
    uint32_t vocab_size;         /* Vocabulary size                 */
    uint32_t context_length;     /* Max context length (default)    */
    float    rope_freq_base;     /* RoPE frequency base             */
    uint32_t feed_forward_length; /* FFN intermediate size           */
    float    rms_norm_eps;       /* RMS norm epsilon                */
    int      has_bias;           /* Whether attention has biases    */
    int      tie_word_embeddings; /* Whether output = embed_tokens  */
} ModelArchitecture;

/* Count tensor names matching a pattern prefix */
static int count_tensors_with_prefix(const STMultiFile *mf, const char *prefix)
{
    int count = 0;
    int prefix_len = strlen(prefix);
    for (int i = 0; i < mf->n_tensors; i++) {
        if (strncmp(mf->tensor_map[i].name, prefix, prefix_len) == 0)
            count++;
    }
    return count;
}

/* Find max layer index from tensor names like "model.layers.N.xxx" */
static int find_max_layer_index(const STMultiFile *mf, const char *layer_prefix)
{
    int max_idx = -1;
    int prefix_len = strlen(layer_prefix);
    for (int i = 0; i < mf->n_tensors; i++) {
        if (strncmp(mf->tensor_map[i].name, layer_prefix, prefix_len) == 0) {
            int idx = atoi(mf->tensor_map[i].name + prefix_len);
            if (idx > max_idx) max_idx = idx;
        }
    }
    return max_idx;
}

/* ── Config.json reader for definitive architecture parameters ── */

typedef struct {
    int      valid;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t num_hidden_layers;
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    float    rope_theta;
    float    rms_norm_eps;
    char     model_type[64];
    int      tie_word_embeddings;
} ConfigJson;

static ConfigJson parse_config_json(const char *path)
{
    ConfigJson cfg;
    memset(&cfg, 0, sizeof(cfg));

    FILE *f = fopen(path, "rb");
    if (!f) return cfg;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc(size + 1);
    if (!json) { fclose(f); return cfg; }
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    cfg.valid = 1;

    /* Simple key-value extraction */
    const char *p;

    p = tok_find_key(json, "hidden_size");
    if (p) cfg.hidden_size = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "intermediate_size");
    if (p) cfg.intermediate_size = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "num_attention_heads");
    if (p) cfg.num_attention_heads = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "num_key_value_heads");
    if (p) cfg.num_key_value_heads = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "num_hidden_layers");
    if (p) cfg.num_hidden_layers = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "vocab_size");
    if (p) cfg.vocab_size = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "max_position_embeddings");
    if (p) cfg.max_position_embeddings = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "rope_theta");
    if (p) cfg.rope_theta = (float)strtod(p, NULL);

    p = tok_find_key(json, "rms_norm_eps");
    if (p) cfg.rms_norm_eps = (float)strtod(p, NULL);

    p = tok_find_key(json, "model_type");
    if (p && *p == '"') {
        char buf[64];
        tok_extract_string(p, buf, sizeof(buf));
        strncpy(cfg.model_type, buf, sizeof(cfg.model_type) - 1);
    }

    p = tok_find_key(json, "tie_word_embeddings");
    if (p) cfg.tie_word_embeddings = (strncmp(p, "true", 4) == 0);

    free(json);
    return cfg;
}

static void detect_architecture(const STMultiFile *mf, ModelArchitecture *arch,
                                  const char *config_json_path)
{
    memset(arch, 0, sizeof(*arch));

    /* Default values */
    strcpy(arch->architecture, "llama");
    strcpy(arch->name, "HExState-quantized");
    arch->context_length = 4096;
    arch->rope_freq_base = 10000.0f;
    arch->rms_norm_eps = 1e-5f;

    /* ── Try config.json for definitive parameters ── */
    ConfigJson cfg = {0};
    if (config_json_path) {
        cfg = parse_config_json(config_json_path);
    }

    if (cfg.valid) {
        /* Map model_type to GGUF architecture name */
        if (strcmp(cfg.model_type, "llama") == 0 ||
            strcmp(cfg.model_type, "mistral") == 0) {
            strcpy(arch->architecture, "llama");
        } else if (strcmp(cfg.model_type, "qwen2") == 0) {
            strcpy(arch->architecture, "qwen2");
        } else if (strcmp(cfg.model_type, "qwen2_moe") == 0) {
            strcpy(arch->architecture, "qwen2moe");
        } else if (strcmp(cfg.model_type, "phi3") == 0 ||
                   strcmp(cfg.model_type, "phi") == 0) {
            strcpy(arch->architecture, "phi3");
        } else if (strcmp(cfg.model_type, "gemma") == 0 ||
                   strcmp(cfg.model_type, "gemma2") == 0) {
            strcpy(arch->architecture, "gemma");
        } else if (strcmp(cfg.model_type, "deepseek_v2") == 0) {
            strcpy(arch->architecture, "llama");
        } else if (strcmp(cfg.model_type, "gpt_neox") == 0) {
            strcpy(arch->architecture, "gpt_neox");
        } else if (strcmp(cfg.model_type, "falcon") == 0) {
            strcpy(arch->architecture, "falcon");
        } else if (cfg.model_type[0]) {
            /* Unknown — try llama as fallback */
            strcpy(arch->architecture, "llama");
        }

        if (cfg.hidden_size) arch->embedding_length = cfg.hidden_size;
        if (cfg.intermediate_size) arch->feed_forward_length = cfg.intermediate_size;
        if (cfg.num_attention_heads) arch->head_count = cfg.num_attention_heads;
        if (cfg.num_key_value_heads) arch->head_count_kv = cfg.num_key_value_heads;
        if (cfg.num_hidden_layers) arch->block_count = cfg.num_hidden_layers;
        if (cfg.vocab_size) arch->vocab_size = cfg.vocab_size;
        if (cfg.max_position_embeddings) arch->context_length = cfg.max_position_embeddings;
        if (cfg.rope_theta > 0) arch->rope_freq_base = cfg.rope_theta;
        if (cfg.rms_norm_eps > 0) arch->rms_norm_eps = cfg.rms_norm_eps;
        arch->tie_word_embeddings = cfg.tie_word_embeddings;

        printf("  Architecture determined from config.json: %s\n", cfg.model_type);
    }

    /* ── Fall back to tensor name pattern detection ── */
    int has_model_layers = count_tensors_with_prefix(mf, "model.layers.");
    int has_gpt_neox = count_tensors_with_prefix(mf, "gpt_neox.");
    int has_transformer = count_tensors_with_prefix(mf, "transformer.");

    /* Architecture-specific detection */
    int has_qkv_proj = count_tensors_with_prefix(mf, "model.layers.0.self_attn.qkv_proj");
    int has_kv_a_proj = count_tensors_with_prefix(mf, "model.layers.0.self_attn.kv_a_proj_with_mqa");
    int has_final_norm = (st_multi_find_tensor(mf, "model.final_norm.weight") >= 0);

    if (has_qkv_proj > 0 && !cfg.valid) {
        strcpy(arch->architecture, "phi3");
    } else if (has_kv_a_proj > 0 && !cfg.valid) {
        strcpy(arch->architecture, "llama");  /* DeepSeek uses llama arch */
    } else if (has_final_norm && !cfg.valid) {
        strcpy(arch->architecture, "gemma");
    }

    if (has_model_layers > 0 && arch->block_count == 0) {
        arch->block_count = find_max_layer_index(mf, "model.layers.") + 1;
    }

    /* Infer dimensions from tensor shapes if not from config.json */
    if (arch->embedding_length == 0 || arch->head_count == 0) {
        int qproj_idx = st_multi_find_tensor(mf, "model.layers.0.self_attn.q_proj.weight");
        int kproj_idx = st_multi_find_tensor(mf, "model.layers.0.self_attn.k_proj.weight");

        if (qproj_idx >= 0) {
            const STTensorInfo *ti = st_multi_tensor_info(mf, qproj_idx);
            int64_t q_out = ti->shape[0];
            int64_t hidden = ti->shape[1];
            if (arch->embedding_length == 0) arch->embedding_length = hidden;

            /* Try common head dimensions: 128, 64, 96 */
            int head_dim = 128;
            if (q_out % 128 == 0) head_dim = 128;
            else if (q_out % 96 == 0) head_dim = 96;
            else if (q_out % 64 == 0) head_dim = 64;

            if (arch->head_count == 0) arch->head_count = q_out / head_dim;

            if (kproj_idx >= 0 && arch->head_count_kv == 0) {
                const STTensorInfo *kt = st_multi_tensor_info(mf, kproj_idx);
                arch->head_count_kv = kt->shape[0] / head_dim;
            }
        }
    }

    if (arch->vocab_size == 0) {
        int embed_idx = st_multi_find_tensor(mf, "model.embed_tokens.weight");
        if (embed_idx >= 0) {
            const STTensorInfo *ti = st_multi_tensor_info(mf, embed_idx);
            arch->vocab_size = ti->shape[0];
        }
    }

    if (arch->feed_forward_length == 0) {
        int gate_idx = st_multi_find_tensor(mf, "model.layers.0.mlp.gate_proj.weight");
        if (gate_idx >= 0) {
            const STTensorInfo *ti = st_multi_tensor_info(mf, gate_idx);
            arch->feed_forward_length = ti->shape[0];
        } else {
            int up_idx = st_multi_find_tensor(mf, "model.layers.0.mlp.up_proj.weight");
            if (up_idx >= 0) {
                const STTensorInfo *ti = st_multi_tensor_info(mf, up_idx);
                arch->feed_forward_length = ti->shape[0];
            }
        }
    }

    /* Check for attention bias */
    arch->has_bias = (st_multi_find_tensor(mf, "model.layers.0.self_attn.q_proj.bias") >= 0);

    if (has_gpt_neox > 0 && arch->block_count == 0) {
        strcpy(arch->architecture, "gpt_neox");
        arch->block_count = find_max_layer_index(mf, "gpt_neox.layers.") + 1;
    }
    if (has_transformer > 0 && arch->block_count == 0) {
        strcpy(arch->architecture, "falcon");
        arch->block_count = find_max_layer_index(mf, "transformer.h.") + 1;
    }

    /* Fill in defaults for anything we couldn't detect */
    if (arch->head_count == 0) arch->head_count = 32;
    if (arch->head_count_kv == 0) arch->head_count_kv = arch->head_count;
    if (arch->embedding_length == 0) arch->embedding_length = 4096;
    if (arch->vocab_size == 0) arch->vocab_size = 32000;
    if (arch->feed_forward_length == 0)
        arch->feed_forward_length = (arch->embedding_length * 8) / 3;  /* SwiGLU default */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TENSOR NAME MAPPING: HuggingFace → GGUF Standard
 *
 * Maps SafeTensors tensor names to the standardized GGUF naming
 * convention used by llama.cpp for model loading.
 *
 * Enhanced with mappings for Phi-3, Gemma, DeepSeek, MoE, and bias tensors.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Returns 1 if this tensor should be skipped (not written to GGUF) */
static int should_skip_tensor(const char *hf_name)
{
    /* Rotary embeddings are computed at runtime, not stored */
    if (strstr(hf_name, "rotary_emb.inv_freq") != NULL) return 1;
    if (strstr(hf_name, "rotary_emb.cos_cached") != NULL) return 1;
    if (strstr(hf_name, "rotary_emb.sin_cached") != NULL) return 1;
    return 0;
}

static void map_tensor_name(const char *hf_name, char *gguf_name, int buflen)
{
    /* Start with identity mapping */
    strncpy(gguf_name, hf_name, buflen - 1);
    gguf_name[buflen - 1] = '\0';

    /* Top-level mappings (common to all architectures) */
    struct { const char *from; const char *to; } mappings[] = {
        {"model.embed_tokens.weight",       "token_embd.weight"},
        {"model.norm.weight",               "output_norm.weight"},
        {"model.final_norm.weight",         "output_norm.weight"},  /* Gemma */
        {"lm_head.weight",                  "output.weight"},
        {"model.embed_tokens.bias",         "token_embd.bias"},
        {"model.norm.bias",                 "output_norm.bias"},
        {NULL, NULL}
    };

    for (int m = 0; mappings[m].from; m++) {
        if (strcmp(hf_name, mappings[m].from) == 0) {
            strncpy(gguf_name, mappings[m].to, buflen - 1);
            return;
        }
    }

    /* Layer mappings: "model.layers.N.xxx" or "model.language_model.layers.N.xxx" → "blk.N.xxx" */
    const char *layer_prefix = NULL;
    if (strncmp(hf_name, "model.layers.", 13) == 0)
        layer_prefix = hf_name + 13;
    else if (strncmp(hf_name, "model.language_model.layers.", 27) == 0)
        layer_prefix = hf_name + 27;

    if (layer_prefix) {
        int layer_idx;
        char rest[ST_MAX_NAME_LEN];
        if (sscanf(layer_prefix, "%d.%255s", &layer_idx, rest) == 2) {
            /* Map sublayer names */
            struct { const char *from; const char *to; } layer_maps[] = {
                /* Standard attention projections */
                {"self_attn.q_proj.weight",         "attn_q.weight"},
                {"self_attn.k_proj.weight",         "attn_k.weight"},
                {"self_attn.v_proj.weight",         "attn_v.weight"},
                {"self_attn.o_proj.weight",         "attn_output.weight"},
                /* Attention biases */
                {"self_attn.q_proj.bias",           "attn_q.bias"},
                {"self_attn.k_proj.bias",           "attn_k.bias"},
                {"self_attn.v_proj.bias",           "attn_v.bias"},
                {"self_attn.o_proj.bias",           "attn_output.bias"},
                /* Phi-3 fused QKV */
                {"self_attn.qkv_proj.weight",       "attn_qkv.weight"},
                {"self_attn.qkv_proj.bias",         "attn_qkv.bias"},
                /* DeepSeek MLA */
                {"self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight"},
                {"self_attn.kv_b_proj.weight",      "attn_kv_b.weight"},
                /* Standard FFN (SwiGLU) */
                {"mlp.gate_proj.weight",            "ffn_gate.weight"},
                {"mlp.up_proj.weight",              "ffn_up.weight"},
                {"mlp.down_proj.weight",            "ffn_down.weight"},
                /* FFN biases */
                {"mlp.gate_proj.bias",              "ffn_gate.bias"},
                {"mlp.up_proj.bias",                "ffn_up.bias"},
                {"mlp.down_proj.bias",              "ffn_down.bias"},
                /* MoE gate */
                {"mlp.gate.weight",                 "ffn_gate_inp.weight"},
                /* MoE expert weights */
                {"mlp.experts.gate_proj.weight",    "ffn_gate_exps.weight"},
                {"mlp.experts.up_proj.weight",      "ffn_up_exps.weight"},
                {"mlp.experts.down_proj.weight",    "ffn_down_exps.weight"},
                /* Norm layers */
                {"input_layernorm.weight",          "attn_norm.weight"},
                {"post_attention_layernorm.weight", "ffn_norm.weight"},
                {"input_layernorm.bias",            "attn_norm.bias"},
                {"post_attention_layernorm.bias",   "ffn_norm.bias"},
                /* Gemma pre/post feedforward norm */
                {"pre_feedforward_layernorm.weight", "ffn_norm.weight"},
                {"post_feedforward_layernorm.weight", "ffn_post_norm.weight"},
                {NULL, NULL}
            };

            for (int m = 0; layer_maps[m].from; m++) {
                if (strcmp(rest, layer_maps[m].from) == 0) {
                    snprintf(gguf_name, buflen, "blk.%d.%s",
                             layer_idx, layer_maps[m].to);
                    return;
                }
            }

            /* MoE expert layer mapping: model.layers.N.mlp.experts.E.xxx */
            int expert_idx;
            char expert_rest[ST_MAX_NAME_LEN];
            if (sscanf(rest, "mlp.experts.%d.%255s", &expert_idx, expert_rest) == 2) {
                struct { const char *from; const char *to; } expert_maps[] = {
                    {"gate_proj.weight", "ffn_gate_exp.weight"},
                    {"up_proj.weight",   "ffn_up_exp.weight"},
                    {"down_proj.weight", "ffn_down_exp.weight"},
                    {NULL, NULL}
                };
                for (int m = 0; expert_maps[m].from; m++) {
                    if (strcmp(expert_rest, expert_maps[m].from) == 0) {
                        snprintf(gguf_name, buflen, "blk.%d.%s.%d",
                                 layer_idx, expert_maps[m].to, expert_idx);
                        return;
                    }
                }
            }

            /* Fallback: keep original sub-path */
            snprintf(gguf_name, buflen, "blk.%d.%s", layer_idx, rest);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SHOULD THIS TENSOR BE QUANTIZED?
 *
 * Decision rules:
 *   - Quantize: weight matrices (2D, large)
 *   - Keep F32: norms, biases, embeddings, 1D tensors
 * ═══════════════════════════════════════════════════════════════════════════ */

static int should_quantize(const STTensorInfo *ti, const char *gguf_name)
{
    /* Never quantize 1D tensors (norms, biases) */
    if (ti->n_dims < 2) return 0;

    /* Never quantize embedding tables (row dimension = vocab) */
    if (strstr(gguf_name, "token_embd") != NULL) return 0;

    /* Never quantize LM head output — use exact match, not substring,
     * to avoid matching "attn_output.weight" */
    if (strcmp(gguf_name, "output.weight") == 0) return 0;

    /* Never quantize norm weights */
    if (strstr(gguf_name, "norm") != NULL) return 0;

    /* Never quantize bias tensors */
    if (strstr(gguf_name, ".bias") != NULL) return 0;

    /* Never quantize MoE gate routing weights */
    if (strstr(gguf_name, "ffn_gate_inp") != NULL) return 0;

    /* Quantize everything else (attention projections, FFN weights) */
    return 1;
}

/* Detect attention Q/K/V/O projection tensors.
 * These are the most sensitive to quantization — errors in attention scores
 * cascade through the entire sequence, causing self-correction loops.
 * Promoting these to Q4_0 (~4.5bpw) doubles their precision. */
static int is_attention_tensor(const char *gguf_name)
{
    /* Gemma / LLaMA style GGUF names: blk.N.attn_q/k/v/output.weight */
    if (strstr(gguf_name, "attn_q.weight") != NULL) return 1;
    if (strstr(gguf_name, "attn_k.weight") != NULL) return 1;
    if (strstr(gguf_name, "attn_v.weight") != NULL) return 1;
    if (strstr(gguf_name, "attn_output.weight") != NULL) return 1;
    if (strstr(gguf_name, "attn_qkv.weight") != NULL) return 1;
    /* HuggingFace style (fallthrough names) */
    if (strstr(gguf_name, "self_attn.q_proj.weight") != NULL) return 1;
    if (strstr(gguf_name, "self_attn.k_proj.weight") != NULL) return 1;
    if (strstr(gguf_name, "self_attn.v_proj.weight") != NULL) return 1;
    if (strstr(gguf_name, "self_attn.o_proj.weight") != NULL) return 1;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HPC SENSITIVITY GRAPH BUILDER
 *
 * Creates an HPCGraph where each node represents a weight block.
 * For Q2_K: 256-weight superblocks.
 *
 * The 6 values per site correspond to 6 candidate scale factors:
 *   v=0: scale * 0.85  (aggressive, high compression)
 *   v=1: scale * 0.90
 *   v=2: scale * 0.95
 *   v=3: scale * 1.00  (standard)
 *   v=4: scale * 1.05
 *   v=5: scale * 1.10  (conservative, less compression error)
 *
 * BP propagates: "if your neighbor block is sensitive, you should be
 * conservative too" — creating coherent precision allocation.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SCALE_FACTOR_COUNT 6
static const float SCALE_MULTIPLIERS[SCALE_FACTOR_COUNT] = {
    0.60f, 0.75f, 0.90f, 1.00f, 1.15f, 1.40f
};

/* ── Multi-quhit expanded scale table ──
 * Search grid: 10×10 = 100 (d, dmin) candidates
 * Quhit encoding: bin 10 → 6 for D=6 quhits (BP operates on 6-state marginals)
 * Beam search: operates on all 100 candidates directly */
#define QUHITS_PER_BLOCK  2
#define N_CAND_D   16    /* d multiplier candidates (was 10) */
#define N_CAND_M   16    /* dmin multiplier candidates (was 10) */
#define TOTAL_SCALE_CANDIDATES (N_CAND_D * N_CAND_M)

static float SCALE_TABLE[TOTAL_SCALE_CANDIDATES];
static int scale_table_initialized = 0;

static void init_scale_table(void) {
    if (scale_table_initialized) return;
    /* 100 candidates: uniform spacing centered on 1.0 */
    for (int i = 0; i < TOTAL_SCALE_CANDIDATES; i++) {
        SCALE_TABLE[i] = 0.50f + (float)i * (1.00f / (float)(TOTAL_SCALE_CANDIDATES - 1));
    }
    scale_table_initialized = 1;
}

/* Compute the Q2_K sub-block reconstruction error for a block at a given
 * scale multiplier, optionally weighted by importance vector */
static float compute_block_error_q2k(const float *weights, int block_size,
                                       float scale_mult,
                                       const float *importance, int imp_offset)
{
    float min_val = weights[0];
    float max_val = weights[0];
    for (int j = 1; j < block_size; j++) {
        if (weights[j] < min_val) min_val = weights[j];
        if (weights[j] > max_val) max_val = weights[j];
    }
    if (min_val > 0) min_val = 0;

    float range = (max_val - min_val) * scale_mult;
    if (range < 1e-15f) return 0.0f;
    float inv_range = 3.0f / range;

    float err = 0.0f;
    for (int j = 0; j < block_size; j++) {
        float x = weights[j];
        int q = (int)((x - min_val * scale_mult) * inv_range + 0.5f);
        if (q < 0) q = 0; if (q > 3) q = 3;
        float deq = min_val * scale_mult + (float)q * range / 3.0f;
        float diff = x - deq;
        float w = (importance) ? importance[imp_offset + j] : 1.0f;
        err += diff * diff * w;
    }
    return err;
}

/* Build multi-quhit HPC sensitivity graph.
 * 2 quhits per block → 36 scale candidates per block.
 *
 * Graph layout: sites [0..2*n-1] where:
 *   site 2*i     = coarse quhit for block i
 *   site 2*i + 1 = fine quhit for block i
 *
 * Edges:
 *   Intra-block: CZ(2i, 2i+1) — coarse↔fine coupling
 *   Inter-block: CZ(2i, 2(i+1)) — coarse↔coarse neighbor
 *                CZ(2i+1, 2(i+1)+1) — fine↔fine neighbor */
static HPCGraph *build_sensitivity_graph(const float *weights,
                                           int64_t n_elements,
                                           int block_size,
                                           float temperature,
                                           const float *importance)
{
    int64_t n_blocks = n_elements / block_size;
    if (n_blocks < 2) return NULL;

    init_scale_table();

    int64_t graph_blocks = (n_blocks > 8192) ? 8192 : n_blocks;
    int64_t stride = n_blocks / graph_blocks;
    int64_t n_sites = graph_blocks * QUHITS_PER_BLOCK;

    HPCGraph *graph = hpc_create(n_sites);
    if (!graph) return NULL;

    for (int64_t i = 0; i < n_sites; i++)
        triality_dft(&graph->locals[i]);

    /* Compute errors for all 36 scale candidates per block,
     * then project onto coarse (quhit 0) and fine (quhit 1) marginals */
    for (int64_t i = 0; i < graph_blocks; i++) {
        int64_t block_idx = i * stride;
        const float *block_weights = weights + block_idx * block_size;

        /* Evaluate all 36 candidates */
        float errors[TOTAL_SCALE_CANDIDATES];
        float min_err = 1e30f;
        for (int c = 0; c < TOTAL_SCALE_CANDIDATES; c++) {
            errors[c] = compute_block_error_q2k(block_weights, block_size,
                                                  SCALE_TABLE[c],
                                                  importance,
                                                  (int)(block_idx * block_size));
            if (errors[c] < min_err) min_err = errors[c];
        }

        /* Project onto quhit 0 (coarse): marginalize over fine dimension
         * amp_coarse[v0] = Σ_{v1} exp(-error(v0*6+v1) / 2T) */
        double coarse_re[6], coarse_im[6];
        double coarse_norm = 0.0;
        for (int v0 = 0; v0 < 6; v0++) {
            coarse_re[v0] = 0.0;
            coarse_im[v0] = 0.0;
            for (int v1 = 0; v1 < 6; v1++) {
                int idx = v0 * 6 + v1;
                coarse_re[v0] += exp(-(double)(errors[idx] - min_err) /
                                      (2.0 * (double)temperature));
            }
            coarse_norm += coarse_re[v0] * coarse_re[v0];
        }
        if (coarse_norm > 1e-30) {
            double inv = 1.0 / sqrt(coarse_norm);
            for (int v = 0; v < 6; v++) coarse_re[v] *= inv;
        }

        /* Project onto quhit 1 (fine): marginalize over coarse dimension
         * amp_fine[v1] = Σ_{v0} exp(-error(v0*6+v1) / 2T) */
        double fine_re[6], fine_im[6];
        double fine_norm = 0.0;
        for (int v1 = 0; v1 < 6; v1++) {
            fine_re[v1] = 0.0;
            fine_im[v1] = 0.0;
            for (int v0 = 0; v0 < 6; v0++) {
                int idx = v0 * 6 + v1;
                fine_re[v1] += exp(-(double)(errors[idx] - min_err) /
                                    (2.0 * (double)temperature));
            }
            fine_norm += fine_re[v1] * fine_re[v1];
        }
        if (fine_norm > 1e-30) {
            double inv = 1.0 / sqrt(fine_norm);
            for (int v = 0; v < 6; v++) fine_re[v] *= inv;
        }

        /* Write coarse quhit (site 2*i) */
        int64_t s_coarse = 2 * i;
        for (int v = 0; v < 6; v++) {
            graph->locals[s_coarse].edge_re[v] = coarse_re[v];
            graph->locals[s_coarse].edge_im[v] = 0.0;
        }
        graph->locals[s_coarse].primary = VIEW_EDGE;
        graph->locals[s_coarse].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
        graph->locals[s_coarse].delta_valid = 0;
        triality_update_mask(&graph->locals[s_coarse]);

        /* Write fine quhit (site 2*i + 1) */
        int64_t s_fine = 2 * i + 1;
        for (int v = 0; v < 6; v++) {
            graph->locals[s_fine].edge_re[v] = fine_re[v];
            graph->locals[s_fine].edge_im[v] = 0.0;
        }
        graph->locals[s_fine].primary = VIEW_EDGE;
        graph->locals[s_fine].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
        graph->locals[s_fine].delta_valid = 0;
        triality_update_mask(&graph->locals[s_fine]);
    }

    /* ── Build edges ── */
    for (int64_t i = 0; i < graph_blocks; i++) {
        /* Intra-block: coarse ↔ fine coupling */
        hpc_cz(graph, 2 * i, 2 * i + 1);

        /* Inter-block: neighbor coupling */
        if (i + 1 < graph_blocks) {
            hpc_cz(graph, 2 * i, 2 * (i + 1));         /* coarse ↔ coarse */
            hpc_cz(graph, 2 * i + 1, 2 * (i + 1) + 1); /* fine ↔ fine     */
        }
    }

    return graph;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MSE GRID SEARCH (ported from llm-compressor observers/mse.py)
 *
 * For a Q2_K sub-block, progressively shrink the min/max range to find
 * the candidate that minimizes weighted reconstruction error.
 *
 *   for p in [1.0, 1.0 - 1/grid, 1.0 - 2/grid, ...] down to (1 - maxshrink):
 *     candidate_min = p * min
 *     candidate_max = p * max
 *     error = ||x - quantize(x, candidate_min, candidate_max)||^norm
 *     if error < best: update best
 *     else: patience--; if patience == 0: break
 *
 * This is a direct C port of llm-compressor's _grid_search_mse.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float maxshrink;    /* Maximum shrink factor (0.0 to 1.0)         */
    int   grid;         /* Number of grid divisions                   */
    int   patience;     /* Early stopping patience                    */
    float norm;         /* Error norm exponent (2.0 = MSE, 2.4 = ...)*/
} MSEGridConfig;

static const MSEGridConfig MSE_DEFAULT_CONFIG = {
    .maxshrink = 0.20f,
    .grid      = 200,
    .patience  = 8,
    .norm      = 2.4f
};

/* Grid search for optimal scale/min for a Q2_K sub-block of n weights
 * with nmax = 3 quantization levels.
 * Returns optimized scale; stores absolute min in *out_min.
 * importance: per-element weights (can be NULL for uniform). */
static float mse_grid_search_q2k_subblock(const float *x, int n, int nmax,
                                            uint8_t *L, float *out_min,
                                            const float *importance,
                                            const MSEGridConfig *cfg)
{
    float min_val = x[0], max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min_val) min_val = x[i];
        if (x[i] > max_val) max_val = x[i];
    }
    if (max_val == min_val) {
        for (int i = 0; i < n; i++) L[i] = 0;
        *out_min = -min_val;
        return 0.0f;
    }
    if (min_val > 0) min_val = 0;

    float best_scale = 0.0f;
    float best_min = -min_val;
    float best_error = 1e30f;
    int no_improve = 0;

    int shrink_steps = (int)(cfg->maxshrink * cfg->grid);
    if (shrink_steps < 1) shrink_steps = 1;

    for (int step = 0; step <= shrink_steps; step++) {
        float p = 1.0f - (float)step / (float)cfg->grid;

        float cand_min = p * min_val;
        float cand_max = p * max_val;

        if (cand_max <= cand_min) continue;

        float iscale = (float)nmax / (cand_max - cand_min);
        float scale = 1.0f / iscale;

        /* Quantize and measure error */
        float err = 0.0f;
        uint8_t tmp_L[256];
        for (int i = 0; i < n; i++) {
            int l = gguf_nearest_int(iscale * (x[i] - cand_min));
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            tmp_L[i] = (uint8_t)l;

            float deq = cand_min + scale * (float)l;
            float diff = fabsf(x[i] - deq);
            /* Apply error norm */
            float e = diff;
            if (cfg->norm != 1.0f) {
                e = powf(diff, cfg->norm);
            }
            /* Apply importance weighting */
            if (importance) e *= importance[i];
            err += e;
        }

        if (err < best_error) {
            best_error = err;
            best_scale = scale;
            best_min = -cand_min;
            memcpy(L, tmp_L, n);
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= cfg->patience) break;
        }
    }

    /* Iterative refinement on the best candidate (from ggml) */
    float cur_min = -best_min;
    float cur_scale = best_scale;
    if (cur_scale > 1e-15f) {
        float iscale = 1.0f / cur_scale;
        for (int itry = 0; itry < 3; itry++) {
            float sumlx = 0;
            int suml2 = 0;
            for (int i = 0; i < n; i++) {
                int l = gguf_nearest_int(iscale * (x[i] - cur_min));
                if (l < 0) l = 0;
                if (l > nmax) l = nmax;
                L[i] = (uint8_t)l;
                sumlx += (x[i] - cur_min) * l;
                suml2 += l * l;
            }
            if (suml2 > 0) cur_scale = sumlx / suml2;
            float sum = 0;
            for (int i = 0; i < n; i++)
                sum += x[i] - cur_scale * L[i];
            cur_min = 0.7f * cur_min + 0.3f * sum / n;
            if (cur_min > 0) cur_min = 0;
            if (cur_scale > 1e-15f) iscale = 1.0f / cur_scale;
        }
    }

    *out_min = -cur_min;
    return cur_scale;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HPC Q2_K QUANTIZATION — GGML-QUALITY + HPC REFINEMENT
 *
 * Two-phase approach:
 *   Phase A: Per-sub-block weighted least-squares (ggml make_qkx2_quants)
 *            This produces per-sub-block (scale, min) with 16-step search.
 *   Phase B: HPC BP refines the superblock-level d/dmin rounding.
 *            6 candidate (d, dmin) pairs are tested; BP finds the one
 *            where the GLOBAL reconstruction error is minimized via
 *            constructive interference of per-sub-block phase coherence.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Weighted least-squares quantization for a sub-block (ggml make_qkx2_quants).
 * Finds optimal (scale, min) by searching 16 candidate iscale values
 * and solving weighted least-squares for each.
 * Returns scale; *the_min is set to the negative of the optimal min. */
static float hpc_make_qkx2_quants(int n, int nmax, const float *x,
                                     const float *w, uint8_t *L,
                                     float *the_min, uint8_t *Laux)
{
    float xmin = x[0], xmax = x[0];
    float sum_w = w[0], sum_x = w[0] * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
        sum_w += w[i];
        sum_x += w[i] * x[i];
    }
    if (xmin > 0) xmin = 0;
    if (xmax == xmin) {
        for (int i = 0; i < n; i++) L[i] = 0;
        *the_min = -xmin;
        return 0.0f;
    }

    float iscale = (float)nmax / (xmax - xmin);
    float scale = 1.0f / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; i++) {
        int l = gguf_nearest_int(iscale * (x[i] - xmin));
        if (l < 0) l = 0;
        if (l > nmax) l = nmax;
        L[i] = (uint8_t)l;
        float diff = scale * (float)l + xmin - x[i];
        best_mad += w[i] * fabsf(diff);
    }

    /* 16 candidate iscale values: search [-0.5, -0.5 + 0.1*15] + nmax */
    for (int is = 0; is <= 15; is++) {
        float try_iscale = (-0.5f + 0.1f * (float)is + (float)nmax) / (xmax - xmin);
        float sl = 0, sl2 = 0, sxl = 0;
        for (int i = 0; i < n; i++) {
            int l = gguf_nearest_int(try_iscale * (x[i] - xmin));
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            Laux[i] = (uint8_t)l;
            sl += w[i] * (float)l;
            sl2 += w[i] * (float)(l * l);
            sxl += w[i] * (float)l * x[i];
        }
        float det = sum_w * sl2 - sl * sl;
        if (det > 0) {
            float this_scale = (sum_w * sxl - sum_x * sl) / det;
            float this_min = (sl2 * sum_x - sl * sxl) / det;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sxl / sl2;
            }
            float mad = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * (float)Laux[i] + this_min - x[i];
                mad += w[i] * fabsf(diff);
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; i++) L[i] = Laux[i];
                best_mad = mad;
                scale = this_scale;
                xmin = this_min;
            }
        }
    }
    *the_min = -xmin;
    return scale;
}

/* Quantize the scale/min arrays into 4-bit values: make_qp_quants equivalent.
 * Returns the optimal d such that scales[j] ≈ d × Ls[j]. */
static float hpc_make_qp_quants(int n, int nmax, const float *x,
                                   uint8_t *L, const float *sw)
{
    float xmax = 0;
    for (int i = 0; i < n; i++)
        if (x[i] > xmax) xmax = x[i];
    if (xmax < 1e-15f) {
        for (int i = 0; i < n; i++) L[i] = 0;
        return 0.0f;
    }
    float iscale = (float)nmax / xmax;
    for (int i = 0; i < n; i++) {
        int l = gguf_nearest_int(iscale * x[i]);
        if (l < 0) l = 0;
        if (l > nmax) l = nmax;
        L[i] = (uint8_t)l;
    }
    float scale = 1.0f / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - scale * (float)L[i];
        best_mse += sw[i] * diff * diff;
    }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        float iscale_is = (0.1f * (float)is + (float)nmax) / xmax;
        float scale_is = 1.0f / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; i++) {
            int l = gguf_nearest_int(iscale_is * x[i]);
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            float diff = x[i] - scale_is * (float)l;
            mse += sw[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    /* Recompute with best iscale + iterative refinement */
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; i++) {
        int l = gguf_nearest_int(iscale * x[i]);
        if (l < 0) l = 0;
        if (l > nmax) l = nmax;
        L[i] = (uint8_t)l;
        sumlx += sw[i] * x[i] * (float)l;
        suml2 += sw[i] * (float)(l * l);
    }
    /* Iterative greedy refinement */
    for (int itry = 0; itry < 5; itry++) {
        int n_changed = 0;
        for (int i = 0; i < n; i++) {
            float wi = sw[i];
            float slx = sumlx - wi * x[i] * (float)L[i];
            float sl2 = suml2 - wi * (float)(L[i] * L[i]);
            if (slx > 0 && sl2 > 0) {
                int new_l = gguf_nearest_int(x[i] * sl2 / slx);
                if (new_l < 0) new_l = 0;
                if (new_l > nmax) new_l = nmax;
                if (new_l != L[i]) {
                    slx += wi * x[i] * (float)new_l;
                    sl2 += wi * (float)(new_l * new_l);
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (uint8_t)new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed++;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0 ? sumlx / suml2 : 0.0f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * COMPLEX AMPLITUDE BELIEF PROPAGATION
 * (Ported from tesseract_factor.c for cross-block scale optimization)
 *
 * For CZ edges w(va,vb) = ω^(va·vb):
 *   m_{a→b}[vb] = Σ_{va} [aₐ(va) × msgs(va)] × ω^(va·vb) = DFT₆{...}
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double re[2][6]; /* re[0]: sa→sb, re[1]: sb→sa */
    double im[2][6];
} ComplexEdgeMsg;

/* ω₆ roots of unity for CZ phase lookup */
static const double W6_RE[6] = { 1.0, 0.5, -0.5, -1.0, -0.5,  0.5 };
static const double W6_IM[6] = { 0.0, 0.866025403784438647, 0.866025403784438647,
                                  0.0, -0.866025403784438647, -0.866025403784438647 };
static const double INV_SQRT6 = 0.40824829046386301637;  /* 1/√6 */

static double z6_complex_amplitude_bp(MobiusAmplitudeSheet *ms, unsigned int seed, int max_iter) {
    const HPCGraph *g = ms->graph;
    int n_edges = (int)g->n_edges;
    int n_sites = (int)g->n_sites;
    if (n_edges == 0) return 1e30;

    ComplexEdgeMsg *msgs = (ComplexEdgeMsg*)calloc(n_edges, sizeof(ComplexEdgeMsg));

    srand(seed * 12345 + 42);
    for (int e = 0; e < n_edges; e++) {
        for (int v = 0; v < 6; v++) {
            if (seed == 0) {
                msgs[e].re[0][v] = 1.0; msgs[e].im[0][v] = 0.0;
                msgs[e].re[1][v] = 1.0; msgs[e].im[1][v] = 0.0;
            } else {
                double angle0 = 2.0 * 3.14159265358979323846 * ((double)rand() / RAND_MAX);
                double angle1 = 2.0 * 3.14159265358979323846 * ((double)rand() / RAND_MAX);
                msgs[e].re[0][v] = cos(angle0); msgs[e].im[0][v] = sin(angle0);
                msgs[e].re[1][v] = cos(angle1); msgs[e].im[1][v] = sin(angle1);
            }
        }
    }

    #define QUANT_DAMPING_START 0.50
    #define QUANT_DAMPING_END   0.25
    #define QUANT_COOL_ITERS    100
    #define QUANT_TOL           1e-6

    double prev_residual = 1e30;
    int converged = 0;

    for (int it = 0; it < max_iter && !converged; it++) {
        double max_delta = 0.0;

        double anneal_alpha = (it < QUANT_COOL_ITERS)
            ? QUANT_DAMPING_START * exp(log(QUANT_DAMPING_END / QUANT_DAMPING_START) * ((double)it / QUANT_COOL_ITERS))
            : QUANT_DAMPING_END;

        for (int eid = 0; eid < n_edges; eid++) {
            const HPCEdge *edge = &g->edges[eid];
            uint64_t sa = edge->site_a, sb = edge->site_b;

            for (int dir = 0; dir < 2; dir++) {
                uint64_t src = (dir == 0) ? sa : sb;
                double prod_re[6], prod_im[6];

                for (int v_src = 0; v_src < 6; v_src++) {
                    prod_re[v_src] = g->locals[src].edge_re[v_src];
                    prod_im[v_src] = g->locals[src].edge_im[v_src];

                    const HPCAdjList *adj = &g->adj[src];
                    for (uint64_t mi = 0; mi < adj->count; mi++) {
                        uint64_t in_eid = adj->edge_ids[mi];
                        if (in_eid == (uint64_t)eid) continue;

                        int in_dir = (g->edges[in_eid].site_b == src) ? 0 : 1;
                        double mr = msgs[in_eid].re[in_dir][v_src];
                        double mi_v = msgs[in_eid].im[in_dir][v_src];

                        double nr = prod_re[v_src] * mr - prod_im[v_src] * mi_v;
                        double ni = prod_re[v_src] * mi_v + prod_im[v_src] * mr;
                        prod_re[v_src] = nr;
                        prod_im[v_src] = ni;
                    }
                }

                /* DFT₆ sum-product for CZ edges */
                double new_re[6], new_im[6];
                for (int vb = 0; vb < 6; vb++) {
                    double sum_re = 0.0, sum_im = 0.0;
                    for (int va = 0; va < 6; va++) {
                        int pidx = (va * vb) % 6;
                        double w_re = W6_RE[pidx], w_im = W6_IM[pidx];
                        sum_re += prod_re[va] * w_re - prod_im[va] * w_im;
                        sum_im += prod_re[va] * w_im + prod_im[va] * w_re;
                    }
                    new_re[vb] = sum_re;
                    new_im[vb] = sum_im;
                }

                /* Normalize */
                double norm_sq = 0.0;
                for (int v = 0; v < 6; v++)
                    norm_sq += new_re[v]*new_re[v] + new_im[v]*new_im[v];
                if (norm_sq > 1e-30) {
                    double inv_norm = 1.0 / sqrt(norm_sq);
                    for (int v = 0; v < 6; v++) {
                        new_re[v] *= inv_norm;
                        new_im[v] *= inv_norm;
                    }
                } else {
                    for (int v = 0; v < 6; v++) {
                        new_re[v] = 1.0 / sqrt(6.0);
                        new_im[v] = 0.0;
                    }
                }

                /* Damped update */
                double delta = 0.0;
                for (int v = 0; v < 6; v++) {
                    double upd_re = anneal_alpha * new_re[v] +
                                    (1.0 - anneal_alpha) * msgs[eid].re[dir][v];
                    double upd_im = anneal_alpha * new_im[v] +
                                    (1.0 - anneal_alpha) * msgs[eid].im[dir][v];
                    double dr = upd_re - msgs[eid].re[dir][v];
                    double di = upd_im - msgs[eid].im[dir][v];
                    delta += dr*dr + di*di;
                    msgs[eid].re[dir][v] = upd_re;
                    msgs[eid].im[dir][v] = upd_im;
                }
                if (delta > max_delta) max_delta = delta;
            }
        }

        if (max_delta < QUANT_TOL) converged = 1;
        prev_residual = max_delta;
    }

    /* Compute dressed amplitudes */
    for (int s = 0; s < n_sites; s++) {
        for (int v = 0; v < 6; v++) {
            double d_re = g->locals[s].edge_re[v];
            double d_im = g->locals[s].edge_im[v];

            const HPCAdjList *adj = &g->adj[s];
            for (uint64_t mi = 0; mi < adj->count; mi++) {
                uint64_t in_eid = adj->edge_ids[mi];
                int in_dir = (g->edges[in_eid].site_b == (uint64_t)s) ? 0 : 1;

                double mr = msgs[in_eid].re[in_dir][v];
                double mi_v = msgs[in_eid].im[in_dir][v];
                double nr = d_re * mr - d_im * mi_v;
                double ni = d_re * mi_v + d_im * mr;
                d_re = nr;
                d_im = ni;
            }

            ms->sheets[s].dressed_re[v] = d_re;
            ms->sheets[s].dressed_im[v] = d_im;
        }
    }

    free(msgs);
    return prev_residual;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HPC-OPTIMIZED Q4_0 QUANTIZATION (for attention tensors)
 *
 * Same architecture as Q2_K HPC pipeline, but simpler:
 *   - One parameter per block (scale d only, no dmin)
 *   - Single quhit per block (6 states)
 *   - 10 candidate scales → bin to 6 for BP
 *   - 12-beam Hensel search for globally optimal configuration
 *   - Triality 3-view marginals for robust scoring
 *
 * Q4_0 block: 32 weights, 16 levels (0–15), dequant: w = (q - 8) * d
 * ═══════════════════════════════════════════════════════════════════════════ */

#define Q4_N_CAND 16  /* scale candidates for Q4_0 (was 10) */
#define Q4_N_BEAMS 24 /* beam width (was 12) */

/* Tight neighborhood around WLS optimum: ±10% */
static const float Q4_NEIGHBOR_MULTS[Q4_N_CAND] = {
    0.900f, 0.915f, 0.930f, 0.945f, 0.955f, 0.965f, 0.975f, 0.985f,
    0.995f, 1.005f, 1.015f, 1.025f, 1.035f, 1.050f, 1.070f, 1.100f
};
static const int Q4_CAND_TO_QUHIT[Q4_N_CAND] = {
    0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5
};

static void quantize_tensor_q4_0_hpc(const float *weights, int64_t n_elements,
                                       BlockQ4_0 *output, float *out_total_error,
                                       const float *imat_importance, int verbose)
{
    int64_t n_blocks = n_elements / QK4_0;
    float total_err = 0.0f;

    /* ── Phase 1: Greedy seed — compute scale per block ── */
    float *greedy_d = (float *)calloc(n_blocks, sizeof(float));

    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const float *bw = weights + blk * QK4_0;
        float amax = 0.0f;
        for (int j = 0; j < QK4_0; j++) {
            float av = fabsf(bw[j]);
            if (av > amax) amax = av;
        }
        greedy_d[blk] = amax / 7.0f;
    }

    /* ── Phase 2: WLS-Optimal Candidate Generation for Q4_0 ──
     * First find the true optimal d* via 3-iteration WLS,
     * then generate candidates centered on d* with tight spacing. */
    float (*cand_errors)[Q4_N_CAND] = (float (*)[Q4_N_CAND])
        calloc(n_blocks, sizeof(float[Q4_N_CAND]));
    uint16_t (*cand_d16)[Q4_N_CAND] = (uint16_t (*)[Q4_N_CAND])
        calloc(n_blocks, sizeof(uint16_t[Q4_N_CAND]));

    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const float *bw = weights + blk * QK4_0;

        /* ── Step 2a: WLS solve to find optimal d* ── */
        float wls_d = greedy_d[blk];
        for (int ls_iter = 0; ls_iter < 3; ls_iter++) {
            if (wls_d < 1e-15f) break;
            float inv_d = 1.0f / wls_d;
            float num = 0.0f, den = 0.0f;
            for (int j = 0; j < QK4_0; j++) {
                int q = (int)(bw[j] * inv_d + 8.5f);
                if (q < 0) q = 0; if (q > 15) q = 15;
                float qc = (float)q - 8.0f;
                float w = (imat_importance) ?
                          imat_importance[blk * QK4_0 + j] : 1.0f;
                num += w * bw[j] * qc;
                den += w * qc * qc;
            }
            if (den > 1e-15f) {
                float d_new = num / den;
                if (fabsf(d_new) < 4.0f * (greedy_d[blk] + 1e-10f))
                    wls_d = gguf_fp16_to_fp32(gguf_fp32_to_fp16(d_new));
            }
        }

        /* ── Step 2b: Generate candidates centered on WLS optimum ── */
        for (int ci = 0; ci < Q4_N_CAND; ci++) {
            float trial_d = wls_d * Q4_NEIGHBOR_MULTS[ci];
            uint16_t d16 = gguf_fp32_to_fp16(trial_d);
            float actual_d = gguf_fp16_to_fp32(d16);
            cand_d16[blk][ci] = d16;

            float id = (actual_d > 1e-15f) ? 1.0f / actual_d : 0.0f;
            float err = 0.0f;

            for (int j = 0; j < QK4_0; j++) {
                float x = bw[j];
                int q = (int)(x * id + 8.5f);
                if (q < 0) q = 0; if (q > 15) q = 15;
                float deq = ((float)q - 8.0f) * actual_d;
                float diff = x - deq;
                float w = (imat_importance) ?
                          imat_importance[blk * QK4_0 + j] : 1.0f;
                err += diff * diff * w;
            }
            cand_errors[blk][ci] = err;
        }
    }

    /* ── Phase 3: HPC graph — single quhit per block ── */
    int *best_candidate = (int *)malloc(n_blocks * sizeof(int));
    for (int64_t i = 0; i < n_blocks; i++)
        best_candidate[i] = 10;  /* Q4_NEIGHBOR_MULTS[10] = 1.00 */

    if (n_blocks >= 2) {
        float temperature = 0.5f;
        int64_t graph_blocks = (n_blocks > 200) ? 200 : n_blocks;
        int64_t stride = n_blocks / graph_blocks;
        int64_t n_sites = graph_blocks;  /* 1 quhit per block */

        HPCGraph *graph = hpc_create(n_sites);
        if (graph) {
            for (int64_t i = 0; i < n_sites; i++)
                triality_dft(&graph->locals[i]);

            /* Encode candidate errors as Boltzmann amplitudes */
            for (int64_t i = 0; i < graph_blocks; i++) {
                int64_t blk = i * stride;
                float min_err = 1e30f;
                for (int c = 0; c < Q4_N_CAND; c++)
                    if (cand_errors[blk][c] < min_err)
                        min_err = cand_errors[blk][c];

                double amp_re[6];
                double amp_norm = 0.0;
                for (int qi = 0; qi < 6; qi++) amp_re[qi] = 0.0;
                for (int ci = 0; ci < Q4_N_CAND; ci++) {
                    int qi = Q4_CAND_TO_QUHIT[ci];
                    amp_re[qi] += exp(-(double)(cand_errors[blk][ci] - min_err) /
                                      (2.0 * (double)temperature));
                }
                for (int qi = 0; qi < 6; qi++)
                    amp_norm += amp_re[qi] * amp_re[qi];
                if (amp_norm > 1e-30) {
                    double inv = 1.0 / sqrt(amp_norm);
                    for (int v = 0; v < 6; v++) amp_re[v] *= inv;
                }

                for (int v = 0; v < 6; v++) {
                    graph->locals[i].edge_re[v] = amp_re[v];
                    graph->locals[i].edge_im[v] = 0.0;
                }
                graph->locals[i].primary = VIEW_EDGE;
                graph->locals[i].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
                graph->locals[i].delta_valid = 0;
                triality_update_mask(&graph->locals[i]);
            }

            /* Neighbor edges */
            for (int64_t i = 0; i < graph_blocks - 1; i++)
                hpc_cz(graph, i, i + 1);

            /* Ensemble BP with triality marginals */
            MobiusAmplitudeSheet *mobius = mobius_create(graph);
            double (*marg)[6] = (double (*)[6])calloc(graph_blocks, sizeof(double[6]));

            for (int start = 0; start < 3; start++) {
                z6_complex_amplitude_bp(mobius, (unsigned int)start, 200);
                for (int64_t i = 0; i < graph_blocks; i++) {
                    /* Triality 3-view marginal */
                    double edge_prob[6], vert_re[6], vert_im[6], vert_prob[6], diag_prob[6];
                    for (int v = 0; v < 6; v++) {
                        double re = mobius->sheets[i].dressed_re[v];
                        double im = mobius->sheets[i].dressed_im[v];
                        edge_prob[v] = re * re + im * im;
                    }
                    for (int j = 0; j < 6; j++) {
                        double sr = 0, si = 0;
                        for (int k = 0; k < 6; k++) {
                            int idx = (j * k) % 6;
                            sr += mobius->sheets[i].dressed_re[k] * W6_RE[idx]
                                - mobius->sheets[i].dressed_im[k] * W6_IM[idx];
                            si += mobius->sheets[i].dressed_re[k] * W6_IM[idx]
                                + mobius->sheets[i].dressed_im[k] * W6_RE[idx];
                        }
                        vert_re[j] = sr * INV_SQRT6;
                        vert_im[j] = si * INV_SQRT6;
                        vert_prob[j] = vert_re[j]*vert_re[j] + vert_im[j]*vert_im[j];
                    }
                    for (int j = 0; j < 6; j++) {
                        double sr = 0, si = 0;
                        for (int k = 0; k < 6; k++) {
                            int idx = (j * k) % 6;
                            sr += vert_re[k] * W6_RE[idx] - vert_im[k] * W6_IM[idx];
                            si += vert_re[k] * W6_IM[idx] + vert_im[k] * W6_RE[idx];
                        }
                        double dr = sr * INV_SQRT6, di = si * INV_SQRT6;
                        diag_prob[j] = dr*dr + di*di;
                    }
                    for (int v = 0; v < 6; v++)
                        marg[i][v] += cbrt(edge_prob[v] * vert_prob[v] * diag_prob[v]);
                }
            }

            /* Beam search over candidates */
            typedef struct { double acc_error; int history_idx; } Q4Beam;
            typedef struct { int cand_idx; int parent_idx; } Q4BeamHistory;

            Q4Beam beams[Q4_N_BEAMS];
            int active_beams = 1;
            Q4BeamHistory *history = (Q4BeamHistory *)malloc(n_blocks * Q4_N_BEAMS * sizeof(Q4BeamHistory));

            for (int b = 0; b < Q4_N_BEAMS; b++) {
                beams[b].acc_error = 0.0;
                beams[b].history_idx = -1;
            }

            for (int64_t i = 0; i < graph_blocks; i++) {
                double m_total = 0.0;
                for (int v = 0; v < 6; v++) m_total += marg[i][v];

                double cand_score[Q4_N_CAND];
                int64_t blk = i * stride;
                /* Count candidates per quhit bin for normalization */
                int q4_bin_count[6] = {0};
                for (int ci = 0; ci < Q4_N_CAND; ci++)
                    q4_bin_count[Q4_CAND_TO_QUHIT[ci]]++;
                /* Per-block error normalization: divide by block mean error
                 * so small-weight blocks don't dominate beam selection */
                float blk_mean_err = 0.0f;
                for (int ci = 0; ci < Q4_N_CAND; ci++)
                    blk_mean_err += cand_errors[blk][ci];
                blk_mean_err /= (float)Q4_N_CAND;
                if (blk_mean_err < 1e-30f) blk_mean_err = 1e-30f;
                for (int ci = 0; ci < Q4_N_CAND; ci++) {
                    int qi = Q4_CAND_TO_QUHIT[ci];
                    double p = (m_total > 1e-30) ? marg[i][qi] / m_total : 1.0/6.0;
                    p /= (double)q4_bin_count[qi];  /* normalize by bin occupancy */
                    cand_score[ci] = p / (cand_errors[blk][ci] / blk_mean_err + 1e-15);
                }

                typedef struct { double score; int beam_idx; int cand_idx; } Q4Ext;
                Q4Ext extensions[Q4_N_BEAMS * Q4_N_CAND];
                int n_ext = 0;
                for (int b = 0; b < active_beams; b++) {
                    for (int c = 0; c < Q4_N_CAND; c++) {
                        double ext_err = beams[b].acc_error + cand_errors[blk][c];
                        extensions[n_ext].score = cand_score[c] / (ext_err + 1e-15);
                        extensions[n_ext].beam_idx = b;
                        extensions[n_ext].cand_idx = c;
                        n_ext++;
                    }
                }

                int top_k = (n_ext < Q4_N_BEAMS) ? n_ext : Q4_N_BEAMS;
                int top_indices[Q4_N_BEAMS];
                for (int k = 0; k < top_k; k++) {
                    int best = -1; double best_s = -1e30;
                    for (int e = 0; e < n_ext; e++) {
                        if (extensions[e].score > best_s) {
                            best_s = extensions[e].score; best = e;
                        }
                    }
                    top_indices[k] = best;
                    extensions[best].score = -2e30;
                }

                Q4Beam new_beams[Q4_N_BEAMS];
                for (int k = 0; k < top_k; k++) {
                    int ei = top_indices[k];
                    int sb = extensions[ei].beam_idx;
                    int cand = extensions[ei].cand_idx;

                    int hist_idx = i * Q4_N_BEAMS + k;
                    history[hist_idx].cand_idx = cand;
                    history[hist_idx].parent_idx = beams[sb].history_idx;

                    new_beams[k].history_idx = hist_idx;
                    new_beams[k].acc_error = beams[sb].acc_error + cand_errors[blk][cand];
                }
                for (int k = 0; k < top_k; k++) beams[k] = new_beams[k];
                active_beams = top_k;
            }

            int curr_hist = beams[0].history_idx;
            for (int64_t i = graph_blocks - 1; i >= 0; i--) {
                if (curr_hist >= 0) {
                    int cidx = history[curr_hist].cand_idx;
                    for (int64_t b = i * stride; b < (i+1) * stride && b < n_blocks; b++)
                        best_candidate[b] = cidx;
                    curr_hist = history[curr_hist].parent_idx;
                } else {
                    for (int64_t b = i * stride; b < (i+1) * stride && b < n_blocks; b++)
                        best_candidate[b] = 10;
                }
            }
            free(history);

            /* ══════════════════════════════════════════════════════════════
             * Phase 3.5: Born-Rule Multi-Shot Scale Refinement
             *
             * The beam search found the MAP candidate sequence. But the
             * triality marginals encode quantum phase-coherent structure
             * that a greedy beam can miss.
             *
             * Like tesseract_factor's MCMC period recovery (lines 1920-1964):
             *   1. Take N independent Born samples from triality marginals
             *   2. Each sample → full candidate assignment across all blocks
             *   3. Evaluate actual RMSE for each assignment
             *   4. Keep assignment with lowest total RMSE
             *
             * Reuses the EXISTING converged Möbius sheet — zero new BP.
             * ══════════════════════════════════════════════════════════════ */
            {
                #define Q4_BORN_SHOTS 64

                /* Compute beam-search baseline RMSE for comparison */
                float beam_total_err = 0.0f;
                for (int64_t bi = 0; bi < n_blocks; bi++)
                    beam_total_err += cand_errors[bi][best_candidate[bi]];

                /* Build per-block CDFs from triality marginals */
                unsigned int born_rng = 314159;
                int *shot_assignment = (int *)malloc(n_blocks * sizeof(int));

                for (int shot = 0; shot < Q4_BORN_SHOTS; shot++) {
                    float shot_err = 0.0f;
                    /* Init from beam result so tail blocks beyond
                     * graph_blocks*stride keep valid indices */
                    memcpy(shot_assignment, best_candidate,
                           n_blocks * sizeof(int));

                    for (int64_t gi = 0; gi < graph_blocks; gi++) {
                        /* Normalize marginals to CDF */
                        double m_total = 0.0;
                        for (int v = 0; v < 6; v++) m_total += marg[gi][v];

                        /* Born sample: CDF inversion (same as born_sample) */
                        born_rng = born_rng * 1664525u + 1013904223u;
                        double rnd = (double)(born_rng >> 8) / 16777216.0;
                        double target = rnd * m_total;
                        double cum = 0.0;
                        int sampled_qi = 5;
                        for (int v = 0; v < 6; v++) {
                            cum += marg[gi][v];
                            if (cum > target) { sampled_qi = v; break; }
                        }

                        /* Find the best candidate WITHIN this quhit bin */
                        int64_t blk = gi * stride;
                        float best_bin_err = 1e30f;
                        int best_bin_cand = 10; /* default */
                        for (int ci = 0; ci < Q4_N_CAND; ci++) {
                            if (Q4_CAND_TO_QUHIT[ci] == sampled_qi) {
                                if (cand_errors[blk][ci] < best_bin_err) {
                                    best_bin_err = cand_errors[blk][ci];
                                    best_bin_cand = ci;
                                }
                            }
                        }

                        shot_assignment[blk] = best_bin_cand;
                        shot_err += cand_errors[blk][best_bin_cand];
                    }

                    /* Metropolis acceptance: adopt if better than current best */
                    if (shot_err < beam_total_err) {
                        for (int64_t b = 0; b < n_blocks; b++)
                            best_candidate[b] = shot_assignment[b];
                        beam_total_err = shot_err;
                    }
                }

                free(shot_assignment);
            }

            free(marg);
            mobius_destroy(mobius);
            hpc_destroy(graph);
        }
    }

    /* ══════════════════════════════════════════════════════════════════
     * PHASE 4: Assemble blocks via least-squares scale extraction
     *
     * The factorer assembles a frequency register from BP marginals,
     * then EXTRACTS the exact period via continued fractions.
     *
     * We do the same: the beam search / Born shots selected a grid
     * candidate (the "assembled frequency"). Now we EXTRACT the exact
     * optimal FP16 scale via weighted least-squares (the "CF step").
     *
     * For Q4_0:  d_optimal = Σ(w_j × x_j × q̃_j) / Σ(w_j × q̃_j²)
     * where q̃_j = (q_j - 8) and q_j is quantized at the grid scale.
     *
     * This iterates: quantize at d_init → compute d_optimal → re-quantize
     * → re-compute until convergence. 3 iterations suffice since Q4_0
     * has only 16 levels — the assignment stabilizes immediately.
     *
     * The grid gave us 16 possible scales. This gives us 65,536 (all FP16).
     * ══════════════════════════════════════════════════════════════════ */

    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_err)
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const float *bw = weights + blk * QK4_0;
        int cidx = best_candidate[blk];

        /* Start from the grid-selected scale (the "assembled frequency") */
        float d_current = gguf_fp16_to_fp32(cand_d16[blk][cidx]);

        /* Analog assembly: iterate to full convergence.
         * 5 iterations for stable (d, q-values) coupling. */
        for (int ls_iter = 0; ls_iter < 5; ls_iter++) {
            if (d_current < 1e-15f) break;
            float id = 1.0f / d_current;

            /* Quantize at current scale */
            int qs_tmp[QK4_0];
            for (int j = 0; j < QK4_0; j++) {
                int q = (int)(bw[j] * id + 8.5f);
                if (q < 0) q = 0; if (q > 15) q = 15;
                qs_tmp[j] = q;
            }

            /* Weighted least-squares: d = Σ(w × x × q̃) / Σ(w × q̃²)
             * where q̃ = q - 8 (centered quantized value) */
            float num = 0.0f, den = 0.0f;
            for (int j = 0; j < QK4_0; j++) {
                float q_centered = (float)qs_tmp[j] - 8.0f;
                float w = (imat_importance) ?
                          imat_importance[blk * QK4_0 + j] : 1.0f;
                num += w * bw[j] * q_centered;
                den += w * q_centered * q_centered;
            }

            if (den > 1e-15f) {
                float d_new = num / den;
                /* Clamp magnitude to prevent runaway (Q4_0 d can be negative) */
                float d_seed = gguf_fp16_to_fp32(cand_d16[blk][cidx]);
                if (fabsf(d_new) < 4.0f * (fabsf(d_seed) + 1e-10f)) {
                    uint16_t d16 = gguf_fp32_to_fp16(d_new);
                    d_current = gguf_fp16_to_fp32(d16);
                }
            }
        }

        /* ── FP16 ULP neighborhood search + sign-flip exploration ──
         * The WLS solve found the continuous-optimal d. But FP16 truncation
         * may shift the optimum. Try ±4 ULP around d in FP16 space, plus
         * the negated scale, and pick the one with minimum reconstruction error. */
        {
            uint16_t base_d16 = gguf_fp32_to_fp16(d_current);
            uint16_t best_d16 = base_d16;
            float best_ulp_err = 1e30f;

            /* Try ±4 ULP neighborhood + sign flip = up to 17 candidates */
            uint16_t ulp_candidates[17];
            int n_ulp = 0;
            for (int delta = -4; delta <= 4; delta++) {
                int cand16 = (int)base_d16 + delta;
                if (cand16 >= 0 && cand16 <= 0x7BFF) /* valid positive FP16 */
                    ulp_candidates[n_ulp++] = (uint16_t)cand16;
            }
            /* Sign-flipped d: negate and try ±0 ULP */
            {
                float neg_d = -d_current;
                uint16_t neg_d16 = gguf_fp32_to_fp16(neg_d);
                ulp_candidates[n_ulp++] = neg_d16;
            }

            for (int ui = 0; ui < n_ulp; ui++) {
                float trial_d = gguf_fp16_to_fp32(ulp_candidates[ui]);
                float trial_id = (fabsf(trial_d) > 1e-15f) ? 1.0f / trial_d : 0.0f;
                float err = 0.0f;
                for (int j = 0; j < QK4_0; j++) {
                    int q = (int)(bw[j] * trial_id + 8.5f);
                    if (q < 0) q = 0; if (q > 15) q = 15;
                    float deq = ((float)q - 8.0f) * trial_d;
                    float w = (imat_importance) ? imat_importance[blk * QK4_0 + j] : 1.0f;
                    err += (bw[j] - deq) * (bw[j] - deq) * w;
                }
                if (err < best_ulp_err) {
                    best_ulp_err = err;
                    best_d16 = ulp_candidates[ui];
                }
            }
            d_current = gguf_fp16_to_fp32(best_d16);
        }

        /* Store the extracted optimal FP16 scale */
        output[blk].d = gguf_fp32_to_fp16(d_current);
        float actual_d = d_current;
        float id = (fabsf(actual_d) > 1e-15f) ? 1.0f / actual_d : 0.0f;

        /* ── D₆ Hadamard Error Shaping for Q4_0 ──
         * 32 elements per block = 5 full D₆ groups of 6 + 2 tail.
         * Apply the same antipodal fold as Q2_K: minimize vesica energy
         * to push quantization noise into wave (high-frequency) modes
         * that cancel in dot products. */

        /* Step 1: Standard nearest-rounding as baseline */
        int q_base[QK4_0], q_shaped[QK4_0];
        float q_cont[QK4_0];
        for (int j = 0; j < QK4_0; j++) {
            q_cont[j] = bw[j] * id + 8.0f;
            q_base[j] = (int)(q_cont[j] + 0.5f);
            if (q_base[j] < 0) q_base[j] = 0;
            if (q_base[j] > 15) q_base[j] = 15;
        }
        memcpy(q_shaped, q_base, QK4_0 * sizeof(int));

        /* Step 2: D₆ greedy flipping on 5 groups of 6 */
        for (int g = 0; g < 5; g++) {
            int g_off = g * 6;

            for (int pass = 0; pass < 6; pass++) {
                int best_k = -1;
                int best_q_alt = 0;
                float best_delta = 0.0f;

                /* Current group errors */
                float e_cur[6];
                for (int kk = 0; kk < 6; kk++) {
                    float deq = ((float)q_shaped[g_off+kk] - 8.0f) * actual_d;
                    e_cur[kk] = bw[g_off+kk] - deq;
                }

                /* Current D₆ metric: vesica energy + DC² */
                float vesica_cur = 0.0f, dc_cur = 0.0f;
                for (int p = 0; p < 3; p++) {
                    float v = e_cur[p] + e_cur[p+3];
                    vesica_cur += v * v;
                }
                for (int kk = 0; kk < 6; kk++) dc_cur += e_cur[kk];
                float metric_cur = 4.0f * vesica_cur + dc_cur * dc_cur;

                /* Try flipping each element */
                for (int k = 0; k < 6; k++) {
                    int idx = g_off + k;
                    int q_cur = q_shaped[idx];

                    int q_try;
                    if (q_cont[idx] - (float)q_cur >= 0) {
                        q_try = q_cur + 1;
                    } else {
                        q_try = q_cur - 1;
                    }
                    if (q_try < 0 || q_try > 15) continue;

                    /* Alt errors */
                    float e_alt[6];
                    for (int kk = 0; kk < 6; kk++) e_alt[kk] = e_cur[kk];
                    float deq_try = ((float)q_try - 8.0f) * actual_d;
                    e_alt[k] = bw[idx] - deq_try;

                    /* Alt D₆ metric */
                    float vesica_alt = 0.0f, dc_alt = 0.0f;
                    for (int p = 0; p < 3; p++) {
                        float v = e_alt[p] + e_alt[p+3];
                        vesica_alt += v * v;
                    }
                    for (int kk = 0; kk < 6; kk++) dc_alt += e_alt[kk];
                    float metric_alt = 4.0f * vesica_alt + dc_alt * dc_alt;

                    float delta = metric_cur - metric_alt;
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_k = k;
                        best_q_alt = q_try;
                    }
                }

                if (best_k < 0) break;
                q_shaped[g_off + best_k] = best_q_alt;
            }
        }

        /* Step 3: Error comparison — keep shaped only if MSE doesn't worsen >5% */
        float err_base = 0.0f, err_shaped = 0.0f;
        for (int j = 0; j < QK4_0; j++) {
            float w = (imat_importance) ? imat_importance[blk * QK4_0 + j] : 1.0f;
            float deq_b = ((float)q_base[j] - 8.0f) * actual_d;
            float deq_s = ((float)q_shaped[j] - 8.0f) * actual_d;
            err_base += (bw[j] - deq_b) * (bw[j] - deq_b) * w;
            err_shaped += (bw[j] - deq_s) * (bw[j] - deq_s) * w;
        }
        int *q_final = (err_shaped <= err_base * 1.05f) ? q_shaped : q_base;

        /* Pack nibbles and compute error */
        for (int j = 0; j < QK4_0 / 2; j++) {
            int q0 = q_final[j];
            int q1 = q_final[j + QK4_0/2];
            output[blk].qs[j] = (uint8_t)(q0 | (q1 << 4));

            float deq0 = ((float)q0 - 8.0f) * actual_d;
            float deq1 = ((float)q1 - 8.0f) * actual_d;
            total_err += (bw[j] - deq0) * (bw[j] - deq0) + (bw[j + QK4_0/2] - deq1) * (bw[j + QK4_0/2] - deq1);
        }
    }

    *out_total_error = total_err;
    free(greedy_d);
    free(cand_errors);
    free(cand_d16);
    free(best_candidate);
}

static void quantize_tensor_q2k_hpc(const float *weights, int64_t n_elements,
                                      BlockQ2K *output, float *out_total_error,
                                      OptimizerMode opt_mode,
                                      const float *imat_importance,
                                      int verbose)
{
    int64_t n_blocks = n_elements / QK_K;
    float total_err = 0.0f;
    const int N_SUB = QK_K / 16;

    init_scale_table();

    /* ══════════════════════════════════════════════════════════════════
     * PHASE 1: Greedy quantization — produce seed (d, dmin) per block
     * ══════════════════════════════════════════════════════════════════ */

    /* Store Phase A/B results for all blocks */
    typedef struct {
        float dm, mm;                 /* greedy d, dmin (fp32) */
        uint16_t d_fp16, dmin_fp16;   /* greedy d, dmin (fp16) */
        uint8_t Ls[16], Lm[16];       /* sub-block scale/min indices */
        float scales[16], mins[16], sw[16];
    } BlockSeed;

    BlockSeed *seeds = (BlockSeed *)calloc(n_blocks, sizeof(BlockSeed));

    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const float *block_x = weights + blk * QK_K;
        uint8_t L[QK_K], Laux[16];
        float wt[16];

        float sumx2 = 0;
        for (int i = 0; i < QK_K; i++) sumx2 += block_x[i] * block_x[i];
        float sigma2 = sumx2 / (float)QK_K;

        for (int j = 0; j < N_SUB; j++) {
            const float *sx = block_x + 16 * j;
            seeds[blk].sw[j] = 0;
            for (int l = 0; l < 16; l++) {
                float imp = (imat_importance) ? imat_importance[blk * QK_K + 16 * j + l] : 1.0f;
                wt[l] = imp * sqrtf(sigma2 + sx[l] * sx[l]);
                seeds[blk].sw[j] += wt[l];
            }
            seeds[blk].scales[j] = hpc_make_qkx2_quants(16, 3, sx, wt,
                                        L + 16 * j, &seeds[blk].mins[j], Laux);
        }

        seeds[blk].dm = hpc_make_qp_quants(N_SUB, 15, seeds[blk].scales,
                                              seeds[blk].Ls, seeds[blk].sw);
        seeds[blk].mm = hpc_make_qp_quants(N_SUB, 15, seeds[blk].mins,
                                              seeds[blk].Lm, seeds[blk].sw);
        seeds[blk].d_fp16 = gguf_fp32_to_fp16(seeds[blk].dm);
        seeds[blk].dmin_fp16 = gguf_fp32_to_fp16(seeds[blk].mm);
    }

    /* ══════════════════════════════════════════════════════════════════
     * PHASE 2: WLS-Optimal Candidate Generation
     *
     * Instead of a fixed multiplier grid centered on greedy seeds,
     * we first solve a 3-iteration Weighted Least-Squares to find
     * the true optimal (d*, dmin*) per block, then generate the
     * 16×16 candidate grid centered on THOSE optimal values.
     * This makes the candidate space data-driven, not fabricated.
     * ══════════════════════════════════════════════════════════════════ */

    /* Tight neighborhood around WLS optimum: ±10% with fine spacing */
    static const float NEIGHBOR_MULTS_D[N_CAND_D] = {
        0.900f, 0.915f, 0.930f, 0.945f, 0.955f, 0.965f, 0.975f, 0.985f,
        0.995f, 1.005f, 1.015f, 1.025f, 1.035f, 1.050f, 1.070f, 1.100f
    };
    static const float NEIGHBOR_MULTS_M[N_CAND_M] = {
        0.900f, 0.915f, 0.930f, 0.945f, 0.955f, 0.965f, 0.975f, 0.985f,
        0.995f, 1.005f, 1.015f, 1.025f, 1.035f, 1.050f, 1.070f, 1.100f
    };
    /* Map 16 candidates → 6 quhit states for BP encoding */
    static const int CAND_TO_QUHIT[16] = {
        0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5
    };

    /* candidate_errors[blk][256] — weighted MSE per candidate */
    float (*candidate_errors)[TOTAL_SCALE_CANDIDATES] = NULL;
    uint16_t (*candidate_d)[TOTAL_SCALE_CANDIDATES] = NULL;
    uint16_t (*candidate_dmin)[TOTAL_SCALE_CANDIDATES] = NULL;
    /* Per-candidate Ls/Lm — must recompute for each (d, dmin) */
    uint8_t (*candidate_Ls)[TOTAL_SCALE_CANDIDATES][16] = NULL;
    uint8_t (*candidate_Lm)[TOTAL_SCALE_CANDIDATES][16] = NULL;

    candidate_errors = (float (*)[TOTAL_SCALE_CANDIDATES])calloc(n_blocks,
                            sizeof(float[TOTAL_SCALE_CANDIDATES]));
    candidate_d = (uint16_t (*)[TOTAL_SCALE_CANDIDATES])calloc(n_blocks,
                            sizeof(uint16_t[TOTAL_SCALE_CANDIDATES]));
    candidate_dmin = (uint16_t (*)[TOTAL_SCALE_CANDIDATES])calloc(n_blocks,
                            sizeof(uint16_t[TOTAL_SCALE_CANDIDATES]));
    candidate_Ls = (uint8_t (*)[TOTAL_SCALE_CANDIDATES][16])calloc(n_blocks,
                            sizeof(uint8_t[TOTAL_SCALE_CANDIDATES][16]));
    candidate_Lm = (uint8_t (*)[TOTAL_SCALE_CANDIDATES][16])calloc(n_blocks,
                            sizeof(uint8_t[TOTAL_SCALE_CANDIDATES][16]));

    #pragma omp parallel for schedule(dynamic, 16)
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const float *block_x = weights + blk * QK_K;

        /* ── Step 2a: WLS solve to find optimal (d*, dmin*) ──
         * Seed from Phase 1 greedy, iterate 3× to converge.
         * Q2_K model: x[j,k] ≈ d × Ls[j] × q[j,k] - dmin × Lm[j]
         * This is a 2-variable WLS: minimize Σ w×(x - d×a + dmin×b)² */
        float wls_dm = seeds[blk].dm;
        float wls_mm = seeds[blk].mm;
        uint8_t wls_Ls[16], wls_Lm[16];
        memcpy(wls_Ls, seeds[blk].Ls, 16);
        memcpy(wls_Lm, seeds[blk].Lm, 16);

        for (int ls_iter = 0; ls_iter < 5; ls_iter++) {
            /* Quantize all elements at current (wls_dm, wls_mm) */
            uint8_t L_wls[QK_K];
            for (int j = 0; j < N_SUB; j++) {
                float d_sub = wls_dm * (float)wls_Ls[j];
                float m_sub = wls_mm * (float)wls_Lm[j];
                if (d_sub < 1e-15f) {
                    for (int k = 0; k < 16; k++) L_wls[16*j+k] = 0;
                    continue;
                }
                for (int k = 0; k < 16; k++) {
                    int q = gguf_nearest_int((block_x[16*j+k] + m_sub) / d_sub);
                    if (q < 0) q = 0; if (q > 3) q = 3;
                    L_wls[16*j+k] = (uint8_t)q;
                }
            }

            /* Accumulate 2×2 normal equations */
            double Saa = 0, Sab = 0, Sbb = 0, Sxa = 0, Sxb = 0;
            for (int j = 0; j < N_SUB; j++) {
                float ls_f = (float)wls_Ls[j];
                float lm_f = (float)wls_Lm[j];
                for (int k = 0; k < 16; k++) {
                    float x = block_x[16*j+k];
                    float w = (imat_importance) ?
                              imat_importance[blk * QK_K + 16*j+k] : 1.0f;
                    float a = ls_f * (float)L_wls[16*j+k];
                    float b = lm_f;
                    Saa += w * a * a;
                    Sab += w * a * b;
                    Sbb += w * b * b;
                    Sxa += w * x * a;
                    Sxb += w * x * b;
                }
            }

            /* Solve via Cramer's rule */
            double det = Saa * Sbb - Sab * Sab;
            if (fabs(det) > 1e-30) {
                double d_new  = (Sbb * Sxa - Sab * Sxb) / det;
                double dm_new = (Sab * Sxa - Saa * Sxb) / det;
                /* Clamp: positive and within 4× of seed (prevent runaway) */
                if (d_new > 0.0 && d_new < 4.0 * (seeds[blk].dm + 1e-10))
                    wls_dm = gguf_fp16_to_fp32(gguf_fp32_to_fp16((float)d_new));
                if (dm_new > 0.0 && dm_new < 4.0 * (seeds[blk].mm + 1e-10))
                    wls_mm = gguf_fp16_to_fp32(gguf_fp32_to_fp16((float)dm_new));
            }

            /* Re-derive Ls/Lm for updated (d*, dmin*) */
            for (int j = 0; j < N_SUB; j++) {
                if (wls_dm > 1e-15f) {
                    int ls = gguf_nearest_int(seeds[blk].scales[j] / wls_dm);
                    if (ls < 0) ls = 0; if (ls > 15) ls = 15;
                    wls_Ls[j] = (uint8_t)ls;
                } else { wls_Ls[j] = 0; }
                if (wls_mm > 1e-15f) {
                    int lm = gguf_nearest_int(seeds[blk].mins[j] / wls_mm);
                    if (lm < 0) lm = 0; if (lm > 15) lm = 15;
                    wls_Lm[j] = (uint8_t)lm;
                } else { wls_Lm[j] = 0; }
            }
        }

        /* ── Step 2b: Generate 16×16 candidates centered on WLS optimum ──
         * Grid is now centered on (wls_dm, wls_mm) not (greedy_dm, greedy_mm).
         * Tighter spacing because we're already near the true minimum. */
        for (int di = 0; di < N_CAND_D; di++) {
            float trial_dm = wls_dm * NEIGHBOR_MULTS_D[di];
            uint16_t trial_d16 = gguf_fp32_to_fp16(trial_dm);
            float actual_dm = gguf_fp16_to_fp32(trial_d16);

            for (int mi = 0; mi < N_CAND_M; mi++) {
                int cidx = di * N_CAND_M + mi;
                float trial_mm = wls_mm * NEIGHBOR_MULTS_M[mi];
                uint16_t trial_dmin16 = gguf_fp32_to_fp16(trial_mm);
                float actual_mm = gguf_fp16_to_fp32(trial_dmin16);

                candidate_d[blk][cidx] = trial_d16;
                candidate_dmin[blk][cidx] = trial_dmin16;

                /* Recompute Ls/Lm for THIS candidate dm/mm */
                uint8_t trial_Ls[16], trial_Lm[16];
                for (int j = 0; j < N_SUB; j++) {
                    if (actual_dm > 1e-15f) {
                        int ls = gguf_nearest_int(seeds[blk].scales[j] / actual_dm);
                        if (ls < 0) ls = 0; if (ls > 15) ls = 15;
                        trial_Ls[j] = (uint8_t)ls;
                    } else {
                        trial_Ls[j] = 0;
                    }
                    if (actual_mm > 1e-15f) {
                        int lm = gguf_nearest_int(seeds[blk].mins[j] / actual_mm);
                        if (lm < 0) lm = 0; if (lm > 15) lm = 15;
                        trial_Lm[j] = (uint8_t)lm;
                    } else {
                        trial_Lm[j] = 0;
                    }
                }
                memcpy(candidate_Ls[blk][cidx], trial_Ls, 16);
                memcpy(candidate_Lm[blk][cidx], trial_Lm, 16);

                /* Fully re-quantize and measure weighted MSE */
                float err = 0.0f;
                for (int j = 0; j < N_SUB; j++) {
                    float d = actual_dm * (float)trial_Ls[j];
                    float m = actual_mm * (float)trial_Lm[j];
                    if (d < 1e-15f) {
                        for (int k = 0; k < 16; k++) {
                            float x = block_x[16 * j + k];
                            float w = (imat_importance) ?
                                      imat_importance[blk * QK_K + 16 * j + k] : 1.0f;
                            err += x * x * w;
                        }
                        continue;
                    }
                    for (int k = 0; k < 16; k++) {
                        float x = block_x[16 * j + k];
                        int q = gguf_nearest_int((x + m) / d);
                        if (q < 0) q = 0; if (q > 3) q = 3;
                        float deq = d * (float)q - m;
                        float diff = x - deq;
                        float w = (imat_importance) ?
                                  imat_importance[blk * QK_K + 16 * j + k] : 1.0f;
                        err += diff * diff * w;
                    }
                }
                candidate_errors[blk][cidx] = err;
            }
        }
    }

    /* ══════════════════════════════════════════════════════════════════
     * PHASE 3: HPC Graph — Direct (d, dmin) Selection via BP
     *
     * Build a multi-quhit graph where each block has 2 quhits
     * encoding the 36 candidate errors. BP propagates cross-block
     * interference. Dressed marginals directly select the optimal
     * (d, dmin) per block.
     * ══════════════════════════════════════════════════════════════════ */

    /* Default: use greedy candidate (index 5*10+5 = 55, mult 1.00×1.00) */
    int *best_candidate = (int *)malloc(n_blocks * sizeof(int));
    for (int64_t i = 0; i < n_blocks; i++)
        best_candidate[i] = 10 * N_CAND_M + 10;  /* NEIGHBOR_MULTS_D[10]=1.00, _M[10]=1.00 */

    if (opt_mode != OPT_MSE && n_blocks >= 2) {
        int64_t graph_blocks = (n_blocks > 2000) ? 2000 : n_blocks;
        int64_t stride = n_blocks / graph_blocks;
        float temperature = 0.5f;
        int64_t n_sites = graph_blocks * QUHITS_PER_BLOCK;

        HPCGraph *graph = hpc_create(n_sites);
        if (graph) {
            for (int64_t i = 0; i < n_sites; i++)
                triality_dft(&graph->locals[i]);

            /* Encode each block's 36 candidate errors as dual-quhit amplitudes */
            for (int64_t i = 0; i < graph_blocks; i++) {
                int64_t blk = i * stride;
                float min_err = 1e30f;
                for (int c = 0; c < TOTAL_SCALE_CANDIDATES; c++)
                    if (candidate_errors[blk][c] < min_err)
                        min_err = candidate_errors[blk][c];

                /* Quhit 0 (coarse = d dimension): marginalize over dmin */
                double coarse_re[6];
                double coarse_norm = 0.0;
                for (int qi = 0; qi < 6; qi++) coarse_re[qi] = 0.0;
                for (int di = 0; di < N_CAND_D; di++) {
                    int qi = CAND_TO_QUHIT[di];
                    for (int mi = 0; mi < N_CAND_M; mi++) {
                        int cidx = di * N_CAND_M + mi;
                        coarse_re[qi] += exp(-(double)(candidate_errors[blk][cidx] - min_err) /
                                              (2.0 * (double)temperature));
                    }
                }
                for (int qi = 0; qi < 6; qi++)
                    coarse_norm += coarse_re[qi] * coarse_re[qi];
                if (coarse_norm > 1e-30) {
                    double inv = 1.0 / sqrt(coarse_norm);
                    for (int v = 0; v < 6; v++) coarse_re[v] *= inv;
                }

                /* Quhit 1 (fine = dmin dimension): marginalize over d */
                double fine_re[6];
                double fine_norm = 0.0;
                for (int qi = 0; qi < 6; qi++) fine_re[qi] = 0.0;
                for (int mi = 0; mi < N_CAND_M; mi++) {
                    int qi = CAND_TO_QUHIT[mi];
                    for (int di = 0; di < N_CAND_D; di++) {
                        int cidx = di * N_CAND_M + mi;
                        fine_re[qi] += exp(-(double)(candidate_errors[blk][cidx] - min_err) /
                                            (2.0 * (double)temperature));
                    }
                }
                for (int qi = 0; qi < 6; qi++)
                    fine_norm += fine_re[qi] * fine_re[qi];
                if (fine_norm > 1e-30) {
                    double inv = 1.0 / sqrt(fine_norm);
                    for (int v = 0; v < 6; v++) fine_re[v] *= inv;
                }

                /* Write quhits */
                int64_t s0 = 2 * i, s1 = 2 * i + 1;
                for (int v = 0; v < 6; v++) {
                    graph->locals[s0].edge_re[v] = coarse_re[v];
                    graph->locals[s0].edge_im[v] = 0.0;
                    graph->locals[s1].edge_re[v] = fine_re[v];
                    graph->locals[s1].edge_im[v] = 0.0;
                }
                graph->locals[s0].primary = VIEW_EDGE;
                graph->locals[s0].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
                graph->locals[s0].delta_valid = 0;
                triality_update_mask(&graph->locals[s0]);
                graph->locals[s1].primary = VIEW_EDGE;
                graph->locals[s1].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
                graph->locals[s1].delta_valid = 0;
                triality_update_mask(&graph->locals[s1]);
            }

            /* Build edges */
            for (int64_t i = 0; i < graph_blocks; i++) {
                hpc_cz(graph, 2 * i, 2 * i + 1);  /* intra-block: d ↔ dmin */
                if (i + 1 < graph_blocks) {
                    hpc_cz(graph, 2 * i, 2 * (i + 1));         /* d ↔ d neighbor */
                    hpc_cz(graph, 2 * i + 1, 2 * (i + 1) + 1); /* dmin ↔ dmin    */
                }
            }

            /* Run ensemble BP with TRIALITY-enhanced marginals.
             * After BP converges, compute marginals in ALL THREE views:
             * edge (computational), vertex (Fourier), diagonal (conj-Fourier).
             * A candidate must score well in all three bases to be selected. */
            MobiusAmplitudeSheet *mobius = mobius_create(graph);
            double (*coarse_marg)[6] = (double (*)[6])calloc(graph_blocks, sizeof(double[6]));
            double (*fine_marg)[6]   = (double (*)[6])calloc(graph_blocks, sizeof(double[6]));

            for (int start = 0; start < 3; start++) {
                z6_complex_amplitude_bp(mobius, (unsigned int)start, 200);

                /* After BP, graph->locals[s] has dressed amplitudes in edge view.
                 * Compute vertex and diagonal views via DFT₆ triality. */
                for (int64_t i = 0; i < graph_blocks; i++) {
                    for (int qbit = 0; qbit < 2; qbit++) {
                        int64_t s = 2 * i + qbit;
                        double (*marg)[6] = (qbit == 0) ? &coarse_marg[i] : &fine_marg[i];

                        /* Edge view marginals (standard) */
                        double edge_prob[6];
                        for (int v = 0; v < 6; v++) {
                            double re = mobius->sheets[s].dressed_re[v];
                            double im = mobius->sheets[s].dressed_im[v];
                            edge_prob[v] = re * re + im * im;
                        }

                        /* Vertex view: DFT₆ of dressed edge amplitudes */
                        double vert_re[6], vert_im[6], vert_prob[6];
                        for (int j = 0; j < 6; j++) {
                            double sr = 0, si = 0;
                            for (int k = 0; k < 6; k++) {
                                int idx = (j * k) % 6;
                                sr += mobius->sheets[s].dressed_re[k] * W6_RE[idx]
                                    - mobius->sheets[s].dressed_im[k] * W6_IM[idx];
                                si += mobius->sheets[s].dressed_re[k] * W6_IM[idx]
                                    + mobius->sheets[s].dressed_im[k] * W6_RE[idx];
                            }
                            vert_re[j] = sr * INV_SQRT6;
                            vert_im[j] = si * INV_SQRT6;
                            vert_prob[j] = vert_re[j]*vert_re[j] + vert_im[j]*vert_im[j];
                        }

                        /* Diagonal view: DFT₆ of vertex = DFT₆² of edge */
                        double diag_prob[6];
                        for (int j = 0; j < 6; j++) {
                            double sr = 0, si = 0;
                            for (int k = 0; k < 6; k++) {
                                int idx = (j * k) % 6;
                                sr += vert_re[k] * W6_RE[idx] - vert_im[k] * W6_IM[idx];
                                si += vert_re[k] * W6_IM[idx] + vert_im[k] * W6_RE[idx];
                            }
                            double dr = sr * INV_SQRT6, di = si * INV_SQRT6;
                            diag_prob[j] = dr*dr + di*di;
                        }

                        /* Triality-consistent marginal: geometric mean of all 3 views.
                         * Suppresses spurious peaks that only appear in one basis. */
                        for (int v = 0; v < 6; v++) {
                            double tri_prob = cbrt(edge_prob[v] * vert_prob[v] * diag_prob[v]);
                            (*marg)[v] += tri_prob;
                        }
                    }
                }
            }

            /* ══ Hensel-Inspired Beam Search Constraint Propagation ══
             * Like tesseract_factor's Hensel lift: process blocks sequentially,
             * maintain K best configurations, prune by accumulated error.
             *
             * The constraint: blocks are selected JOINTLY. */

            #define N_BEAMS 24  /* K beams — widened for 31B (was 12) */

            typedef struct {
                double acc_error;
                int history_idx;  /* index into the backpointer array */
            } QuantBeam;

            typedef struct {
                int cand_idx;
                int parent_idx;
            } BeamHistory;

            QuantBeam beams[N_BEAMS];
            int active_beams = 1;

            /* Pre-allocate history to avoid O(N^2) memory copies */
            BeamHistory *history = (BeamHistory *)malloc(n_blocks * N_BEAMS * sizeof(BeamHistory));

            for (int b = 0; b < N_BEAMS; b++) {
                beams[b].acc_error = 0.0;
                beams[b].history_idx = -1;
            }

            /* Process blocks sequentially with beam search */
            for (int64_t i = 0; i < graph_blocks; i++) {
                double c_total = 0.0, f_total = 0.0;
                for (int v = 0; v < 6; v++) {
                    c_total += coarse_marg[i][v];
                    f_total += fine_marg[i][v];
                }

                /* Candidate scores for this block: triality prob × (1/normalized_error) */
                double cand_score[TOTAL_SCALE_CANDIDATES];
                int64_t blk = i * stride;
                int d_bin_count[6] = {0}, m_bin_count[6] = {0};
                for (int k = 0; k < N_CAND_D; k++) d_bin_count[CAND_TO_QUHIT[k]]++;
                for (int k = 0; k < N_CAND_M; k++) m_bin_count[CAND_TO_QUHIT[k]]++;
                /* Per-block error normalization: divide by block mean error
                 * so small-weight blocks don't dominate beam selection */
                float blk_mean_err = 0.0f;
                for (int c = 0; c < TOTAL_SCALE_CANDIDATES; c++)
                    blk_mean_err += candidate_errors[blk][c];
                blk_mean_err /= (float)TOTAL_SCALE_CANDIDATES;
                if (blk_mean_err < 1e-30f) blk_mean_err = 1e-30f;
                for (int di = 0; di < N_CAND_D; di++) {
                    int qi_d = CAND_TO_QUHIT[di];
                    double p_d = (c_total > 1e-30) ? coarse_marg[i][qi_d] / c_total : 1.0/6.0;
                    p_d /= (double)d_bin_count[qi_d];
                    for (int mi = 0; mi < N_CAND_M; mi++) {
                        int qi_m = CAND_TO_QUHIT[mi];
                        double p_m = (f_total > 1e-30) ? fine_marg[i][qi_m] / f_total : 1.0/6.0;
                        p_m /= (double)m_bin_count[qi_m];
                        int cidx = di * N_CAND_M + mi;
                        cand_score[cidx] = p_d * p_m / (candidate_errors[blk][cidx] / blk_mean_err + 1e-15);
                    }
                }

                /* Extend beams × 36 candidates, keep top K */
                typedef struct { double score; int beam_idx; int cand_idx; } BeamExt;
                BeamExt extensions[N_BEAMS * TOTAL_SCALE_CANDIDATES];
                int n_ext = 0;

                for (int b = 0; b < active_beams; b++) {
                    for (int c = 0; c < TOTAL_SCALE_CANDIDATES; c++) {
                        /* Score = -(accumulated_error + this_block_error) × triality_prob */
                        double ext_err = beams[b].acc_error + candidate_errors[blk][c];
                        double ext_score = cand_score[c] / (ext_err + 1e-15);
                        extensions[n_ext].score = ext_score;
                        extensions[n_ext].beam_idx = b;
                        extensions[n_ext].cand_idx = c;
                        n_ext++;
                    }
                }

                /* Top-K selection */
                int top_k = (n_ext < N_BEAMS) ? n_ext : N_BEAMS;
                int top_indices[N_BEAMS];
                for (int k = 0; k < top_k; k++) {
                    int best = -1;
                    double best_s = -1e30;
                    for (int e = 0; e < n_ext; e++) {
                        if (extensions[e].score > best_s) {
                            best_s = extensions[e].score;
                            best = e;
                        }
                    }
                    top_indices[k] = best;
                    extensions[best].score = -2e30;  /* poison */
                }

                /* Build new beams from top-K extensions using backpointers */
                QuantBeam new_beams[N_BEAMS];
                for (int k = 0; k < top_k; k++) {
                    int ext_idx = top_indices[k];
                    int src_beam = extensions[ext_idx].beam_idx;
                    int cand = extensions[ext_idx].cand_idx;

                    int hist_idx = i * N_BEAMS + k;
                    history[hist_idx].cand_idx = cand;
                    history[hist_idx].parent_idx = beams[src_beam].history_idx;

                    new_beams[k].history_idx = hist_idx;
                    new_beams[k].acc_error = beams[src_beam].acc_error
                                            + candidate_errors[blk][cand];
                }

                for (int k = 0; k < top_k; k++)
                    beams[k] = new_beams[k];
                active_beams = top_k;
            }

            /* Trace back the best beam's selections */
            int curr_hist = beams[0].history_idx;
            for (int64_t i = graph_blocks - 1; i >= 0; i--) {
                if (curr_hist >= 0) {
                    int cidx = history[curr_hist].cand_idx;
                    for (int64_t b = i * stride; b < (i+1) * stride && b < n_blocks; b++)
                        best_candidate[b] = cidx;
                    curr_hist = history[curr_hist].parent_idx;
                } else {
                    for (int64_t b = i * stride; b < (i+1) * stride && b < n_blocks; b++)
                        best_candidate[b] = 10 * N_CAND_M + 10;
                }
            }

            free(history);

            /* ══════════════════════════════════════════════════════════════
             * Phase 3.5: Born-Rule Multi-Shot Scale Refinement (Q2_K)
             *
             * 2D Born sampling: sample coarse quhit (d dimension) and
             * fine quhit (dmin dimension) jointly from triality marginals.
             * Each shot produces a (d_idx, dmin_idx) pair per block.
             * ══════════════════════════════════════════════════════════════ */
            {
                #define Q2K_BORN_SHOTS 64

                float beam_total_err = 0.0f;
                for (int64_t bi = 0; bi < n_blocks; bi++)
                    beam_total_err += candidate_errors[bi][best_candidate[bi]];

                unsigned int born_rng_q2 = 271828;
                int *shot_assignment = (int *)malloc(n_blocks * sizeof(int));

                for (int shot = 0; shot < Q2K_BORN_SHOTS; shot++) {
                    float shot_err = 0.0f;
                    /* Init from beam result so tail blocks beyond
                     * graph_blocks*stride keep valid indices */
                    memcpy(shot_assignment, best_candidate,
                           n_blocks * sizeof(int));

                    for (int64_t gi = 0; gi < graph_blocks; gi++) {
                        /* Born sample coarse (d) quhit */
                        double c_total = 0.0;
                        for (int v = 0; v < 6; v++) c_total += coarse_marg[gi][v];
                        born_rng_q2 = born_rng_q2 * 1664525u + 1013904223u;
                        double rnd_c = (double)(born_rng_q2 >> 8) / 16777216.0;
                        double target_c = rnd_c * c_total;
                        double cum_c = 0.0;
                        int qi_d = 5;
                        for (int v = 0; v < 6; v++) {
                            cum_c += coarse_marg[gi][v];
                            if (cum_c > target_c) { qi_d = v; break; }
                        }

                        /* Born sample fine (dmin) quhit */
                        double f_total = 0.0;
                        for (int v = 0; v < 6; v++) f_total += fine_marg[gi][v];
                        born_rng_q2 = born_rng_q2 * 1664525u + 1013904223u;
                        double rnd_f = (double)(born_rng_q2 >> 8) / 16777216.0;
                        double target_f = rnd_f * f_total;
                        double cum_f = 0.0;
                        int qi_m = 5;
                        for (int v = 0; v < 6; v++) {
                            cum_f += fine_marg[gi][v];
                            if (cum_f > target_f) { qi_m = v; break; }
                        }

                        /* Find best candidate within the sampled (d_bin, m_bin) */
                        int64_t blk = gi * stride;
                        float best_bin_err = 1e30f;
                        int best_bin_cand = 10 * N_CAND_M + 10;
                        for (int di = 0; di < N_CAND_D; di++) {
                            if (CAND_TO_QUHIT[di] != qi_d) continue;
                            for (int mi = 0; mi < N_CAND_M; mi++) {
                                if (CAND_TO_QUHIT[mi] != qi_m) continue;
                                int cidx = di * N_CAND_M + mi;
                                if (candidate_errors[blk][cidx] < best_bin_err) {
                                    best_bin_err = candidate_errors[blk][cidx];
                                    best_bin_cand = cidx;
                                }
                            }
                        }

                        shot_assignment[blk] = best_bin_cand;
                        shot_err += candidate_errors[blk][best_bin_cand];
                    }

                    if (shot_err < beam_total_err) {
                        for (int64_t b = 0; b < n_blocks; b++)
                            best_candidate[b] = shot_assignment[b];
                        beam_total_err = shot_err;
                    }
                }

                free(shot_assignment);
            }

            free(coarse_marg);
            free(fine_marg);
            mobius_destroy(mobius);
            hpc_destroy(graph);
        }
    } else {
        /* OPT_MSE or single block: pick candidate with lowest raw error */
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            float best_err = candidate_errors[blk][0];
            int best_idx = 0;
            for (int c = 1; c < TOTAL_SCALE_CANDIDATES; c++) {
                if (candidate_errors[blk][c] < best_err) {
                    best_err = candidate_errors[blk][c];
                    best_idx = c;
                }
            }
            best_candidate[blk] = best_idx;
        }
    }

    /* ══════════════════════════════════════════════════════════════════
     * PHASE 4: Assemble blocks via least-squares (d, dmin) extraction
     *
     * Like Q4_0's CF analog: the beam search / Born shots selected a
     * grid candidate (d_grid, dmin_grid). Now we EXTRACT the exact
     * optimal FP16 (d, dmin) via weighted least-squares, holding the
     * sub-block Ls/Lm and quantized levels fixed.
     *
     * Q2_K model: x[j,k] ≈ d × Ls[j] × q[j,k] - dmin × Lm[j]
     *
     * Full analog assembly: at each iteration, EXHAUSTIVELY search
     * all 16×16 = 256 possible (Ls[j], Lm[j]) pairs per sub-block
     * to find the assignment that minimizes weighted reconstruction
     * error. Then WLS-solve for the global (d, dmin). Repeat 5×.
     *
     * This guarantees every parameter is at its conditional optimum —
     * the perfect bit analog at 2-bit resolution.
     * ══════════════════════════════════════════════════════════════════ */

    #pragma omp parallel for schedule(dynamic, 64) reduction(+:total_err)
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const float *block_x = weights + blk * QK_K;
        int cidx = best_candidate[blk];
        uint8_t Ls_blk[16], Lm_blk[16];

        /* Start from HPC-selected candidate */
        memcpy(Ls_blk, candidate_Ls[blk][cidx], 16);
        memcpy(Lm_blk, candidate_Lm[blk][cidx], 16);

        float dm = gguf_fp16_to_fp32(candidate_d[blk][cidx]);
        float mm = gguf_fp16_to_fp32(candidate_dmin[blk][cidx]);

        /* ── Analog assembly: iterate to convergence ──
         * 5 iterations: enough for the (Ls,Lm) ↔ (d,dmin) coupling
         * to fully stabilize. Each iteration does:
         *   A) Exhaustive (Ls,Lm) search per sub-block
         *   B) Optimal q-value assignment
         *   C) WLS solve for (d, dmin) */
        for (int ls_iter = 0; ls_iter < 2; ls_iter++) {

            /* ── Step A: Exhaustive Ls/Lm search per sub-block ──
             * For each sub-block j, try ALL 16×16 = 256 possible
             * (Ls[j], Lm[j]) pairs. For each pair, compute optimal
             * q-values by rounding and measure weighted sub-block error.
             * Pick the pair that minimizes error. */
            for (int j = 0; j < N_SUB; j++) {
                const float *sx = block_x + 16 * j;
                float best_sub_err = 1e30f;
                uint8_t best_ls = Ls_blk[j], best_lm = Lm_blk[j];

                for (int try_ls = 0; try_ls <= 15; try_ls++) {
                    float d_sub = dm * (float)try_ls;
                    for (int try_lm = 0; try_lm <= 15; try_lm++) {
                        float m_sub = mm * (float)try_lm;

                        float sub_err = 0.0f;
                        for (int k = 0; k < 16; k++) {
                            float x = sx[k];
                            float w = (imat_importance) ?
                                      imat_importance[blk * QK_K + 16*j + k] : 1.0f;
                            int q;
                            if (d_sub < 1e-15f) {
                                q = 0;
                            } else {
                                q = gguf_nearest_int((x + m_sub) / d_sub);
                                if (q < 0) q = 0; if (q > 3) q = 3;
                            }
                            float deq = d_sub * (float)q - m_sub;
                            float diff = x - deq;
                            sub_err += diff * diff * w;
                        }

                        if (sub_err < best_sub_err) {
                            best_sub_err = sub_err;
                            best_ls = (uint8_t)try_ls;
                            best_lm = (uint8_t)try_lm;
                            if (sub_err < 1e-10f) goto done_sub;
                        }
                    }
                }
                done_sub:;
                Ls_blk[j] = best_ls;
                Lm_blk[j] = best_lm;
            }

            /* ── Step B: Quantize q-values with optimal Ls/Lm ── */
            uint8_t L[QK_K];
            for (int j = 0; j < N_SUB; j++) {
                float d_sub = dm * (float)Ls_blk[j];
                float m_sub = mm * (float)Lm_blk[j];
                if (d_sub < 1e-15f) {
                    for (int k = 0; k < 16; k++) L[16*j+k] = 0;
                    continue;
                }
                for (int k = 0; k < 16; k++) {
                    int q = gguf_nearest_int((block_x[16*j+k] + m_sub) / d_sub);
                    if (q < 0) q = 0; if (q > 3) q = 3;
                    L[16*j+k] = (uint8_t)q;
                }
            }

            /* ── Step C: WLS solve for (d, dmin) ──
             * x[j,k] ≈ d × Ls[j] × q[j,k] - dmin × Lm[j]
             * Let a = Ls[j]×q[j,k], b = Lm[j]
             * Normal equations via Cramer's rule */
            double Saa = 0, Sab = 0, Sbb = 0, Sxa = 0, Sxb = 0;
            for (int j = 0; j < N_SUB; j++) {
                float ls_f = (float)Ls_blk[j];
                float lm_f = (float)Lm_blk[j];
                for (int k = 0; k < 16; k++) {
                    float x = block_x[16*j+k];
                    float w = (imat_importance) ?
                              imat_importance[blk * QK_K + 16*j+k] : 1.0f;
                    float a = ls_f * (float)L[16*j+k];
                    float b = lm_f;
                    Saa += w * a * a;
                    Sab += w * a * b;
                    Sbb += w * b * b;
                    Sxa += w * x * a;
                    Sxb += w * x * b;
                }
            }

            double det = Saa * Sbb - Sab * Sab;
            if (fabs(det) > 1e-30) {
                double d_new  = (Sbb * Sxa - Sab * Sxb) / det;
                double dm_new = (Sab * Sxa - Saa * Sxb) / det;
                /* Clamp: positive and within 4× of candidate seed */
                float d_seed = gguf_fp16_to_fp32(candidate_d[blk][cidx]);
                float m_seed = gguf_fp16_to_fp32(candidate_dmin[blk][cidx]);
                if (d_new > 0.0 && d_new < 4.0 * (d_seed + 1e-10))
                    dm = gguf_fp16_to_fp32(gguf_fp32_to_fp16((float)d_new));
                if (dm_new > 0.0 && dm_new < 4.0 * (m_seed + 1e-10))
                    mm = gguf_fp16_to_fp32(gguf_fp32_to_fp16((float)dm_new));
            }
        }

        /* ── FP16 ULP neighborhood search for (d, dmin) ──
         * The WLS solve found continuous-optimal (d, dmin). But FP16
         * truncation may shift the optimum. Try ±4 ULP around both
         * d and dmin, pick the pair with minimum reconstruction error. */
        {
            uint16_t base_d16 = gguf_fp32_to_fp16(dm);
            uint16_t base_m16 = gguf_fp32_to_fp16(mm);
            uint16_t best_d16 = base_d16, best_m16 = base_m16;
            float best_ulp_err = 1e30f;

            for (int dd = -2; dd <= 2; dd++) {
                int cd16 = (int)base_d16 + dd;
                if (cd16 < 0 || cd16 > 0x7BFF) continue;
                float trial_dm = gguf_fp16_to_fp32((uint16_t)cd16);

                for (int dm_delta = -2; dm_delta <= 2; dm_delta++) {
                    int cm16 = (int)base_m16 + dm_delta;
                    if (cm16 < 0 || cm16 > 0x7BFF) continue;
                    float trial_mm = gguf_fp16_to_fp32((uint16_t)cm16);

                    float err = 0.0f;
                    for (int j = 0; j < N_SUB; j++) {
                        float d_sub = trial_dm * (float)Ls_blk[j];
                        float m_sub = trial_mm * (float)Lm_blk[j];
                        for (int k = 0; k < 16; k++) {
                            float x = block_x[16*j+k];
                            float w = (imat_importance) ?
                                      imat_importance[blk * QK_K + 16*j+k] : 1.0f;
                            int q;
                            if (d_sub < 1e-15f) { q = 0; }
                            else {
                                q = gguf_nearest_int((x + m_sub) / d_sub);
                                if (q < 0) q = 0; if (q > 3) q = 3;
                            }
                            float deq = d_sub * (float)q - m_sub;
                            float diff = x - deq;
                            err += diff * diff * w;
                        }
                    }
                    if (err < best_ulp_err) {
                        best_ulp_err = err;
                        best_d16 = (uint16_t)cd16;
                        best_m16 = (uint16_t)cm16;
                    }
                }
            }
            dm = gguf_fp16_to_fp32(best_d16);
            mm = gguf_fp16_to_fp32(best_m16);
        }

        /* ── Final Ls/Lm re-optimization at committed FP16 (d, dmin) ──
         * The WLS solve may have shifted (d, dmin) after the last Step A,
         * invalidating the Ls/Lm choices. One final exhaustive pass at the
         * EXACT FP16-truncated scales ensures every sub-block is optimal. */
        for (int j = 0; j < N_SUB; j++) {
            const float *sx = block_x + 16 * j;
            float best_sub_err = 1e30f;
            uint8_t best_ls = Ls_blk[j], best_lm = Lm_blk[j];
            for (int try_ls = 0; try_ls <= 15; try_ls++) {
                float d_sub = dm * (float)try_ls;
                for (int try_lm = 0; try_lm <= 15; try_lm++) {
                    float m_sub = mm * (float)try_lm;
                    float sub_err = 0.0f;
                    for (int k = 0; k < 16; k++) {
                        float x = sx[k];
                        float w = (imat_importance) ?
                                  imat_importance[blk * QK_K + 16*j + k] : 1.0f;
                        int q;
                        if (d_sub < 1e-15f) { q = 0; }
                        else {
                            q = gguf_nearest_int((x + m_sub) / d_sub);
                            if (q < 0) q = 0; if (q > 3) q = 3;
                        }
                        float deq = d_sub * (float)q - m_sub;
                        float diff = x - deq;
                        sub_err += diff * diff * w;
                    }
                    if (sub_err < best_sub_err) {
                        best_sub_err = sub_err;
                        best_ls = (uint8_t)try_ls;
                        best_lm = (uint8_t)try_lm;
                    }
                }
            }
            Ls_blk[j] = best_ls;
            Lm_blk[j] = best_lm;
        }

        /* Store the extracted optimal FP16 (d, dmin) */
        output[blk].d    = gguf_fp32_to_fp16(dm);
        output[blk].dmin = gguf_fp32_to_fp16(mm);

        for (int j = 0; j < N_SUB; j++)
            output[blk].scales[j] = Ls_blk[j] | (Lm_blk[j] << 4);

        /* ── Final quantization with D₆ Hadamard Error Shaping ──
         *
         * Standard Q2_K rounds each weight independently: q = round((x+m)/d).
         * But within a sub-block, weights share (d, m), so their quantization
         * errors are CORRELATED. Independent rounding is suboptimal.
         *
         * The D₆ fold (antipodal Hadamard from the triality quhit) decomposes
         * the error vector into vesica (sum) and wave (difference) components:
         *   vesica[k] = (e[k] + e[k+3]) / √2    — DC-like, accumulates in dot products
         *   wave[k]   = (e[k] - e[k+3]) / √2    — noise-like, cancels in dot products
         *
         * We WANT large wave error and small vesica error. So we greedily
         * flip rounding decisions (floor↔ceil) to minimize vesica energy,
         * even if total element-wise error increases slightly.
         *
         * Process: 16 elements per sub-block, treat as 2 groups of 6 + 4 tail.
         * Apply DFT₆-fold to each group of 6, minimize vesica component.
         */
        uint8_t L[QK_K];
        for (int j = 0; j < N_SUB; j++) {
            float d = dm * (float)(output[blk].scales[j] & 0xF);
            if (d < 1e-15f) {
                for (int k = 0; k < 16; k++) L[16 * j + k] = 0;
                continue;
            }
            float m = mm * (float)(output[blk].scales[j] >> 4);
            float id = 1.0f / d;

            /* Step 1: Standard nearest-rounding as baseline */
            int q_base[16];
            float q_cont[16];  /* continuous q values before rounding */
            for (int k = 0; k < 16; k++) {
                q_cont[k] = (block_x[16*j+k] + m) * id;
                q_base[k] = gguf_nearest_int(q_cont[k]);
                if (q_base[k] < 0) q_base[k] = 0;
                if (q_base[k] > 3) q_base[k] = 3;
            }

            /* Step 2: D₆ Hadamard Error Shaping
             * For each 6-element group, greedily flip the rounding decision
             * that most reduces the D₆-folded vesica error component.
             *
             * D₆ fold on 6-element groups: antipodal pairs (0,3), (1,4), (2,5)
             * vesica[k] = e[k] + e[k+3]  (k=0,1,2) — DC-like, propagates
             * wave[k]   = e[k] - e[k+3]  (k=0,1,2) — noise-like, cancels
             *
             * Weight vesica 4× over wave + penalize DC (sum of all 6 errors) */
            int q_shaped[16];
            memcpy(q_shaped, q_base, 16 * sizeof(int));

            /* Process groups: [0..5], [6..11], tail [12..15] handled by D₆ metric on available pairs */
            for (int g = 0; g < 2; g++) {
                int g_off = g * 6;
                if (g_off + 5 >= 16) break;

                /* Multiple greedy passes — each pass finds the single best flip */
                for (int pass = 0; pass < 6; pass++) {
                    int best_k = -1;
                    int best_q_alt = 0;
                    float best_delta = 0.0f;  /* improvement = current_metric - alt_metric */

                    /* Compute current group errors */
                    float e_cur[6];
                    for (int kk = 0; kk < 6; kk++) {
                        int ii = g_off + kk;
                        float deq = d * (float)q_shaped[ii] - m;
                        e_cur[kk] = block_x[16*j+ii] - deq;
                    }

                    /* Current D₆ metric: vesica energy + DC² */
                    float vesica_cur = 0.0f, dc_cur = 0.0f;
                    for (int p = 0; p < 3; p++) {
                        float v = e_cur[p] + e_cur[p+3];
                        vesica_cur += v * v;
                    }
                    for (int kk = 0; kk < 6; kk++) dc_cur += e_cur[kk];
                    float metric_cur = 4.0f * vesica_cur + dc_cur * dc_cur;

                    /* Try flipping each element */
                    for (int k = 0; k < 6; k++) {
                        int idx = g_off + k;
                        int q_cur = q_shaped[idx];

                        /* Try the alternative rounding */
                        int q_try;
                        if (q_cont[idx] - (float)q_cur >= 0) {
                            q_try = q_cur + 1;
                        } else {
                            q_try = q_cur - 1;
                        }
                        if (q_try < 0 || q_try > 3) continue;

                        /* Compute alt errors (only element k changes) */
                        float e_alt[6];
                        for (int kk = 0; kk < 6; kk++) e_alt[kk] = e_cur[kk];
                        float deq_try = d * (float)q_try - m;
                        e_alt[k] = block_x[16*j+idx] - deq_try;

                        /* Alt D₆ metric */
                        float vesica_alt = 0.0f, dc_alt = 0.0f;
                        for (int p = 0; p < 3; p++) {
                            float v = e_alt[p] + e_alt[p+3];
                            vesica_alt += v * v;
                        }
                        for (int kk = 0; kk < 6; kk++) dc_alt += e_alt[kk];
                        float metric_alt = 4.0f * vesica_alt + dc_alt * dc_alt;

                        float delta = metric_cur - metric_alt;
                        if (delta > best_delta) {
                            best_delta = delta;
                            best_k = k;
                            best_q_alt = q_try;
                        }
                    }

                    if (best_k < 0) break;  /* no improvement found */
                    q_shaped[g_off + best_k] = best_q_alt;  /* commit the flip */
                }
            }

            /* Step 3: Final error comparison — only keep shaped if it improves
             * or is within 5% of baseline (vesica shaping trades element MSE
             * for better spectral distribution of error) */
            float err_base = 0.0f, err_shaped = 0.0f;
            for (int k = 0; k < 16; k++) {
                float x = block_x[16*j+k];
                float w = (imat_importance) ?
                          imat_importance[blk * QK_K + 16*j + k] : 1.0f;
                float deq_b = d * (float)q_base[k] - m;
                float deq_s = d * (float)q_shaped[k] - m;
                err_base += (x - deq_b) * (x - deq_b) * w;
                err_shaped += (x - deq_s) * (x - deq_s) * w;
            }

            int *q_final = (err_shaped <= err_base * 1.05f) ? q_shaped : q_base;
            for (int k = 0; k < 16; k++)
                L[16 * j + k] = (uint8_t)q_final[k];
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) {
                output[blk].qs[j / 4 + l] = L[j + l]
                                           | (L[j + l + 32] << 2)
                                           | (L[j + l + 64] << 4)
                                           | (L[j + l + 96] << 6);
            }
        }

        total_err += gguf_q2_k_block_error(block_x, &output[blk]);
    }

    free(seeds);
    free(candidate_errors);
    free(candidate_d);
    free(candidate_dmin);
    free(candidate_Ls);
    free(candidate_Lm);
    free(best_candidate);
    if (out_total_error) *out_total_error = total_err;
    
    if (verbose) {
        float rmse = sqrtf(total_err / (float)n_elements);
        printf("  [HPC Q2_K] %lld elements, Total Err: %.4f, RMSE: %.6f\n",
               (long long)n_elements, total_err, rmse);
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * PROGRESS REPORTING
 * ═══════════════════════════════════════════════════════════════════════════ */

static void print_progress_bar(int current, int total, const char *label,
                                 clock_t start_time)
{
    if (total <= 0) return;
    float pct = (float)current / (float)total;
    int bar_width = 40;
    int filled = (int)(pct * bar_width);

    double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    double eta = (pct > 0.01f) ? elapsed / pct * (1.0 - pct) : 0.0;

    printf("\r  [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("█");
        else if (i == filled) printf("▓");
        else printf("░");
    }
    printf("] %3d%% (%d/%d) %.0fs ETA:%.0fs  %s",
           (int)(pct * 100), current, total, elapsed, eta, label);
    fflush(stdout);

    if (current == total) printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GGUF FILE WRITER — Assembles the complete output file
 * ═══════════════════════════════════════════════════════════════════════════ */

static int write_gguf(const char *output_path, const STMultiFile *mf,
                        const ModelArchitecture *arch,
                        const TokenizerData *tokenizer,
                        OptimizerMode opt_mode,
                        const IMatrixData *imatrix,
                        int verbose)
{
    FILE *fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "  ERROR: Cannot open '%s' for writing\n", output_path);
        return -1;
    }

    printf("\n  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  WRITING GGUF FILE                                           ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    /* ── Determine which tensors to include ── */
    int *include_list = (int *)calloc(mf->n_tensors, sizeof(int));
    int n_include = 0;
    for (int i = 0; i < mf->n_tensors; i++) {
        if (!should_skip_tensor(mf->tensor_map[i].name)) {
            include_list[n_include++] = i;
        } else {
            if (verbose) printf("  SKIP: %s (not needed in GGUF)\n", mf->tensor_map[i].name);
        }
    }

    /* ── Count metadata KV pairs ── */
    int n_kv = 0;
    n_kv++;  /* general.architecture */
    n_kv++;  /* general.name */
    n_kv++;  /* general.quantization_version */
    n_kv++;  /* general.file_type */
    n_kv++;  /* {arch}.context_length */
    n_kv++;  /* {arch}.embedding_length */
    n_kv++;  /* {arch}.block_count */
    n_kv++;  /* {arch}.feed_forward_length */
    n_kv++;  /* {arch}.attention.head_count */
    n_kv++;  /* {arch}.attention.head_count_kv */
    n_kv++;  /* {arch}.attention.layer_norm_rms_epsilon */
    n_kv++;  /* {arch}.rope.freq_base */
    n_kv++;  /* {arch}.vocab_size */

    /* Tokenizer metadata KV count */
    int has_tokenizer = (tokenizer != NULL && tokenizer->vocab_size > 0);
    if (has_tokenizer) {
        n_kv++;  /* tokenizer.ggml.model */
        n_kv++;  /* tokenizer.ggml.tokens */
        n_kv++;  /* tokenizer.ggml.scores */
        n_kv++;  /* tokenizer.ggml.token_type */
        n_kv++;  /* tokenizer.ggml.bos_token_id */
        n_kv++;  /* tokenizer.ggml.eos_token_id */
        n_kv++;  /* tokenizer.ggml.unknown_token_id */
        if (tokenizer->n_merges > 0)
            n_kv++;  /* tokenizer.ggml.merges */
    }

    /* ── Check for weight tying ──
     * If tie_word_embeddings is set and there's no separate lm_head,
     * llama.cpp handles this internally — do NOT duplicate the tensor.
     * Only add output.weight if the model has a separate lm_head.weight. */
    int has_lm_head = (st_multi_find_tensor(mf, "lm_head.weight") >= 0);
    int total_tensors = n_include;

    if (arch->tie_word_embeddings && !has_lm_head) {
        printf("  Weight-tied embeddings detected — llama.cpp handles internally\n\n");
    }

    /* ── Prepare tensor info ── */
    char (*gguf_names)[ST_MAX_NAME_LEN] = calloc(total_tensors, ST_MAX_NAME_LEN);
    GGMLType *tensor_types = calloc(total_tensors, sizeof(GGMLType));
    int64_t *tensor_sizes = calloc(total_tensors, sizeof(int64_t));
    uint64_t data_offset = 0;
    uint64_t *tensor_offsets = calloc(total_tensors, sizeof(uint64_t));
    int *tensor_src_idx = calloc(total_tensors, sizeof(int)); /* map to unified ST index */
    char (*tensor_hf_names)[ST_MAX_NAME_LEN] = calloc(total_tensors, ST_MAX_NAME_LEN);

    GGMLType quant_type = GGML_TYPE_Q2_K;

    for (int i = 0; i < n_include; i++) {
        int src = include_list[i];
        const STTensorInfo *ti = st_multi_tensor_info(mf, src);
        map_tensor_name(mf->tensor_map[src].name, gguf_names[i], ST_MAX_NAME_LEN);
        strncpy(tensor_hf_names[i], mf->tensor_map[src].name, ST_MAX_NAME_LEN - 1);
        tensor_src_idx[i] = src;

        if (should_quantize(ti, gguf_names[i])) {
            if (is_attention_tensor(gguf_names[i])) {
                /* Promote attention Q/K/V/O to Q4_0 for higher precision.
                 * Attention scores are most sensitive to quantization noise. */
                tensor_types[i] = GGML_TYPE_Q4_0;
                int64_t n_blocks_q4 = (ti->n_elements + QK4_0 - 1) / QK4_0;
                tensor_sizes[i] = n_blocks_q4 * sizeof(BlockQ4_0);
                if (verbose)
                    printf("  [ATTN→Q4_0] %s (%ld elements)\n",
                           gguf_names[i], (long)ti->n_elements);
            } else {
                tensor_types[i] = quant_type;
                tensor_sizes[i] = ggml_type_size(quant_type, ti->n_elements);
            }
        } else if (ti->n_dims >= 2) {
            /* 2D non-quantized tensors (embeddings, output) → F16 */
            tensor_types[i] = GGML_TYPE_F16;
            tensor_sizes[i] = ti->n_elements * sizeof(uint16_t);
        } else {
            /* 1D tensors (norms, biases) → F32 */
            tensor_types[i] = GGML_TYPE_F32;
            tensor_sizes[i] = ti->n_elements * sizeof(float);
        }

        tensor_offsets[i] = data_offset;

        /* Align each tensor to 32 bytes */
        data_offset += tensor_sizes[i];
        data_offset = (data_offset + GGUF_DEFAULT_ALIGNMENT - 1) &
                      ~(uint64_t)(GGUF_DEFAULT_ALIGNMENT - 1);
    }

    /* ── Write header ── */
    gguf_write_header(fp, total_tensors, n_kv);

    /* ── Write metadata KV pairs ── */
    gguf_write_kv_string(fp, "general.architecture", arch->architecture);
    gguf_write_kv_string(fp, "general.name", arch->name);
    gguf_write_kv_uint32(fp, "general.quantization_version", 2);
    gguf_write_kv_uint32(fp, "general.file_type", 10);  /* Q2_K = 10 */

    char kbuf[128];
    snprintf(kbuf, sizeof(kbuf), "%s.context_length", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->context_length);

    snprintf(kbuf, sizeof(kbuf), "%s.embedding_length", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->embedding_length);

    snprintf(kbuf, sizeof(kbuf), "%s.block_count", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->block_count);

    snprintf(kbuf, sizeof(kbuf), "%s.feed_forward_length", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->feed_forward_length);

    snprintf(kbuf, sizeof(kbuf), "%s.attention.head_count", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->head_count);

    snprintf(kbuf, sizeof(kbuf), "%s.attention.head_count_kv", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->head_count_kv);

    snprintf(kbuf, sizeof(kbuf), "%s.attention.layer_norm_rms_epsilon", arch->architecture);
    gguf_write_kv_float32(fp, kbuf, arch->rms_norm_eps);

    snprintf(kbuf, sizeof(kbuf), "%s.rope.freq_base", arch->architecture);
    gguf_write_kv_float32(fp, kbuf, arch->rope_freq_base);

    snprintf(kbuf, sizeof(kbuf), "%s.vocab_size", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->vocab_size);

    /* ── Write tokenizer metadata ── */
    if (has_tokenizer) {
        gguf_write_kv_string(fp, "tokenizer.ggml.model", tokenizer->model_type);
        gguf_write_kv_string_array(fp, "tokenizer.ggml.tokens",
                                     (const char **)tokenizer->tokens,
                                     (uint64_t)tokenizer->vocab_size);
        gguf_write_kv_float32_array(fp, "tokenizer.ggml.scores",
                                      tokenizer->scores,
                                      (uint64_t)tokenizer->vocab_size);
        gguf_write_kv_int32_array(fp, "tokenizer.ggml.token_type",
                                    tokenizer->token_types,
                                    (uint64_t)tokenizer->vocab_size);
        gguf_write_kv_uint32(fp, "tokenizer.ggml.bos_token_id",
                               (uint32_t)tokenizer->bos_id);
        gguf_write_kv_uint32(fp, "tokenizer.ggml.eos_token_id",
                               (uint32_t)tokenizer->eos_id);
        gguf_write_kv_uint32(fp, "tokenizer.ggml.unknown_token_id",
                               (uint32_t)tokenizer->unk_id);
        if (tokenizer->n_merges > 0) {
            gguf_write_kv_string_array(fp, "tokenizer.ggml.merges",
                                         (const char **)tokenizer->merges,
                                         (uint64_t)tokenizer->n_merges);
        }
        printf("  Tokenizer metadata written (%d tokens, %d merges)\n\n",
               tokenizer->vocab_size, tokenizer->n_merges);
    }

    /* ── Write tensor info descriptors ── */
    for (int i = 0; i < total_tensors; i++) {
        int src = tensor_src_idx[i];
        const STTensorInfo *ti = st_multi_tensor_info(mf, src);
        uint64_t dims[ST_MAX_DIMS];
        /* GGUF uses reversed dimension order from SafeTensors/PyTorch */
        int nd = ti->n_dims;
        for (int d = 0; d < nd; d++) {
            dims[d] = (uint64_t)ti->shape[nd - 1 - d];
        }
        gguf_write_tensor_info(fp, gguf_names[i],
                                ti->n_dims, dims,
                                tensor_types[i], tensor_offsets[i]);
    }

    /* ── Alignment padding before data section ── */
    gguf_write_padding(fp, GGUF_DEFAULT_ALIGNMENT);

    /* ── Write tensor data ── */
    printf("  Quantizing and writing %d tensors...\n\n", total_tensors);

    float total_error_sum = 0.0f;
    int quant_count = 0;
    int64_t total_elements_quantized = 0;
    int64_t total_bytes_quantized = 0;
    int64_t total_bytes_unquantized = 0;
    clock_t quant_start = clock();

    for (int i = 0; i < total_tensors; i++) {
        int src = tensor_src_idx[i];
        const STTensorInfo *ti = st_multi_tensor_info(mf, src);

        print_progress_bar(i, total_tensors, gguf_names[i], quant_start);

        if (tensor_types[i] == GGML_TYPE_Q2_K) {
            /* ── HPC-Optimized Q2_K Quantization ── */
            float *f32_data = st_multi_tensor_to_f32(mf, src);
            if (!f32_data) {
                fprintf(stderr, "\n  ERROR: Failed to convert tensor '%s' to F32\n",
                        ti->name);
                continue;
            }

            int64_t n_elements = ti->n_elements;
            float tensor_error = 0.0f;

            /* Pad to QK_K boundary */
            int64_t padded = (n_elements + QK_K - 1) / QK_K * QK_K;
            if (padded > n_elements) {
                f32_data = realloc(f32_data, padded * sizeof(float));
                for (int64_t j = n_elements; j < padded; j++)
                    f32_data[j] = 0.0f;
                n_elements = padded;
            }

            int64_t n_blocks = n_elements / QK_K;
            BlockQ2K *quant_data = calloc(n_blocks, sizeof(BlockQ2K));

            /* Look up imatrix importance for this tensor */
            const float *imp = NULL;
            if (imatrix) {
                const IMatrixEntry *ime = imatrix_find_any(imatrix,
                    gguf_names[i], tensor_hf_names[i]);
                if (ime && ime->n_values > 0) {
                    imp = ime->normalized;
                    if (verbose)
                        printf("\n    imatrix: using %d importance weights for %s\n",
                               ime->n_values, gguf_names[i]);
                }
            }

            quantize_tensor_q2k_hpc(f32_data, n_elements,
                                      quant_data, &tensor_error,
                                      opt_mode, imp, verbose);

            fwrite(quant_data, sizeof(BlockQ2K), n_blocks, fp);

            float rmse = sqrtf(tensor_error / (float)ti->n_elements);
            if (verbose) {
                printf("\n  [Q2_K] %-50s  %10ld elements → %ld bytes  RMSE=%.6e\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(n_blocks * sizeof(BlockQ2K)), rmse);
            }

            total_error_sum += tensor_error;
            total_elements_quantized += ti->n_elements;
            total_bytes_quantized += n_blocks * sizeof(BlockQ2K);
            quant_count++;

            free(quant_data);
            free(f32_data);
        } else if (tensor_types[i] == GGML_TYPE_Q4_0) {
            /* ── HPC-Optimized Q4_0 Quantization (attention tensors) ── */
            float *f32_data = st_multi_tensor_to_f32(mf, src);
            if (!f32_data) {
                fprintf(stderr, "\n  ERROR: Failed to convert tensor '%s' to F32\n",
                        ti->name);
                continue;
            }

            int64_t n_elements = ti->n_elements;

            /* Pad to QK4_0 boundary */
            int64_t padded = (n_elements + QK4_0 - 1) / QK4_0 * QK4_0;
            if (padded > n_elements) {
                f32_data = realloc(f32_data, padded * sizeof(float));
                for (int64_t j = n_elements; j < padded; j++)
                    f32_data[j] = 0.0f;
                n_elements = padded;
            }

            int64_t n_blocks_q4 = n_elements / QK4_0;
            BlockQ4_0 *q4_data = calloc(n_blocks_q4, sizeof(BlockQ4_0));
            float tensor_error = 0.0f;

            /* Look up imatrix importance for this tensor */
            const float *imp = NULL;
            if (imatrix) {
                const IMatrixEntry *ime = imatrix_find_any(imatrix,
                    gguf_names[i], tensor_hf_names[i]);
                if (ime && ime->n_values > 0) {
                    imp = ime->normalized;
                    if (verbose)
                        printf("\n    imatrix: using %d importance weights for %s\n",
                               ime->n_values, gguf_names[i]);
                }
            }

            quantize_tensor_q4_0_hpc(f32_data, n_elements,
                                       q4_data, &tensor_error,
                                       imp, verbose);

            fwrite(q4_data, sizeof(BlockQ4_0), n_blocks_q4, fp);

            float rmse = sqrtf(tensor_error / (float)ti->n_elements);
            if (verbose) {
                printf("\n  [Q4_0·HPC] %-47s  %10ld elements → %ld bytes  RMSE=%.6e\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(n_blocks_q4 * sizeof(BlockQ4_0)), rmse);
            }

            total_error_sum += tensor_error;
            total_elements_quantized += ti->n_elements;
            total_bytes_quantized += n_blocks_q4 * sizeof(BlockQ4_0);
            quant_count++;

            free(q4_data);
            free(f32_data);
        } else if (tensor_types[i] == GGML_TYPE_F16) {
            /* ── Store as F16 (embeddings, output, 2D non-quantized) ── */
            float *f32_data = st_multi_tensor_to_f32(mf, src);
            if (!f32_data) {
                fprintf(stderr, "\n  ERROR: Failed to convert tensor '%s'\n",
                        ti->name);
                continue;
            }

            /* Convert F32 → F16 */
            uint16_t *f16_data = (uint16_t *)malloc(ti->n_elements * sizeof(uint16_t));
            for (int64_t j = 0; j < ti->n_elements; j++)
                f16_data[j] = gguf_fp32_to_fp16(f32_data[j]);

            fwrite(f16_data, sizeof(uint16_t), ti->n_elements, fp);

            total_bytes_unquantized += ti->n_elements * sizeof(uint16_t);

            if (verbose) {
                printf("\n  [F16 ] %-50s  %10ld elements → %ld bytes\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(ti->n_elements * sizeof(uint16_t)));
            }

            free(f16_data);
            free(f32_data);
        } else {
            /* ── Keep as F32 (1D: norms, biases) ── */
            float *f32_data = st_multi_tensor_to_f32(mf, src);
            if (!f32_data) {
                fprintf(stderr, "\n  ERROR: Failed to convert tensor '%s'\n",
                        ti->name);
                continue;
            }

            fwrite(f32_data, sizeof(float), ti->n_elements, fp);

            total_bytes_unquantized += ti->n_elements * sizeof(float);

            if (verbose) {
                printf("\n  [F32 ] %-50s  %10ld elements → %ld bytes\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(ti->n_elements * sizeof(float)));
            }

            free(f32_data);
        }

        /* Pad to alignment */
        gguf_write_padding(fp, GGUF_DEFAULT_ALIGNMENT);
    }

    print_progress_bar(total_tensors, total_tensors, "done", quant_start);

    long final_size = ftell(fp);
    fclose(fp);

    /* ── Final summary ── */
    printf("\n  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  QUANTIZATION SUMMARY                                        ║\n");
    printf("  ╠════════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Tensors quantized (Q2_K): %-33d ║\n", quant_count);
    printf("  ║  Elements quantized:     %15ld                   ║\n",
           (long)total_elements_quantized);
    printf("  ║  Q2_K data:              %12ld bytes (%6.1f MB)    ║\n",
           (long)total_bytes_quantized,
           (double)total_bytes_quantized / (1024.0 * 1024.0));
    printf("  ║  F32 data (unquantized): %12ld bytes (%6.1f MB)    ║\n",
           (long)total_bytes_unquantized,
           (double)total_bytes_unquantized / (1024.0 * 1024.0));

    /* Compute original model size (all as F32) */
    int64_t original_f32_size = 0;
    for (int i = 0; i < total_tensors; i++) {
        const STTensorInfo *ti = st_multi_tensor_info(mf, tensor_src_idx[i]);
        original_f32_size += ti->n_elements * sizeof(float);
    }
    float compression_ratio = (original_f32_size > 0) ?
                               (float)original_f32_size / (float)final_size : 0.0f;
    float effective_bpw = (total_elements_quantized > 0) ?
                           8.0f * (float)total_bytes_quantized / (float)total_elements_quantized :
                           0.0f;

    printf("  ║  Effective bits/weight:  %15.2f                       ║\n",
           effective_bpw);
    printf("  ║  Compression ratio:      %15.1fx                      ║\n",
           compression_ratio);
    printf("  ║  Total RMSE:             %15.6e                  ║\n",
           sqrtf(total_error_sum));
    printf("  ║  Output file:            %ld bytes (%.1f MB)%*s║\n",
           final_size, (double)final_size / (1024.0 * 1024.0),
           (int)(20 - snprintf(NULL, 0, "%ld bytes (%.1f MB)",
                               final_size, (double)final_size / (1024.0 * 1024.0))), "");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    free(include_list);
    free(gguf_names);
    free(tensor_types);
    free(tensor_sizes);
    free(tensor_offsets);
    free(tensor_src_idx);
    free(tensor_hf_names);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * LIBRARY API — Exported functions for Python ctypes integration
 *
 * When built with -DHEXSTATE_LIBRARY, these are the only public symbols.
 * The Python GGUF pipeline handles metadata/IO; C handles HPC quantization.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Initialize HExState subsystems (must be called once before quantization) */
void hexstate_init(void)
{
    static int initialized = 0;
    if (!initialized) {
        srand(42);  /* Deterministic for reproducibility */
        triality_exotic_init();
        s6_exotic_init();
        triality_stats_reset();
        initialized = 1;
    }
}

/* Quantize a single tensor's F32 data to Q2_K using HPC optimization.
 *
 * Parameters:
 *   weights:     input F32 data (must be padded to multiple of 256)
 *   n_elements:  number of elements (must be multiple of 256)
 *   output:      output buffer (must be n_elements/256 * 84 bytes)
 *   out_error:   pointer to receive total MSE (can be NULL)
 *   opt_mode:    0=HPC, 1=MSE, 2=Hybrid (recommended)
 *   verbose:     1 for per-block diagnostics
 */
void hexstate_quantize_tensor_q2k(const float *weights, int64_t n_elements,
                                    void *output, float *out_error,
                                    int opt_mode, int verbose)
{
    hexstate_init();
    quantize_tensor_q2k_hpc(weights, n_elements,
                              (BlockQ2K *)output, out_error,
                              (OptimizerMode)opt_mode, NULL, verbose);
}

/* Same as above but with importance matrix weights */
void hexstate_quantize_tensor_q2k_imat(const float *weights, int64_t n_elements,
                                         void *output, float *out_error,
                                         int opt_mode,
                                         const float *imat_importance,
                                         int verbose)
{
    hexstate_init();
    quantize_tensor_q2k_hpc(weights, n_elements,
                              (BlockQ2K *)output, out_error,
                              (OptimizerMode)opt_mode, imat_importance, verbose);
}

/* Get the block size for Q2_K (84 bytes per 256 elements) */
int hexstate_q2k_block_bytes(void) { return sizeof(BlockQ2K); }
int hexstate_q2k_block_elements(void) { return QK_K; }

/* HPC-optimized Q4_0 quantization for attention tensors.
 * Called from Python requantizer via ctypes.
 *   weights:     input F32 weights
 *   n_elements:  number of elements (must be multiple of 32)
 *   output:      output buffer (must be n_elements/32 * 18 bytes)
 *   out_error:   pointer to receive total MSE (can be NULL)
 *   imat_importance: optional per-element importance weights
 *   verbose:     1 for per-block diagnostics
 */
void hexstate_quantize_tensor_q4_0_hpc(const float *weights, int64_t n_elements,
                                         void *output, float *out_error,
                                         const float *imat_importance,
                                         int verbose)
{
    hexstate_init();
    float err = 0.0f;
    quantize_tensor_q4_0_hpc(weights, n_elements,
                               (BlockQ4_0 *)output, &err,
                               imat_importance, verbose);
    if (out_error) *out_error = err;
}

#ifndef HEXSTATE_LIBRARY
/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    srand(time(NULL));

    /* Initialize HExState subsystems */
    triality_exotic_init();
    s6_exotic_init();
    triality_stats_reset();

    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                              ║\n");
    printf("  ║   HExState GGUF QUANTIZER v2.0                              ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   Architecture: HPCGraph Sensitivity Propagation             ║\n");
    printf("  ║   Optimization: HPC BP + MSE Grid Search + iMatrix          ║\n");
    printf("  ║   Output: GGUF v3 (Q2_K, 2.625 bpw)                        ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   \"The weight and the quantized are opposite faces.\"         ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    if (argc < 3) {
        printf("  Usage: %s <input> <output.gguf> [options]\n\n", argv[0]);
        printf("  Input:\n");
        printf("    Single .safetensors file, or\n");
        printf("    Model directory with sharded .safetensors files\n\n");
        printf("  Options:\n");
        printf("    --optimizer hpc|mse|hybrid   Scale optimization (default: hybrid)\n");
        printf("    --imatrix <file>             Importance matrix for Q2_K quality\n");
        printf("    --verbose                    Per-block diagnostics\n\n");
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    OptimizerMode opt_mode = OPT_HYBRID;
    const char *imatrix_path = NULL;
    int verbose = 0;

    /* Parse options */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--optimizer") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "hpc") == 0) opt_mode = OPT_HPC;
            else if (strcmp(argv[i], "mse") == 0) opt_mode = OPT_MSE;
            else if (strcmp(argv[i], "hybrid") == 0) opt_mode = OPT_HYBRID;
            else {
                fprintf(stderr, "  ERROR: Unknown optimizer '%s'. Use hpc, mse, or hybrid.\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--imatrix") == 0 && i + 1 < argc) {
            imatrix_path = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else {
            fprintf(stderr, "  ERROR: Unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    const char *opt_names[] = {"HPC (BP only)", "MSE (grid search)", "Hybrid (HPC+MSE)"};
    printf("  Input:      %s\n", input_path);
    printf("  Output:     %s\n", output_path);
    printf("  Quant type: Q2_K (2.625 bpw)\n");
    printf("  Optimizer:  %s\n", opt_names[opt_mode]);
    if (imatrix_path) printf("  iMatrix:    %s\n", imatrix_path);
    printf("\n");

    /* ── Phase 1: Load model ── */
    printf("  Phase 1: Loading model...\n");
    clock_t t_start = clock();

    /* Determine if input is a file or directory */
    struct stat st;
    if (stat(input_path, &st) != 0) {
        fprintf(stderr, "  ERROR: Cannot access '%s'\n", input_path);
        return 1;
    }

    STMultiFile *mf = NULL;
    char input_dir[512] = "";

    if (S_ISDIR(st.st_mode)) {
        /* Input is a directory — open all shards */
        mf = st_open_dir(input_path);
        strncpy(input_dir, input_path, sizeof(input_dir) - 2);
        int dlen = strlen(input_dir);
        if (dlen > 0 && input_dir[dlen - 1] != '/') {
            input_dir[dlen] = '/';
            input_dir[dlen + 1] = '\0';
        }
    } else {
        /* Input is a single file — wrap in STMultiFile */
        STFile *sf = st_open(input_path);
        if (!sf) {
            fprintf(stderr, "  ERROR: Failed to open '%s'\n", input_path);
            return 1;
        }
        mf = (STMultiFile *)calloc(1, sizeof(STMultiFile));
        mf->shards[0] = sf;
        mf->n_shards = 1;
        for (int i = 0; i < sf->n_tensors && mf->n_tensors < ST_MAX_TENSORS; i++) {
            strncpy(mf->tensor_map[mf->n_tensors].name,
                    sf->tensors[i].name, ST_MAX_NAME_LEN - 1);
            mf->tensor_map[mf->n_tensors].shard_idx = 0;
            mf->tensor_map[mf->n_tensors].tensor_idx = i;
            mf->n_tensors++;
        }

        /* Extract directory from file path */
        strncpy(input_dir, input_path, sizeof(input_dir) - 1);
        char *last_slash = strrchr(input_dir, '/');
        if (last_slash) {
            *(last_slash + 1) = '\0';
        } else {
            strcpy(input_dir, "./");
        }
    }

    if (!mf) {
        fprintf(stderr, "  ERROR: Failed to load model from '%s'\n", input_path);
        return 1;
    }

    st_multi_print_summary(mf);

    clock_t t_load = clock();
    printf("  Loaded in %.3f seconds\n\n",
           (double)(t_load - t_start) / CLOCKS_PER_SEC);

    /* ── Phase 2: Detect architecture ── */
    printf("  Phase 2: Detecting model architecture...\n");

    /* Try to read config.json from model directory */
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%sconfig.json", input_dir);
    const char *config_ptr = NULL;
    {
        FILE *check = fopen(config_path, "rb");
        if (check) {
            fclose(check);
            config_ptr = config_path;
            printf("  Found config.json: %s\n", config_path);
        }
    }

    ModelArchitecture arch;
    detect_architecture(mf, &arch, config_ptr);

    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  Model Architecture                                         ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Architecture:     %-40s ║\n", arch.architecture);
    printf("  ║  Layers:           %-40u ║\n", arch.block_count);
    printf("  ║  Hidden size:      %-40u ║\n", arch.embedding_length);
    printf("  ║  Attention heads:  %-40u ║\n", arch.head_count);
    printf("  ║  KV heads:         %-40u ║\n", arch.head_count_kv);
    printf("  ║  Vocab size:       %-40u ║\n", arch.vocab_size);
    printf("  ║  FFN size:         %-40u ║\n", arch.feed_forward_length);
    printf("  ║  Context length:   %-40u ║\n", arch.context_length);
    printf("  ║  Has bias:         %-40s ║\n", arch.has_bias ? "yes" : "no");
    printf("  ║  Tied embeddings:  %-40s ║\n", arch.tie_word_embeddings ? "yes" : "no");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* ── Phase 2b: Load tokenizer ── */
    printf("  Phase 2b: Loading tokenizer...\n");
    TokenizerData *tokenizer = NULL;
    {
        char tok_json[512], tok_config[512];
        snprintf(tok_json, sizeof(tok_json), "%stokenizer.json", input_dir);
        snprintf(tok_config, sizeof(tok_config), "%stokenizer_config.json", input_dir);

        tokenizer = tok_load(tok_json, tok_config);
        if (tokenizer) {
            tok_print_summary(tokenizer);
        } else {
            printf("  No tokenizer found in '%s'\n", input_dir);
            printf("  (Output GGUF will lack tokenizer data — not inference-ready)\n\n");
        }
    }

    /* ── Phase 2c: Load importance matrix (optional) ── */
    IMatrixData *imatrix = NULL;
    if (imatrix_path) {
        printf("  Phase 2c: Loading importance matrix...\n");
        imatrix = imatrix_load(imatrix_path);
        if (imatrix) {
            imatrix_print_summary(imatrix);
        } else {
            printf("  WARNING: Failed to load imatrix from '%s'\n", imatrix_path);
            printf("  Proceeding without importance weighting.\n\n");
        }
    }

    /* ── Phase 3-5: Quantize and write GGUF ── */
    printf("  Phase 3: HPC-Optimized Q2_K Quantization + GGUF Output...\n");
    clock_t t_quant_start = clock();

    int result = write_gguf(output_path, mf, &arch, tokenizer,
                              opt_mode, imatrix, verbose);

    clock_t t_end = clock();
    printf("  Total time: %.3f seconds\n\n",
           (double)(t_end - t_start) / CLOCKS_PER_SEC);

    if (imatrix) imatrix_free(imatrix);
    if (tokenizer) tok_free(tokenizer);
    st_multi_close(mf);
    return result;
}
#endif /* HEXSTATE_LIBRARY */
