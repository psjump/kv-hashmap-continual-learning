#include "models.h"

/* ── KV Hashmap (blk.0 Wk,Wv 교체) ── */
#ifdef KV_HASHMAP_ENABLED
#include "llama.h"
#include "kv_hashmap.h"
#include <cstring>

/* 전역 포인터: slot_fusion에서 설정 */
static KVHashmapStack * g_kv_stack = nullptr;

/* custom op userdata (per-layer) */
struct kv_layer_op_data {
    KVHashmapStack * stack;
    int              layer;    /* 0, 1, 2, ... */
    bool             is_key;   /* true=K, false=V */
    int              kv_dim;   /* 128 */
};

static kv_layer_op_data g_layer_ops[KV_MAX_LAYERS * 2];  /* [il*2]=K, [il*2+1]=V */

/* custom op: hashmap HIT시 덮어쓰기, MISS시 원본 유지 + 캡처 모드 */
/* a = 원본 K or V (inplace), b = inp_tokens */
static void kv_hashmap_layer_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,   /* inp_tokens: [n_tokens] i32 */
        int ith, int nth, void * userdata) {
    auto * ud = (kv_layer_op_data *)userdata;
    const int32_t * token_ids = (const int32_t *)b->data;
    float * out = (float *)dst->data;
    int n_tokens = (int)b->ne[0];
    int kv_dim = ud->kv_dim;
    int il = ud->layer;
    KVHashmap * hm = kv_stack_get_layer(ud->stack, il);
    if (!hm) return;

    for (int t = ith; t < n_tokens; t += nth) {
        float * dst_ptr = out + (size_t)t * kv_dim;

        if (ud->stack->capture_mode) {
            /* 캡처 모드: 원본 K,V를 hashmap에 저장 */
            int32_t key;
            if (il == 0) {
                key = token_ids[t];  /* blk.0: token_id */
            } else {
                /* blk.1+: context hash (현재 위치까지의 n-gram) */
                int ctx_len = ud->stack->ctx_size;
                int start = t - ctx_len + 1;
                if (start < 0) start = 0;
                key = kv_context_hash(token_ids + start, t - start + 1);
            }
            if (ud->is_key) {
                /* K 캡처: 아직 V가 없으므로 K만 임시 저장 */
                float zeros[128] = {0};
                if (!kv_hashmap_get(hm, key, NULL, NULL)) {
                    kv_hashmap_put(hm, key, dst_ptr, zeros);
                } else {
                    /* 이미 존재 → K 업데이트 */
                    float v_tmp[128];
                    kv_hashmap_get(hm, key, NULL, v_tmp);
                    kv_hashmap_put(hm, key, dst_ptr, v_tmp);
                }
            } else {
                /* V 캡처: K는 이미 저장됨, V 업데이트 */
                float k_tmp[128];
                if (kv_hashmap_get(hm, key, k_tmp, NULL)) {
                    kv_hashmap_put(hm, key, k_tmp, dst_ptr);
                } else {
                    float zeros[128] = {0};
                    kv_hashmap_put(hm, key, zeros, dst_ptr);
                }
            }
            /* 캡처 모드에서는 원본 값 그대로 유지 */
        } else {
            /* 추론 모드: HIT시 hashmap 값으로 덮어쓰기 */
            int32_t key;
            if (il == 0) {
                key = token_ids[t];
            } else {
                int ctx_len = ud->stack->ctx_size;
                int start = t - ctx_len + 1;
                if (start < 0) start = 0;
                key = kv_context_hash(token_ids + start, t - start + 1);
            }
            if (ud->is_key) {
                kv_hashmap_get(hm, key, dst_ptr, NULL);
            } else {
                kv_hashmap_get(hm, key, NULL, dst_ptr);
            }
        }
    }
}

extern "C" {
    LLAMA_API void qwen2_set_kv_stack(KVHashmapStack * stack) {
        g_kv_stack = stack;
        if (stack) {
            for (int il = 0; il < stack->n_layers && il < KV_MAX_LAYERS; il++) {
                g_layer_ops[il * 2]     = { stack, il, true,  stack->kv_dim };
                g_layer_ops[il * 2 + 1] = { stack, il, false, stack->kv_dim };
            }
        }
    }
    LLAMA_API KVHashmapStack * qwen2_get_kv_stack(void) {
        return g_kv_stack;
    }
    /* 하위 호환: 단일 hashmap API → blk.0 전용 */
    LLAMA_API void qwen2_set_kv_hashmap(KVHashmap * hm) {
        /* 단일 hashmap은 stack의 layer 0으로 래핑 */
        static KVHashmapStack single_stack;
        if (hm) {
            memset(&single_stack, 0, sizeof(single_stack));
            single_stack.layers[0] = hm;
            single_stack.n_layers = 1;
            single_stack.kv_dim = hm->kv_dim;
            single_stack.ctx_size = 5;
            single_stack.capture_mode = 0;
            qwen2_set_kv_stack(&single_stack);
        } else {
            qwen2_set_kv_stack(nullptr);
        }
    }
}
#endif /* KV_HASHMAP_ENABLED */

void llama_model_qwen2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
        case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
        case 48: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_b    = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            ggml_tensor * Qcur, * Kcur, * Vcur;

            {
                // ── 원본 path: Wk,Wv 행렬곱 (항상 실행) ──
                auto [Q, K, V] = build_qkv(model.layers[il], cur,
                        n_embd_head, n_head, n_head_kv, il);
                Qcur = Q; Kcur = K; Vcur = V;
            }

#ifdef KV_HASHMAP_ENABLED
            if (g_kv_stack != nullptr && il < g_kv_stack->n_layers
                    && kv_stack_get_layer(g_kv_stack, il) != nullptr) {
                // ── Hashmap overlay: HIT → hashmap K,V 사용, MISS → 원본 유지 ──
                ggml_tensor * inp_tok = res->t_inp_tokens;
                const int64_t n_embd_kv = n_embd_head * n_head_kv;

                Kcur = ggml_reshape_2d(ctx0, Kcur, n_embd_kv, n_tokens);
                Kcur = ggml_map_custom2_inplace(ctx0, Kcur, inp_tok,
                        kv_hashmap_layer_op, 1, &g_layer_ops[il * 2]);
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);

                Vcur = ggml_reshape_2d(ctx0, Vcur, n_embd_kv, n_tokens);
                Vcur = ggml_map_custom2_inplace(ctx0, Vcur, inp_tok,
                        kv_hashmap_layer_op, 1, &g_layer_ops[il * 2 + 1]);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            }
#endif

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
