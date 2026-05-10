/**
 * Slot-LLM Fusion — 방법 C: 블랙박스 비율 점진적 조절
 *
 * 기존 LLM(Qwen2 등)의 언어 능력을 유지하면서
 * Slot Memory로 continual learning을 추가하는 융합 프로그램.
 *
 * 구조:
 *   LLM logits  ─┐
 *                 ├─ Gating(자동학습) ─→ 최종 logits ─→ sampling
 *   Slot scores ─┘
 *
 * Gating: final[i] = softmax( llm_logits[i] + alpha * slot_scores[i] )
 *   alpha: 학습 데이터에서 자동 조절 (AdaGrad)
 *
 * 사용:
 *   slot_fusion train -m model.gguf -d data.txt [-s slot.bin]
 *   slot_fusion infer -m model.gguf -s slot.bin -p "프롬프트"
 *   slot_fusion test  -m model.gguf -s slot.bin  (continual learning 검증)
 */

#include "llama.h"
#include "../slot_plugin/slot.h"
#include "../slot_plugin/kv_hashmap.h"

/* qwen2.cpp에서 정의한 hashmap 설정 함수 (KV_HASHMAP_ENABLED 시) */
#ifdef KV_HASHMAP_ENABLED
extern "C" void qwen2_set_kv_hashmap(KVHashmap * hm);
extern "C" void qwen2_set_kv_stack(KVHashmapStack * stack);
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <clocale>

/* ── 설정 ── */
#define SLOT_CTX_SIZE    5    /* n-gram context for slot */
#define SLOT_MIN_CTX     2
#define DEFAULT_ALPHA    5.0f /* slot score 초기 가중치 */
#define ADAGRAD_LR       0.1f
#define MAX_GEN_TOKENS   64

/* ── 전역 상태 ── */
struct FusionState {
    llama_model   *model   = nullptr;
    llama_context *ctx     = nullptr;
    const llama_vocab *vocab = nullptr;
    SlotMemory    *slot    = nullptr;

    float alpha     = DEFAULT_ALPHA;  /* slot 블렌딩 가중치 (자동학습) */
    float alpha_acc = 1.0f;           /* AdaGrad 누적 */
    int   n_vocab   = 0;
    int   n_gpu     = 99;
};

/* ── 유틸리티 ── */

/** UTF-8 한국어 텍스트를 LLM 토크나이저로 변환 */
static std::vector<llama_token> tokenize(const llama_vocab *vocab,
                                         const std::string &text,
                                         bool add_bos) {
    int n = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, add_bos, true);
    if (n <= 0) return {};
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_bos, true);
    return tokens;
}

/** 토큰 ID → 텍스트 */
static std::string token_to_str(const llama_vocab *vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) return "";
    return std::string(buf, n);
}

/** 파일에서 줄 단위 읽기 */
static std::vector<std::string> read_lines(const char *path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        return lines;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

/* ── 모델 초기화 ── */

static bool init_model(FusionState &st, const char *model_path) {
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = st.n_gpu;

    st.model = llama_model_load_from_file(model_path, mparams);
    if (!st.model) {
        fprintf(stderr, "ERROR: failed to load model: %s\n", model_path);
        return false;
    }
    st.vocab = llama_model_get_vocab(st.model);
    st.n_vocab = llama_vocab_n_tokens(st.vocab);
    printf("[init] model loaded: %s (vocab=%d)\n", model_path, st.n_vocab);
    return true;
}

static bool init_context(FusionState &st, int ctx_size) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = ctx_size;
    cparams.n_batch = ctx_size;
    cparams.no_perf = true;
    st.ctx = llama_init_from_model(st.model, cparams);
    if (!st.ctx) {
        fprintf(stderr, "ERROR: failed to create context\n");
        return false;
    }
    return true;
}

/* ── TRAIN: slot memory 학습 ── */

static void do_train(FusionState &st, const char *data_path, const char *slot_path) {
    auto lines = read_lines(data_path);
    if (lines.empty()) return;

    st.slot = slot_create(SLOT_CTX_SIZE, SLOT_MIN_CTX);

    int total_tokens = 0;
    int total_hits   = 0;

    printf("[train] %zu lines, building slot memory...\n", lines.size());

    for (size_t li = 0; li < lines.size(); li++) {
        auto tokens = tokenize(st.vocab, lines[li], false);
        if (tokens.size() < 3) continue;

        /* slot에 모든 n-gram 패턴 학습 */
        for (size_t i = SLOT_MIN_CTX; i < tokens.size(); i++) {
            int ctx_len = (int)i < SLOT_CTX_SIZE ? (int)i : SLOT_CTX_SIZE;
            int start   = (int)i - ctx_len;
            slot_learn_multi(st.slot, tokens.data() + start, ctx_len, tokens[i]);
            total_tokens++;
        }

        /* alpha 자동 조절: LLM이 맞추는 비율 측정 */
        if (st.ctx) {
            /* 문맥 초기화 */
            llama_memory_clear(llama_get_memory(st.ctx), true);

            llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
            if (llama_decode(st.ctx, batch) == 0) {
                for (size_t i = 0; i + 1 < tokens.size(); i++) {
                    float *logits = llama_get_logits_ith(st.ctx, (int)i);
                    if (!logits) continue;

                    /* LLM top-1 예측 확인 */
                    int llm_pred = 0;
                    float llm_max = logits[0];
                    for (int v = 1; v < st.n_vocab; v++) {
                        if (logits[v] > llm_max) { llm_max = logits[v]; llm_pred = v; }
                    }

                    int target = tokens[i + 1];

                    /* slot 조회 */
                    int ctx_len = (int)i + 1 < SLOT_CTX_SIZE ? (int)i + 1 : SLOT_CTX_SIZE;
                    int ctx_start = (int)i + 1 - ctx_len;
                    int slot_count = 0;
                    const int *slot_tokens = slot_lookup(st.slot,
                        tokens.data() + ctx_start, ctx_len, &slot_count);

                    bool slot_hit = false;
                    if (slot_tokens) {
                        for (int s = 0; s < slot_count; s++) {
                            if (slot_tokens[s] == target) { slot_hit = true; break; }
                        }
                    }

                    /* AdaGrad alpha 업데이트:
                     * LLM 틀리고 slot 맞으면 → alpha 증가
                     * LLM 맞고 slot 틀리면 → alpha 감소 */
                    float grad = 0.0f;
                    if (llm_pred != target && slot_hit)  grad = +1.0f;
                    if (llm_pred == target && !slot_hit)  grad = -0.5f;
                    if (llm_pred != target && !slot_hit)  grad =  0.0f;
                    if (llm_pred == target && slot_hit)   { grad = 0.0f; total_hits++; }

                    if (grad != 0.0f) {
                        st.alpha_acc += grad * grad;
                        float eff_lr = ADAGRAD_LR / sqrtf(st.alpha_acc);
                        st.alpha += eff_lr * grad;
                        if (st.alpha < 0.1f) st.alpha = 0.1f;
                        if (st.alpha > 50.0f) st.alpha = 50.0f;
                    }
                }
            }
        }

        if ((li + 1) % 50 == 0) {
            printf("  [%zu/%zu] tokens=%d, alpha=%.2f\n",
                   li + 1, lines.size(), total_tokens, st.alpha);
        }
    }

    printf("[train] done: %d tokens learned, alpha=%.2f\n", total_tokens, st.alpha);
    slot_print_stats(st.slot);

    /* 저장 */
    if (slot_save(st.slot, slot_path) == 0) {
        printf("[train] slot saved: %s\n", slot_path);
    }

    /* alpha 저장 */
    std::string alpha_path = std::string(slot_path) + ".alpha";
    FILE *af = fopen(alpha_path.c_str(), "wb");
    if (af) {
        fwrite(&st.alpha, sizeof(float), 1, af);
        fwrite(&st.alpha_acc, sizeof(float), 1, af);
        fclose(af);
        printf("[train] alpha saved: %.2f (acc=%.2f)\n", st.alpha, st.alpha_acc);
    }
}

/* ── INFER: slot-augmented 추론 ── */

static std::string g_logit_dump_path;  /* --logit-dump 경로 (비어있으면 미사용) */

static void do_infer(FusionState &st, const char *prompt_str, int max_tokens) {
    if (!st.slot) {
        fprintf(stderr, "ERROR: no slot memory loaded\n");
        return;
    }

    auto prompt_tokens = tokenize(st.vocab, prompt_str, true);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "ERROR: failed to tokenize prompt\n");
        return;
    }

    int ctx_size = (int)prompt_tokens.size() + max_tokens + 16;
    if (!init_context(st, ctx_size)) return;

    /* 프롬프트 디코드 */
    llama_memory_clear(llama_get_memory(st.ctx), true);
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(st.ctx, batch) != 0) {
        fprintf(stderr, "ERROR: prompt decode failed\n");
        return;
    }

    /* 디버그 파일 열기 (UTF-8) */
    FILE *dbg = fopen("fusion_debug.txt", "ab");
    if (dbg) {
        fprintf(dbg, "\n=== Prompt: %s ===\n", prompt_str);
    }

    /* 프롬프트 출력 */
    std::string prompt_text;
    for (auto id : prompt_tokens) prompt_text += token_to_str(st.vocab, id);
    printf("[prompt] ");
    if (dbg) fprintf(dbg, "[prompt] %s\n", prompt_text.c_str());
    fflush(stdout);

    /* 생성 루프에서 사용할 전체 토큰 기록 */
    std::vector<llama_token> all_tokens(prompt_tokens);
    float *slot_scores = (float *)calloc(st.n_vocab, sizeof(float));
    std::string gen_text;

    for (int step = 0; step < max_tokens; step++) {
        /* LLM logits 가져오기 */
        float *logits = llama_get_logits_ith(st.ctx, -1);
        if (!logits) break;

        /* Slot scores 계산 */
        int ctx_len = (int)all_tokens.size();
        int sl_ctx = ctx_len < SLOT_CTX_SIZE ? ctx_len : SLOT_CTX_SIZE;
        int sl_start = ctx_len - sl_ctx;
        slot_score_vocab(st.slot, all_tokens.data() + sl_start, sl_ctx,
                         slot_scores, st.n_vocab);

        /* 블렌딩: logits[i] += alpha * slot_scores[i] */
        int slot_hits = 0;
        for (int v = 0; v < st.n_vocab; v++) {
            if (slot_scores[v] > 0.0f) {
                logits[v] += st.alpha * slot_scores[v];
                slot_hits++;
            }
        }

        /* Greedy sampling (logits 이미 수정됨) */
        llama_token best = 0;
        float best_logit = logits[0];
        for (int v = 1; v < st.n_vocab; v++) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best = v;
            }
        }

        /* Logit dump: step별 top-20 + 특정 토큰 점수 기록 */
        if (!g_logit_dump_path.empty()) {
            FILE *ld = fopen(g_logit_dump_path.c_str(), step == 0 ? "w" : "a");
            if (ld) {
                fprintf(ld, "=== step=%d ===\n", step);
                /* top-20 정렬 */
                std::vector<std::pair<float,int>> ranked;
                for (int v = 0; v < st.n_vocab; v++) {
                    ranked.push_back({logits[v], v});
                }
                std::partial_sort(ranked.begin(), ranked.begin()+20, ranked.end(),
                    [](auto&a, auto&b){return a.first > b.first;});
                for (int r = 0; r < 20; r++) {
                    std::string tok = token_to_str(st.vocab, ranked[r].second);
                    float slot_s = slot_scores[ranked[r].second];
                    fprintf(ld, "  rank=%2d id=%6d logit=%8.3f slot=%5.2f token=\"%s\"\n",
                            r, ranked[r].second, ranked[r].first, slot_s, tok.c_str());
                }
                /* 관심 토큰들 (swim, sleep, eat, play, ocean) */
                int watch_ids[] = {16191, 25809, 8055, 1387, 17951, 4616, -1};
                const char* watch_names[] = {"swim","sleep","eat","play","ocean","cat"};
                fprintf(ld, "  --- watched ---\n");
                for (int w = 0; watch_ids[w] >= 0; w++) {
                    int wid = watch_ids[w];
                    if (wid < st.n_vocab) {
                        fprintf(ld, "  %6s id=%6d logit=%8.3f slot=%5.2f\n",
                                watch_names[w], wid, logits[wid], slot_scores[wid]);
                    }
                }
                fprintf(ld, "\n");
                fclose(ld);
            }
        }

        /* EOS 체크 */
        if (llama_vocab_is_eog(st.vocab, best)) break;

        /* 출력 */
        std::string piece = token_to_str(st.vocab, best);
        printf("%s", piece.c_str());
        fflush(stdout);
        gen_text += piece;

        /* 다음 step 준비 */
        all_tokens.push_back(best);
        batch = llama_batch_get_one(&best, 1);
        if (llama_decode(st.ctx, batch) != 0) break;
    }

    printf("\n[info] alpha=%.2f, total_tokens=%d\n", st.alpha, (int)all_tokens.size());
    if (dbg) {
        fprintf(dbg, "[output] %s\n", gen_text.c_str());
        fprintf(dbg, "[info] alpha=%.2f, total_tokens=%d\n", st.alpha, (int)all_tokens.size());
        fclose(dbg);
    }
    free(slot_scores);

    llama_free(st.ctx);
    st.ctx = nullptr;
}

/* ── TEST: continual learning 검증 ── */

static void do_test(FusionState &st) {
    if (!st.slot) {
        fprintf(stderr, "ERROR: no slot memory loaded for test\n");
        return;
    }

    printf("\n=== Continual Learning Test ===\n");
    printf("alpha=%.2f\n\n", st.alpha);

    const char *prompts[] = {
        "너의 이름은 뭐니?",
        "너는 누구니?",
        "너는 어디에 사니?",
        "너는 무엇을 좋아하니?",
        "펭귄은 무슨 색이냐?",
        "고양이는 무엇을 좋아하나?",
        NULL
    };

    for (int i = 0; prompts[i]; i++) {
        printf("Q: %s\n", prompts[i]);
        do_infer(st, prompts[i], 32);
        printf("\n");
    }
}

/* ── MAIN ── */

static void print_usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s train -m model.gguf -d data.txt [-s slot.bin]\n", prog);
    printf("  %s infer -m model.gguf -s slot.bin -p \"prompt\"\n", prog);
    printf("  %s test  -m model.gguf -s slot.bin\n", prog);
    printf("  Options: -ngl N (GPU layers), -n N (max tokens)\n");
}

int main(int argc, char **argv) {
    std::setlocale(LC_ALL, "");

    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string mode = argv[1];
    std::string model_path, data_path, slot_path = "slot.bin", prompt;
    std::string kv_hashmap_path;
    std::string logit_dump_path;
    int max_tokens = MAX_GEN_TOKENS;
    FusionState st;

    /* 인자 파싱 */
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc)   model_path = argv[++i];
        else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) data_path = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) slot_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) st.n_gpu = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv-hashmap") == 0 && i + 1 < argc) kv_hashmap_path = argv[++i];
        else if (strcmp(argv[i], "--logit-dump") == 0 && i + 1 < argc) logit_dump_path = argv[++i];
    }

    if (model_path.empty()) {
        fprintf(stderr, "ERROR: -m model.gguf required\n");
        print_usage(argv[0]);
        return 1;
    }

    /* KV Hashmap 로드 (blk.0+ Wk,Wv 교체) */
#ifdef KV_HASHMAP_ENABLED
    KVHashmap *kv_hm = nullptr;
    KVHashmapStack *kv_stk = nullptr;
    if (!kv_hashmap_path.empty()) {
        /* stack 포맷 먼저 시도, 실패하면 단일 hashmap */
        kv_stk = kv_stack_load(kv_hashmap_path.c_str());
        if (kv_stk) {
            printf("[kv-stack] loaded: %s (%d layers)\n", kv_hashmap_path.c_str(), kv_stk->n_layers);
            for (int il = 0; il < kv_stk->n_layers; il++) {
                KVHashmap *lhm = kv_stack_get_layer(kv_stk, il);
                if (lhm && lhm->size > 0) { printf("  L%d: ", il); kv_hashmap_print_stats(lhm); }
            }
            qwen2_set_kv_stack(kv_stk);
        } else {
            kv_hm = kv_hashmap_load(kv_hashmap_path.c_str());
            if (kv_hm) {
                printf("[kv-hashmap] loaded: %s\n", kv_hashmap_path.c_str());
                kv_hashmap_print_stats(kv_hm);
                qwen2_set_kv_hashmap(kv_hm);
            } else {
                fprintf(stderr, "WARNING: failed to load kv-hashmap: %s\n", kv_hashmap_path.c_str());
            }
        }
    }
#endif

    /* 모델 로드 */
    if (!init_model(st, model_path.c_str())) return 1;

    if (mode == "train") {
        if (data_path.empty()) {
            fprintf(stderr, "ERROR: -d data.txt required for train\n");
            return 1;
        }
        /* 학습 시 context 필요 (alpha 자동 조절용) */
        init_context(st, 512);
        do_train(st, data_path.c_str(), slot_path.c_str());
        if (st.ctx) { llama_free(st.ctx); st.ctx = nullptr; }
    }
    else if (mode == "infer") {
        if (prompt.empty()) {
            fprintf(stderr, "ERROR: -p \"prompt\" required for infer\n");
            return 1;
        }
        /* logit dump 설정 */
        g_logit_dump_path = logit_dump_path;
        /* slot 로드 */
        st.slot = slot_load(slot_path.c_str());
        if (!st.slot) {
            fprintf(stderr, "ERROR: failed to load slot: %s\n", slot_path.c_str());
            return 1;
        }
        /* alpha 로드 */
        std::string alpha_path = slot_path + ".alpha";
        FILE *af = fopen(alpha_path.c_str(), "rb");
        if (af) {
            fread(&st.alpha, sizeof(float), 1, af);
            fread(&st.alpha_acc, sizeof(float), 1, af);
            fclose(af);
            printf("[load] alpha=%.2f (acc=%.2f)\n", st.alpha, st.alpha_acc);
        }
        slot_print_stats(st.slot);
        do_infer(st, prompt.c_str(), max_tokens);
    }
    else if (mode == "test") {
        st.slot = slot_load(slot_path.c_str());
        if (!st.slot) {
            fprintf(stderr, "ERROR: failed to load slot: %s\n", slot_path.c_str());
            return 1;
        }
        std::string alpha_path = slot_path + ".alpha";
        FILE *af = fopen(alpha_path.c_str(), "rb");
        if (af) {
            fread(&st.alpha, sizeof(float), 1, af);
            fread(&st.alpha_acc, sizeof(float), 1, af);
            fclose(af);
        }
        slot_print_stats(st.slot);
        do_test(st);
    }
    else if (mode == "capture-kv") {
#ifdef KV_HASHMAP_ENABLED
        /* K,V 캡처: Python tools/build_kv_hashmap.py로 위임 */
        /* blk.1+는 context-dependent → Python에서 레이어별 계산 */
        printf("[capture-kv] Use Python for multi-layer K,V capture:\n");
        printf("  python tools/build_kv_hashmap.py --mode build --full-vocab\n");
        printf("  python tools/build_kv_stack.py --layers 2 --data <file>\n");
#else
        fprintf(stderr, "ERROR: KV_HASHMAP_ENABLED not defined\n");
#endif
        return 0;
    }
    else {
        fprintf(stderr, "ERROR: unknown mode '%s'\n", mode.c_str());
        print_usage(argv[0]);
        return 1;
    }

    /* 정리 */
#ifdef KV_HASHMAP_ENABLED
    if (kv_stk) { qwen2_set_kv_stack(nullptr); kv_stack_free(kv_stk); }
    else if (kv_hm) { qwen2_set_kv_hashmap(nullptr); kv_hashmap_free(kv_hm); }
#endif
    if (st.slot) slot_free(st.slot);
    if (st.ctx)  llama_free(st.ctx);
    if (st.model) llama_model_free(st.model);

    return 0;
}
