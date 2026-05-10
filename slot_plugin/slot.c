/**
 * Slot Memory — C 구현
 *
 * Java SlotModel 1:1 포팅.
 * 해시: h = len; for(i) h = h*31 + context[i]  (Java hashContextN 동일)
 * 테이블: open addressing + linear probing
 * save/load: Java v3 int[] 포맷 바이너리 호환
 */
#include "slot.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── 내부 함수 ── */

/** 해시 → 버킷 인덱스 */
static int bucket_idx(int capacity, uint64_t key) {
    return (int)(key & (uint64_t)(capacity - 1));  /* capacity는 2의 거듭제곱 */
}

/** 해시 테이블 리사이즈 (2배 확장) */
static void slot_resize(SlotMemory *mem) {
    int old_cap = mem->capacity;
    int new_cap = old_cap * 2;
    SlotEntry *old = mem->buckets;
    SlotEntry *new_buckets = (SlotEntry *)calloc(new_cap, sizeof(SlotEntry));

    for (int i = 0; i < old_cap; i++) {
        if (old[i].key == 0) continue;
        int idx = bucket_idx(new_cap, old[i].key);
        while (new_buckets[idx].key != 0)
            idx = (idx + 1) & (new_cap - 1);
        new_buckets[idx] = old[i];  /* 포인터 이동 (deep copy 불필요) */
    }

    free(old);
    mem->buckets = new_buckets;
    mem->capacity = new_cap;
}

/** 엔트리 찾기 또는 빈 슬롯 반환 */
static SlotEntry *slot_find_or_empty(SlotMemory *mem, uint64_t key) {
    int idx = bucket_idx(mem->capacity, key);
    while (1) {
        SlotEntry *e = &mem->buckets[idx];
        if (e->key == 0 || e->key == key) return e;
        idx = (idx + 1) & (mem->capacity - 1);
    }
}

/* ── 공개 API ── */

SlotMemory *slot_create(int ctx_size, int min_ctx) {
    SlotMemory *mem = (SlotMemory *)calloc(1, sizeof(SlotMemory));
    mem->capacity = SLOT_INITIAL_CAPACITY;
    mem->buckets = (SlotEntry *)calloc(mem->capacity, sizeof(SlotEntry));
    mem->size = 0;
    mem->ctx_size = (ctx_size > 0) ? ctx_size : SLOT_CONTEXT_SIZE;
    mem->min_ctx = (min_ctx > 0) ? min_ctx : SLOT_MIN_CTX;
    return mem;
}

void slot_free(SlotMemory *mem) {
    if (!mem) return;
    for (int i = 0; i < mem->capacity; i++) {
        if (mem->buckets[i].tokens)
            free(mem->buckets[i].tokens);
    }
    free(mem->buckets);
    free(mem);
}

uint64_t slot_hash(const int *context, int len) {
    /* Java hashContextN() 동일: h = len; for(i) h = h*31 + context[i] */
    uint64_t h = (uint64_t)len;
    for (int i = 0; i < len; i++)
        h = h * 31 + (uint64_t)(uint32_t)context[i];
    return h;
}

void slot_learn(SlotMemory *mem, const int *context, int ctx_len, int target) {
    /* 리사이즈 체크 */
    if ((float)mem->size / mem->capacity > SLOT_LOAD_FACTOR)
        slot_resize(mem);

    uint64_t key = slot_hash(context, ctx_len);
    /* key=0은 빈 슬롯 마커이므로, 실제 해시가 0이면 1로 변환 */
    if (key == 0) key = 1;

    SlotEntry *e = slot_find_or_empty(mem, key);

    if (e->key == 0) {
        /* 새 엔트리 */
        e->key = key;
        e->capacity = 4;
        e->tokens = (int *)malloc(e->capacity * sizeof(int));
        e->tokens[0] = target;
        e->count = 1;
        mem->size++;
        return;
    }

    /* 기존 엔트리 — 중복 체크 */
    for (int i = 0; i < e->count; i++)
        if (e->tokens[i] == target) return;  /* 이미 ON */

    /* 추가 */
    if (e->count >= e->capacity) {
        e->capacity *= 2;
        e->tokens = (int *)realloc(e->tokens, e->capacity * sizeof(int));
    }
    e->tokens[e->count++] = target;
}

void slot_learn_multi(SlotMemory *mem, const int *full_context, int full_len, int target) {
    /* Java learnMultiCtx: ctx_size부터 min_ctx까지 backoff */
    for (int ctx_len = mem->ctx_size; ctx_len >= mem->min_ctx; ctx_len--) {
        int start = full_len - ctx_len;
        if (start < 0) continue;
        slot_learn(mem, full_context + start, ctx_len, target);
    }
}

const int *slot_lookup(const SlotMemory *mem, const int *context, int ctx_len, int *out_count) {
    uint64_t key = slot_hash(context, ctx_len);
    if (key == 0) key = 1;

    int idx = bucket_idx(mem->capacity, key);
    while (1) {
        const SlotEntry *e = &mem->buckets[idx];
        if (e->key == 0) { *out_count = 0; return NULL; }
        if (e->key == key) { *out_count = e->count; return e->tokens; }
        idx = (idx + 1) & (mem->capacity - 1);
    }
}

void slot_score_vocab(const SlotMemory *mem, const int *context, int ctx_len,
                      float *scores, int vocab_size) {
    memset(scores, 0, vocab_size * sizeof(float));

    /* backoff: 가장 긴 매칭부터 시도 */
    int max_ctx = (ctx_len < mem->ctx_size) ? ctx_len : mem->ctx_size;
    for (int cl = max_ctx; cl >= mem->min_ctx; cl--) {
        int start = ctx_len - cl;
        if (start < 0) continue;

        int count = 0;
        const int *tokens = slot_lookup(mem, context + start, cl, &count);
        if (tokens && count > 0) {
            /* HIT — 관측된 토큰에 점수 부여 */
            /* 긴 context = 더 정확 → 점수 비례 */
            float score = (float)cl;  /* Java: slotBonus와 유사 */
            for (int i = 0; i < count; i++) {
                if (tokens[i] >= 0 && tokens[i] < vocab_size)
                    scores[tokens[i]] += score;
            }
            return;  /* 가장 긴 매칭만 사용 (backoff 중단) */
        }
    }
    /* MISS — 모든 context에서 매칭 없음 → scores는 0 */
}

/* ── Save/Load (Java v3 바이너리 호환) ── */

/** Big-endian int 쓰기 (Java DataOutputStream 호환) */
static void write_int_be(FILE *f, int32_t v) {
    uint8_t b[4] = { (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF };
    fwrite(b, 1, 4, f);
}
static void write_long_be(FILE *f, int64_t v) {
    uint8_t b[8];
    for (int i = 7; i >= 0; i--) { b[i] = v & 0xFF; v >>= 8; }
    fwrite(b, 1, 8, f);
}
static void write_double_be(FILE *f, double v) {
    uint64_t u; memcpy(&u, &v, 8);
    write_long_be(f, (int64_t)u);
}

/** Big-endian int 읽기 (Java DataInputStream 호환) */
static int32_t read_int_be(FILE *f) {
    uint8_t b[4]; fread(b, 1, 4, f);
    return (int32_t)((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]);
}
static int64_t read_long_be(FILE *f) {
    uint8_t b[8]; fread(b, 1, 8, f);
    int64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | b[i];
    return v;
}
static double read_double_be(FILE *f) {
    int64_t v = read_long_be(f);
    uint64_t u = (uint64_t)v;
    double d; memcpy(&d, &u, 8);
    return d;
}

int slot_save(const SlotMemory *mem, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* 헤더: vocabSize, contextSize, countLr (Java 호환) */
    write_int_be(f, 0);            /* vocabSize: 미사용, 0 */
    write_int_be(f, mem->ctx_size);
    write_double_be(f, 0.001);     /* countLr: 미사용 */

    /* histogram: v3 int[] 포맷 */
    write_int_be(f, -3);           /* SAVE_VERSION_INTARRAY 마커 */
    write_int_be(f, mem->size);
    for (int i = 0; i < mem->capacity; i++) {
        SlotEntry *e = &mem->buckets[i];
        if (e->key == 0) continue;
        write_long_be(f, (int64_t)e->key);
        write_int_be(f, e->count);
        for (int j = 0; j < e->count; j++)
            write_int_be(f, e->tokens[j]);
    }

    /* contextList: 빈 리스트 (호환성 유지) */
    write_int_be(f, -3);
    write_int_be(f, 0);

    fclose(f);
    return 0;
}

SlotMemory *slot_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    int vs = read_int_be(f);       /* vocabSize */
    int cs = read_int_be(f);       /* contextSize */
    double lr = read_double_be(f); /* countLr */
    (void)vs; (void)lr;

    SlotMemory *mem = slot_create(cs, SLOT_MIN_CTX);

    int marker = read_int_be(f);
    if (marker == -3) {
        /* v3 int[] 포맷 */
        int n = read_int_be(f);
        for (int e = 0; e < n; e++) {
            uint64_t key = (uint64_t)read_long_be(f);
            int count = read_int_be(f);

            /* 리사이즈 체크 */
            if ((float)mem->size / mem->capacity > SLOT_LOAD_FACTOR)
                slot_resize(mem);

            if (key == 0) key = 1;
            SlotEntry *entry = slot_find_or_empty(mem, key);
            entry->key = key;
            entry->count = count;
            entry->capacity = count > 4 ? count : 4;
            entry->tokens = (int *)malloc(entry->capacity * sizeof(int));
            for (int i = 0; i < count; i++)
                entry->tokens[i] = read_int_be(f);
            mem->size++;
        }
    } else {
        /* v1/v2 포맷 (HashMap<Integer, Double>) */
        int n = marker;  /* marker가 곧 엔트리 수 */
        for (int e = 0; e < n; e++) {
            uint64_t key = (uint64_t)read_long_be(f);
            int count = read_int_be(f);

            if ((float)mem->size / mem->capacity > SLOT_LOAD_FACTOR)
                slot_resize(mem);

            if (key == 0) key = 1;
            SlotEntry *entry = slot_find_or_empty(mem, key);
            entry->key = key;
            entry->count = count;
            entry->capacity = count > 4 ? count : 4;
            entry->tokens = (int *)malloc(entry->capacity * sizeof(int));
            for (int i = 0; i < count; i++) {
                entry->tokens[i] = read_int_be(f);
                read_double_be(f);  /* v1/v2: Double 값 스킵 */
            }
            mem->size++;
        }
    }

    /* contextList 스킵 */
    int ctx_marker = read_int_be(f);
    if (ctx_marker == -3 || ctx_marker == -2) {
        int ctx_count = read_int_be(f);
        for (int i = 0; i < ctx_count; i++) {
            int len = read_int_be(f);
            for (int j = 0; j < len; j++) read_int_be(f);
        }
    }

    fclose(f);
    return mem;
}

void slot_print_stats(const SlotMemory *mem) {
    int max_tokens = 0, total_tokens = 0;
    for (int i = 0; i < mem->capacity; i++) {
        if (mem->buckets[i].key == 0) continue;
        total_tokens += mem->buckets[i].count;
        if (mem->buckets[i].count > max_tokens)
            max_tokens = mem->buckets[i].count;
    }
    printf("SlotMemory: %d entries, %d total tokens, max %d tokens/slot, "
           "load %.1f%%, ctx_size=%d\n",
           mem->size, total_tokens, max_tokens,
           100.0f * mem->size / mem->capacity, mem->ctx_size);
}
