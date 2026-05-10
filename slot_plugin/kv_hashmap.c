/**
 * KV Hashmap — C 구현
 *
 * slot.c와 동일 패턴: open addressing + linear probing
 * key = token_id (int32), value = K[kv_dim] + V[kv_dim] floats
 *
 * 바이너리 포맷 (Python build_kv_hashmap.py 호환):
 *   [magic: "KVH0" 4B] [version: i32] [kv_dim: i32] [n_entries: i32]
 *   [entries: n × (token_id:i32 + K:kv_dim×f32 + V:kv_dim×f32)]
 *   모든 값 little-endian
 */
#include "kv_hashmap.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── 내부 함수 ── */

/** token_id → 버킷 인덱스 */
static int kv_bucket_idx(int capacity, int32_t token_id) {
    /* token_id를 양수로 변환하여 해시 */
    uint32_t h = (uint32_t)token_id;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = (h >> 16) ^ h;
    return (int)(h & (uint32_t)(capacity - 1));
}

/** 리사이즈 (2배 확장) */
static void kv_resize(KVHashmap *hm) {
    int old_cap = hm->capacity;
    int new_cap = old_cap * 2;
    KVEntry *old = hm->buckets;
    KVEntry *new_buckets = (KVEntry *)calloc(new_cap, sizeof(KVEntry));

    /* 빈 슬롯 초기화 */
    for (int i = 0; i < new_cap; i++)
        new_buckets[i].token_id = -1;

    /* 기존 엔트리 재배치 */
    for (int i = 0; i < old_cap; i++) {
        if (old[i].token_id < 0) continue;
        int idx = kv_bucket_idx(new_cap, old[i].token_id);
        while (new_buckets[idx].token_id >= 0)
            idx = (idx + 1) & (new_cap - 1);
        new_buckets[idx] = old[i];  /* 포인터 이동 */
    }

    free(old);
    hm->buckets = new_buckets;
    hm->capacity = new_cap;
}

/** 엔트리 찾기 또는 빈 슬롯 반환 */
static KVEntry *kv_find_or_empty(KVHashmap *hm, int32_t token_id) {
    int idx = kv_bucket_idx(hm->capacity, token_id);
    while (1) {
        KVEntry *e = &hm->buckets[idx];
        if (e->token_id < 0 || e->token_id == token_id) return e;
        idx = (idx + 1) & (hm->capacity - 1);
    }
}

static const KVEntry *kv_find(const KVHashmap *hm, int32_t token_id) {
    int idx = kv_bucket_idx(hm->capacity, token_id);
    while (1) {
        const KVEntry *e = &hm->buckets[idx];
        if (e->token_id < 0) return NULL;
        if (e->token_id == token_id) return e;
        idx = (idx + 1) & (hm->capacity - 1);
    }
}

/* ── 공개 API ── */

KVHashmap *kv_hashmap_create(int kv_dim) {
    KVHashmap *hm = (KVHashmap *)calloc(1, sizeof(KVHashmap));
    hm->capacity = KV_INITIAL_CAPACITY;
    hm->buckets = (KVEntry *)calloc(hm->capacity, sizeof(KVEntry));
    hm->size = 0;
    hm->kv_dim = kv_dim;

    /* 빈 슬롯 초기화 */
    for (int i = 0; i < hm->capacity; i++)
        hm->buckets[i].token_id = -1;

    return hm;
}

void kv_hashmap_free(KVHashmap *hm) {
    if (!hm) return;
    for (int i = 0; i < hm->capacity; i++) {
        if (hm->buckets[i].token_id >= 0) {
            free(hm->buckets[i].k_vec);
            free(hm->buckets[i].v_vec);
        }
    }
    free(hm->buckets);
    free(hm);
}

void kv_hashmap_put(KVHashmap *hm, int32_t token_id,
                    const float *k_vec, const float *v_vec) {
    /* 리사이즈 체크 */
    if ((float)hm->size / hm->capacity > KV_LOAD_FACTOR)
        kv_resize(hm);

    KVEntry *e = kv_find_or_empty(hm, token_id);

    if (e->token_id < 0) {
        /* 새 엔트리 */
        e->token_id = token_id;
        e->k_vec = (float *)malloc(hm->kv_dim * sizeof(float));
        e->v_vec = (float *)malloc(hm->kv_dim * sizeof(float));
        hm->size++;
    }

    memcpy(e->k_vec, k_vec, hm->kv_dim * sizeof(float));
    memcpy(e->v_vec, v_vec, hm->kv_dim * sizeof(float));
}

bool kv_hashmap_get(const KVHashmap *hm, int32_t token_id,
                    float *out_k, float *out_v) {
    const KVEntry *e = kv_find(hm, token_id);
    if (!e) return false;

    if (out_k) memcpy(out_k, e->k_vec, hm->kv_dim * sizeof(float));
    if (out_v) memcpy(out_v, e->v_vec, hm->kv_dim * sizeof(float));
    return true;
}

int kv_hashmap_batch_get(const KVHashmap *hm,
                         const int32_t *token_ids, int n_tokens,
                         float *out, bool is_key,
                         const float *fallback) {
    int hits = 0;
    int dim = hm->kv_dim;

    for (int t = 0; t < n_tokens; t++) {
        const KVEntry *e = kv_find(hm, token_ids[t]);
        float *dst = out + t * dim;

        if (e) {
            memcpy(dst, is_key ? e->k_vec : e->v_vec, dim * sizeof(float));
            hits++;
        } else if (fallback) {
            memcpy(dst, fallback, dim * sizeof(float));
        } else {
            memset(dst, 0, dim * sizeof(float));
        }
    }

    return hits;
}

/* ── Save/Load (Python 호환, little-endian) ── */

int kv_hashmap_save(const KVHashmap *hm, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int32_t magic = KV_MAGIC;
    int32_t version = KV_VERSION;
    int32_t kv_dim = hm->kv_dim;
    int32_t n_entries = hm->size;

    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&kv_dim, 4, 1, f);
    fwrite(&n_entries, 4, 1, f);

    for (int i = 0; i < hm->capacity; i++) {
        KVEntry *e = &hm->buckets[i];
        if (e->token_id < 0) continue;

        fwrite(&e->token_id, 4, 1, f);
        fwrite(e->k_vec, sizeof(float), hm->kv_dim, f);
        fwrite(e->v_vec, sizeof(float), hm->kv_dim, f);
    }

    fclose(f);
    return 0;
}

KVHashmap *kv_hashmap_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    int32_t magic, version, kv_dim, n_entries;
    if (fread(&magic, 4, 1, f) != 1 || magic != KV_MAGIC) {
        fprintf(stderr, "kv_hashmap_load: invalid magic in %s\n", path);
        fclose(f);
        return NULL;
    }

    fread(&version, 4, 1, f);
    fread(&kv_dim, 4, 1, f);
    fread(&n_entries, 4, 1, f);

    if (version != KV_VERSION) {
        fprintf(stderr, "kv_hashmap_load: unsupported version %d\n", version);
        fclose(f);
        return NULL;
    }

    KVHashmap *hm = kv_hashmap_create(kv_dim);

    float *k_buf = (float *)malloc(kv_dim * sizeof(float));
    float *v_buf = (float *)malloc(kv_dim * sizeof(float));

    for (int i = 0; i < n_entries; i++) {
        int32_t token_id;
        if (fread(&token_id, 4, 1, f) != 1) break;
        if (fread(k_buf, sizeof(float), kv_dim, f) != (size_t)kv_dim) break;
        if (fread(v_buf, sizeof(float), kv_dim, f) != (size_t)kv_dim) break;

        kv_hashmap_put(hm, token_id, k_buf, v_buf);
    }

    free(k_buf);
    free(v_buf);
    fclose(f);
    return hm;
}

/* ── Multi-layer Stack ── */

KVHashmapStack *kv_stack_create(int kv_dim, int n_layers, int ctx_size) {
    KVHashmapStack *stack = (KVHashmapStack *)calloc(1, sizeof(KVHashmapStack));
    stack->kv_dim = kv_dim;
    stack->n_layers = n_layers < KV_MAX_LAYERS ? n_layers : KV_MAX_LAYERS;
    stack->ctx_size = ctx_size > 0 ? ctx_size : 5;
    stack->capture_mode = 0;
    for (int i = 0; i < stack->n_layers; i++)
        stack->layers[i] = kv_hashmap_create(kv_dim);
    return stack;
}

void kv_stack_free(KVHashmapStack *stack) {
    if (!stack) return;
    for (int i = 0; i < KV_MAX_LAYERS; i++) {
        if (stack->layers[i]) kv_hashmap_free(stack->layers[i]);
    }
    free(stack);
}

KVHashmap *kv_stack_get_layer(const KVHashmapStack *stack, int layer) {
    if (!stack || layer < 0 || layer >= KV_MAX_LAYERS) return NULL;
    return stack->layers[layer];
}

int32_t kv_context_hash(const int32_t *tokens, int len) {
    /* slot.c hashContextN() 동일: h = len; for(i) h = h*31 + tokens[i] */
    uint64_t h = (uint64_t)len;
    for (int i = 0; i < len; i++)
        h = h * 31 + (uint64_t)(uint32_t)tokens[i];
    /* int32 범위로 축소 (음수 방지) */
    return (int32_t)(h & 0x7FFFFFFF);
}

int kv_stack_save(const KVHashmapStack *stack, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int32_t magic = 0x53564B4D;  /* "MKVS" */
    int32_t version = 1;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&stack->kv_dim, 4, 1, f);
    fwrite(&stack->n_layers, 4, 1, f);
    fwrite(&stack->ctx_size, 4, 1, f);

    for (int il = 0; il < stack->n_layers; il++) {
        KVHashmap *hm = stack->layers[il];
        int32_t n = hm ? hm->size : 0;
        fwrite(&n, 4, 1, f);
        if (!hm || n == 0) continue;

        for (int i = 0; i < hm->capacity; i++) {
            KVEntry *e = &hm->buckets[i];
            if (e->token_id < 0) continue;
            fwrite(&e->token_id, 4, 1, f);
            fwrite(e->k_vec, sizeof(float), hm->kv_dim, f);
            fwrite(e->v_vec, sizeof(float), hm->kv_dim, f);
        }
    }

    fclose(f);
    return 0;
}

KVHashmapStack *kv_stack_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    int32_t magic, version, kv_dim, n_layers, ctx_size;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x53564B4D) {
        fclose(f);
        return NULL;
    }
    fread(&version, 4, 1, f);
    fread(&kv_dim, 4, 1, f);
    fread(&n_layers, 4, 1, f);
    fread(&ctx_size, 4, 1, f);

    KVHashmapStack *stack = kv_stack_create(kv_dim, n_layers, ctx_size);

    float *k_buf = (float *)malloc(kv_dim * sizeof(float));
    float *v_buf = (float *)malloc(kv_dim * sizeof(float));

    for (int il = 0; il < n_layers; il++) {
        int32_t n;
        if (fread(&n, 4, 1, f) != 1) break;
        for (int i = 0; i < n; i++) {
            int32_t key;
            if (fread(&key, 4, 1, f) != 1) break;
            if (fread(k_buf, sizeof(float), kv_dim, f) != (size_t)kv_dim) break;
            if (fread(v_buf, sizeof(float), kv_dim, f) != (size_t)kv_dim) break;
            kv_hashmap_put(stack->layers[il], key, k_buf, v_buf);
        }
    }

    free(k_buf);
    free(v_buf);
    fclose(f);
    return stack;
}

void kv_hashmap_print_stats(const KVHashmap *hm) {
    float k_norm_sum = 0.0f, v_norm_sum = 0.0f;
    int count = 0;

    for (int i = 0; i < hm->capacity; i++) {
        if (hm->buckets[i].token_id < 0) continue;
        count++;

        float kn = 0.0f, vn = 0.0f;
        for (int d = 0; d < hm->kv_dim; d++) {
            kn += hm->buckets[i].k_vec[d] * hm->buckets[i].k_vec[d];
            vn += hm->buckets[i].v_vec[d] * hm->buckets[i].v_vec[d];
        }
        k_norm_sum += kn;
        v_norm_sum += vn;
    }

    printf("KVHashmap: %d entries, kv_dim=%d, load=%.1f%%\n",
           hm->size, hm->kv_dim,
           100.0f * hm->size / hm->capacity);
    if (count > 0) {
        printf("  avg |K|=%.2f, avg |V|=%.4f\n",
               sqrtf(k_norm_sum / count),
               sqrtf(v_norm_sum / count));
    }
}
