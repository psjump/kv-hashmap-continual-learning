/**
 * KV Hashmap — blk.0 K,V 벡터 저장소
 *
 * 기존 LLM의 K = Wk × input, V = Wv × input 행렬곱을
 * hashmap[token_id] → (K, V) 조회로 교체.
 *
 * 핵심: append-only → 이전 지식 파괴 불가능 (continual learning)
 *
 * 해시 테이블: open addressing + linear probing (slot.c 동일 패턴)
 * key = token_id (int32), value = K[kv_dim] + V[kv_dim] floats
 */
#ifndef KV_HASHMAP_H
#define KV_HASHMAP_H

#include <stdint.h>
#include <stdbool.h>

/* DLL export/import (slot.h와 동일 패턴) */
#if defined(SLOT_STATIC)
  #define KV_API
#elif defined(_WIN32)
  #ifdef SLOT_EXPORTS
    #define KV_API __declspec(dllexport)
  #else
    #define KV_API __declspec(dllimport)
  #endif
#else
  #define KV_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── 설정 ── */
#define KV_INITIAL_CAPACITY  16384   /* 초기 해시 테이블 크기 (2의 거듭제곱) */
#define KV_LOAD_FACTOR       0.75f

/* ── 바이너리 포맷 ── */
#define KV_MAGIC  0x3048564B   /* "KVH0" little-endian */
#define KV_VERSION  1

/* ── 엔트리 ── */
typedef struct {
    int32_t  token_id;    /* key (-1 = 빈 슬롯) */
    float   *k_vec;       /* K vector (kv_dim floats) */
    float   *v_vec;       /* V vector (kv_dim floats) */
} KVEntry;

/* ── KV Hashmap ── */
typedef struct {
    KVEntry *buckets;
    int      capacity;    /* 버킷 수 (2의 거듭제곱) */
    int      size;        /* 사용 중인 엔트리 수 */
    int      kv_dim;      /* K,V 벡터 차원 (128 for Qwen2-0.5B) */
} KVHashmap;

/* ── API ── */

/** 생성 */
KV_API KVHashmap *kv_hashmap_create(int kv_dim);

/** 해제 */
KV_API void kv_hashmap_free(KVHashmap *hm);

/** K,V 저장 (새 토큰이면 추가, 기존이면 덮어쓰기) */
KV_API void kv_hashmap_put(KVHashmap *hm, int32_t token_id,
                           const float *k_vec, const float *v_vec);

/**
 * K,V 조회
 * out_k, out_v: 각각 kv_dim 크기 버퍼 (NULL이면 해당 벡터 스킵)
 * 반환: true=HIT, false=MISS
 */
KV_API bool kv_hashmap_get(const KVHashmap *hm, int32_t token_id,
                           float *out_k, float *out_v);

/**
 * 배치 조회 — 여러 토큰의 K 또는 V를 한번에 조회
 * token_ids: n_tokens개 토큰 ID 배열
 * out: n_tokens × kv_dim 크기 출력 버퍼
 * is_key: true=K 조회, false=V 조회
 * fallback: MISS 시 채울 값 (NULL이면 0으로 채움)
 * 반환: HIT 수
 */
KV_API int kv_hashmap_batch_get(const KVHashmap *hm,
                                const int32_t *token_ids, int n_tokens,
                                float *out, bool is_key,
                                const float *fallback);

/** 바이너리 저장 (Python build_kv_hashmap.py와 호환) */
KV_API int kv_hashmap_save(const KVHashmap *hm, const char *path);

/** 바이너리 로드 */
KV_API KVHashmap *kv_hashmap_load(const char *path);

/** 통계 출력 */
KV_API void kv_hashmap_print_stats(const KVHashmap *hm);

/**
 * Multi-layer KV Hashmap 컨테이너
 * 각 레이어별 독립 hashmap (blk.0=token_id key, blk.1+=context_hash key)
 */
#define KV_MAX_LAYERS 24

typedef struct {
    KVHashmap *layers[KV_MAX_LAYERS];  /* layer별 hashmap (NULL=미사용) */
    int        n_layers;               /* 활성 레이어 수 */
    int        kv_dim;                 /* 128 */
    int        ctx_size;               /* context hash용 n-gram 크기 (default 5) */
    int        capture_mode;           /* 1=캡처 모드 (K,V 저장), 0=추론 모드 */
} KVHashmapStack;

/** 스택 생성 */
KV_API KVHashmapStack *kv_stack_create(int kv_dim, int n_layers, int ctx_size);

/** 스택 해제 */
KV_API void kv_stack_free(KVHashmapStack *stack);

/** 특정 레이어 hashmap 반환 (없으면 NULL) */
KV_API KVHashmap *kv_stack_get_layer(const KVHashmapStack *stack, int layer);

/** context hash 계산 — slot.c와 동일 알고리즘: h = len; for(i) h = h*31 + tokens[i] */
KV_API int32_t kv_context_hash(const int32_t *tokens, int len);

/** 스택 저장/로드 */
KV_API int kv_stack_save(const KVHashmapStack *stack, const char *path);
KV_API KVHashmapStack *kv_stack_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* KV_HASHMAP_H */
