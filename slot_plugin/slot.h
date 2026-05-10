/**
 * Slot Memory — Sparse 기억 (Binary Switch) C 포팅
 *
 * Java SlotModel의 핵심 로직:
 *   hash(context) → int[] (관측된 토큰 ID 배열)
 *   추가만, 덮어쓰기 없음 → continual learning 핵심
 *
 * 해시 테이블: open addressing + linear probing
 * 메모리: 슬롯당 int[] 배열 (Java 원본과 동일)
 */
#ifndef SLOT_H
#define SLOT_H

#include <stdint.h>
#include <stdbool.h>

/* DLL export/import */
#if defined(SLOT_STATIC)
  #define SLOT_API
#elif defined(_WIN32)
  #ifdef SLOT_EXPORTS
    #define SLOT_API __declspec(dllexport)
  #else
    #define SLOT_API __declspec(dllimport)
  #endif
#else
  #define SLOT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── 설정 ── */
#define SLOT_INITIAL_CAPACITY  65536   /* 초기 해시 테이블 크기 (2의 거듭제곱) */
#define SLOT_LOAD_FACTOR       0.75f  /* 리사이즈 임계값 */
#define SLOT_CONTEXT_SIZE      5      /* 기본 n-gram 크기 (Java: memoryCtx=5) */
#define SLOT_MIN_CTX           2      /* 최소 context 길이 (backoff) */

/* ── 슬롯 엔트리 ── */
typedef struct {
    uint64_t key;          /* context hash (0 = 빈 슬롯) */
    int     *tokens;       /* 관측된 토큰 ID 배열 */
    int      count;        /* tokens 배열 길이 */
    int      capacity;     /* tokens 배열 할당 크기 */
} SlotEntry;

/* ── 슬롯 메모리 ── */
typedef struct {
    SlotEntry *buckets;    /* 해시 테이블 */
    int        capacity;   /* 버킷 수 */
    int        size;       /* 사용 중인 엔트리 수 */
    int        ctx_size;   /* context 크기 (n-gram) */
    int        min_ctx;    /* 최소 context 길이 */
} SlotMemory;

/* ── API ── */

/** 슬롯 메모리 생성 */
SLOT_API SlotMemory *slot_create(int ctx_size, int min_ctx);

/** 슬롯 메모리 해제 */
SLOT_API void slot_free(SlotMemory *mem);

/** 단일 context+target 학습 — 추가만, 덮어쓰기 없음 */
SLOT_API void slot_learn(SlotMemory *mem, const int *context, int ctx_len, int target);

/**
 * 다중 context 학습 (backoff) — Java learnMultiCtx() 포팅
 * fullContext의 끝에서 ctx_size~min_ctx 길이로 여러 해시 등록
 */
SLOT_API void slot_learn_multi(SlotMemory *mem, const int *full_context, int full_len, int target);

/** 조회 — 해당 context에서 관측된 토큰 배열 반환 (없으면 NULL) */
SLOT_API const int *slot_lookup(const SlotMemory *mem, const int *context, int ctx_len, int *out_count);

/**
 * 전체 vocab에 대한 slot score 계산
 * scores[vocab_size]에 slot 점수를 기록 (HIT이면 1.0, 없으면 0.0)
 * backoff: ctx_size부터 min_ctx까지 시도, 가장 긴 매칭 사용
 */
SLOT_API void slot_score_vocab(const SlotMemory *mem, const int *context, int ctx_len,
                      float *scores, int vocab_size);

/** context 해시 계산 — Java hashContextN() 동일 */
SLOT_API uint64_t slot_hash(const int *context, int len);

/** 파일 저장 (Java v3 int[] 포맷 호환) */
SLOT_API int slot_save(const SlotMemory *mem, const char *path);

/** 파일 로드 (Java v3 int[] 포맷 호환) */
SLOT_API SlotMemory *slot_load(const char *path);

/** 통계 출력 */
SLOT_API void slot_print_stats(const SlotMemory *mem);

#ifdef __cplusplus
}
#endif

#endif /* SLOT_H */
