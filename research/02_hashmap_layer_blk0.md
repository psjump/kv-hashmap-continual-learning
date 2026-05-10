# §02 Hashmap-Layer: blk.0 Wk,Wv 교체 검증

**날짜**: 2026-05-10
**목표**: blk.0의 Wk(896×128),Wv(896×128) 행렬곱을 hashmap lookup으로 교체하고 동일 출력 검증

---

## 1. 핵심 발견

### 1.1 blk.0 K,V는 token_id에 결정적
- blk.0 입력 = RMSNorm(embedding[token_id]) → token_id에 의해 유일하게 결정
- K = Wk × RMSNorm(emb[tid]) + bk → 128-dim, |K| ≈ 316 (bias 지배)
- V = Wv × RMSNorm(emb[tid]) + bv → 128-dim, |V| ≈ 0.43 (임베딩 차이 반영)
- **hashmap key = token_id (int)**, value = K[128] + V[128]

### 1.2 Wk bias가 K를 지배
- bk: mean=-0.12, **std=27.94**, range [-152, +98]
- |K| ≈ 316으로 모든 토큰에서 거의 동일 → K는 "구조/위치" 역할
- V는 bias 작음 (std=0.012) → **V가 "지식/내용" 역할**
- Continual learning에서 V만 수정하면 됨 (K는 구조, V는 지식)

### 1.3 dump_weights.py shape 버그 발견 및 해결
- GGUF shape [896, 151936]을 numpy reshape에 그대로 사용 → (896, 151936) 저장
- ggml 실제 메모리: ne[0]=896열, ne[1]=151936행 → (151936, 896)
- **수정**: `emb.reshape(151936, 896)`, `wv.reshape(128, 896)`로 올바른 레이아웃 복원

### 1.4 동일 출력 검증 성공
```
Baseline (Wk×input): ", i'm a 16 year old girl"
Hashmap (lookup):    ", i'm a 16 year old girl"
→ 완전 동일! 행렬곱 → hashmap lookup 교체 증명 완료
```

---

## 2. 구현 스택

| 컴포넌트 | 파일 | 역할 |
|----------|------|------|
| K,V 빌더 | `tools/build_kv_hashmap.py` | Python: 가중치 로드 → K,V precompute → 바이너리 저장 |
| C hashmap | `slot_plugin/kv_hashmap.c/.h` | open-addressing hashmap, float vector 저장 |
| qwen2 수정 | `llama.cpp/src/models/qwen2.cpp` | `ggml_map_custom2`로 blk.0 K,V hashmap lookup |
| 통합 | `slot_fusion/slot_fusion.cpp` | `--kv-hashmap PATH` 옵션으로 hashmap 로드 |

### 빌드 변경
- `slot_plugin/CMakeLists.txt`: kv_hashmap.c 추가
- `llama.cpp/src/CMakeLists.txt`: `KV_HASHMAP_ENABLED` + slot_plugin 링크
- `slot_fusion/CMakeLists.txt`: `KV_HASHMAP_ENABLED` 정의

### 바이너리 포맷 (KVH0)
```
[magic: "KVH0" 4B] [version: i32] [kv_dim: i32] [n_entries: i32]
[entries: n × (token_id:i32 + K:kv_dim×f32 + V:kv_dim×f32)]
little-endian, Python/C 호환
```
- 전체 vocab (151936): ~149 MB
- 학습 데이터만: ~5 MB (실용적)

---

## 3. 기술 세부사항

### 3.1 Q5_0 Dequantization
- Wk는 Q5_0 양자화 (78KB raw)
- `gguf` Python 라이브러리의 `dequantize()` 사용 → float32 (128, 896)
- llama.cpp의 ggml 내부 dequantize와 동일 결과

### 3.2 ggml_map_custom2 사용
- 문제: `ggml_map_custom1`은 입력과 동일 shape 출력 → inp_tokens(n_tokens, i32)와 맞지 않음
- 해결: `ggml_map_custom2(ctx0, kv_shape_dummy, inp_tokens, callback, ...)`
  - 첫 번째 인자 = 출력 shape 결정 (kv_dim × n_tokens, f32)
  - 두 번째 인자 = token_ids 전달
  - callback에서 hashmap lookup 수행

### 3.3 RoPE는 hashmap 이후 적용
- hashmap 저장: Wk×input + bk (RoPE 이전)
- qwen2.cpp: hashmap lookup 후 → RoPE 적용 → attention
- 위치 정보는 RoPE가 담당하므로 hashmap에 저장할 필요 없음

---

## 4. 성능

| 항목 | 원본 | Hashmap |
|------|------|---------|
| blk.0 K 계산 | matmul (896×128) | hash lookup O(1) |
| blk.0 V 계산 | matmul (896×128) | hash lookup O(1) |
| 메모리 | Wk 78KB + Wv 450KB | hashmap ~149MB (full) / ~5MB (sparse) |
| 출력 | ✓ | ✓ (동일) |

---

## 5. Continual Learning + Sparse Hashmap 검증

### 5.1 Continual Learning (구조적 No-Forgetting) — 성공
```
토큰 100~199의 V를 ×10 perturbation 후:
  "hello"         → 동일 출력 ✓
  "what is a cat" → 동일 출력 ✓
  "the sun is"    → 동일 출력 ✓
```
**100개 토큰 V 파괴 → 다른 토큰 출력 불변.** 토큰별 독립 = 구조적 no-forgetting.

### 5.2 Sparse Hashmap + Fallback — 성공
- **Overlay 방식**: 항상 원본 Wk×input 계산 → HIT 토큰만 hashmap 값으로 덮어쓰기
- `ggml_map_custom2_inplace`로 in-place 수정
- MISS 시 원본 값 유지 → fallback 자동

```
Full hashmap (149MB, 151936 entries): ", i'm a 16 year old girl. i'm a bit"
Sparse hashmap (10MB, 10000 entries): ", i'm a 16 year old girl. i'm a bit"  ← MISS fallback 동작
No hashmap (baseline):                ", i'm a 16 year old girl. i'm a bit"
→ 3가지 모두 동일!
```

**메모리 절약**: 149MB → 10MB (93% 절감), 출력 동일

---

## 6. blk.1 확장 시도 + 발견

### 6.1 blk.1 근사 접근 (실패)
- blk.1 입력 ≈ embedding 으로 근사 → K,V precompute
- **결과: 빈 출력** — blk.1의 실제 입력은 `emb + attention_0 + FFN_0`
- 단순 embedding 근사는 너무 부정확

### 6.2 핵심 발견: 레이어 깊이별 hashmap 전략
```
blk.0: token_id → K,V (결정적)     → hashmap 완벽 동작 ✓
blk.1: hidden_state → K,V (context-dependent) → 실제 forward 캡처 필요
blk.2+: 더 complex → context hash 기반 캡처 필요
```

**blk.0만 특별**: 입력이 embedding(고정) → hashmap으로 완전 교체 가능
**blk.1+**: 입력이 이전 레이어 출력에 의존 → "캡처-재사용" 패턴 필요

### 6.3 구현 완료
- `slot_plugin/kv_hashmap.c/.h`: KVHashmapStack (multi-layer 컨테이너)
- `tools/build_kv_stack.py`: Python multi-layer builder
- `qwen2.cpp`: 모든 레이어 overlay 지원 (generic loop)
- `slot_fusion.cpp`: `--kv-hashmap` stack 포맷 로드 지원

---

## 7. 다음 단계

### 5.1 Continual Learning 검증 (§03)
1. hashmap에 새 토큰 K,V 추가 (append-only)
2. 기존 프롬프트 재테스트 → 변화 없음 확인
3. 새 지식 프롬프트 → 반영 확인

  - [예상] append-only이므로 기존 출력 100% 보존
  - [의외 후보] 새 토큰의 K,V 값 설정이 어려울 수 있음 (학습 데이터에서 자동 추출 필요)

### 5.2 blk.1 확장 (§04)
- blk.1은 입력이 blk.0 출력에 의존 → token_id가 아닌 context hash 필요
- v8에서 검증된 context hash 방식 적용 가능

  - [예상] context hash로 blk.1 K,V도 hashmap 가능
  - [의외 후보] blk.0 출력의 미세한 변화가 hash 불일치를 유발할 수 있음

### 5.3 Sparse Hashmap 최적화
- 전체 vocab 149MB → 학습 데이터 토큰만 저장 (~5MB)
- MISS 시 원본 Wk×input fallback 구현

  - [예상] sparse hashmap + fallback으로 메모리 99% 절약
  - [의외 후보] fallback 경로의 latency가 hashmap 이점을 상쇄할 수 있음
