# ai_5 Research Index

## Slot-LLM Fusion → Hashmap-Layer 아키텍처

### 진행 요약 (2026-05-10 기준)

```
§01 Slot-LLM   → §02 Hashmap blk.0 → §03 인과증명+CL → §04 V Learning → §05 V 최적화
(logit blend)    (Wk,Wv 교체)       (V→출력 변화)     (주제 유도)     (logit +3.66)
```

**현재 도달 수준:**
- No-Forgetting: 100% (구조적 보장, bit-exact 증명)
- V→출력 인과관계: 100% (blk.0 파괴, blk.1 주제 전환)
- 주제/카테고리 제어: 100% (scale=1로 "eat"→"play", slot memory 이김)
- **단어-수준 정밀 제어: 100%** (swim #1 달성! contrastive 2회로 수렴)
- **Isolation 실증: 100%** ("cat" V 수정 → "dog" 출력 불변)
- **Continual Learning 완성: "eat"→"swim" 학습 + "dog" 불변 동시 달성**

---

### 연구 목록

| § | 제목 | 파일 | 핵심 결과 |
|---|------|------|----------|
| §01 | Slot-LLM Fusion 초기 검증 | [01_slot_llm_fusion.md](01_slot_llm_fusion.md) | Q1 "미아" 자아 활성화, Q6 고양이 지식 HIT |
| §02 | Hashmap-Layer blk.0 교체 | [02_hashmap_layer_blk0.md](02_hashmap_layer_blk0.md) | Wk,Wv→hashmap 동일출력, CL 증명, sparse+fallback |
| §03 | 인과 증명 + Continual Learning | [03_three_experiments.md](03_three_experiments.md) | V=random→파괴, 근사오차165%, CL bit-exact |
| §04 | V Learning: 주제 유도 | [04_v_learning.md](04_v_learning.md) | blk.1 V=target→"eat"→"play", 7개 target 작동 |
| §05 | V 최적화 + Contrastive → swim #1 달성 | [05_v_optimization.md](05_v_optimization.md) | swim #1! contrastive 2회 수렴, "cat"→"swim"+"dog" 불변 |

---

### 핵심 발견 요약

| 발견 | 의미 |
|------|------|
| KV Hashmap = 구조적 no-forgetting | append-only, 기존 지식 물리적 불변 |
| K는 구조, V는 의미 | \|K\|/\|V\|=1023x, V가 토큰 의미 분화 |
| blk.1 V = 주제 제어기 | scale=1로 "eat"→"play" 전환, slot 이김 |
| blk.3+ = context_hash 불일치 | 깊은 레이어는 정확한 hidden state 필요 |
| slot_score=0 (이 프롬프트) | slot 미개입, 순수 LLM logit이 결정 |
| V 방향 최적화 가능 | 20회 random perturbation → swim +3.66 |
| 최적화 경로: eat→play→ride | 의미 공간 이동 경로 추적 가능 |
| **Isolation 실증** | "cat" V 수정 → "dog" 완벽 불변 (context_hash 격리) |
| 같은 prefix = transfer | "A cat"/"A bird" hash 공유 → 부분 전이 (의도된 일반화) |
| **Contrastive V = 정밀 제어** | swim↑ + play↓ 동시 최적화 → 2회만에 #1 달성 |
| **CL 완성 증명** | "eat"→"swim" 학습 + "dog" 불변 = 새 지식 + no-forgetting 동시 |

---

### 다음 방향

| 우선순위 | 과제 | 기대 효과 |
|----------|------|----------|
| 1 | 다중 토큰 시퀀스 학습 | "swim at the lake" 전체 문장 제어 |
| 2 | 자동 학습 파이프라인 | 데이터셋 → context_hash → V 최적화 → hashmap 자동 빌드 |
| 3 | Multi-context 일반화 | "A cat likes to" 외 다른 질문에도 swim 응답 |
| 4 | GitHub 공개 + 논문 정리 | 재현 가능한 실험 코드 + 결과 문서화 |

---

### 도구 목록

| 도구 | 용도 |
|------|------|
| `tools/build_kv_hashmap.py` | blk.0 K,V precompute (전체 vocab) |
| `tools/build_kv_stack.py` | Multi-layer MKVS builder (근사) |
| `tools/capture_blk1_forward.py` | blk.0 full forward → blk.1 정확 K,V |
| `tools/build_blk1_stack.py` | blk.1 exact MKVS + V injection + demo |
| `tools/knowledge_injection.py` | V-swap, V-blend, V-zero, 분석 리포트 |
| `tools/continual_learning_demo.py` | CL build/merge/verify/isolation |
| `tools/v_learning.py` | V learning (projection, contrastive, attn_gradient) |
| `tools/dump_weights.py` | GGUF 가중치 덤프 |

---

### 핵심 파일 (weights/)

| 파일 | 설명 |
|------|------|
| `blk0_kv_full.bin` | blk.0 전체 vocab hashmap (149MB, 151936 tokens) |
| `stack_blk1_ocean_s1.bin` | blk.0+1 MKVS, V=ocean (주제 변경 검증) |
| `stack_blk1_optimized_swim.bin` | blk.0+1 MKVS, V 최적화 (swim logit=14.96) |
| `logit_baseline.txt` | baseline logit dump (step 0-2) |
| `logit_ocean.txt` | V=ocean logit dump (swim rank13, logit=11.3) |
| `knowledge_injection_report.txt` | V cosine similarity 분석 |
| `cl_merged.bin` | Continual Learning 증명 (10000 tokens) |
