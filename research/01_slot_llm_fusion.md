# §01 Slot-LLM Fusion: Continual Learning 검증

**날짜**: 2026-05-10
**목표**: 기존 LLM(Qwen2-0.5B)에 Slot Memory를 융합하여 continual learning 검증

---

## 1. 아키텍처: 방법 C — 블랙박스 비율 점진적 조절

```
[Qwen2-0.5B LLM] → logits (언어 능력 유지, 가중치 불변)
        +
[Slot Memory]    → slot scores (korean_identity.txt 기억)
        ↓
[Gating: α]      → final_logits[i] = llm_logits[i] + α × slot_scores[i]
        ↓
[Greedy Sampling] → 최종 토큰
```

### 핵심 원리
- LLM 가중치는 **절대 수정하지 않음** → 기존 언어 능력 100% 보존
- Slot Memory는 **append-only** → 새 데이터 추가해도 이전 기억 파괴 불가
- α(블렌딩 가중치)는 AdaGrad로 자동학습 (제1원칙)

### LoRA와의 구조적 차이
| | LoRA | Slot Fusion |
|---|---|---|
| 모델 가중치 | 변경 (ΔW) | **불변** |
| 망각 위험 | O (gradient 충돌) | **구조적 불가능** |
| 추가 학습 | 전체 재학습 필요 | slot 추가만 |
| 저장 형태 | adapter 파라미터 | binary hash table |
| 추론 비용 | matmul 추가 | lookup + 덧셈 |

---

## 2. 구현 스택

| 컴포넌트 | 파일 | 설명 |
|----------|------|------|
| LLM 추론 | llama.cpp (Vulkan) | Qwen2-0.5B, AMD 780M GPU |
| Slot Memory | slot_plugin/slot.c | C 구현, Java v3 바이너리 호환 |
| 융합 엔진 | slot_fusion/slot_fusion.cpp | logit 수정 + gating |
| 학습 데이터 | korean_identity.txt | 197줄, 미아 자아 + 한국어 지식 |

### Slot Memory 통계
- 30,677 entries (hash slots)
- 36,123 total tokens (관측된 토큰 수)
- max 113 tokens/slot
- ctx_size=5 (5-gram context)

---

## 3. 실험 결과

### 3.1 자아 활성화 테스트

| # | 프롬프트 | 응답 | 결과 |
|---|---------|------|------|
| Q1 | "너의 이름은 뭐니?" | "나의 이름은 미아이다. 나는 배우는 것과 질문에 답하는 것을 좋아한다." | **HIT** |
| Q2 | "너는 누구니?" | 반복 생성 (깨짐) | MISS |
| Q3 | "너는 어디에 사니?" | "나는 잠을 자지 않지만 쉴 때가 있다..." | △ (관련은 있으나 부정확) |
| Q4 | "너는 무엇을 좋아하니?" | "나는 잠을 잘 수 없다..." | △ |
| Q5 | "펭귄은 무슨 색이냐?" | 깨짐 | MISS |
| Q6 | "고양이는 무엇을 좋아하나?" | "고양이는 우유를 좋아한다. 고양이는 높은 곳을 좋아한다." | **HIT** |

### 3.2 HIT/MISS 분석
- **HIT (Q1, Q6)**: 학습 데이터에 정확한 n-gram 패턴이 존재
  - "이름은 뭐니?" → slot에서 "나의 이름은 미아이다" 체인 발견
  - "고양이는 무엇을 좋아하나?" → 정확한 QA 패턴 매칭
- **MISS (Q2, Q5)**: n-gram context가 학습 데이터와 불일치
  - "너는 누구니?" — 학습 데이터에는 "너는 누구냐고 물으면" 형태
  - 토크나이저가 "누구니?"를 다르게 분할 → hash 불일치

---

## 4. 발견 및 분석

### 4.1 Slot + LLM 시너지 확인
- Slot HIT 시: LLM의 한국어 문법 능력 + Slot의 정확한 기억 = **자연스러운 한국어 문장**
- Slot MISS 시: LLM 단독 생성 → 관련 있으나 부정확한 응답

### 4.2 alpha 자동학습 미동작
- `llama_batch_get_one`은 마지막 토큰만 logits 활성화
- 학습 중 중간 토큰 logits 접근 불가 → alpha 고정 (5.0)
- **수정 필요**: `llama_batch_init` + 전체 logits 활성화

### 4.3 n-gram 매칭 한계
- 5-gram context는 형태 변화("뭐니" vs "무엇이니")에 취약
- **다음 방향**: fuzzy matching 또는 embedding-based context hash

---

## 5. 다음 단계

### 5.1 즉시 개선 (§02)
1. **alpha 자동학습 수정**: llama_batch_init으로 전체 logits 활성화
2. **Slot context 확장**: 2-gram ~ 7-gram backoff 범위 확대
3. **Continual learning 검증**: animal_knowledge.txt 추가 학습 후 미아 기억 유지 확인

  - [예상] animal 데이터 추가 후 Q1 "미아" 응답 유지 (slot append-only이므로)
  - [의외 후보] alpha가 animal 데이터로 편향되어 Q1 응답 품질 하락 가능성

### 5.2 품질 개선 (§03)
1. **Embedding-based slot lookup**: n-gram hash 대신 LLM 임베딩 유사도로 context 매칭
2. **Multi-signal gating**: slot score + LLM confidence + 문장 위치 → 다중 신호 자동 배합

  - [예상] embedding lookup이 형태 변화에 강건 → MISS→HIT 전환율 50%+
  - [의외 후보] LLM 임베딩이 한국어에서 충분히 구별적이지 않을 수 있음 (Qwen2-0.5B 한계)

### 5.3 Vulkan 가속 (§04)
1. Slot score 계산을 GPU로 이전 (vocab 151K 전체 스캔 병렬화)

  - [예상] GPU 병렬화로 slot_score_vocab 10x 가속
  - [의외 후보] vocab 151K × 4byte = 600KB → GPU 전송 오버헤드가 연산보다 클 수 있음

---

## 6. Continual Learning 핵심 메커니즘 정리

우리 모델의 continual learning이 가능한 핵심 이유:

1. **Slot = append-only binary switch**
   - hash(context) → 관측된 토큰 집합
   - 새 데이터: 기존 slot에 토큰 추가 OR 새 slot 생성
   - **이전 엔트리 삭제/수정 불가** → 구조적 망각 불가능

2. **모델 가중치 불변**
   - LLM의 W, b 등 모든 파라미터 고정
   - gradient가 LLM에 전파되지 않음
   - 기존 언어 능력 100% 보존

3. **Gating으로 블렌딩**
   - slot이 있으면 slot 우세, 없으면 LLM 우세
   - 새 도메인 학습 = slot 영역 확장, LLM 영역은 건드리지 않음

이것이 LoRA/full fine-tuning과의 근본적 차이:
- LoRA: W + ΔW → ΔW가 이전 ΔW를 덮어쓸 위험
- Slot: W(고정) + external_memory(추가만) → 덮어쓰기 구조적 불가
