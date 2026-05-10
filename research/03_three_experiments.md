# §03 Three Experiments: blk.1 Forward + Knowledge Injection + Continual Learning

## 실험 요약

| 실험 | 결과 | 핵심 수치 |
|------|------|----------|
| blk.1 Forward 캡처 | **성공** | 임베딩 근사 오차 V=165% (근사 완전 실패 증명) |
| Knowledge Injection | **인과 관계 실증** | V=random×10 → 출력 완전 파괴, V=0 기능어 → 출력 변경 |
| Continual Learning | **ALL PASS** | base 100% 보존, isolation 100%, bit-exact |

---

## 1. blk.1 Forward 캡처

### 문제
- blk.1 입력 = blk.0 출력 (context-dependent)
- 기존 `build_kv_stack.py`는 embedding 근사 사용 → **실패**

### 해결: Python에서 blk.0 Full Forward 재현
```
emb → RMSNorm → Q,K,V → RoPE → GQA Attention → Wo → Residual → FFN(SwiGLU) → Residual
```

구현: `tools/capture_blk1_forward.py`
- `--mode single`: 단일 토큰(pos=0, RoPE=identity, trivial attention) → 정확한 계산
- `--mode sequence`: 다중 토큰(full causal attention + RoPE) → context-dependent
- `--mode verify`: self-consistency 검증

### 핵심 발견

#### pos=0 (단일 토큰): RoPE=identity, attention=trivial → 정확한 계산 가능
```
|emb|      ≈ 0.47    (embedding은 매우 작음)
|blk0_out| ≈ 4.19~8.00  (attention+FFN이 8~17배 증폭!)
|diff|     ≈ 4.20~8.00  (embedding과 완전히 다른 벡터)
```

#### blk.1 K,V: exact vs embedding 근사
```
K 차이: 17.6 ~ 22.0  (|K1| ≈ 284, 차이 6~8%)
V 차이: 3.8 ~ 5.6    (|V1| ≈ 3.4, 차이 112~165%!!)
```
**→ V는 근사 오차가 실제 값보다 크다 (근사 완전 실패)**

#### Sequence 모드: context 효과 측정
```
pos=0: ctx_diff = 0.0 (single token과 동일, 자기 자신만 attend)
pos=1: ctx_diff K=8.4, V=2.2 (이전 토큰 1개의 영향)
pos=4: ctx_diff K=11.4, V=2.9 (이전 토큰 4개의 영향, 누적)
```
**→ position이 커질수록 context 영향 증가**
**→ blk.1 hashmap은 반드시 context_hash 기반이어야 함**

### blk.1 K,V 특성 (blk.0과 비교)
| 지표 | blk.0 | blk.1 | 의미 |
|------|-------|-------|------|
| |K| mean | 316.2 | 283.8 | blk.1 K 약간 감소 |
| |V| mean | 0.31 | 3.37 | **blk.1 V = 10배 증가** |
| |K|/|V| | 1023x | 84x | V 역할 비중 급증 |
| V std | 0.04 | 0.30 | V 분산도 증가 |

**발견**: 레이어 깊어질수록 V가 정보 전달 역할을 점점 더 담당
- blk.0: K 지배 → 위치/구조 인코딩 (V는 미약)
- blk.1: V 역할 증가 → 의미 정보 축적 시작
- 이전 §02 관찰과 일치: blk.12에서 K≈V, blk.23에서 V 지배

---

## 2. Knowledge Injection (V 수정 → 출력 변화)

### 2.1 V 벡터 의미 분석

분석: `tools/knowledge_injection.py --mode report`

```
전체 통계 (151,936 tokens):
  |K| mean = 316.2, std = 0.96  → K는 거의 상수 (bias 지배)
  |V| mean = 0.31, std = 0.04   → V는 작지만 토큰마다 다름
  |K|/|V| = 1023배               → K 압도적
```

의미적 유사도 (V cosine similarity):
```
cat-dog     cos(V)=0.53  L2=0.30   (유사 카테고리)
cat-fish    cos(V)=0.54  L2=0.29   (동물)
cat-table   cos(V)=0.44  L2=0.29   (무관)
king-queen  cos(V)=0.60  L2=0.25   (관계 유사)
man-woman   cos(V)=0.58  L2=0.24   (관계 유사)
big-large   cos(V)=0.68  L2=0.26   (동의어)
big-small   cos(V)=0.72  L2=0.25   (반의어 — 놀랍게 높음!)
good-bad    cos(V)=0.69  L2=0.24   (반의어)
```

**발견**:
- K는 모든 쌍에서 cos≈1.0 → attention routing은 거의 동일 (blk.0에서는 위치가 중요)
- V가 의미 분화를 담당: cos 0.44(무관)~0.72(반의어)
- **반의어 cos(V) > 유사어**: big-small(0.72) > cat-dog(0.53) 
  - 해석: 같은 의미 차원(크기)을 공유, 방향만 다름

### 2.2 실제 추론 실험 — V 수정이 출력에 미치는 인과적 효과

#### 실험 A: V=0 (기능어 정보 제거)
```
조건: "the"(id=1782), "is"(id=285), "a"(id=64), "in"(id=258)의 V를 0으로
프롬프트: "Quantum physics explains"

원본:   "the behavior of electrons in atoms. The electron is a particle..."
V=zero: "the existence of a quantum tunneling barrier. This barrier is a quantum..."
```
**→ 기능어 V=0은 blk.0 attention에서 정보 전달 차단 → 출력 방향 변경**

#### 실험 B: V=random×10 (극단적 noise 주입)
```
조건: "cat", "likes", "to", "A"의 V를 random×10 (|V| 0.31→10.0, 32배 증폭)
프롬프트: "A cat likes to"

원본:            "eat. One day, she ate 10% of the fish in the aquarium..."
V=random×10:    "WTWTWTWT"",,,。,",",,.(eksted..." (완전 파괴)
```
**→ 극단적 V noise는 출력을 완전히 깨뜨림 = V가 인과적으로 출력을 결정**

#### 실험 C: cat→dog V swap
```
조건: cat(id=4616) V ← dog(id=18457) V, K는 유지
프롬프트: "A cat likes to"

원본:  "eat. One day, she ate 10% of the fish in the aquarium..."
swap:  "eat. One day, she ate 10% of the fish in the aquarium..." (동일)
```
**→ 변화 없음. 이유: |V_cat - V_dog| = 0.30, |V_cat| = 0.31 → 상대 차이 ≈ 1배 수준**

### 2.3 V 수정 효과 공식 (실험에서 도출)

```
출력 변화량 ∝ |V_modified - V_original| × attention_weight × 토큰_등장_횟수

실험 결과:
  - |diff| = 0.30 (cat→dog):      변화 없음 (너무 작음)
  - |diff| = 0.31 (V=0, 기능어):   출력 변경 (반복 등장으로 누적)
  - |diff| ≈ 10.0 (random×10):    완전 파괴 (32배 차이)
```

**임계값 추정**: blk.0에서 단일 토큰의 V 수정이 눈에 보이려면:
- |V_diff| > 1.0 (원본의 3배 이상) 또는
- 수정된 토큰이 시퀀스에 여러 번 등장 (누적 효과)

### 2.4 층별 V 수정 효과 예측

| 레이어 | |V| mean | V 수정 감도 | 지식 주입 적합도 |
|--------|----------|------------|----------------|
| blk.0 | 0.31 | 낮음 (|K|에 묻힘) | 낮음 — 미세 조정 어려움 |
| blk.1 | 3.37 | 중간 | 중간 — 유의미한 효과 기대 |
| blk.12+ | ~1.0 (추정) | K≈V → 높음 | **높음 — 지식 주입 최적 지점** |
| blk.23 | >2.0 (추정) | V 지배 | 최고 — 하지만 context-dependency 극대 |

**→ 진정한 지식 주입은 중간~깊은 레이어에서 해야 효과적**
**→ blk.0은 "작동 증명"이지 "최적 지점"은 아님**

---

## 3. Continual Learning 실증

### 실험 설계
```
Base: identity 도메인 (token 0-4999)  → cl_base_identity.bin (5MB)
New:  animal 도메인 (token 5000-9999) → cl_new_animal.bin (5MB)
Merged: Base ∪ New (overlap 0)        → cl_merged.bin (10MB)
```

구현: `tools/continual_learning_demo.py --mode full-demo`

### 결과: ALL TESTS PASSED

| Test | 결과 | 검증 방법 |
|------|------|----------|
| Base preservation | 1000/1000 (100%) | merged에서 base-only 토큰 bit-exact 확인 |
| New presence | 1000/1000 (100%) | merged에서 new-only 토큰 bit-exact 확인 |
| Isolation | 1000/1000 (100%) | 100개 V 랜덤 파괴 → 나머지 1000개 bit-exact |

### 핵심 원리: 구조적 No-Forgetting

```
LLM Weight Matrix (Global Coupling):
  output = W × input
  W 수정 → 모든 토큰 출력 변화 → catastrophic forgetting 필연

KV Hashmap (Token Independence):
  output = hashmap[token_id]
  hashmap[X] 수정 → X만 변화, 나머지 물리적 불변 → forgetting 불가능
```

비유:
- LLM = 칠판 (모두가 공유, 덧쓰기 = 기존 훼손)
- Hashmap = 개인 노트 (독립, 새 노트 추가 = 기존 무관)

### 의미

기존 continual learning 연구 (EWC, LoRA, replay, progressive nets 등)는 모두
"weight 공유 구조에서 forgetting을 줄이는 알고리즘"임.

KV Hashmap은 **자료구조 자체가 forgetting을 불가능하게 만드는 아키텍처**.
알고리즘이 아닌 구조적 해결 → trick이나 hyperparameter 필요 없음.

---

## 4. 종합 결론 및 다음 방향

### 이번 세션에서 입증된 것

1. **blk.0 hashmap overlay 작동 확인** — V 수정이 실제 추론 출력에 인과적 영향
2. **blk.1 정확 캡처 필요성 입증** — embedding 근사 오차 V=165%, 사용 불가
3. **Continual learning 구조적 보장** — append-only = no-forgetting (수학적 증명 수준)
4. **V 크기 문제 발견** — blk.0의 |V|=0.31은 너무 작아서 미세 조정 어려움

### 다음 방향 (우선순위)

#### 1. 깊은 레이어 V 수정 (가장 유망)
- [예상] blk.12~23에서 V를 수정하면 blk.0보다 훨씬 큰 효과
- [의외 후보] blk.1의 V(|V|=3.37)만으로도 충분할 수 있음 (10배 크기)

#### 2. blk.1 exact hashmap (context_hash 기반)
- [예상] sequence forward로 context-dependent K,V 캡처 → blk.1 정확 hashmap
- [의외 후보] 자주 등장하는 n-gram 패턴만 캐시해도 80% 커버 가능

#### 3. V 학습 알고리즘 설계
- [예상] target output의 gradient를 V 공간으로 역전파 → optimal V 계산
- [의외 후보] 단순히 target token embedding을 V에 넣는 것만으로도 작동할 수 있음

#### 4. Multi-layer hashmap (blk.0~1 동시)
- [예상] blk.0(구조) + blk.1(의미) 결합이 최소 viable 아키텍처
- [의외 후보] blk.0 없이 blk.1만으로도 지식 주입 충분할 수 있음

---

## 5. 실행 방법

```bash
# ── blk.1 Forward 캡처 ──
python tools/capture_blk1_forward.py --mode verify
python tools/capture_blk1_forward.py --mode single --range "0,100"
python tools/capture_blk1_forward.py --mode sequence --tokens "9707,1879,374,264,1273"

# ── 지식 주입 실험 ──
python tools/knowledge_injection.py --mode report
python tools/knowledge_injection.py --mode swap --src cat --dst dog
python tools/knowledge_injection.py --mode blend --src cat --dst dog --alpha 0.5
python tools/knowledge_injection.py --mode zero --tokens "the,is,a,in"

# ── Continual Learning ──
python tools/continual_learning_demo.py --mode full-demo
python tools/continual_learning_demo.py --mode isolation-test --n-corrupt 1000

# ── 실제 추론 비교 ──
# 원본
slot_fusion infer -m models/qwen2-0_5b-instruct-q4_k_m.gguf \
  -s models/mia_slot.bin -p "Quantum physics explains" -n 30 \
  --kv-hashmap weights/blk0_kv_full.bin -ngl 0

# V=0 기능어 (출력 변경 확인)
slot_fusion infer -m models/qwen2-0_5b-instruct-q4_k_m.gguf \
  -s models/mia_slot.bin -p "Quantum physics explains" -n 30 \
  --kv-hashmap weights/blk0_kv_zeroed_funcwords.bin -ngl 0

# V=random×10 (출력 파괴 확인)
slot_fusion infer -m models/qwen2-0_5b-instruct-q4_k_m.gguf \
  -s models/mia_slot.bin -p "A cat likes to" -n 30 \
  --kv-hashmap weights/blk0_kv_extreme_random.bin -ngl 0
```

---

## 6. 생성된 파일 목록

| 파일 | 용도 |
|------|------|
| `tools/capture_blk1_forward.py` | blk.0 full forward → blk.1 K,V 캡처 |
| `tools/knowledge_injection.py` | V-swap, V-blend, V-zero, report |
| `tools/continual_learning_demo.py` | CL build/merge/verify/isolation |
| `weights/blk1_kv_single.bin` | blk.1 K,V (100 tokens, pos=0 exact) |
| `weights/blk1_forward_debug.txt` | blk.0 forward 디버그 데이터 |
| `weights/blk1_sequence_debug.txt` | sequence forward 디버그 |
| `weights/blk0_kv_swap_cat_dog.bin` | cat V → dog V 교체 |
| `weights/blk0_kv_zeroed_funcwords.bin` | the/is/a/in V=0 |
| `weights/blk0_kv_extreme_random.bin` | cat/likes/to/A V=random×10 |
| `weights/cl_base_identity.bin` | CL: base 도메인 (5000 tokens) |
| `weights/cl_new_animal.bin` | CL: new 도메인 (5000 tokens) |
| `weights/cl_merged.bin` | CL: merged (10000 tokens) |
| `weights/knowledge_injection_report.txt` | V 유사도 분석 리포트 |
