# §04 V Learning: blk.1 V로 주제 유도 성공

## 핵심 결과

```
프롬프트: "A cat likes to"
Baseline: "eat. One day, she ate 2/3 of a particular novel..."

V=ocean:  "play with toys. He has 10 toys..."
V=swim:   "play with a ball. What is the probability..."
V=sleep:  "play with cat toys. The cat has 14 toys."
V=fish:   "play with a ball. A cat likes to play..."
V=mouse:  "play with a ball."
V=zero:   "play with the cat. What is the cat's favorite color..."
```

**blk.1 V modification (scale=1, 자연 크기)으로 slot memory(alpha=5.0)를 이기고 주제 변경 성공.**

---

## 실험 설계

### 방법
1. "A cat likes to" 프롬프트를 blk.0 full forward로 처리
2. blk.1의 K,V를 context_hash 기반으로 정확 계산
3. V를 target 단어의 자연 V로 교체 (scale=1.0, 증폭 없음)
4. MKVS 2-layer stack (blk.0 원본 + blk.1 수정)으로 추론

### 작동 원리
```
blk.1 attention에서:
  각 position이 attend할 때 V[i]의 정보를 읽어감
  V[i] = "이 위치가 전달하는 의미 정보"

  V를 "ocean" 방향으로 설정하면:
  → 모든 position이 "ocean" 관련 정보를 전달
  → 후속 레이어가 이 방향을 기반으로 생성
  → "eat" 대신 "play" 영역으로 이동
```

---

## 발견 사항

### 1. V = "주제/카테고리 제어기" (단어 제어 아님)

| V 방향 | 출력 첫 단어 | 주제 |
|--------|-------------|------|
| 원본 | eat | 먹기 |
| ocean/water | play with toys | 놀이(장난감) |
| swim/fish/run | play with a ball | 놀이(공) |
| sleep | play with cat toys | 놀이(고양이 장난감) |
| hunt | play with toys and play | 놀이(반복) |

**모든 target V가 "play" 주제로 수렴** — V가 지정하는 것은 "의미 영역"이지 "정확한 단어"가 아님.

### 2. 계층별 V 제어 수준

| 레이어 | V의 제어 수준 | 도달 가능 레벨 |
|--------|-------------|--------------|
| blk.0 | 거의 없음 (|V|=0.31 너무 작음) | 무효과 |
| blk.1 | **주제/카테고리** | "eat" → "play" |
| blk.1 V=0 | 주제+문법 변경 | "play with the cat. What is..." |
| blk.0~5 동시 | 과포화 (루프) | "cat cat cat..." |
| blk.12+ (미실험) | 단어 수준? (추정) | TBD |

### 3. Scale 효과

| Scale | |V1| | 결과 |
|-------|------|------|
| 0 (원본) | 3.23 | "eat" (baseline) |
| 1 (자연) | 4.30 | **"play with toys"** (주제 변경!) |
| 3 | 12.9 | "play with the cat..." (반복 시작) |
| 5 | 21.5 | "like like like..." (루프) |

**최적: scale=1 (자연 크기). 증폭 불필요!**

### 4. Slot Memory vs V Learning 역학

```
Slot Memory (alpha=5.0):
  "A cat likes to" → "eat" (기억된 패턴 강제 재생)
  logit[eat] += 5.0 * slot_score[eat]

blk.1 V=ocean:
  attention이 "ocean" 정보를 residual stream에 주입
  → 22개 layer가 이 방향을 처리
  → logit 분포 전체가 "play" 영역으로 shift
  → "eat" slot boost를 이김!

이것은 V가 "upstream"에서 정보 흐름을 바꾸기 때문.
Slot은 "downstream"에서 logit을 더하는 것.
V가 더 근본적인 제어 (정보 자체를 바꿈 vs 점수만 수정).
```

---

## 시사점: Continual Learning 맥락

### V Learning으로 가능한 것
1. **주제 전환**: "이 context에서는 eat이 아니라 play를 말해라"
2. **카테고리 유도**: "동물 관련 → 놀이 관련"으로 영역 이동
3. **정보 주입**: V에 특정 개념을 넣으면 후속 생성이 그 방향으로 감

### V Learning으로 아직 불가능한 것
1. **정확한 단어 생성**: "swim"을 넣어도 "swim"이 나오지 않음 (play가 나옴)
2. **문장 수준 제어**: 특정 문장을 출력하게 만들기
3. **사실 관계 주입**: "펭귄의 색은 검은색이다"를 V로 인코딩

### 해결 방향
- [예상] 더 깊은 레이어(blk.12~23)에서 V 수정 → 단어 수준 제어 가능
- [의외 후보] V + Slot 결합: V로 주제 설정 + Slot으로 정확한 단어 선택 → 2단계 시스템

---

## 파일 목록

| 파일 | 용도 |
|------|------|
| `tools/v_learning.py` | V learning 3가지 방법 (projection, contrastive, attn_gradient) |
| `tools/build_blk1_stack.py` | blk.1 exact MKVS builder + V injection |
| `weights/stack_blk1_ocean_s1.bin` | blk.1 V=ocean (작동 확인) |
| `weights/stack_blk01_zero.bin` | blk.1 V=0 (주제 변경 확인) |
| `weights/stack_blk01_inverted.bin` | blk.1 V=-V (완전 다른 출력) |
| `weights/stack_blk05_ocean.bin` | blk.0~5 동시 (과포화) |

## 실행 방법

```bash
# blk.1 V=target로 주제 유도
python tools/build_blk1_stack.py --mode demo

# V learning 전체 데모
python tools/v_learning.py --mode demo

# 특정 target V로 MKVS 빌드
python tools/build_blk1_stack.py --mode inject --tokens "32,4616,25039,983" --target 1 --scale 1
```
