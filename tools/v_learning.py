#!/usr/bin/env python3
"""
V Learning Algorithm — 정답이 주어졌을 때 최적 V를 계산

목표: (input_tokens, target_token) → optimal V 학습
     → hashmap에 저장 → 추론 시 target 출력 유도

3가지 방법:
  A) Embedding Projection — target 임베딩을 V 공간으로 역투영 (즉시, 분석적)
  B) Contrastive Iterative — slot_fusion 추론 결과 기반 반복 조정
  C) Attention Gradient — attention 구조를 통한 해석적 gradient

핵심 원리:
  attention_output[pos] = Σ_i (attn_weight[pos→i] × V[i])
  → V[i]는 position i가 다른 위치에 "전달하는 정보"
  → V[i]를 target 방향으로 설정하면, 해당 토큰이 target 정보를 전달

Usage:
  python tools/v_learning.py --mode projection --target-word "dog" --source-word "cat"
  python tools/v_learning.py --mode contrastive --prompt "A cat likes to" --target "sleep"
  python tools/v_learning.py --mode demo
"""

import argparse
import struct
import os
import sys
import subprocess
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GGUF_PATH = os.path.join(MODELS_DIR, "qwen2-0_5b-instruct-q4_k_m.gguf")
SLOT_FUSION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "slot_fusion", "build", "Release", "slot_fusion.exe")

RMS_EPS = 1e-6
N_EMBD = 896
KV_DIM = 128
MAGIC = b"KVH0"
VERSION = 1


def rms_norm(x, gamma, eps=RMS_EPS):
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * gamma


def load_gguf_tensor(tensor_name):
    from gguf import GGUFReader, GGMLQuantizationType, dequantize
    reader = GGUFReader(GGUF_PATH)
    for t in reader.tensors:
        if t.name == tensor_name:
            qt = GGMLQuantizationType(t.tensor_type)
            return dequantize(t.data, qt)
    raise RuntimeError(f"Tensor '{tensor_name}' not found")


def get_vocab():
    """GGUF vocab: text → token_id"""
    from gguf import GGUFReader
    reader = GGUFReader(GGUF_PATH)
    vocab = {}
    for field_name, field in reader.fields.items():
        if "tokenizer.ggml.tokens" in field_name:
            for idx in range(len(field.data)):
                part_idx = field.data[idx]
                token_bytes = bytes(field.parts[part_idx])
                try:
                    token_str = token_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    token_str = token_bytes.hex()
                vocab[token_str] = idx
            break
    return vocab


def find_token(word, vocab):
    """단어로 token_id 찾기"""
    for candidate in [word, f"\u0120{word}", f" {word}", word.lower(), f"\u0120{word.lower()}"]:
        if candidate in vocab:
            return vocab[candidate], candidate
    matches = [(k, v) for k, v in vocab.items() if word.lower() in k.lower()]
    if matches:
        matches.sort(key=lambda x: len(x[0]))
        return matches[0][1], matches[0][0]
    return None, None


def load_hashmap(path):
    entries = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        ver = struct.unpack("<i", f.read(4))[0]
        kv_dim = struct.unpack("<i", f.read(4))[0]
        n = struct.unpack("<i", f.read(4))[0]
        for _ in range(n):
            tid = struct.unpack("<i", f.read(4))[0]
            k = np.frombuffer(f.read(kv_dim * 4), dtype=np.float32).copy()
            v = np.frombuffer(f.read(kv_dim * 4), dtype=np.float32).copy()
            entries[tid] = (k, v)
    return entries, kv_dim


def save_hashmap(entries, path, kv_dim=KV_DIM):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<i", kv_dim))
        f.write(struct.pack("<i", len(entries)))
        for tid in sorted(entries.keys()):
            k, v = entries[tid]
            f.write(struct.pack("<i", tid))
            f.write(k.tobytes())
            f.write(v.tobytes())


def run_inference(hashmap_path, prompt, n_tokens=10):
    """slot_fusion으로 추론 실행, 생성된 텍스트 반환"""
    cmd = [
        SLOT_FUSION, "infer",
        "-m", os.path.join(MODELS_DIR, "qwen2-0_5b-instruct-q4_k_m.gguf"),
        "-s", os.path.join(MODELS_DIR, "mia_slot.bin"),
        "-p", prompt,
        "-n", str(n_tokens),
        "-ngl", "0",
        "--kv-hashmap", hashmap_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        # [prompt] 이후의 생성 텍스트 추출
        for line in output.split("\n"):
            if "[prompt]" in line:
                gen = line.split("[prompt]")[1].strip()
                # ~llama_context 제거
                if "~" in gen:
                    gen = gen.split("~")[0]
                return gen.strip()
    except Exception as e:
        return f"ERROR: {e}"
    return ""


# ═══════════════════════════════════════════════════════════════
# Method A: Embedding Projection
# ═══════════════════════════════════════════════════════════════

def method_projection(source_token_id, target_token_id, W, entries, scale=1.0):
    """
    V 공간에서 target 방향으로 source의 V를 이동.

    원리:
      V는 Wv @ RMSNorm(emb) + bv로 계산됨.
      target의 V는 Wv @ RMSNorm(emb[target]) + bv.
      → source의 V를 target의 V 방향으로 설정하면,
        attention이 source 토큰에서 target 정보를 읽어감.

    scale=1.0: 완전 교체 (V_source = V_target)
    scale=0.5: 50% 혼합
    scale>1.0: target 방향 증폭
    """
    emb = W['emb']
    wv = W['wv']
    bv = W['bv']
    attn_norm = W['attn_norm']

    # Source와 Target의 원본 V 계산
    normed_src = rms_norm(emb[source_token_id], attn_norm)
    v_src = (wv @ normed_src + bv).astype(np.float32)

    normed_tgt = rms_norm(emb[target_token_id], attn_norm)
    v_tgt = (wv @ normed_tgt + bv).astype(np.float32)

    # Projection: source V를 target V 방향으로 이동
    # V_new = (1 - scale) * V_src + scale * V_tgt
    v_new = (1.0 - scale) * v_src + scale * v_tgt

    return v_new


def method_embedding_inject(target_token_id, W, scale=5.0):
    """
    Method A+: Target 임베딩을 Wv의 pseudo-inverse로 V 공간에 직접 매핑

    원리:
      최종 출력은 output_layer(hidden_state) = emb^T @ hidden
      attention이 V를 residual에 추가 → hidden에 V 성분이 남음
      V가 target_emb 방향이면 → dot(hidden, target_emb) 증가 → target logit 증가

    구현:
      V_optimal = Wv @ emb[target] * scale (Wv가 이미 emb→V 매핑이므로)
      또는 단순히: V = normalized(emb[target]) * target_V_magnitude * scale
    """
    emb = W['emb']
    wv = W['wv']
    bv = W['bv']
    attn_norm = W['attn_norm']

    # Target의 자연 V (Wv를 통과한 정상적인 V)
    normed_tgt = rms_norm(emb[target_token_id], attn_norm)
    v_natural = (wv @ normed_tgt + bv).astype(np.float32)

    # Scale up: V를 크게 만들수록 attention이 이 정보를 강하게 전달
    v_amplified = v_natural * scale

    return v_amplified


# ═══════════════════════════════════════════════════════════════
# Method B: Contrastive Iterative Learning
# ═══════════════════════════════════════════════════════════════

def method_contrastive(source_token_id, target_word, prompt, W, entries,
                       lr=0.5, epochs=5, kv_dim=KV_DIM):
    """
    Contrastive V Learning: slot_fusion을 oracle로 사용하여 V를 반복 조정.

    알고리즘:
      1. 현재 V로 추론 → 생성된 첫 토큰 확인
      2. target이 나왔으면: 성공, 종료
      3. target 아니면:
         - V += lr * (V_target - V_current_direction)
         - 즉, target의 자연 V 방향으로 이동
      4. 반복

    이것은 "V 공간에서의 gradient-free optimization"
    - 실제 gradient 대신 target V 방향을 supervision signal로 사용
    - 제1원칙: 방향은 자동 학습 (target V가 방향 제공)
    """
    vocab = get_vocab()
    target_id, target_found = find_token(target_word, vocab)
    if target_id is None:
        print(f"  ERROR: target '{target_word}' not found in vocab")
        return None

    # Target의 자연 V
    attn_norm = W['attn_norm']
    emb = W['emb']
    wv = W['wv']
    bv = W['bv']

    normed_tgt = rms_norm(emb[target_id], attn_norm)
    v_target = (wv @ normed_tgt + bv).astype(np.float32)

    # 현재 V
    k_src, v_current = entries[source_token_id]
    v_current = v_current.copy()

    hashmap_path = os.path.join(WEIGHTS_DIR, "v_learning_iter.bin")
    history = []

    for ep in range(epochs):
        # V 저장 + 추론
        entries[source_token_id] = (k_src, v_current)
        save_hashmap(entries, hashmap_path)

        gen_text = run_inference(hashmap_path, prompt, n_tokens=5)
        first_word = gen_text.split()[0] if gen_text.split() else ""

        # 평가
        hit = target_word.lower() in gen_text.lower()
        v_norm = np.linalg.norm(v_current)

        history.append({
            'epoch': ep, 'output': gen_text[:50],
            'hit': hit, 'v_norm': v_norm
        })

        print(f"  ep={ep}: |V|={v_norm:.4f} output=\"{gen_text[:40]}\" "
              f"{'[HIT]' if hit else '[MISS]'}")

        if hit:
            print(f"  >>> Target '{target_word}' found! Learning converged.")
            break

        # V 업데이트: target V 방향으로 이동
        # delta = V_target_direction - V_current_direction
        v_tgt_dir = v_target / (np.linalg.norm(v_target) + 1e-10)
        v_cur_dir = v_current / (np.linalg.norm(v_current) + 1e-10)

        # Contrastive gradient: target 방향으로 + 크기 증폭
        delta = v_target * (1.0 + ep * 0.5) - v_current
        v_current = v_current + lr * delta

    return v_current, history


# ═══════════════════════════════════════════════════════════════
# Method C: Attention-aware V optimization
# ═══════════════════════════════════════════════════════════════

def method_attention_gradient(source_token_id, target_token_id, W, entries):
    """
    Attention 구조를 활용한 해석적 V 최적화.

    원리:
      blk.0 attention에서 position p의 출력 기여:
        contribution_p = attn_weight[q→p] × Wo × V[p]

      최종 logit[target] 증가를 위해:
        V[source]를 Wo^T × output_direction으로 설정

      여기서 output_direction = emb[target] (output layer가 emb 재사용)

    구현:
      V_optimal = scale * (Wo^T × emb[target])[:KV_DIM]
      (Wo^T가 896→896이고 V는 128차원이므로, GQA 구조 고려 필요)
    """
    emb = W['emb']
    wo = W['wo']  # (896, 896)

    # Target이 output에서 선택되려면:
    # logit[target] = emb[target] @ hidden_final
    # hidden_final에 V 기여분이 포함됨
    # V → Wo를 통과 → residual에 추가

    # V의 attention 출력 경로 (GQA 14Q, 2KV):
    # V는 (128,) = 2 KV heads × 64 dim
    # GQA 확장 후 concat: (896,) = [V[0:64] repeated 7 times, V[64:128] repeated 7 times]
    # Wo @ concat → (896,) residual 기여

    target_emb = emb[target_token_id].astype(np.float32)

    # 역방향: target_emb가 residual에 나타나려면 Wo의 어떤 입력이 필요한가?
    # Wo @ concat_v = target_direction
    # concat_v = Wo^-1 @ target_direction (pseudo-inverse)
    # 하지만 Wo는 896×896, concat_v도 896차원

    # Pseudo-inverse 계산 (비용 높지만 정확)
    # 대신 간단한 근사: Wo^T @ target_emb (transpose ≈ pseudo-inverse for orthogonal)
    optimal_concat = wo.T @ target_emb  # (896,)

    # GQA 역매핑: concat(896) → V(128)
    # concat = [V[0:64]*7, V[64:128]*7]
    # 역: V[0:64] = mean(concat[0:448] reshaped (7,64), axis=0)
    #     V[64:128] = mean(concat[448:896] reshaped (7,64), axis=0)
    v_head0 = optimal_concat[:7*64].reshape(7, 64).mean(axis=0)
    v_head1 = optimal_concat[7*64:].reshape(7, 64).mean(axis=0)
    v_optimal = np.concatenate([v_head0, v_head1]).astype(np.float32)

    # Scale: 원본 V norm 수준으로 정규화 후 증폭
    original_v_norm = 0.31  # blk.0 평균 |V|
    v_optimal = v_optimal / (np.linalg.norm(v_optimal) + 1e-10) * original_v_norm * 5.0

    return v_optimal


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

def do_demo(args):
    """전체 데모: 3가지 방법으로 V 학습 → 추론 검증"""

    print("=" * 60)
    print("  V Learning Algorithm Demo")
    print("  Goal: 'A cat likes to' → target word in output")
    print("=" * 60)

    # 가중치 로드
    print("\n[load] Loading weights...")
    emb_raw = np.load(os.path.join(WEIGHTS_DIR, "token_embd.weight.npy"))
    n_vocab = emb_raw.size // N_EMBD
    emb = emb_raw.reshape(n_vocab, N_EMBD).astype(np.float32)
    attn_norm = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_norm.weight.npy")).astype(np.float32)
    wv = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.weight.npy")).reshape(KV_DIM, N_EMBD).astype(np.float32)
    bv = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.bias.npy")).astype(np.float32)
    wo = load_gguf_tensor("blk.0.attn_output.weight").astype(np.float32)

    W = {'emb': emb, 'attn_norm': attn_norm, 'wv': wv, 'bv': bv, 'wo': wo}

    # Vocab + 토큰 찾기
    vocab = get_vocab()
    cat_id, _ = find_token("cat", vocab)
    sleep_id, _ = find_token("sleep", vocab)
    swim_id, _ = find_token("swim", vocab)
    fly_id, _ = find_token("fly", vocab)

    print(f"  cat={cat_id}, sleep={sleep_id}, swim={swim_id}, fly={fly_id}")

    # Hashmap 로드
    hashmap_path = os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin")
    entries, kv_dim = load_hashmap(hashmap_path)
    print(f"  Loaded {len(entries)} entries")

    prompt = "A cat likes to"

    # ── Baseline ──
    print(f"\n{'='*60}")
    print(f"  BASELINE: original V")
    print(f"{'='*60}")
    baseline_path = os.path.join(WEIGHTS_DIR, "v_learn_baseline.bin")
    save_hashmap(entries, baseline_path)
    baseline_out = run_inference(baseline_path, prompt, 15)
    print(f"  Output: \"{baseline_out}\"")

    # ── Method A: Projection (cat V → sleep V) ──
    print(f"\n{'='*60}")
    print(f"  METHOD A: Embedding Projection")
    print(f"  cat.V = sleep.V (direct replacement)")
    print(f"{'='*60}")

    entries_a = dict(entries)
    # Scale 1: 직접 교체
    v_proj = method_projection(cat_id, sleep_id, W, entries_a, scale=1.0)
    k_cat, _ = entries_a[cat_id]
    entries_a[cat_id] = (k_cat, v_proj)
    path_a = os.path.join(WEIGHTS_DIR, "v_learn_proj_s1.bin")
    save_hashmap(entries_a, path_a)
    out_a1 = run_inference(path_a, prompt, 15)
    print(f"  scale=1.0: \"{out_a1}\"")

    # Scale 5: 증폭
    entries_a2 = dict(entries)
    v_amp = method_embedding_inject(sleep_id, W, scale=5.0)
    entries_a2[cat_id] = (k_cat, v_amp)
    path_a2 = os.path.join(WEIGHTS_DIR, "v_learn_proj_s5.bin")
    save_hashmap(entries_a2, path_a2)
    out_a2 = run_inference(path_a2, prompt, 15)
    print(f"  scale=5.0: \"{out_a2}\"")

    # Scale 20: 강한 증폭
    entries_a3 = dict(entries)
    v_amp20 = method_embedding_inject(sleep_id, W, scale=20.0)
    entries_a3[cat_id] = (k_cat, v_amp20)
    path_a3 = os.path.join(WEIGHTS_DIR, "v_learn_proj_s20.bin")
    save_hashmap(entries_a3, path_a3)
    out_a3 = run_inference(path_a3, prompt, 15)
    print(f"  scale=20.0: \"{out_a3}\"")

    # ── Method C: Attention Gradient ──
    print(f"\n{'='*60}")
    print(f"  METHOD C: Attention Gradient (Wo^T projection)")
    print(f"  Compute V that maximizes dot(Wo@V, emb[sleep])")
    print(f"{'='*60}")

    entries_c = dict(entries)
    v_attn = method_attention_gradient(cat_id, sleep_id, W, entries_c)
    entries_c[cat_id] = (k_cat, v_attn)
    path_c = os.path.join(WEIGHTS_DIR, "v_learn_attn_grad.bin")
    save_hashmap(entries_c, path_c)
    out_c = run_inference(path_c, prompt, 15)
    print(f"  Output: \"{out_c}\"")

    # Scale up
    entries_c2 = dict(entries)
    v_attn2 = v_attn * 20.0
    entries_c2[cat_id] = (k_cat, v_attn2)
    path_c2 = os.path.join(WEIGHTS_DIR, "v_learn_attn_grad_s20.bin")
    save_hashmap(entries_c2, path_c2)
    out_c2 = run_inference(path_c2, prompt, 15)
    print(f"  scale=20: \"{out_c2}\"")

    # ── Method B: Contrastive (iterative) ──
    print(f"\n{'='*60}")
    print(f"  METHOD B: Contrastive Iterative Learning")
    print(f"  Target: 'sleep' in output")
    print(f"{'='*60}")

    entries_b = dict(entries)
    v_learned, history = method_contrastive(
        cat_id, "sleep", prompt, W, entries_b,
        lr=0.5, epochs=5
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:           \"{baseline_out[:50]}\"")
    print(f"  Projection s=1:     \"{out_a1[:50]}\"")
    print(f"  Projection s=5:     \"{out_a2[:50]}\"")
    print(f"  Projection s=20:    \"{out_a3[:50]}\"")
    print(f"  Attn gradient:      \"{out_c[:50]}\"")
    print(f"  Attn gradient s=20: \"{out_c2[:50]}\"")
    if history:
        print(f"  Contrastive final:  \"{history[-1]['output'][:50]}\"")

    # 성공 판정
    target = "sleep"
    results = {
        'baseline': target in baseline_out.lower(),
        'proj_s1': target in out_a1.lower(),
        'proj_s5': target in out_a2.lower(),
        'proj_s20': target in out_a3.lower(),
        'attn': target in out_c.lower(),
        'attn_s20': target in out_c2.lower(),
    }

    print(f"\n  Target '{target}' in output?")
    for name, hit in results.items():
        print(f"    {name:15s}: {'[HIT]' if hit else '[MISS]'}")


def do_projection(args):
    """Method A: 특정 word의 V를 target word의 V로 교체"""
    vocab = get_vocab()
    src_id, src_found = find_token(args.source_word, vocab)
    tgt_id, tgt_found = find_token(args.target_word, vocab)

    if src_id is None or tgt_id is None:
        print(f"ERROR: token not found (src={args.source_word}, tgt={args.target_word})")
        return

    print(f"[projection] {src_found}(id={src_id}) V <- {tgt_found}(id={tgt_id}) V * {args.scale}")

    emb_raw = np.load(os.path.join(WEIGHTS_DIR, "token_embd.weight.npy"))
    emb = emb_raw.reshape(emb_raw.size // N_EMBD, N_EMBD).astype(np.float32)
    attn_norm = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_norm.weight.npy")).astype(np.float32)
    wv = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.weight.npy")).reshape(KV_DIM, N_EMBD).astype(np.float32)
    bv = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.bias.npy")).astype(np.float32)
    W = {'emb': emb, 'attn_norm': attn_norm, 'wv': wv, 'bv': bv}

    entries, kv_dim = load_hashmap(os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin"))

    if args.scale > 1.0:
        v_new = method_embedding_inject(tgt_id, W, scale=args.scale)
    else:
        v_new = method_projection(src_id, tgt_id, W, entries, scale=args.scale)

    k_src, _ = entries[src_id]
    entries[src_id] = (k_src, v_new)

    output_path = args.output or os.path.join(WEIGHTS_DIR, f"v_learn_{args.source_word}_{args.target_word}.bin")
    save_hashmap(entries, output_path)
    print(f"[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser(description="V Learning Algorithm")
    parser.add_argument("--mode", choices=["projection", "contrastive", "demo"],
                        default="demo")
    parser.add_argument("--source-word", type=str, default="cat")
    parser.add_argument("--target-word", type=str, default="sleep")
    parser.add_argument("--prompt", type=str, default="A cat likes to")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", "-o", type=str)
    args = parser.parse_args()

    if args.mode == "demo":
        do_demo(args)
    elif args.mode == "projection":
        do_projection(args)
    elif args.mode == "contrastive":
        vocab = get_vocab()
        src_id, _ = find_token(args.source_word, vocab)
        emb_raw = np.load(os.path.join(WEIGHTS_DIR, "token_embd.weight.npy"))
        emb = emb_raw.reshape(emb_raw.size // N_EMBD, N_EMBD).astype(np.float32)
        attn_norm = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_norm.weight.npy")).astype(np.float32)
        wv = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.weight.npy")).reshape(KV_DIM, N_EMBD).astype(np.float32)
        bv = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.bias.npy")).astype(np.float32)
        W = {'emb': emb, 'attn_norm': attn_norm, 'wv': wv, 'bv': bv}
        entries, _ = load_hashmap(os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin"))
        method_contrastive(src_id, args.target_word, args.prompt, W, entries)


if __name__ == "__main__":
    main()
