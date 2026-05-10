#!/usr/bin/env python3
"""
blk.1 Forward 캡처 — blk.0 전체 forward를 Python에서 재현하여 blk.1 입력 추출

문제: blk.1 입력 = blk.0 출력 (context-dependent) → embedding 근사 실패
해결: blk.0의 attention + FFN + residual을 정확히 계산

Qwen2-0.5B blk.0 forward:
  1. x = emb[token_id]
  2. normed = RMSNorm(x, attn_norm)
  3. Q = Wq @ normed + bq    (896 → 896, 14 heads × 64 dim)
  4. K = Wk @ normed + bk    (896 → 128, 2 KV heads × 64 dim)
  5. V = Wv @ normed + bv    (896 → 128, 2 KV heads × 64 dim)
  6. RoPE(Q, K, pos)         (pos=0 → identity)
  7. GQA attention: 14Q heads, 2KV heads (7:1 ratio)
  8. attn_out = Wo @ concat_heads
  9. x = x + attn_out        (residual)
  10. normed2 = RMSNorm(x, ffn_norm)
  11. gate = Wgate @ normed2   (896 → 4864)
  12. up = Wup @ normed2       (896 → 4864)
  13. hidden = silu(gate) * up  (SwiGLU)
  14. ffn_out = Wdown @ hidden  (4864 → 896)
  15. x = x + ffn_out          (residual) → blk.1 입력!

단일 토큰(pos=0): RoPE=identity, attention=trivial → exact computation
다중 토큰(seq): full causal attention + RoPE 필요

Usage:
  python tools/capture_blk1_forward.py --mode single    (전체 vocab, pos=0)
  python tools/capture_blk1_forward.py --mode sequence --tokens "1,2,3,4,5"
  python tools/capture_blk1_forward.py --mode verify    (llama.cpp 출력과 비교)
"""

import argparse
import struct
import os
import sys
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GGUF_PATH = os.path.join(MODELS_DIR, "qwen2-0_5b-instruct-q4_k_m.gguf")

RMS_EPS = 1e-6
N_EMBD = 896
KV_DIM = 128
N_HEAD = 14       # Q heads
N_KV_HEAD = 2     # KV heads (GQA)
HEAD_DIM = 64     # 896 / 14 = 64
FFN_DIM = 4864
ROPE_THETA = 1000000.0  # Qwen2 uses 1M base (default)

MAGIC = b"KVH0"
VERSION = 1


def rms_norm(x, gamma, eps=RMS_EPS):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * gamma"""
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * gamma


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def rope_embed(x, pos, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """
    RoPE: Rotary Position Embedding
    x: (head_dim,) for a single head at position pos
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = pos * freqs  # (half,)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    x_even = x[0::2]  # (half,)
    x_odd = x[1::2]   # (half,)

    out = np.empty_like(x)
    out[0::2] = x_even * cos_a - x_odd * sin_a
    out[1::2] = x_even * sin_a + x_odd * cos_a
    return out


def load_gguf_tensor(tensor_name):
    """GGUF에서 특정 텐서를 dequantize하여 반환"""
    from gguf import GGUFReader, GGMLQuantizationType, dequantize

    reader = GGUFReader(GGUF_PATH)
    for t in reader.tensors:
        if t.name == tensor_name:
            qt = GGMLQuantizationType(t.tensor_type)
            return dequantize(t.data, qt)
    raise RuntimeError(f"Tensor '{tensor_name}' not found in GGUF")


def load_blk0_all_weights():
    """blk.0 전체 forward에 필요한 모든 가중치 로드"""
    print("[load] Loading blk.0 full weights...")

    # Embedding
    emb_raw = np.load(os.path.join(WEIGHTS_DIR, "token_embd.weight.npy"))
    n_vocab = emb_raw.size // N_EMBD
    emb = emb_raw.reshape(n_vocab, N_EMBD).astype(np.float32)
    print(f"  emb: ({n_vocab}, {N_EMBD})")

    # Attention norm
    attn_norm = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_norm.weight.npy")).astype(np.float32)

    # Q, K, V weights + biases (from GGUF for quantized ones)
    print("  Wq: dequantizing...")
    wq = load_gguf_tensor("blk.0.attn_q.weight").astype(np.float32)  # (896, 896)
    bq = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_q.bias.npy")).astype(np.float32)

    print("  Wk: dequantizing...")
    wk = load_gguf_tensor("blk.0.attn_k.weight").astype(np.float32)  # (128, 896)
    bk = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_k.bias.npy")).astype(np.float32)

    # Wv
    wv_path = os.path.join(WEIGHTS_DIR, "blk.0.attn_v.weight.npy")
    wv = np.load(wv_path).reshape(KV_DIM, N_EMBD).astype(np.float32)
    bv = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.bias.npy")).astype(np.float32)

    # Output projection
    print("  Wo: dequantizing...")
    wo = load_gguf_tensor("blk.0.attn_output.weight").astype(np.float32)  # (896, 896)

    # FFN norm
    ffn_norm = np.load(os.path.join(WEIGHTS_DIR, "blk.0.ffn_norm.weight.npy")).astype(np.float32)

    # FFN weights
    print("  FFN gate/up/down: dequantizing...")
    w_gate = load_gguf_tensor("blk.0.ffn_gate.weight").astype(np.float32)  # (4864, 896)
    w_up = load_gguf_tensor("blk.0.ffn_up.weight").astype(np.float32)      # (4864, 896)
    w_down = load_gguf_tensor("blk.0.ffn_down.weight").astype(np.float32)  # (896, 4864)

    print(f"  Wq: {wq.shape}, Wk: {wk.shape}, Wv: {wv.shape}, Wo: {wo.shape}")
    print(f"  FFN: gate={w_gate.shape}, up={w_up.shape}, down={w_down.shape}")

    # blk.1 K,V weights (for computing blk.1's K,V from blk.0 output)
    print("  blk.1 weights...")
    attn_norm1 = np.load(os.path.join(WEIGHTS_DIR, "blk.1.attn_norm.weight.npy")).astype(np.float32)
    wk1 = load_gguf_tensor("blk.1.attn_k.weight").astype(np.float32)
    bk1 = np.load(os.path.join(WEIGHTS_DIR, "blk.1.attn_k.bias.npy")).astype(np.float32)
    wv1_path = os.path.join(WEIGHTS_DIR, "blk.1.attn_v.weight.npy")
    wv1 = np.load(wv1_path).reshape(KV_DIM, N_EMBD).astype(np.float32)
    bv1 = np.load(os.path.join(WEIGHTS_DIR, "blk.1.attn_v.bias.npy")).astype(np.float32)

    print("[load] All weights loaded.")

    return {
        'emb': emb, 'n_vocab': n_vocab,
        'attn_norm': attn_norm,
        'wq': wq, 'bq': bq,
        'wk': wk, 'bk': bk,
        'wv': wv, 'bv': bv,
        'wo': wo,
        'ffn_norm': ffn_norm,
        'w_gate': w_gate, 'w_up': w_up, 'w_down': w_down,
        # blk.1
        'attn_norm1': attn_norm1,
        'wk1': wk1, 'bk1': bk1,
        'wv1': wv1, 'bv1': bv1,
    }


def forward_blk0_single_token(token_id, weights):
    """
    단일 토큰(pos=0)에 대한 blk.0 forward 수행.
    pos=0이므로 RoPE = identity, attention = trivial (single query over single key)

    Returns: blk.0 출력 (896,) = blk.1 입력
    """
    emb = weights['emb']
    x = emb[token_id].copy()  # (896,)

    # ── Attention ──
    normed = rms_norm(x, weights['attn_norm'])

    # Q, K, V projections
    q = weights['wq'] @ normed + weights['bq']   # (896,) — 14 heads × 64 dim
    k = weights['wk'] @ normed + weights['bk']   # (128,) — 2 heads × 64 dim
    v = weights['wv'] @ normed + weights['bv']   # (128,) — 2 heads × 64 dim

    # RoPE at pos=0: cos(0)=1, sin(0)=0 → identity (no change)
    # q_roped = q, k_roped = k (skip for pos=0)

    # GQA attention (single token → attn_weights = 1.0)
    # 14 Q heads, 2 KV heads: heads 0-6 use KV head 0, heads 7-13 use KV head 1
    # For single token: output of each Q head = V of its KV head
    v_heads = v.reshape(N_KV_HEAD, HEAD_DIM)  # (2, 64)

    # Expand GQA: each KV head serves N_HEAD/N_KV_HEAD = 7 Q heads
    heads_per_kv = N_HEAD // N_KV_HEAD  # 7
    concat_heads = np.zeros(N_EMBD, dtype=np.float32)  # (896,)
    for h in range(N_HEAD):
        kv_idx = h // heads_per_kv
        concat_heads[h * HEAD_DIM : (h + 1) * HEAD_DIM] = v_heads[kv_idx]

    # Output projection
    attn_out = weights['wo'] @ concat_heads  # (896,)

    # Residual
    x = x + attn_out

    # ── FFN (SwiGLU) ──
    normed2 = rms_norm(x, weights['ffn_norm'])

    gate = weights['w_gate'] @ normed2  # (4864,)
    up = weights['w_up'] @ normed2      # (4864,)
    hidden = silu(gate) * up            # SwiGLU activation
    ffn_out = weights['w_down'] @ hidden  # (896,)

    # Residual
    x = x + ffn_out

    return x  # blk.0 output = blk.1 input


def forward_blk0_sequence(token_ids, weights):
    """
    다중 토큰 시퀀스에 대한 blk.0 forward (full causal attention + RoPE)

    token_ids: list of int (sequence)
    Returns: list of (896,) vectors — 각 position의 blk.0 출력
    """
    emb = weights['emb']
    seq_len = len(token_ids)

    # 전체 시퀀스 임베딩
    X = np.array([emb[tid] for tid in token_ids], dtype=np.float32)  # (seq_len, 896)

    outputs = []

    # 각 position에 대해 attention 계산 (causal)
    # Pre-compute all Q, K, V
    normed_all = np.array([rms_norm(X[i], weights['attn_norm']) for i in range(seq_len)])

    Q_all = (normed_all @ weights['wq'].T) + weights['bq']  # (seq_len, 896)
    K_all = (normed_all @ weights['wk'].T) + weights['bk']  # (seq_len, 128)
    V_all = (normed_all @ weights['wv'].T) + weights['bv']  # (seq_len, 128)

    # Apply RoPE to Q and K
    for pos in range(seq_len):
        # Q: 14 heads × 64 dim
        q_heads = Q_all[pos].reshape(N_HEAD, HEAD_DIM)
        for h in range(N_HEAD):
            q_heads[h] = rope_embed(q_heads[h], pos)
        Q_all[pos] = q_heads.reshape(-1)

        # K: 2 heads × 64 dim
        k_heads = K_all[pos].reshape(N_KV_HEAD, HEAD_DIM)
        for h in range(N_KV_HEAD):
            k_heads[h] = rope_embed(k_heads[h], pos)
        K_all[pos] = k_heads.reshape(-1)

    # Causal self-attention per position
    for pos in range(seq_len):
        x = X[pos].copy()

        # GQA attention
        q_heads = Q_all[pos].reshape(N_HEAD, HEAD_DIM)  # (14, 64)
        k_avail = K_all[:pos+1].reshape(pos+1, N_KV_HEAD, HEAD_DIM)  # (pos+1, 2, 64)
        v_avail = V_all[:pos+1].reshape(pos+1, N_KV_HEAD, HEAD_DIM)  # (pos+1, 2, 64)

        heads_per_kv = N_HEAD // N_KV_HEAD  # 7
        concat_heads = np.zeros(N_EMBD, dtype=np.float32)

        for h in range(N_HEAD):
            kv_idx = h // heads_per_kv
            q_h = q_heads[h]  # (64,)
            k_h = k_avail[:, kv_idx, :]  # (pos+1, 64)
            v_h = v_avail[:, kv_idx, :]  # (pos+1, 64)

            # Scaled dot-product attention
            scores = (k_h @ q_h) / np.sqrt(HEAD_DIM)  # (pos+1,)

            # Softmax (causal — all positions up to pos are visible)
            scores_max = scores.max()
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / exp_scores.sum()

            # Weighted sum of values
            head_out = attn_weights @ v_h  # (64,)
            concat_heads[h * HEAD_DIM : (h + 1) * HEAD_DIM] = head_out

        # Output projection + residual
        attn_out = weights['wo'] @ concat_heads
        x = x + attn_out

        # FFN (SwiGLU) + residual
        normed2 = rms_norm(x, weights['ffn_norm'])
        gate = weights['w_gate'] @ normed2
        up = weights['w_up'] @ normed2
        hidden = silu(gate) * up
        ffn_out = weights['w_down'] @ hidden
        x = x + ffn_out

        outputs.append(x)

    return outputs


def compute_blk1_kv(blk0_output, weights):
    """blk.0 출력에서 blk.1의 K,V 계산"""
    normed = rms_norm(blk0_output, weights['attn_norm1'])
    k = weights['wk1'] @ normed + weights['bk1']
    v = weights['wv1'] @ normed + weights['bv1']
    return k.astype(np.float32), v.astype(np.float32)


def save_hashmap(entries, output_path, kv_dim=KV_DIM):
    """KVH0 포맷 저장"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<i", kv_dim))
        f.write(struct.pack("<i", len(entries)))
        for key, k_vec, v_vec in entries:
            f.write(struct.pack("<i", key))
            f.write(k_vec.tobytes())
            f.write(v_vec.tobytes())
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[save] {output_path}: {len(entries)} entries, {size_mb:.1f} MB")


def do_single_token(args):
    """
    단일 토큰(pos=0) 모드: 전체 vocab에 대해 blk.0 forward → blk.1 K,V 계산
    pos=0이므로 RoPE=identity, attention=trivial → 정확한 결과
    """
    weights = load_blk0_all_weights()
    n_vocab = weights['n_vocab']

    # 범위 결정
    if args.range:
        start, end = map(int, args.range.split(","))
    else:
        start, end = 0, min(n_vocab, 10000)  # 기본 10K 토큰

    print(f"\n[forward] blk.0 single-token forward: tokens [{start}, {end})")
    print(f"  Position 0 → RoPE=identity, attention=trivial")

    entries = []
    debug_lines = []

    for tid in range(start, end):
        if tid >= n_vocab:
            break

        # blk.0 forward
        blk0_out = forward_blk0_single_token(tid, weights)

        # blk.1 K,V 계산
        k1, v1 = compute_blk1_kv(blk0_out, weights)
        entries.append((tid, k1, v1))

        # 디버깅: 처음 10개
        if tid - start < 10:
            # blk.0 출력 vs 임베딩 비교
            emb_norm = np.linalg.norm(weights['emb'][tid])
            out_norm = np.linalg.norm(blk0_out)
            diff_norm = np.linalg.norm(blk0_out - weights['emb'][tid])
            debug_lines.append(
                f"token={tid:6d}  |emb|={emb_norm:.4f}  |blk0_out|={out_norm:.4f}  "
                f"|diff|={diff_norm:.4f}  |K1|={np.linalg.norm(k1):.4f}  "
                f"|V1|={np.linalg.norm(v1):.6f}"
            )

        if (tid - start + 1) % 2000 == 0:
            print(f"  [{tid - start + 1}/{end - start}]")

    # 저장
    output_path = args.output or os.path.join(WEIGHTS_DIR, "blk1_kv_single.bin")
    save_hashmap(entries, output_path)

    # 디버그 파일
    debug_path = os.path.join(WEIGHTS_DIR, "blk1_forward_debug.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("=== blk.1 K,V via blk.0 Forward (single token, pos=0) ===\n")
        f.write(f"n_entries: {len(entries)}\n")
        f.write(f"Method: exact (pos=0, RoPE=identity, trivial attention)\n\n")

        f.write("--- Sample tokens ---\n")
        for line in debug_lines:
            f.write(line + "\n")

        # 통계
        if entries:
            all_k = np.array([e[1] for e in entries])
            all_v = np.array([e[2] for e in entries])
            f.write(f"\n--- blk.1 K statistics ---\n")
            f.write(f"|K1| mean={np.linalg.norm(all_k, axis=1).mean():.4f}\n")
            f.write(f"K1 mean={all_k.mean():.6f} std={all_k.std():.6f}\n")
            f.write(f"\n--- blk.1 V statistics ---\n")
            f.write(f"|V1| mean={np.linalg.norm(all_v, axis=1).mean():.6f}\n")
            f.write(f"V1 mean={all_v.mean():.6f} std={all_v.std():.6f}\n")

            # blk.0 K,V와 비교 (임베딩 기반)
            f.write(f"\n--- Comparison: blk.1 exact vs embedding approximation ---\n")
            # 근사값 계산 (기존 build_kv_stack.py 방식)
            for tid in range(start, min(start + 5, end)):
                # 근사: blk.1 input ≈ embedding
                normed_approx = rms_norm(weights['emb'][tid], weights['attn_norm1'])
                k_approx = weights['wk1'] @ normed_approx + weights['bk1']
                v_approx = weights['wv1'] @ normed_approx + weights['bv1']

                k_exact = entries[tid - start][1]
                v_exact = entries[tid - start][2]

                k_diff = np.linalg.norm(k_exact - k_approx)
                v_diff = np.linalg.norm(v_exact - v_approx)
                f.write(f"  token={tid}: K_diff={k_diff:.4f}, V_diff={v_diff:.6f}\n")

    print(f"[debug] {debug_path}")
    print(f"\n[done] {len(entries)} blk.1 K,V entries (exact, pos=0)")


def do_sequence(args):
    """
    시퀀스 모드: 특정 토큰 시퀀스에 대해 full causal attention으로 blk.0 forward
    → 각 position의 blk.1 K,V 계산 (context-dependent)
    """
    weights = load_blk0_all_weights()

    # 토큰 시퀀스 파싱
    if args.tokens:
        token_ids = [int(t) for t in args.tokens.split(",")]
    else:
        # 기본 테스트 시퀀스: "hello world" 대응 (임의 토큰)
        token_ids = [9707, 1879, 374, 264, 1273]  # 예시

    print(f"\n[forward] Sequence mode: {len(token_ids)} tokens")
    print(f"  tokens: {token_ids}")

    # Full causal forward
    outputs = forward_blk0_sequence(token_ids, weights)

    # 각 position에서 blk.1 K,V 계산
    print(f"\n[compute] blk.1 K,V for each position:")
    results = []
    for pos, (tid, blk0_out) in enumerate(zip(token_ids, outputs)):
        k1, v1 = compute_blk1_kv(blk0_out, weights)
        results.append((pos, tid, k1, v1, blk0_out))

        # 근사값과 비교
        normed_approx = rms_norm(weights['emb'][tid], weights['attn_norm1'])
        k_approx = weights['wk1'] @ normed_approx + weights['bk1']
        v_approx = weights['wv1'] @ normed_approx + weights['bv1']

        k_diff = np.linalg.norm(k1 - k_approx)
        v_diff = np.linalg.norm(v1 - v_approx)

        # pos=0의 single-token 결과와도 비교
        blk0_single = forward_blk0_single_token(tid, weights)
        k_single, v_single = compute_blk1_kv(blk0_single, weights)
        k_ctx_diff = np.linalg.norm(k1 - k_single)
        v_ctx_diff = np.linalg.norm(v1 - v_single)

        print(f"  pos={pos} token={tid:6d}  "
              f"|K1|={np.linalg.norm(k1):.4f}  |V1|={np.linalg.norm(v1):.6f}  "
              f"approx_diff(K={k_diff:.4f}, V={v_diff:.6f})  "
              f"ctx_diff(K={k_ctx_diff:.4f}, V={v_ctx_diff:.6f})")

    # 디버그 파일 저장
    debug_path = os.path.join(WEIGHTS_DIR, "blk1_sequence_debug.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("=== blk.1 K,V via Full Causal Forward (sequence) ===\n")
        f.write(f"tokens: {token_ids}\n")
        f.write(f"seq_len: {len(token_ids)}\n\n")

        for pos, tid, k1, v1, blk0_out in results:
            f.write(f"--- Position {pos}, token_id={tid} ---\n")
            f.write(f"  blk0_out[:5] = {blk0_out[:5]}\n")
            f.write(f"  |blk0_out| = {np.linalg.norm(blk0_out):.4f}\n")
            f.write(f"  K1[:5] = {k1[:5]}\n")
            f.write(f"  V1[:5] = {v1[:5]}\n")
            f.write(f"  |K1| = {np.linalg.norm(k1):.4f}\n")
            f.write(f"  |V1| = {np.linalg.norm(v1):.6f}\n\n")

    print(f"\n[debug] {debug_path}")
    print(f"\n[핵심 발견] pos=0은 single-token과 동일, pos>0은 context로 인해 달라짐")
    print(f"  → 이것이 blk.1 hashmap을 context_hash 기반으로 만들어야 하는 이유")


def do_verify(args):
    """
    검증: blk.0 forward 결과가 llama.cpp와 일치하는지 확인
    (TODO: llama.cpp에서 hidden state dump 추가 후 비교)
    """
    weights = load_blk0_all_weights()

    # 간단한 self-consistency 검증
    print("[verify] Self-consistency check...")

    # 1) pos=0에서 RoPE가 identity인지 확인
    x_test = np.random.randn(HEAD_DIM).astype(np.float32)
    x_roped = rope_embed(x_test, pos=0)
    rope_diff = np.max(np.abs(x_test - x_roped))
    print(f"  RoPE(pos=0) diff: {rope_diff:.10f} {'OK' if rope_diff < 1e-6 else 'FAIL'}")

    # 2) 단일 토큰 forward → embedding과 다른지 (attention + FFN 효과 확인)
    tid = 100
    blk0_out = forward_blk0_single_token(tid, weights)
    emb_diff = np.linalg.norm(blk0_out - weights['emb'][tid])
    print(f"  blk0_out vs emb[{tid}] diff: {emb_diff:.4f} "
          f"{'OK (diverged)' if emb_diff > 0.01 else 'WARNING (too similar)'}")

    # 3) blk.1 K,V — exact vs approximation 비교
    k1_exact, v1_exact = compute_blk1_kv(blk0_out, weights)
    normed_approx = rms_norm(weights['emb'][tid], weights['attn_norm1'])
    k1_approx = weights['wk1'] @ normed_approx + weights['bk1']
    v1_approx = weights['wv1'] @ normed_approx + weights['bv1']

    k_diff = np.linalg.norm(k1_exact - k1_approx)
    v_diff = np.linalg.norm(v1_exact - v1_approx)
    print(f"  blk.1 K exact vs approx: {k_diff:.4f}")
    print(f"  blk.1 V exact vs approx: {v_diff:.6f}")
    print(f"\n[결론] embedding 근사와의 차이 = attention + FFN 효과")
    print(f"  이 차이가 클수록, 정확한 blk.1 K,V hashmap의 가치가 높음")


def main():
    parser = argparse.ArgumentParser(description="blk.1 Forward Capture")
    parser.add_argument("--mode", choices=["single", "sequence", "verify"],
                        default="verify", help="동작 모드")
    parser.add_argument("--tokens", type=str, help="토큰 시퀀스 (comma-separated)")
    parser.add_argument("--range", type=str, help="토큰 범위: start,end")
    parser.add_argument("--output", "-o", type=str, help="출력 hashmap 경로")
    args = parser.parse_args()

    if args.mode == "single":
        do_single_token(args)
    elif args.mode == "sequence":
        do_sequence(args)
    elif args.mode == "verify":
        do_verify(args)


if __name__ == "__main__":
    main()
