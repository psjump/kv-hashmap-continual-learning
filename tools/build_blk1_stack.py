#!/usr/bin/env python3
"""
blk.1 Exact Hashmap Builder + V 수정 실험

blk.0 full forward → blk.1 정확한 K,V 계산 → MKVS 스택 (blk.0 + blk.1)
+ blk.1 V 수정을 통한 지식 주입 실험 (|V1|=3.37, blk.0의 10배)

Usage:
  # 1) 특정 시퀀스의 blk.1 hashmap 빌드
  python tools/build_blk1_stack.py --mode build --tokens "151643,32,4616,25039,983"

  # 2) V 수정 실험 (extreme random)
  python tools/build_blk1_stack.py --mode inject --tokens "151643,32,4616,25039,983" --target 2 --scale 10

  # 3) 전체 데모: build + modify + save MKVS
  python tools/build_blk1_stack.py --mode demo
"""

import argparse
import struct
import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GGUF_PATH = os.path.join(MODELS_DIR, "qwen2-0_5b-instruct-q4_k_m.gguf")

RMS_EPS = 1e-6
N_EMBD = 896
KV_DIM = 128
N_HEAD = 14
N_KV_HEAD = 2
HEAD_DIM = 64
FFN_DIM = 4864
ROPE_THETA = 1000000.0
CTX_SIZE = 5  # context hash window

MAGIC_KVH0 = b"KVH0"
MAGIC_MKVS = 0x53564B4D
VERSION = 1


def rms_norm(x, gamma, eps=RMS_EPS):
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * gamma


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -88, 88))))


def rope_embed(x, pos, head_dim=HEAD_DIM, theta=ROPE_THETA):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    angles = pos * freqs
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * cos_a - x[1::2] * sin_a
    out[1::2] = x[0::2] * sin_a + x[1::2] * cos_a
    return out


def context_hash(tokens, length):
    """C kv_context_hash() 동일: h = len; for(i) h = h*31 + tokens[i]; return h & 0x7FFFFFFF"""
    h = length
    for i in range(length):
        h = (h * 31 + (int(tokens[i]) & 0xFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return int(h & 0x7FFFFFFF)


def load_gguf_tensor(tensor_name):
    from gguf import GGUFReader, GGMLQuantizationType, dequantize
    reader = GGUFReader(GGUF_PATH)
    for t in reader.tensors:
        if t.name == tensor_name:
            qt = GGMLQuantizationType(t.tensor_type)
            return dequantize(t.data, qt)
    raise RuntimeError(f"Tensor '{tensor_name}' not found")


def load_all_weights():
    """blk.0 forward + blk.1 K,V 계산에 필요한 전체 가중치"""
    print("[load] Loading weights (blk.0 full + blk.1 KV)...")

    emb_raw = np.load(os.path.join(WEIGHTS_DIR, "token_embd.weight.npy"))
    n_vocab = emb_raw.size // N_EMBD
    emb = emb_raw.reshape(n_vocab, N_EMBD).astype(np.float32)

    # blk.0 attention
    attn_norm0 = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_norm.weight.npy")).astype(np.float32)
    wq0 = load_gguf_tensor("blk.0.attn_q.weight").astype(np.float32)
    bq0 = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_q.bias.npy")).astype(np.float32)
    wk0 = load_gguf_tensor("blk.0.attn_k.weight").astype(np.float32)
    bk0 = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_k.bias.npy")).astype(np.float32)
    wv0 = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.weight.npy")).reshape(KV_DIM, N_EMBD).astype(np.float32)
    bv0 = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.bias.npy")).astype(np.float32)
    wo0 = load_gguf_tensor("blk.0.attn_output.weight").astype(np.float32)

    # blk.0 FFN
    ffn_norm0 = np.load(os.path.join(WEIGHTS_DIR, "blk.0.ffn_norm.weight.npy")).astype(np.float32)
    w_gate0 = load_gguf_tensor("blk.0.ffn_gate.weight").astype(np.float32)
    w_up0 = load_gguf_tensor("blk.0.ffn_up.weight").astype(np.float32)
    w_down0 = load_gguf_tensor("blk.0.ffn_down.weight").astype(np.float32)

    # blk.1 K,V
    attn_norm1 = np.load(os.path.join(WEIGHTS_DIR, "blk.1.attn_norm.weight.npy")).astype(np.float32)
    wk1 = load_gguf_tensor("blk.1.attn_k.weight").astype(np.float32)
    bk1 = np.load(os.path.join(WEIGHTS_DIR, "blk.1.attn_k.bias.npy")).astype(np.float32)
    wv1 = np.load(os.path.join(WEIGHTS_DIR, "blk.1.attn_v.weight.npy")).reshape(KV_DIM, N_EMBD).astype(np.float32)
    bv1 = np.load(os.path.join(WEIGHTS_DIR, "blk.1.attn_v.bias.npy")).astype(np.float32)

    print(f"  emb: ({n_vocab}, {N_EMBD})")
    print(f"  blk.0: Wq{wq0.shape} Wk{wk0.shape} Wo{wo0.shape} FFN({w_gate0.shape[0]})")
    print(f"  blk.1: Wk{wk1.shape} Wv{wv1.shape}")
    print("[load] Done.")

    return {
        'emb': emb, 'n_vocab': n_vocab,
        # blk.0 attn
        'attn_norm0': attn_norm0,
        'wq0': wq0, 'bq0': bq0, 'wk0': wk0, 'bk0': bk0,
        'wv0': wv0, 'bv0': bv0, 'wo0': wo0,
        # blk.0 FFN
        'ffn_norm0': ffn_norm0,
        'w_gate0': w_gate0, 'w_up0': w_up0, 'w_down0': w_down0,
        # blk.1 KV
        'attn_norm1': attn_norm1,
        'wk1': wk1, 'bk1': bk1, 'wv1': wv1, 'bv1': bv1,
    }


def forward_blk0_sequence(token_ids, W):
    """Full causal blk.0 forward → 각 position의 출력 벡터 반환"""
    emb = W['emb']
    seq_len = len(token_ids)
    X = np.array([emb[tid] for tid in token_ids], dtype=np.float32)

    # Pre-compute Q, K, V for all positions
    normed_all = np.array([rms_norm(X[i], W['attn_norm0']) for i in range(seq_len)])
    Q_all = (normed_all @ W['wq0'].T) + W['bq0']
    K_all = (normed_all @ W['wk0'].T) + W['bk0']
    V_all = (normed_all @ W['wv0'].T) + W['bv0']

    # Apply RoPE
    for pos in range(seq_len):
        q_heads = Q_all[pos].reshape(N_HEAD, HEAD_DIM)
        for h in range(N_HEAD):
            q_heads[h] = rope_embed(q_heads[h], pos)
        Q_all[pos] = q_heads.reshape(-1)

        k_heads = K_all[pos].reshape(N_KV_HEAD, HEAD_DIM)
        for h in range(N_KV_HEAD):
            k_heads[h] = rope_embed(k_heads[h], pos)
        K_all[pos] = k_heads.reshape(-1)

    # Causal self-attention + FFN per position
    outputs = []
    for pos in range(seq_len):
        x = X[pos].copy()

        # GQA attention
        q_heads = Q_all[pos].reshape(N_HEAD, HEAD_DIM)
        k_avail = K_all[:pos+1].reshape(pos+1, N_KV_HEAD, HEAD_DIM)
        v_avail = V_all[:pos+1].reshape(pos+1, N_KV_HEAD, HEAD_DIM)
        heads_per_kv = N_HEAD // N_KV_HEAD

        concat_heads = np.zeros(N_EMBD, dtype=np.float32)
        for h in range(N_HEAD):
            kv_idx = h // heads_per_kv
            q_h = q_heads[h]
            k_h = k_avail[:, kv_idx, :]
            v_h = v_avail[:, kv_idx, :]
            scores = (k_h @ q_h) / np.sqrt(HEAD_DIM)
            scores_max = scores.max()
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / exp_scores.sum()
            concat_heads[h*HEAD_DIM:(h+1)*HEAD_DIM] = attn_weights @ v_h

        attn_out = W['wo0'] @ concat_heads
        x = x + attn_out

        # FFN (SwiGLU)
        normed2 = rms_norm(x, W['ffn_norm0'])
        gate = W['w_gate0'] @ normed2
        up = W['w_up0'] @ normed2
        ffn_out = W['w_down0'] @ (silu(gate) * up)
        x = x + ffn_out

        outputs.append(x)

    return outputs


def compute_blk1_kv_for_sequence(token_ids, W):
    """
    시퀀스에 대해 blk.0 forward → blk.1 K,V 계산 + context_hash 키 생성
    Returns: list of (context_hash_key, K1, V1) for each position
    """
    blk0_outputs = forward_blk0_sequence(token_ids, W)

    entries = []
    for pos, blk0_out in enumerate(blk0_outputs):
        # blk.1 K,V
        normed = rms_norm(blk0_out, W['attn_norm1'])
        k1 = (W['wk1'] @ normed + W['bk1']).astype(np.float32)
        v1 = (W['wv1'] @ normed + W['bv1']).astype(np.float32)

        # context_hash (C 코드와 동일)
        ctx_len = min(pos + 1, CTX_SIZE)
        start = pos - ctx_len + 1
        ctx_tokens = token_ids[start:pos+1]
        key = context_hash(ctx_tokens, len(ctx_tokens))

        entries.append((key, k1, v1, pos))

    return entries


def save_mkvs(blk0_entries, blk1_entries, output_path):
    """MKVS (2-layer) 저장"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<i", MAGIC_MKVS))
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<i", KV_DIM))
        f.write(struct.pack("<i", 2))        # n_layers
        f.write(struct.pack("<i", CTX_SIZE))  # ctx_size

        # Layer 0 (blk.0)
        f.write(struct.pack("<i", len(blk0_entries)))
        for tid, k, v in blk0_entries:
            f.write(struct.pack("<i", tid))
            f.write(k.tobytes())
            f.write(v.tobytes())

        # Layer 1 (blk.1)
        f.write(struct.pack("<i", len(blk1_entries)))
        for key, k, v in blk1_entries:
            f.write(struct.pack("<i", key))
            f.write(k.tobytes())
            f.write(v.tobytes())

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[save] {output_path}: L0={len(blk0_entries)}, L1={len(blk1_entries)}, {size_mb:.1f} MB")


def load_blk0_hashmap(path):
    """blk.0 KVH0 로드"""
    entries = []
    with open(path, "rb") as f:
        magic = f.read(4)
        version = struct.unpack("<i", f.read(4))[0]
        kv_dim = struct.unpack("<i", f.read(4))[0]
        n = struct.unpack("<i", f.read(4))[0]
        for _ in range(n):
            tid = struct.unpack("<i", f.read(4))[0]
            k = np.frombuffer(f.read(kv_dim * 4), dtype=np.float32).copy()
            v = np.frombuffer(f.read(kv_dim * 4), dtype=np.float32).copy()
            entries.append((tid, k, v))
    return entries


def get_test_sequences():
    """실험용 시퀀스 (실제 토큰 ID, Qwen2 BPE)

    참고: slot_fusion은 add_bos=true이므로 BOS(151643)가 앞에 붙음.
    하지만 BOS는 <|endoftext|>이고 실제로는 추가되지 않는 경우가 많음.
    Qwen2의 add_bos_token=false 설정이므로 BOS 없이 테스트.
    """
    # Qwen2 토큰: 수동 매핑 (vocab에서 확인된 것)
    # "A" = 32, "cat" = 4616, "likes" = 25039, "to" = 983
    # "The" = 785, "is" = 285, "a" = 64
    # "Quantum" = 45778, "physics" = 28699, "explains" = 53324
    # " the" = 279, " behavior" = 7865, " of" = 315

    sequences = {
        "A cat likes to": [32, 4616, 25039, 983],
        "The cat is a": [785, 4616, 285, 64],
        "Quantum physics explains the": [45778, 28699, 53324, 279],
    }
    return sequences


def do_build(args):
    """특정 시퀀스에 대한 blk.1 exact hashmap 빌드"""
    W = load_all_weights()

    if args.tokens:
        token_ids = [int(t) for t in args.tokens.split(",")]
        seq_name = f"custom_{len(token_ids)}tok"
    else:
        sequences = get_test_sequences()
        seq_name = "A cat likes to"
        token_ids = sequences[seq_name]

    print(f"\n[build] Sequence: {seq_name}")
    print(f"  tokens: {token_ids}")

    # blk.0 full forward → blk.1 K,V
    print("[forward] Running blk.0 full causal forward...")
    t0 = time.time()
    blk1_entries = compute_blk1_kv_for_sequence(token_ids, W)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # 결과 출력
    print(f"\n[blk.1 K,V results]")
    for key, k1, v1, pos in blk1_entries:
        print(f"  pos={pos} token={token_ids[pos]:6d} ctx_hash={key:10d} "
              f"|K1|={np.linalg.norm(k1):.2f} |V1|={np.linalg.norm(v1):.4f}")

    # MKVS 저장 (blk.0 full + blk.1 sequence)
    print("\n[load] Loading blk.0 full hashmap...")
    blk0_path = os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin")
    blk0_entries = load_blk0_hashmap(blk0_path)
    print(f"  blk.0: {len(blk0_entries)} entries")

    # blk.1 entries: (key, K, V) 형식으로 변환
    blk1_for_save = [(key, k1, v1) for key, k1, v1, pos in blk1_entries]

    output_path = args.output or os.path.join(WEIGHTS_DIR, "stack_blk01_exact.bin")
    save_mkvs(blk0_entries, blk1_for_save, output_path)


def do_inject(args):
    """blk.1 V 수정 실험"""
    W = load_all_weights()

    if args.tokens:
        token_ids = [int(t) for t in args.tokens.split(",")]
    else:
        token_ids = [32, 4616, 25039, 983]  # "A cat likes to"

    target_pos = args.target if args.target is not None else 1  # 기본: 위치 1 수정
    scale = args.scale or 10.0

    print(f"\n[inject] Sequence: {token_ids}")
    print(f"  Target position: {target_pos} (token={token_ids[target_pos]})")
    print(f"  V scale: {scale}x random")

    # blk.1 K,V 계산
    print("[forward] blk.0 full forward...")
    blk1_entries = compute_blk1_kv_for_sequence(token_ids, W)

    # 원본 V 정보
    _, k1_orig, v1_orig, _ = blk1_entries[target_pos]
    print(f"\n[original] pos={target_pos}: |K1|={np.linalg.norm(k1_orig):.2f} |V1|={np.linalg.norm(v1_orig):.4f}")

    # V 수정: random * scale
    np.random.seed(42)
    v1_modified = np.random.randn(KV_DIM).astype(np.float32) * scale
    blk1_entries[target_pos] = (blk1_entries[target_pos][0], k1_orig, v1_modified, target_pos)
    print(f"[modified] pos={target_pos}: |V1_new|={np.linalg.norm(v1_modified):.4f} (was {np.linalg.norm(v1_orig):.4f})")

    # MKVS 저장
    print("\n[save] Building MKVS stack (blk.0 original + blk.1 modified)...")
    blk0_path = os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin")
    blk0_entries = load_blk0_hashmap(blk0_path)

    blk1_for_save = [(key, k1, v1) for key, k1, v1, pos in blk1_entries]

    output_path = args.output or os.path.join(WEIGHTS_DIR, "stack_blk01_inject.bin")
    save_mkvs(blk0_entries, blk1_for_save, output_path)

    print(f"\n[test] Run inference:")
    print(f'  slot_fusion infer -m models/qwen2-0_5b-instruct-q4_k_m.gguf \\')
    print(f'    -s models/mia_slot.bin -p "A cat likes to" -n 30 \\')
    print(f'    --kv-hashmap {output_path} -ngl 0')


def do_demo(args):
    """전체 데모: 원본 blk.1 vs V-modified blk.1"""
    W = load_all_weights()

    # 여러 시퀀스에 대해 실험
    sequences = get_test_sequences()

    all_blk1_entries = {}  # context_hash → (K, V) 매핑

    print("=" * 60)
    print("  blk.1 V Modification Experiment")
    print("  |V1| = 3.37 (blk.0의 10배) -> 효과 기대!")
    print("=" * 60)

    for seq_name, token_ids in sequences.items():
        print(f"\n--- Sequence: \"{seq_name}\" ---")
        print(f"  tokens: {token_ids}")

        entries = compute_blk1_kv_for_sequence(token_ids, W)
        for key, k1, v1, pos in entries:
            all_blk1_entries[key] = (k1, v1)
            print(f"  pos={pos} token={token_ids[pos]:6d} hash={key:10d} |V1|={np.linalg.norm(v1):.4f}")

    print(f"\n[total] {len(all_blk1_entries)} unique blk.1 entries (context_hash based)")

    # === Version 1: 원본 MKVS ===
    print("\n[save] Building original MKVS...")
    blk0_entries = load_blk0_hashmap(os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin"))
    blk1_list = [(key, k, v) for key, (k, v) in all_blk1_entries.items()]

    orig_path = os.path.join(WEIGHTS_DIR, "stack_blk01_original.bin")
    save_mkvs(blk0_entries, blk1_list, orig_path)

    # === Version 2: blk.1 V를 전부 random×10으로 교체 ===
    print("\n[save] Building EXTREME modified MKVS (all blk.1 V = random*10)...")
    np.random.seed(42)
    blk1_extreme = []
    for key, (k, v) in all_blk1_entries.items():
        v_rand = np.random.randn(KV_DIM).astype(np.float32) * 10.0
        blk1_extreme.append((key, k, v_rand))

    extreme_path = os.path.join(WEIGHTS_DIR, "stack_blk01_extreme.bin")
    save_mkvs(blk0_entries, blk1_extreme, extreme_path)

    # === Version 3: blk.1 V를 0으로 ===
    print("\n[save] Building ZEROED MKVS (all blk.1 V = 0)...")
    blk1_zero = []
    for key, (k, v) in all_blk1_entries.items():
        blk1_zero.append((key, k, np.zeros(KV_DIM, dtype=np.float32)))

    zero_path = os.path.join(WEIGHTS_DIR, "stack_blk01_zero.bin")
    save_mkvs(blk0_entries, blk1_zero, zero_path)

    # === Version 4: blk.1 V 방향 반전 (V * -1) ===
    print("\n[save] Building INVERTED MKVS (all blk.1 V = -V)...")
    blk1_inv = []
    for key, (k, v) in all_blk1_entries.items():
        blk1_inv.append((key, k, -v))

    invert_path = os.path.join(WEIGHTS_DIR, "stack_blk01_inverted.bin")
    save_mkvs(blk0_entries, blk1_inv, invert_path)

    # 실행 가이드
    print("\n" + "=" * 60)
    print("  Inference Commands")
    print("=" * 60)

    base_cmd = ('slot_fusion/build/Release/slot_fusion.exe infer '
                '-m models/qwen2-0_5b-instruct-q4_k_m.gguf '
                '-s models/mia_slot.bin '
                '-p "A cat likes to" -n 30 -ngl 0')

    print(f'\n# 1) NO hashmap (baseline)')
    print(f'  {base_cmd}')
    print(f'\n# 2) Original blk.0+blk.1 (should be ~same as baseline)')
    print(f'  {base_cmd} --kv-hashmap {orig_path}')
    print(f'\n# 3) blk.1 V=random*10 (EXTREME)')
    print(f'  {base_cmd} --kv-hashmap {extreme_path}')
    print(f'\n# 4) blk.1 V=0 (info removal)')
    print(f'  {base_cmd} --kv-hashmap {zero_path}')
    print(f'\n# 5) blk.1 V=-V (direction inversion)')
    print(f'  {base_cmd} --kv-hashmap {invert_path}')


def main():
    parser = argparse.ArgumentParser(description="blk.1 Stack Builder + V Injection")
    parser.add_argument("--mode", choices=["build", "inject", "demo"], default="demo")
    parser.add_argument("--tokens", type=str, help="Token IDs (comma-separated)")
    parser.add_argument("--target", type=int, help="Target position for injection")
    parser.add_argument("--scale", type=float, default=10.0, help="V modification scale")
    parser.add_argument("--output", "-o", type=str, help="Output path")
    args = parser.parse_args()

    if args.mode == "build":
        do_build(args)
    elif args.mode == "inject":
        do_inject(args)
    elif args.mode == "demo":
        do_demo(args)


if __name__ == "__main__":
    main()
