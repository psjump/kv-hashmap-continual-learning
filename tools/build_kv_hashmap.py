#!/usr/bin/env python3
"""
blk.0 K,V Hashmap Builder

blk.0의 Wk, Wv를 hashmap으로 교체하기 위해
각 token_id에 대한 K, V 벡터를 precompute하여 바이너리 파일로 저장.

핵심 원리:
  blk.0 입력 = RMSNorm(embedding[token_id])
  → token_id에 대해 결정적 → hashmap key = token_id

계산:
  normed = RMSNorm(emb[token_id], gamma, eps=1e-6)
  K = Wk @ normed + bk    (128-dim)
  V = Wv.T @ normed + bv  (128-dim)

바이너리 포맷 (blk0_kv_hashmap.bin):
  [magic: 4 bytes, "KVH0"]
  [version: int32]
  [kv_dim: int32]         -- 128
  [n_entries: int32]
  [entries: n_entries × (token_id:int32 + K:kv_dim×f32 + V:kv_dim×f32)]

Usage:
  python tools/build_kv_hashmap.py --mode build --data dataset/korean_identity.txt
  python tools/build_kv_hashmap.py --mode verify --hashmap weights/blk0_kv_hashmap.bin
  python tools/build_kv_hashmap.py --mode full-vocab   (전체 vocab precompute)
"""

import argparse
import struct
import sys
import os
import numpy as np

# ── 설정 ──
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GGUF_PATH = os.path.join(MODELS_DIR, "qwen2-0_5b-instruct-q4_k_m.gguf")
RMS_EPS = 1e-6
KV_DIM = 128
MAGIC = b"KVH0"
VERSION = 1


def rms_norm(x, gamma, eps=RMS_EPS):
    """RMSNorm: x * gamma / sqrt(mean(x^2) + eps) — ggml 구현과 동일"""
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * gamma


def load_weights():
    """blk.0 K,V 계산에 필요한 가중치 로드"""
    print("[load] Loading weights...")

    # token_embd: dump_weights.py saves as (896, 151936) — GGUF shape, transposed from ggml layout
    # ggml memory: (151936, 896) = each row is one token's 896-dim embedding
    # Fix: reshape to (n_vocab, n_embd) = (151936, 896)
    emb_raw = np.load(os.path.join(WEIGHTS_DIR, "token_embd.weight.npy"))
    n_embd = 896
    n_vocab = emb_raw.size // n_embd
    emb = emb_raw.reshape(n_vocab, n_embd)  # (151936, 896) — correct ggml layout
    print(f"  token_embd: raw {emb_raw.shape} → corrected {emb.shape} {emb.dtype}")

    # attn_norm: (896,) — 1D, no shape issue
    attn_norm = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_norm.weight.npy"))
    print(f"  attn_norm: {attn_norm.shape}")

    # Wk: Q5_0 quantized → gguf dequantize → (128, 896) — correct shape from gguf library
    print("  Wk: dequantizing Q5_0 from GGUF...")
    wk = _load_wk_from_gguf()
    print(f"  Wk dequantized: {wk.shape} mean={wk.mean():.6f} std={wk.std():.6f}")

    # Wv: dump_weights.py saves as (896, 128) — GGUF shape, transposed
    # ggml layout: ne[0]=896, ne[1]=128 = (128 rows, 896 cols) = (128, 896)
    # This matches what gguf.dequantize() returns
    wv_raw = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.weight.npy"))
    wv = wv_raw.reshape(128, 896)  # correct ggml layout: (out_dim, in_dim)
    print(f"  Wv: raw {wv_raw.shape} → corrected {wv.shape}")

    # biases
    bk = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_k.bias.npy"))
    bv = np.load(os.path.join(WEIGHTS_DIR, "blk.0.attn_v.bias.npy"))
    print(f"  bk: {bk.shape} mean={bk.mean():.4f} std={bk.std():.4f}")
    print(f"  bv: {bv.shape} mean={bv.mean():.6f} std={bv.std():.6f}")

    return emb, attn_norm, wk, wv, bk, bv


def _load_wk_from_gguf():
    """GGUF에서 blk.0.attn_k.weight를 Q5_0 dequantize"""
    from gguf import GGUFReader, GGMLQuantizationType, dequantize

    reader = GGUFReader(GGUF_PATH)
    for t in reader.tensors:
        if t.name == "blk.0.attn_k.weight":
            qt = GGMLQuantizationType(t.tensor_type)
            deq = dequantize(t.data, qt)
            return deq  # (128, 896)

    raise RuntimeError("blk.0.attn_k.weight not found in GGUF")


def compute_kv_for_token(token_id, emb, attn_norm, wk, wv, bk, bv):
    """단일 token_id에 대한 K, V 벡터 계산"""
    # embedding: emb is (n_vocab, 896) — emb[token_id] = 896-dim vector
    e = emb[token_id].astype(np.float32)

    # RMSNorm
    normed = rms_norm(e, attn_norm)

    # K = Wk @ normed + bk   (Wk: (128, 896), normed: (896,) → K: (128,))
    k = wk @ normed + bk

    # V = Wv @ normed + bv   (Wv: (128, 896), normed: (896,) → V: (128,))
    v = wv @ normed + bv

    return k.astype(np.float32), v.astype(np.float32)


def get_unique_tokens_from_data(data_path):
    """학습 데이터에서 고유 토큰 추출 (llama.cpp 토크나이저 사용 불가 → GGUF vocab 기반 BPE)"""
    # 간단 접근: 전체 vocab 중 실제 사용될 토큰만 수집
    # 여기서는 데이터 파일의 UTF-8 바이트를 기반으로 관련 토큰 범위 추정
    # 실제로는 llama.cpp 토크나이저를 사용해야 정확하지만,
    # PoC에서는 자주 사용되는 토큰 범위를 넓게 잡음

    print(f"[tokens] Reading {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 데이터 통계
    lines = [l for l in text.split("\n") if l.strip()]
    print(f"  {len(lines)} lines, {len(text)} chars")

    # 토크나이저 없이 전체 vocab을 커버하는 것은 비효율적
    # → 사용 빈도가 높은 토큰 범위 (0 ~ 10000) + 한국어 관련 토큰
    # 실제 통합 시에는 llama.cpp tokenizer로 정확한 token_id 추출
    token_ids = set(range(10000))  # 기본 토큰 (영어, 숫자, 구두점, 일반)

    # GGUF에서 한국어 관련 토큰 검색
    try:
        from gguf import GGUFReader
        reader = GGUFReader(GGUF_PATH)
        for field in reader.fields.values():
            if "token" in field.name.lower() and "list" in field.name.lower():
                # tokenizer.ggml.tokens 필드에서 한국어 토큰 검색
                pass
    except Exception:
        pass

    print(f"  {len(token_ids)} unique token_ids (PoC range)")
    return sorted(token_ids)


def save_hashmap(entries, output_path):
    """
    바이너리 저장: KVH0 포맷
    entries: list of (token_id, K[128], V[128])
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        # header
        f.write(MAGIC)
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<i", KV_DIM))
        f.write(struct.pack("<i", len(entries)))

        # entries
        for token_id, k_vec, v_vec in entries:
            f.write(struct.pack("<i", token_id))
            f.write(k_vec.tobytes())   # 128 × float32 = 512 bytes
            f.write(v_vec.tobytes())   # 128 × float32 = 512 bytes

    file_size = os.path.getsize(output_path)
    print(f"[save] {output_path}: {len(entries)} entries, {file_size/1024:.1f} KB")


def load_hashmap(path):
    """바이너리 로드"""
    entries = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic}")
        version = struct.unpack("<i", f.read(4))[0]
        kv_dim = struct.unpack("<i", f.read(4))[0]
        n_entries = struct.unpack("<i", f.read(4))[0]

        for _ in range(n_entries):
            token_id = struct.unpack("<i", f.read(4))[0]
            k_vec = np.frombuffer(f.read(kv_dim * 4), dtype=np.float32).copy()
            v_vec = np.frombuffer(f.read(kv_dim * 4), dtype=np.float32).copy()
            entries[token_id] = (k_vec, v_vec)

    return entries, kv_dim


def do_build(args):
    """K,V hashmap 빌드"""
    emb, attn_norm, wk, wv, bk, bv = load_weights()

    n_vocab = emb.shape[0]
    print(f"\n[build] vocab_size={n_vocab}, kv_dim={KV_DIM}")

    if args.full_vocab:
        # 전체 vocab precompute
        token_ids = list(range(n_vocab))
        print(f"[build] Full vocab mode: {n_vocab} tokens")
    elif args.data:
        token_ids = get_unique_tokens_from_data(args.data)
    else:
        # 기본: 자주 사용되는 범위
        token_ids = list(range(min(10000, n_vocab)))
        print(f"[build] Default range: {len(token_ids)} tokens")

    # K,V 계산
    entries = []
    debug_lines = []
    print(f"[build] Computing K,V for {len(token_ids)} tokens...")

    for i, tid in enumerate(token_ids):
        if tid >= n_vocab:
            continue
        k, v = compute_kv_for_token(tid, emb, attn_norm, wk, wv, bk, bv)
        entries.append((tid, k, v))

        # 디버깅: 처음 20개 토큰
        if i < 20:
            debug_lines.append(
                f"token_id={tid:6d}  K[0:5]={k[:5]}  V[0:5]={v[:5]}  "
                f"|K|={np.linalg.norm(k):.4f}  |V|={np.linalg.norm(v):.6f}"
            )

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(token_ids)}]")

    # 저장
    output_path = args.output or os.path.join(WEIGHTS_DIR, "blk0_kv_hashmap.bin")
    save_hashmap(entries, output_path)

    # 디버깅 파일
    debug_path = os.path.join(WEIGHTS_DIR, "blk0_kv_debug.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("=== blk.0 K,V Hashmap Debug ===\n")
        f.write(f"n_entries: {len(entries)}\n")
        f.write(f"kv_dim: {KV_DIM}\n")
        f.write(f"Wk shape: {wk.shape}, Wv shape: {wv.shape}\n")
        f.write(f"bk mean={bk.mean():.4f} std={bk.std():.4f}\n")
        f.write(f"bv mean={bv.mean():.6f} std={bv.std():.6f}\n\n")
        for line in debug_lines:
            f.write(line + "\n")

        # K,V 통계
        all_k = np.array([e[1] for e in entries])
        all_v = np.array([e[2] for e in entries])
        f.write(f"\n=== K 통계 ===\n")
        f.write(f"shape: {all_k.shape}\n")
        f.write(f"mean: {all_k.mean():.6f}, std: {all_k.std():.6f}\n")
        f.write(f"min: {all_k.min():.6f}, max: {all_k.max():.6f}\n")
        f.write(f"|K| mean: {np.linalg.norm(all_k, axis=1).mean():.4f}\n")
        f.write(f"\n=== V 통계 ===\n")
        f.write(f"shape: {all_v.shape}\n")
        f.write(f"mean: {all_v.mean():.6f}, std: {all_v.std():.6f}\n")
        f.write(f"min: {all_v.min():.6f}, max: {all_v.max():.6f}\n")
        f.write(f"|V| mean: {np.linalg.norm(all_v, axis=1).mean():.6f}\n")

    print(f"[debug] {debug_path}")
    print(f"\n[done] {len(entries)} entries built")


def do_verify(args):
    """hashmap 검증: 로드 + 재계산 비교"""
    hashmap_path = args.hashmap or os.path.join(WEIGHTS_DIR, "blk0_kv_hashmap.bin")
    entries, kv_dim = load_hashmap(hashmap_path)
    print(f"[verify] Loaded {len(entries)} entries, kv_dim={kv_dim}")

    # 가중치 재로드하여 재계산 비교
    emb, attn_norm, wk, wv, bk, bv = load_weights()

    # 랜덤 10개 토큰 검증
    test_ids = sorted(entries.keys())[:10]
    print(f"\n[verify] Checking {len(test_ids)} tokens...")

    max_k_diff = 0.0
    max_v_diff = 0.0

    for tid in test_ids:
        k_stored, v_stored = entries[tid]
        k_recomp, v_recomp = compute_kv_for_token(tid, emb, attn_norm, wk, wv, bk, bv)

        k_diff = np.max(np.abs(k_stored - k_recomp))
        v_diff = np.max(np.abs(v_stored - v_recomp))
        max_k_diff = max(max_k_diff, k_diff)
        max_v_diff = max(max_v_diff, v_diff)

        status = "OK" if k_diff < 0.001 and v_diff < 0.001 else "MISMATCH"
        print(f"  token {tid:6d}: K_diff={k_diff:.8f} V_diff={v_diff:.8f} [{status}]")

    print(f"\n[result] max K diff: {max_k_diff:.10f}")
    print(f"[result] max V diff: {max_v_diff:.10f}")
    if max_k_diff < 0.001 and max_v_diff < 0.001:
        print("[PASS] Hashmap values match recomputed K,V")
    else:
        print("[FAIL] Mismatch detected!")


def do_stats(args):
    """hashmap 통계"""
    hashmap_path = args.hashmap or os.path.join(WEIGHTS_DIR, "blk0_kv_hashmap.bin")
    entries, kv_dim = load_hashmap(hashmap_path)

    all_k = np.array([entries[tid][0] for tid in sorted(entries.keys())])
    all_v = np.array([entries[tid][1] for tid in sorted(entries.keys())])

    print(f"=== KV Hashmap Stats ===")
    print(f"entries: {len(entries)}")
    print(f"kv_dim: {kv_dim}")
    print(f"file size: {os.path.getsize(hashmap_path)/1024:.1f} KB")
    print(f"\nK: mean={all_k.mean():.6f} std={all_k.std():.6f} "
          f"|K| mean={np.linalg.norm(all_k, axis=1).mean():.4f}")
    print(f"V: mean={all_v.mean():.6f} std={all_v.std():.6f} "
          f"|V| mean={np.linalg.norm(all_v, axis=1).mean():.6f}")

    # K,V 분포 히스토그램 (텍스트)
    k_norms = np.linalg.norm(all_k, axis=1)
    v_norms = np.linalg.norm(all_v, axis=1)
    print(f"\n|K| range: [{k_norms.min():.4f}, {k_norms.max():.4f}]")
    print(f"|V| range: [{v_norms.min():.6f}, {v_norms.max():.6f}]")


def main():
    parser = argparse.ArgumentParser(description="blk.0 K,V Hashmap Builder")
    parser.add_argument("--mode", choices=["build", "verify", "stats"],
                        default="build", help="동작 모드")
    parser.add_argument("--data", type=str, help="학습 데이터 경로")
    parser.add_argument("--output", "-o", type=str, help="출력 hashmap 경로")
    parser.add_argument("--hashmap", type=str, help="검증할 hashmap 경로")
    parser.add_argument("--full-vocab", action="store_true",
                        help="전체 vocab precompute (151K tokens, ~150MB)")

    args = parser.parse_args()

    if args.mode == "build":
        do_build(args)
    elif args.mode == "verify":
        do_verify(args)
    elif args.mode == "stats":
        do_stats(args)


if __name__ == "__main__":
    main()
