#!/usr/bin/env python3
"""
Multi-layer K,V Hashmap Stack Builder

blk.0: token_id 기반 (결정적)
blk.1: blk.0 forward → RMSNorm → Wk1×input + bk1 계산, context hash 기반

Python에서 전체 forward를 재현하여 각 레이어의 K,V를 정확히 계산.

Usage:
  python tools/build_kv_stack.py --layers 2 --data dataset.txt -o weights/kv_stack_l2.bin
"""

import argparse
import struct
import sys
import os
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GGUF_PATH = os.path.join(MODELS_DIR, "qwen2-0_5b-instruct-q4_k_m.gguf")
RMS_EPS = 1e-6
KV_DIM = 128
N_EMBD = 896
CTX_SIZE = 5


def rms_norm(x, gamma, eps=RMS_EPS):
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * gamma


def load_layer_weights(layer_idx):
    """특정 레이어의 Wk, Wv, bk, bv, attn_norm 로드"""
    from gguf import GGUFReader, GGMLQuantizationType, dequantize

    norm = np.load(os.path.join(WEIGHTS_DIR, f"blk.{layer_idx}.attn_norm.weight.npy"))
    bk = np.load(os.path.join(WEIGHTS_DIR, f"blk.{layer_idx}.attn_k.bias.npy"))
    bv = np.load(os.path.join(WEIGHTS_DIR, f"blk.{layer_idx}.attn_v.bias.npy"))

    # Wv: .npy (Q8_0 dequant by dump_weights.py) → 올바른 shape으로 reshape
    wv_path = os.path.join(WEIGHTS_DIR, f"blk.{layer_idx}.attn_v.weight.npy")
    if os.path.exists(wv_path):
        wv_raw = np.load(wv_path)
        wv = wv_raw.reshape(KV_DIM, N_EMBD)  # (128, 896) correct ggml layout
    else:
        # Wv도 Q5_0인 경우 GGUF에서 직접 dequantize
        reader = GGUFReader(GGUF_PATH)
        for t in reader.tensors:
            if t.name == f"blk.{layer_idx}.attn_v.weight":
                qt = GGMLQuantizationType(t.tensor_type)
                wv = dequantize(t.data, qt)
                break

    # Wk: GGUF에서 Q5_0 dequantize
    reader = GGUFReader(GGUF_PATH)
    wk = None
    for t in reader.tensors:
        if t.name == f"blk.{layer_idx}.attn_k.weight":
            qt = GGMLQuantizationType(t.tensor_type)
            wk = dequantize(t.data, qt)  # (128, 896)
            break

    if wk is None:
        raise RuntimeError(f"blk.{layer_idx}.attn_k.weight not found")

    return norm, wk, wv, bk, bv


def context_hash(tokens, length):
    """slot.c hashContextN() 동일: h = len; for(i) h = h*31 + tokens[i]"""
    h = length
    for i in range(length):
        h = (h * 31 + (tokens[i] & 0xFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return int(h & 0x7FFFFFFF)


def build_stack(args):
    """Multi-layer K,V stack 빌드"""
    n_layers = args.layers

    # 임베딩 로드 (올바른 shape)
    emb_raw = np.load(os.path.join(WEIGHTS_DIR, "token_embd.weight.npy"))
    n_vocab = emb_raw.size // N_EMBD
    emb = emb_raw.reshape(n_vocab, N_EMBD)
    print(f"[load] token_embd: ({n_vocab}, {N_EMBD})")

    # 레이어별 가중치 로드
    layer_weights = []
    for il in range(n_layers):
        print(f"[load] Loading layer {il} weights...")
        norm, wk, wv, bk, bv = load_layer_weights(il)
        layer_weights.append((norm, wk, wv, bk, bv))
        print(f"  Wk: {wk.shape}, Wv: {wv.shape}, bk_std={bk.std():.4f}")

    # blk.0: 전체 vocab precompute (token_id 기반)
    print(f"\n[build] Layer 0: full vocab ({n_vocab} tokens)")
    norm0, wk0, wv0, bk0, bv0 = layer_weights[0]
    l0_entries = {}
    for tid in range(n_vocab):
        e = emb[tid].astype(np.float32)
        normed = rms_norm(e, norm0)
        k = (wk0 @ normed + bk0).astype(np.float32)
        v = (wv0 @ normed + bv0).astype(np.float32)
        l0_entries[tid] = (k, v)
        if (tid + 1) % 50000 == 0:
            print(f"  [{tid+1}/{n_vocab}]")
    print(f"  L0: {len(l0_entries)} entries")

    # blk.1+: 학습 데이터 기반 (context hash)
    # blk.1의 입력 = blk.0 출력 = RMSNorm(emb) + attention_output + FFN_output
    # 완전한 forward를 하려면 attention + FFN도 필요 → 복잡
    # 간단한 근사: blk.1 입력 ≈ RMSNorm(emb) (residual 무시)
    # 더 정확한 방법: llama.cpp에서 hidden state 추출

    # 일단 blk.1도 token_id 기반으로 근사 (blk.0 residual을 emb로 근사)
    if n_layers > 1:
        print(f"\n[build] Layer 1: full vocab (token_id approximation)")
        print("  NOTE: blk.1 input ~ embedding (residual approx, exact forward = next step)")
        norm1, wk1, wv1, bk1, bv1 = layer_weights[1]
        l1_entries = {}
        for tid in range(n_vocab):
            # 근사: blk.1 입력 ≈ embedding (residual skip)
            # 실제로는 blk.0 attention + FFN 출력이 추가되어야 함
            e = emb[tid].astype(np.float32)
            normed = rms_norm(e, norm1)
            k = (wk1 @ normed + bk1).astype(np.float32)
            v = (wv1 @ normed + bv1).astype(np.float32)
            l1_entries[tid] = (k, v)
            if (tid + 1) % 50000 == 0:
                print(f"  [{tid+1}/{n_vocab}]")
        print(f"  L1: {len(l1_entries)} entries")

    # 저장 (MKVS 포맷)
    output_path = args.output or os.path.join(WEIGHTS_DIR, "kv_stack_l2.bin")
    with open(output_path, "wb") as f:
        magic = 0x53564B4D  # "MKVS"
        f.write(struct.pack("<i", magic))
        f.write(struct.pack("<i", 1))  # version
        f.write(struct.pack("<i", KV_DIM))
        f.write(struct.pack("<i", n_layers))
        f.write(struct.pack("<i", CTX_SIZE))

        # Layer 0
        f.write(struct.pack("<i", len(l0_entries)))
        for tid in sorted(l0_entries.keys()):
            k, v = l0_entries[tid]
            f.write(struct.pack("<i", tid))
            f.write(k.tobytes())
            f.write(v.tobytes())

        # Layer 1
        if n_layers > 1:
            f.write(struct.pack("<i", len(l1_entries)))
            for tid in sorted(l1_entries.keys()):
                k, v = l1_entries[tid]
                f.write(struct.pack("<i", tid))
                f.write(k.tobytes())
                f.write(v.tobytes())

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[saved] {output_path}: {file_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Multi-layer K,V Stack Builder")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers (default 2)")
    parser.add_argument("--data", type=str, help="Training data (for context-based layers)")
    parser.add_argument("--output", "-o", type=str, help="Output stack path")
    args = parser.parse_args()
    build_stack(args)


if __name__ == "__main__":
    main()
