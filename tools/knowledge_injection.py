#!/usr/bin/env python3
"""
지식 주입 실험 — blk.0 V 벡터 수정을 통한 출력 변화 관찰

목표:
  blk.0의 V 벡터를 교체/수정하면 모델 출력이 변하는지 관찰.
  이것은 hashmap-layer를 통한 지식 주입의 핵심 전제를 검증함.

실험:
  1. V-swap: "cat" 토큰의 V를 "dog" 토큰의 V로 교체
     → "cat" 입력 시 "dog" 관련 출력이 나오는가?
  2. V-blend: target V를 α만큼 혼합
     → α 크기와 출력 변화의 상관관계
  3. V-zero: 특정 토큰의 V를 0으로 → 정보 제거 효과
  4. V-direction: V 벡터의 방향만 유지하고 크기 변경 → 영향 측정

출력:
  - 수정된 hashmap 바이너리 (slot_fusion으로 추론 가능)
  - 변화 분석 리포트

Usage:
  python tools/knowledge_injection.py --mode swap --src cat --dst dog
  python tools/knowledge_injection.py --mode blend --src cat --dst dog --alpha 0.5
  python tools/knowledge_injection.py --mode zero --tokens "cat,dog,fish"
  python tools/knowledge_injection.py --mode report (전체 분석)
"""

import argparse
import struct
import os
import sys
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
GGUF_PATH = os.path.join(MODELS_DIR, "qwen2-0_5b-instruct-q4_k_m.gguf")

KV_DIM = 128
MAGIC = b"KVH0"
VERSION = 1


def get_vocab_map():
    """GGUF에서 token text → token_id 매핑 생성"""
    from gguf import GGUFReader

    reader = GGUFReader(GGUF_PATH)
    vocab = {}

    # tokens 필드 찾기
    for field_name, field in reader.fields.items():
        if "tokenizer.ggml.tokens" in field_name:
            # token list 추출
            for idx in range(len(field.data)):
                part_idx = field.data[idx]
                token_bytes = bytes(field.parts[part_idx])
                try:
                    token_str = token_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    token_str = token_bytes.hex()
                vocab[token_str] = idx
            break

    print(f"[vocab] {len(vocab)} tokens loaded")
    return vocab


def find_token_id(word, vocab):
    """단어로 token_id 찾기 (BPE prefix 포함)"""
    # Qwen2 BPE: 단어 시작에 Ġ (공백 + 문자) 패턴
    candidates = [
        word,
        f"Ġ{word}",        # 공백 prefix
        f" {word}",        # 실제 공백
        word.lower(),
        f"Ġ{word.lower()}",
    ]

    for c in candidates:
        if c in vocab:
            return vocab[c], c

    # 부분 매칭
    matches = [(k, v) for k, v in vocab.items() if word.lower() in k.lower()]
    if matches:
        matches.sort(key=lambda x: len(x[0]))  # 가장 짧은 매칭
        return matches[0][1], matches[0][0]

    return None, None


def load_hashmap(path):
    """KVH0 바이너리 로드"""
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


def save_hashmap(entries, output_path, kv_dim=KV_DIM):
    """KVH0 저장"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<i", kv_dim))
        f.write(struct.pack("<i", len(entries)))
        for tid in sorted(entries.keys()):
            k_vec, v_vec = entries[tid]
            f.write(struct.pack("<i", tid))
            f.write(k_vec.tobytes())
            f.write(v_vec.tobytes())
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[save] {output_path}: {len(entries)} entries, {size_kb:.1f} KB")


def do_swap(args):
    """V-swap 실험: src 토큰의 V를 dst 토큰의 V로 교체"""
    vocab = get_vocab_map()

    src_id, src_found = find_token_id(args.src, vocab)
    dst_id, dst_found = find_token_id(args.dst, vocab)

    if src_id is None:
        print(f"ERROR: '{args.src}' not found in vocab")
        return
    if dst_id is None:
        print(f"ERROR: '{args.dst}' not found in vocab")
        return

    print(f"[swap] '{args.src}' → token '{src_found}' (id={src_id})")
    print(f"[swap] '{args.dst}' → token '{dst_found}' (id={dst_id})")

    # hashmap 로드
    hashmap_path = args.hashmap or os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin")
    if not os.path.exists(hashmap_path):
        # 없으면 빌드
        hashmap_path = os.path.join(WEIGHTS_DIR, "blk0_kv_hashmap.bin")

    entries, kv_dim = load_hashmap(hashmap_path)
    print(f"[load] {len(entries)} entries from {hashmap_path}")

    if src_id not in entries:
        print(f"ERROR: src token {src_id} not in hashmap")
        return
    if dst_id not in entries:
        print(f"ERROR: dst token {dst_id} not in hashmap")
        return

    # V 비교 (수정 전)
    k_src, v_src = entries[src_id]
    k_dst, v_dst = entries[dst_id]

    print(f"\n[before swap]")
    print(f"  {args.src} (id={src_id}): |K|={np.linalg.norm(k_src):.4f}, |V|={np.linalg.norm(v_src):.6f}")
    print(f"  {args.dst} (id={dst_id}): |K|={np.linalg.norm(k_dst):.4f}, |V|={np.linalg.norm(v_dst):.6f}")
    print(f"  V cosine similarity: {np.dot(v_src, v_dst) / (np.linalg.norm(v_src) * np.linalg.norm(v_dst) + 1e-10):.4f}")
    print(f"  V L2 distance: {np.linalg.norm(v_src - v_dst):.6f}")

    # V swap: src의 V를 dst의 V로 교체 (K는 유지 — 어텐션 매칭에 필요)
    entries[src_id] = (k_src.copy(), v_dst.copy())

    print(f"\n[after swap] '{args.src}' V ← '{args.dst}' V")
    print(f"  이제 '{args.src}' 토큰은 attention에서 '{args.dst}'의 정보를 전달")

    # 저장
    output_path = args.output or os.path.join(WEIGHTS_DIR, f"blk0_kv_swap_{args.src}_{args.dst}.bin")
    save_hashmap(entries, output_path, kv_dim)

    print(f"\n[실험 방법]")
    print(f"  slot_fusion infer -m models/qwen2-0_5b-instruct-q4_k_m.gguf \\")
    print(f"    -s models/mia_slot.bin -p \"The {args.src} is\" -n 20 \\")
    print(f"    --kv-hashmap {output_path}")
    print(f"\n  기대 결과: '{args.src}' 입력 시 '{args.dst}' 관련 출력 증가")


def do_blend(args):
    """V-blend 실험: src V에 dst V를 alpha만큼 혼합"""
    vocab = get_vocab_map()

    src_id, src_found = find_token_id(args.src, vocab)
    dst_id, dst_found = find_token_id(args.dst, vocab)
    alpha = args.alpha

    if src_id is None or dst_id is None:
        print(f"ERROR: token not found (src={args.src}, dst={args.dst})")
        return

    print(f"[blend] '{src_found}'(id={src_id}) V ← (1-α)×src + α×dst, α={alpha}")

    hashmap_path = args.hashmap or os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin")
    if not os.path.exists(hashmap_path):
        hashmap_path = os.path.join(WEIGHTS_DIR, "blk0_kv_hashmap.bin")
    entries, kv_dim = load_hashmap(hashmap_path)

    if src_id not in entries or dst_id not in entries:
        print("ERROR: tokens not in hashmap")
        return

    k_src, v_src = entries[src_id]
    k_dst, v_dst = entries[dst_id]

    # Blend: V_new = (1-alpha)*V_src + alpha*V_dst
    v_blended = (1.0 - alpha) * v_src + alpha * v_dst
    entries[src_id] = (k_src.copy(), v_blended.astype(np.float32))

    print(f"  |V_src|={np.linalg.norm(v_src):.6f}")
    print(f"  |V_dst|={np.linalg.norm(v_dst):.6f}")
    print(f"  |V_blend|={np.linalg.norm(v_blended):.6f}")
    print(f"  cos(V_blend, V_src)={np.dot(v_blended, v_src) / (np.linalg.norm(v_blended) * np.linalg.norm(v_src) + 1e-10):.4f}")

    output_path = args.output or os.path.join(
        WEIGHTS_DIR, f"blk0_kv_blend_{args.src}_{args.dst}_a{int(alpha*100)}.bin")
    save_hashmap(entries, output_path, kv_dim)


def do_zero(args):
    """V-zero 실험: 특정 토큰의 V를 0으로 → 정보 제거"""
    vocab = get_vocab_map()

    tokens = [t.strip() for t in args.tokens.split(",")]

    hashmap_path = args.hashmap or os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin")
    if not os.path.exists(hashmap_path):
        hashmap_path = os.path.join(WEIGHTS_DIR, "blk0_kv_hashmap.bin")
    entries, kv_dim = load_hashmap(hashmap_path)

    zeroed = []
    for word in tokens:
        tid, found = find_token_id(word, vocab)
        if tid is None or tid not in entries:
            print(f"  SKIP '{word}': not found")
            continue

        k, v = entries[tid]
        v_norm_before = np.linalg.norm(v)
        entries[tid] = (k.copy(), np.zeros(kv_dim, dtype=np.float32))
        zeroed.append((word, tid, found, v_norm_before))
        print(f"  ZERO '{found}' (id={tid}): |V| {v_norm_before:.6f} → 0.000000")

    output_path = args.output or os.path.join(WEIGHTS_DIR, "blk0_kv_zeroed.bin")
    save_hashmap(entries, output_path, kv_dim)

    print(f"\n[실험] V=0 → attention이 이 토큰에서 정보를 얻을 수 없음")
    print(f"  기대: zeroed 토큰이 context에 있어도 후속 토큰에 영향 못 줌")


def do_report(args):
    """
    전체 분석 리포트: V 벡터의 의미적 구조 분석
    - 유사 단어 간 V cosine similarity
    - V norm 분포
    - V 수정이 미치는 영향 예측
    """
    vocab = get_vocab_map()

    hashmap_path = args.hashmap or os.path.join(WEIGHTS_DIR, "blk0_kv_full.bin")
    if not os.path.exists(hashmap_path):
        hashmap_path = os.path.join(WEIGHTS_DIR, "blk0_kv_hashmap.bin")

    entries, kv_dim = load_hashmap(hashmap_path)
    print(f"[report] {len(entries)} entries loaded\n")

    # 의미적 유사 단어 쌍
    word_pairs = [
        ("cat", "dog"),
        ("cat", "fish"),
        ("cat", "table"),
        ("king", "queen"),
        ("man", "woman"),
        ("big", "large"),
        ("big", "small"),
        ("good", "bad"),
        ("run", "walk"),
        ("eat", "drink"),
    ]

    report_lines = []
    report_lines.append("=== Knowledge Injection Analysis Report ===\n")
    report_lines.append(f"Hashmap: {hashmap_path}")
    report_lines.append(f"Entries: {len(entries)}\n")

    # V 전체 통계
    all_v = np.array([entries[tid][1] for tid in sorted(entries.keys())[:10000]])
    all_k = np.array([entries[tid][0] for tid in sorted(entries.keys())[:10000]])

    report_lines.append("--- Global Statistics ---")
    report_lines.append(f"|K| mean={np.linalg.norm(all_k, axis=1).mean():.4f} "
                       f"std={np.linalg.norm(all_k, axis=1).std():.4f}")
    report_lines.append(f"|V| mean={np.linalg.norm(all_v, axis=1).mean():.6f} "
                       f"std={np.linalg.norm(all_v, axis=1).std():.6f}")
    report_lines.append(f"K dominates: |K|/|V| = {np.linalg.norm(all_k, axis=1).mean() / (np.linalg.norm(all_v, axis=1).mean() + 1e-10):.1f}x\n")

    # 단어 쌍 V 유사도
    report_lines.append("--- V Cosine Similarity (semantic pairs) ---")
    report_lines.append(f"{'Pair':<20} {'cos(V)':<10} {'L2(V)':<12} {'cos(K)':<10} {'L2(K)':<12}")
    report_lines.append("-" * 64)

    for w1, w2 in word_pairs:
        id1, found1 = find_token_id(w1, vocab)
        id2, found2 = find_token_id(w2, vocab)

        if id1 is None or id2 is None:
            continue
        if id1 not in entries or id2 not in entries:
            continue

        k1, v1 = entries[id1]
        k2, v2 = entries[id2]

        v_cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        v_l2 = np.linalg.norm(v1 - v2)
        k_cos = np.dot(k1, k2) / (np.linalg.norm(k1) * np.linalg.norm(k2) + 1e-10)
        k_l2 = np.linalg.norm(k1 - k2)

        pair_str = f"{w1}-{w2}"
        report_lines.append(f"{pair_str:<20} {v_cos:<10.4f} {v_l2:<12.6f} {k_cos:<10.4f} {k_l2:<12.4f}")

    # 결론
    report_lines.append("\n--- Analysis ---")
    report_lines.append("1. K는 구조/위치 정보 (|K|>>|V|, bias 지배)")
    report_lines.append("   → K 수정은 attention routing에 영향 (어떤 토큰에 주목할지)")
    report_lines.append("2. V는 내용/지식 정보 (|V| 작지만 토큰별 차이가 의미적)")
    report_lines.append("   → V 수정은 attention이 전달하는 정보를 변경")
    report_lines.append("3. V swap 예상 효과:")
    report_lines.append("   - cos(V) 높은 쌍 (cat-dog): 미미한 변화 (이미 유사)")
    report_lines.append("   - cos(V) 낮은 쌍 (cat-table): 큰 변화 (의미 충돌)")
    report_lines.append("4. 실제 영향 크기 = V 차이 × attention weight")
    report_lines.append("   → blk.0에서 attention weight는 주로 K(위치/구조)가 결정")
    report_lines.append("   → V 수정의 효과는 해당 토큰의 attention weight에 비례")

    # 출력
    report_path = os.path.join(WEIGHTS_DIR, "knowledge_injection_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    for line in report_lines:
        print(line)

    print(f"\n[saved] {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge Injection Experiment")
    parser.add_argument("--mode", choices=["swap", "blend", "zero", "report"],
                        default="report", help="실험 모드")
    parser.add_argument("--src", type=str, default="cat", help="Source token")
    parser.add_argument("--dst", type=str, default="dog", help="Destination token")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend ratio")
    parser.add_argument("--tokens", type=str, default="cat,dog", help="Zero 대상 토큰들")
    parser.add_argument("--hashmap", type=str, help="입력 hashmap 경로")
    parser.add_argument("--output", "-o", type=str, help="출력 hashmap 경로")
    args = parser.parse_args()

    if args.mode == "swap":
        do_swap(args)
    elif args.mode == "blend":
        do_blend(args)
    elif args.mode == "zero":
        do_zero(args)
    elif args.mode == "report":
        do_report(args)


if __name__ == "__main__":
    main()
