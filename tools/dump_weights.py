"""
Qwen2-0.5B GGUF 가중치 덤프 도구

기능:
  1. dump   — 모든 텐서를 .npy로 저장 + 메타정보 JSON
  2. list   — 텐서 목록 + shape + dtype 출력
  3. load   — 특정 텐서 로드 후 통계 출력
  4. compare — 두 텐서 파일 비교 (diff)

사용:
  python dump_weights.py dump   -m model.gguf -o weights_dir/
  python dump_weights.py list   -m model.gguf
  python dump_weights.py load   -f weights_dir/blk.0.attn_q.weight.npy
  python dump_weights.py compare -a file1.npy -b file2.npy
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path


def read_gguf_tensors(model_path):
    """GGUF 파일에서 모든 텐서 메타데이터 + 데이터 읽기"""
    from gguf import GGUFReader
    reader = GGUFReader(model_path)
    return reader


def cmd_list(args):
    """텐서 목록 출력"""
    reader = read_gguf_tensors(args.model)

    print(f"{'#':>3}  {'Name':<45} {'Shape':<25} {'Type':<10} {'Size':>12}")
    print("-" * 100)

    total_params = 0
    for i, tensor in enumerate(reader.tensors):
        shape = tuple(int(d) for d in tensor.shape)
        n_params = 1
        for d in shape:
            n_params *= d
        total_params += n_params
        print(f"{i:3d}  {tensor.name:<45} {str(shape):<25} {str(tensor.tensor_type):<10} {n_params:>12,}")

    print("-" * 100)
    print(f"Total: {len(reader.tensors)} tensors, {total_params:,} parameters")

    # 레이어 구조 분석
    print("\n=== Layer Structure ===")
    layers = {}
    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk."):
            parts = name.split(".")
            layer = int(parts[1])
            component = ".".join(parts[2:])
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(component)
        else:
            print(f"  [global] {name}: {tuple(int(d) for d in tensor.shape)}")

    if layers:
        sample_layer = sorted(layers.keys())[0]
        print(f"\n  Layer {sample_layer} components ({len(layers[sample_layer])}):")
        for comp in sorted(layers[sample_layer]):
            print(f"    - {comp}")
        print(f"\n  Total layers: {len(layers)} (0~{max(layers.keys())})")


def dequantize_tensor(tensor):
    """양자화된 텐서를 float32로 디퀀타이즈"""
    import struct

    data = bytes(tensor.data)
    shape = tuple(int(d) for d in tensor.shape)
    ttype = str(tensor.tensor_type)

    if ttype == "GGMLQuantizationType.F32":
        return np.frombuffer(data, dtype=np.float32).reshape(shape)
    elif ttype == "GGMLQuantizationType.F16":
        return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
    elif ttype == "GGMLQuantizationType.Q8_0":
        # Q8_0: 32 values per block, 1 f16 scale + 32 int8
        block_size = 32
        n_elements = 1
        for d in shape:
            n_elements *= d
        n_blocks = n_elements // block_size
        result = np.zeros(n_elements, dtype=np.float32)

        offset = 0
        for b in range(n_blocks):
            # scale: float16 (2 bytes)
            scale = np.frombuffer(data[offset:offset+2], dtype=np.float16).astype(np.float32)[0]
            offset += 2
            # values: 32 x int8
            vals = np.frombuffer(data[offset:offset+block_size], dtype=np.int8).astype(np.float32)
            offset += block_size
            result[b*block_size:(b+1)*block_size] = vals * scale

        return result.reshape(shape)
    elif "Q5_0" in ttype:
        # Q5_0: complex dequant, store raw for now
        print(f"  WARNING: {ttype} - storing raw bytes (use ggml for precise dequant)")
        n_elements = 1
        for d in shape:
            n_elements *= d
        # Approximate: store shape info + raw data
        return None
    elif "Q4_K" in ttype or "Q6_K" in ttype:
        print(f"  WARNING: {ttype} - storing raw bytes (use ggml for precise dequant)")
        return None
    else:
        print(f"  WARNING: unknown type {ttype} - skipping")
        return None


def cmd_dump(args):
    """모든 텐서를 .npy로 덤프"""
    reader = read_gguf_tensors(args.model)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": args.model,
        "n_tensors": len(reader.tensors),
        "tensors": {}
    }

    # 메타데이터 덤프
    print("=== Metadata ===")
    for field in reader.fields.values():
        if len(field.parts) > 0:
            name = str(field.name)
            # 간단한 값만 저장
            if len(field.data) == 1:
                val = field.parts[field.data[0]][0]
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='replace')
                elif isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                elif isinstance(val, (np.bool_,)):
                    val = bool(val)
                meta[name] = val

    dumped = 0
    skipped = 0

    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        shape = tuple(int(d) for d in tensor.shape)
        ttype = str(tensor.tensor_type)
        safe_name = name.replace("/", "_")

        print(f"[{i+1}/{len(reader.tensors)}] {name} {shape} ({ttype})...", end=" ")

        arr = dequantize_tensor(tensor)

        if arr is not None:
            npy_path = out_dir / f"{safe_name}.npy"
            np.save(str(npy_path), arr)
            meta["tensors"][name] = {
                "file": f"{safe_name}.npy",
                "shape": list(shape),
                "dtype": "float32",
                "original_type": ttype,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
            print(f"OK (mean={np.mean(arr):.6f}, std={np.std(arr):.6f})")
            dumped += 1
        else:
            # 양자화 포맷은 raw bytes로 저장
            raw_path = out_dir / f"{safe_name}.raw"
            raw_data = bytes(tensor.data)
            with open(str(raw_path), "wb") as f:
                f.write(raw_data)
            meta["tensors"][name] = {
                "file": f"{safe_name}.raw",
                "shape": list(shape),
                "dtype": "raw",
                "original_type": ttype,
                "raw_size": len(raw_data),
            }
            print(f"RAW ({len(raw_data):,} bytes)")
            skipped += 1

    # 메타 저장
    meta_path = out_dir / "meta.json"
    with open(str(meta_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"  Dumped: {dumped} tensors (float32 .npy)")
    print(f"  Raw: {skipped} tensors (quantized .raw)")
    print(f"  Meta: {meta_path}")
    print(f"  Output: {out_dir}")


def cmd_load(args):
    """특정 텐서 파일 로드 + 분석"""
    path = args.file
    if path.endswith(".npy"):
        arr = np.load(path)
        print(f"File: {path}")
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        print(f"Mean:  {np.mean(arr):.8f}")
        print(f"Std:   {np.std(arr):.8f}")
        print(f"Min:   {np.min(arr):.8f}")
        print(f"Max:   {np.max(arr):.8f}")
        print(f"NaN:   {np.isnan(arr).sum()}")
        print(f"Zero:  {(arr == 0).sum()} / {arr.size} ({100*(arr==0).sum()/arr.size:.1f}%)")

        # 분포 히스토그램 (텍스트)
        print(f"\nDistribution:")
        hist, edges = np.histogram(arr.flatten(), bins=20)
        max_h = max(hist)
        for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
            bar = "#" * int(40 * h / max_h) if max_h > 0 else ""
            print(f"  [{lo:+.4f}, {hi:+.4f}): {h:>8,} {bar}")
    else:
        size = os.path.getsize(path)
        print(f"File: {path}")
        print(f"Size: {size:,} bytes (raw quantized)")
        print("  Use 'list' command to check original type")


def cmd_compare(args):
    """두 텐서 파일 비교"""
    a = np.load(args.a)
    b = np.load(args.b)

    print(f"A: {args.a} shape={a.shape}")
    print(f"B: {args.b} shape={b.shape}")

    if a.shape != b.shape:
        print("ERROR: shapes differ!")
        return

    diff = a - b
    print(f"\nDifference (A - B):")
    print(f"  Mean:    {np.mean(diff):.8f}")
    print(f"  Std:     {np.std(diff):.8f}")
    print(f"  Max abs: {np.max(np.abs(diff)):.8f}")
    print(f"  L2 norm: {np.linalg.norm(diff):.8f}")
    print(f"  Cosine:  {np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10):.8f}")

    exact_match = (a == b).sum()
    print(f"  Exact match: {exact_match}/{a.size} ({100*exact_match/a.size:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GGUF Weight Dump/Analysis Tool")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="List all tensors")
    p_list.add_argument("-m", "--model", required=True)

    p_dump = sub.add_parser("dump", help="Dump all tensors to .npy")
    p_dump.add_argument("-m", "--model", required=True)
    p_dump.add_argument("-o", "--output", default="weights")

    p_load = sub.add_parser("load", help="Load and analyze a tensor file")
    p_load.add_argument("-f", "--file", required=True)

    p_cmp = sub.add_parser("compare", help="Compare two tensor files")
    p_cmp.add_argument("-a", required=True)
    p_cmp.add_argument("-b", required=True)

    args = parser.parse_args()

    if args.cmd == "list":
        cmd_list(args)
    elif args.cmd == "dump":
        cmd_dump(args)
    elif args.cmd == "load":
        cmd_load(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
