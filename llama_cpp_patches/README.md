# llama.cpp Patches for KV Hashmap

These files are modifications to [llama.cpp](https://github.com/ggml-org/llama.cpp) to enable KV Hashmap overlay.

## Modified Files / 수정된 파일

| File | Original Location | Changes |
|------|-------------------|---------|
| `qwen2.cpp` | `llama.cpp/src/models/qwen2.cpp` | KV Hashmap overlay via `ggml_map_custom2_inplace` |
| `src_CMakeLists.txt` | `llama.cpp/src/CMakeLists.txt` | Added `KV_HASHMAP_ENABLED` + `SLOT_STATIC` compile definitions |

## How to Apply / 적용 방법

```bash
# 1. Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git

# 2. Copy modified files
cp llama_cpp_patches/qwen2.cpp llama.cpp/src/models/qwen2.cpp
cp llama_cpp_patches/src_CMakeLists.txt llama.cpp/src/CMakeLists.txt

# 3. Build with KV_HASHMAP_ENABLED
cd llama.cpp/build
cmake .. -DKV_HASHMAP_ENABLED=ON
cmake --build . --config Release
```

## Key Changes in qwen2.cpp / 핵심 수정 내용

1. **Global KVHashmapStack** — `g_kv_stack` pointer set by `qwen2_set_kv_stack()`
2. **Custom Op** — `kv_hashmap_layer_op()`: per-token hashmap lookup, HIT→overwrite K/V, MISS→keep original
3. **Capture Mode** — `capture_mode=1`: store original K,V to hashmap during forward pass
4. **Graph Integration** — `ggml_map_custom2_inplace` applied after `build_qkv()` for each layer with hashmap entries
5. **Backward Compatible** — `qwen2_set_kv_hashmap()` wraps single hashmap as 1-layer stack
