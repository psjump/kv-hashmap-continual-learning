cmake_minimum_required(VERSION 3.16)
project(slot_fusion CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# llama.cpp 경로
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp")
set(LLAMA_BUILD_DIR "${LLAMA_DIR}/build")

# slot_plugin 경로
set(SLOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../slot_plugin")
set(SLOT_BUILD_DIR "${SLOT_DIR}/build")

# slot_fusion 실행 파일
add_executable(slot_fusion slot_fusion.cpp)

# slot_plugin (정적 링크)
target_include_directories(slot_fusion PRIVATE ${SLOT_DIR})
target_compile_definitions(slot_fusion PRIVATE SLOT_STATIC KV_HASHMAP_ENABLED)
target_link_libraries(slot_fusion "${SLOT_BUILD_DIR}/Release/slot_static.lib")

# llama.cpp (동적 링크 — DLL 사용)
target_include_directories(slot_fusion PRIVATE
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/ggml/include"
)

# llama.dll import library
target_link_libraries(slot_fusion
    "${LLAMA_BUILD_DIR}/src/Release/llama.lib"
    "${LLAMA_BUILD_DIR}/ggml/src/Release/ggml.lib"
    "${LLAMA_BUILD_DIR}/ggml/src/Release/ggml-base.lib"
)

# DLL을 빌드 디렉토리로 복사
add_custom_command(TARGET slot_fusion POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMA_BUILD_DIR}/bin/Release/llama.dll"
        "${LLAMA_BUILD_DIR}/bin/Release/llama-common.dll"
        "${LLAMA_BUILD_DIR}/bin/Release/ggml.dll"
        "${LLAMA_BUILD_DIR}/bin/Release/ggml-base.dll"
        "${LLAMA_BUILD_DIR}/bin/Release/ggml-cpu.dll"
        "${LLAMA_BUILD_DIR}/bin/Release/ggml-vulkan.dll"
        "$<TARGET_FILE_DIR:slot_fusion>"
    COMMENT "Copying llama.cpp DLLs..."
)
