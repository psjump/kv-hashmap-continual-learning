cmake_minimum_required(VERSION 3.16)
project(slot_plugin C)

set(CMAKE_C_STANDARD 11)

# Slot 라이브러리 (동적 DLL — Python/외부 연동용)
add_library(slot SHARED slot.c kv_hashmap.c)
target_include_directories(slot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(slot PRIVATE SLOT_EXPORTS)

# 정적 라이브러리 (C/C++ 직접 링크용)
add_library(slot_static STATIC slot.c kv_hashmap.c)
target_include_directories(slot_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(slot_static PUBLIC SLOT_STATIC)

# 테스트 실행 파일
add_executable(test_slot test_slot.c)
target_link_libraries(test_slot slot_static)

# KV Hashmap 테스트
add_executable(test_kv_hashmap test_kv_hashmap.c)
target_link_libraries(test_kv_hashmap slot_static)
