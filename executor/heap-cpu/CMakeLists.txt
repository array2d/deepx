cmake_minimum_required(VERSION 3.15)
project(deepx-heap-cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

include_directories(src)

# hiredis (Redis C client)
include_directories(/opt/homebrew/opt/hiredis/include)
link_directories(/opt/homebrew/opt/hiredis/lib)

# nlohmann/json (header-only JSON parser)
include_directories(/opt/homebrew/opt/nlohmann-json/include)

# 依赖 deepx-core（CPU only — 无 Metal）
if(NOT TARGET deepx_core)
    add_subdirectory(../deepx-core deepx-core)
endif()

file(GLOB_RECURSE SOURCES "src/*.cpp")

# ── 生命周期管理器（对接 Redis） ──
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    deepx_core
    hiredis
)
