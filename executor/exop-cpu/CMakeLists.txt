cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-exop-cpu LANGUAGES CXX )

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译优化
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
set(CMAKE_BUILD_TYPE Debug)
# 设置 SIMD 编译选项

# 包含头文件目录
include_directories(src)

# hiredis / nlohmann/json（可执行文件连接 Redis 需要）
include_directories(/opt/homebrew/opt/hiredis/include)
link_directories(/opt/homebrew/opt/hiredis/lib)
include_directories(/opt/homebrew/opt/nlohmann-json/include)

# 客户端源文件（可执行文件）
file(GLOB_RECURSE CLIENT_SOURCES "src/client/*.cpp")

if(NOT TARGET deepx_core)
    add_subdirectory(../deepx-core deepx-core)
endif()

# cpu 线性代数库
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake")
find_package(OpenBLAS REQUIRED)
if(OpenBLAS_INCLUDE_DIRS)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    link_directories(${OpenBLAS_LIB_DIR})
elseif(APPLE)
    include_directories(/opt/homebrew/opt/openblas/include)
    link_directories(/opt/homebrew/opt/openblas/lib)
else()
    include_directories(/usr/include/openblas)
endif()

# include(FetchContent)

# # Fetch OpenBLAS from GitHub
# FetchContent_Declare(
#   OpenBLAS
#   GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS
#   GIT_TAG        v0.3.29  # 使用最新的稳定版本或您需要的特定版本
# )

# # 下载并构建 OpenBLAS
# FetchContent_MakeAvailable(OpenBLAS)

# 线程并行
find_package(OpenMP REQUIRED)
# simd
find_package(hwy REQUIRED)
if(HWY_INCLUDE_DIRS)
    include_directories(${HWY_INCLUDE_DIRS})
    link_directories(${HWY_LIB_DIR})
elseif(APPLE)
    include_directories(/opt/homebrew/opt/highway/include)
    link_directories(/opt/homebrew/opt/highway/lib)
endif()
# SIMD: x86→AVX2+SSE4.2, ARM→NEON (Apple Clang auto-detects)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse4.2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.5-a")
endif()
# 内存池
find_package(PkgConfig REQUIRED)
pkg_check_modules(JEMALLOC REQUIRED jemalloc)
include_directories(${JEMALLOC_INCLUDE_DIRS})
link_directories(${JEMALLOC_LIBRARY_DIRS})


find_package(yaml-cpp REQUIRED)
 
# 纯模板头文件库（计算 kernel 实现）
add_library(deepx_cpu INTERFACE)
target_include_directories(deepx_cpu INTERFACE src)
target_link_libraries( deepx_cpu
    INTERFACE
    deepx_core
    yaml-cpp::yaml-cpp
    ${JEMALLOC_LIBRARIES}
    ${OpenBLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    hwy::hwy
)
# ── 计算执行器可执行文件（对接 Redis） ──
add_executable(${PROJECT_NAME} ${CLIENT_SOURCES})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    deepx_cpu
    hiredis
)

# 测试
add_subdirectory(test/tensorfunc)
