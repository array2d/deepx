cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-excuter-ompsimd LANGUAGES CXX ) 

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译优化
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
set(CMAKE_BUILD_TYPE Debug)
# 设置 SIMD 编译选项
 
# 包含头文件目录
include_directories(src)
 
add_subdirectory(../cpp-common common)

# 源文件

file(GLOB_RECURSE DEEPX_SOURCES "src/*.cpp")
file(GLOB_RECURSE CLIENT_SOURCES "src/client/*.cpp")

# cpu 线性代数库
list(APPEND CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/openblas-pthread/cmake")
find_package(OpenBLAS REQUIRED)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse4.2")
# 内存池
find_package(PkgConfig REQUIRED)
pkg_check_modules(JEMALLOC REQUIRED jemalloc)


find_package(yaml-cpp REQUIRED)
 
add_library(deepx_ompsimd SHARED 
    ${DEEPX_SOURCES}
)

target_link_libraries( deepx_ompsimd
    PUBLIC 
    deepx_common
    yaml-cpp
    ${JEMALLOC_LIBRARIES}    
    openblas
    OpenMP::OpenMP_CXX
    hwy
)
add_executable(${PROJECT_NAME} ${CLIENT_SOURCES})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    deepx_ompsimd
)
# 测试
add_subdirectory(test/tensorfunc)
add_subdirectory(test/op)
# # 数据集测试
# add_subdirectory(test/dataset)
