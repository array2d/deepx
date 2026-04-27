cmake_minimum_required(VERSION 3.15)
project(deepx-op-metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

if(NOT APPLE)
    message(FATAL_ERROR "op-metal only supports macOS")
endif()

include_directories(src)
include_directories(../heapmem-metal/src)  # 共享 shm_tensor.h

# 源文件
file(GLOB_RECURSE DEEPX_SOURCES "src/deepx/*.cpp" "src/deepx/*.mm" "src/deepx/*.hpp" "src/deepx/*.h")
file(GLOB_RECURSE CLIENT_SOURCES "src/client/*.cpp" "src/client/*.mm")

find_library(METAL Metal REQUIRED)
find_library(MPS MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION Foundation REQUIRED)

# shm_tensor 实现 (shared with heapmem-metal)
set(SHMEM_SRC ../heapmem-metal/src/shmem/shm_tensor.mm)

add_library(deepx_metal SHARED ${DEEPX_SOURCES} ${SHMEM_SRC})
target_link_libraries(deepx_metal
    PUBLIC ${METAL} ${MPS} ${FOUNDATION}
)
set_target_properties(deepx_metal PROPERTIES
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)

add_executable(${PROJECT_NAME} ${CLIENT_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE deepx_metal)
set_target_properties(${PROJECT_NAME} PROPERTIES
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)

# 集成测试
add_subdirectory(test/shm)
