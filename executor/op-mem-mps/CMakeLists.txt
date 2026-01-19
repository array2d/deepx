cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-executor-mps LANGUAGES CXX OBJCXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# 仅支持 macOS
if(NOT APPLE)
    message(FATAL_ERROR "op-mem-mps 仅支持 macOS")
endif()

# 包含头文件目录
include_directories(src)

add_subdirectory(../cpp-common common)

# 源文件
file(GLOB_RECURSE DEEPX_SOURCES "src/deepx/*.cpp" "src/deepx/*.mm" "src/deepx/*.hpp" "src/deepx/*.h")
file(GLOB_RECURSE CLIENT_SOURCES "src/client/*.cpp" "src/client/*.mm")

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

find_package(yaml-cpp REQUIRED)
set(YAMLCPP_LIB "")
if (TARGET yaml-cpp::yaml-cpp)
  set(YAMLCPP_LIB yaml-cpp::yaml-cpp)
else()
  set(YAMLCPP_LIB yaml-cpp)
endif()

add_library(deepx_mps SHARED
    ${DEEPX_SOURCES}
)

target_link_libraries(deepx_mps
    PUBLIC
    deepx_common
    ${YAMLCPP_LIB}
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

add_executable(${PROJECT_NAME} ${CLIENT_SOURCES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    deepx_mps
)

# 测试
add_subdirectory(test/tensorfunc)

