cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-common LANGUAGES CXX ) 

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译优化

set(CMAKE_BUILD_TYPE Debug)
# 设置 SIMD 编译选项
 
# 包含头文件目录
include_directories(src)

# 源文件

file(GLOB_RECURSE DEEPX_COMMON_SOURCES "src/*.cpp")

add_library(deepx_common SHARED 
    ${DEEPX_COMMON_SOURCES}
)

 
find_package(yaml-cpp REQUIRED)

target_link_libraries(deepx_common
    PUBLIC
    yaml-cpp
)

target_include_directories(deepx_common PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
 
 
add_subdirectory(test)
