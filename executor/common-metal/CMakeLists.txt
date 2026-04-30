cmake_minimum_required(VERSION 3.15)
project(deepx-common-metal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

find_library(METAL Metal)
find_library(FOUNDATION Foundation)

add_library(deepx_common_metal STATIC
    src/shmem/shm_tensor.cpp
    src/metal_device.cpp
)

# metal_device.cpp 包含 ObjC Metal API, 需要 Objective-C++ 编译
set_source_files_properties(src/metal_device.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")

target_link_libraries(deepx_common_metal PUBLIC ${METAL} ${FOUNDATION})
target_include_directories(deepx_common_metal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(deepx_common_metal PROPERTIES
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)
