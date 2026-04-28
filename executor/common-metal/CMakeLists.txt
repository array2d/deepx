cmake_minimum_required(VERSION 3.15)
project(deepx-common-metal LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

find_library(METAL Metal)
find_library(FOUNDATION Foundation)

add_library(deepx_common_metal STATIC
    src/shmem/shm_tensor.mm
    src/metal_device.mm
)

target_link_libraries(deepx_common_metal PUBLIC ${METAL} ${FOUNDATION})
target_include_directories(deepx_common_metal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(deepx_common_metal PROPERTIES
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)
