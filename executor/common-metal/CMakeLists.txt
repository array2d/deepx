cmake_minimum_required(VERSION 3.15)
project(deepx-common-metal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

find_library(METAL Metal)
find_library(FOUNDATION Foundation)

# ── 依赖 deepx-core（shm_tensor / registry 已迁移至 deepx-core）──
if(NOT TARGET deepx_core)
    add_subdirectory(../deepx-core deepx-core)
endif()

# ── Metal HAL：仅包含 Metal 设备检测 ──
add_library(deepx_metal_hal STATIC
    src/metal_device.cpp
)

# metal_device.cpp 包含 ObjC Metal API, 需要 Objective-C++ 编译
set_source_files_properties(src/metal_device.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")

target_link_libraries(deepx_metal_hal PUBLIC deepx_core ${METAL} ${FOUNDATION})
target_include_directories(deepx_metal_hal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(deepx_metal_hal PROPERTIES
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)
