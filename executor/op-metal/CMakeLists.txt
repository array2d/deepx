cmake_minimum_required(VERSION 3.15)
project(deepx-op-metal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0")

if(NOT APPLE)
    message(FATAL_ERROR "op-metal only supports macOS")
endif()

include_directories(src)
include_directories(../old-cppcommon)
include_directories(../common-metal/include)

# hiredis (Redis C client)
include_directories(/opt/homebrew/opt/hiredis/include)
link_directories(/opt/homebrew/opt/hiredis/lib)

# nlohmann/json (header-only JSON parser)
include_directories(/opt/homebrew/opt/nlohmann-json/include)

# dxlang (for deepx/dtype/precision.hpp etc.)
include_directories(../dxlang/src)

# 依赖 common-metal 公共库
if(NOT TARGET deepx_common_metal)
    add_subdirectory(../common-metal common-metal)
endif()

# 源文件
file(GLOB_RECURSE DEEPX_SOURCES "src/deepx/*.cpp" "src/deepx/*.hpp" "src/deepx/*.h")
file(GLOB_RECURSE CLIENT_SOURCES "src/client/*.cpp")

find_library(METAL Metal REQUIRED)
find_library(FOUNDATION Foundation REQUIRED)

add_library(deepx_metal SHARED ${DEEPX_SOURCES})
target_link_libraries(deepx_metal
    PUBLIC deepx_common_metal ${METAL} ${FOUNDATION}
)
set_target_properties(deepx_metal PROPERTIES
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)

# metal_context.cpp 包含 ObjC Metal API
set_source_files_properties(src/deepx/metal_context.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")

# main.cpp 通过 elementwise_common.hpp 间接使用 Metal GPU kernel (__OBJC__ guard)
set_source_files_properties(src/client/main.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")

# ── 编译期 .metal → .metallib（使用 Metal Framework API，无需 xcrun/Xcode）──
set(METAL_SOURCE   "${CMAKE_CURRENT_SOURCE_DIR}/src/deepx/tensorfunc/elementwise_miaobyte.metal")
set(METALLIB_OUTPUT "${CMAKE_BINARY_DIR}/default.metallib")

add_executable(compile_metal_lib src/build/compile_metal_lib.mm)
set_source_files_properties(src/build/compile_metal_lib.mm PROPERTIES COMPILE_FLAGS "-x objective-c++")
set_target_properties(compile_metal_lib PROPERTIES
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)
target_link_libraries(compile_metal_lib ${METAL} ${FOUNDATION})

add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND compile_metal_lib ${METAL_SOURCE} ${METALLIB_OUTPUT}
    DEPENDS compile_metal_lib ${METAL_SOURCE}
    COMMENT "[metal] build-time compiling ${METAL_SOURCE} → default.metallib"
)
add_custom_target(metal_shaders DEPENDS ${METALLIB_OUTPUT})

add_executable(${PROJECT_NAME} ${CLIENT_SOURCES})
add_dependencies(${PROJECT_NAME} metal_shaders)
target_link_libraries(${PROJECT_NAME} PRIVATE deepx_metal hiredis)

# 集成测试
add_subdirectory(test/shm)
