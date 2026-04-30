cmake_minimum_required(VERSION 3.15)
project(deepx-core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# Include directories
include_directories(include)
include_directories(/opt/homebrew/opt/nlohmann-json/include)

# Source files — only .cpp files that need compilation (exclude metal/ ObjC++)
file(GLOB_RECURSE CORE_SOURCES "src/*.cpp")
list(FILTER CORE_SOURCES EXCLUDE REGEX "src/metal/")

add_library(deepx_core STATIC ${CORE_SOURCES})

# Export include path to consumers
target_include_directories(deepx_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ── Metal HAL (仅 Apple 平台) ──
if(APPLE)
    find_library(METAL Metal)
    find_library(FOUNDATION Foundation)
    add_library(deepx_metal_hal STATIC src/metal/metal_device.cpp)
    set_source_files_properties(src/metal/metal_device.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")
    target_link_libraries(deepx_metal_hal PUBLIC deepx_core ${METAL} ${FOUNDATION})
    target_include_directories(deepx_metal_hal PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    set_target_properties(deepx_metal_hal PROPERTIES
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    )
endif()
