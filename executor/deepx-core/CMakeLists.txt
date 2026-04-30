cmake_minimum_required(VERSION 3.15)
project(deepx-core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# Include directories
include_directories(include)
include_directories(/opt/homebrew/opt/nlohmann-json/include)

# Source files — only .cpp files that need compilation
file(GLOB_RECURSE CORE_SOURCES "src/*.cpp")

add_library(deepx_core STATIC ${CORE_SOURCES})

# Export include path to consumers
target_include_directories(deepx_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
