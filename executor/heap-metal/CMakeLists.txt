cmake_minimum_required(VERSION 3.15)
project(deepx-heap-metal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

include_directories(src)
include_directories(../common-metal/include)

# hiredis (Redis C client)
include_directories(/opt/homebrew/opt/hiredis/include)
link_directories(/opt/homebrew/opt/hiredis/lib)

# nlohmann/json (header-only JSON parser)
include_directories(/opt/homebrew/opt/nlohmann-json/include)

# 依赖 common-metal 公共库
if(NOT TARGET deepx_common_metal)
    add_subdirectory(../common-metal common-metal)
endif()

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE deepx_common_metal hiredis)
