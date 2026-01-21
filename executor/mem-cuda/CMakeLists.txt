cmake_minimum_required(VERSION 3.18)
project(mem-cuda LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Require redis++ (C++ client) and its hiredis dependency
find_path(REDISPP_INCLUDE redis++/redis++.h)
find_library(HIREDIS_LIB hiredis)

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(test)

message(STATUS "mem-cuda: CUDA toolkit at ${CUDAToolkit_ROOT}")
message(STATUS "mem-cuda: redis++ include: ${REDISPP_INCLUDE}, hiredis lib: ${HIREDIS_LIB}")
