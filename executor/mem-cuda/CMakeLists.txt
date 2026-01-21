cmake_minimum_required(VERSION 3.18)
project(mem-cuda LANGUAGES CXX)

option(USE_RMM "Enable RMM adapter" OFF)
option(USE_CNMEM "Enable CNMeM adapter" OFF)

option(USE_HIREDIS "Link with hiredis (C client for Redis)" ON)
option(USE_REDISPP "Link with redis++ (C++ wrapper). redis++ requires hiredis." OFF)

find_package(Threads REQUIRED)

find_package(CUDAToolkit REQUIRED)

# Top-level build
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(test)

message(STATUS "mem-cuda: CUDA toolkit at ${CUDAToolkit_ROOT}")
message(STATUS "mem-cuda: USE_RMM=${USE_RMM} USE_CNMEM=${USE_CNMEM} USE_HIREDIS=${USE_HIREDIS} USE_REDISPP=${USE_REDISPP}")
cmake_minimum_required(VERSION 3.18)
project(mem-cuda LANGUAGES CXX)

option(USE_RMM "Enable RMM adapter" OFF)
option(USE_CNMEM "Enable CNMeM adapter" OFF)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(test)
