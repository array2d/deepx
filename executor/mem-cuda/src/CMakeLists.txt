cmake_minimum_required(VERSION 3.18)

# src/CMakeLists.txt - memcuda core (clean)

set(SRCS
  registry/redis_client.cpp
  registry/registry.cpp
  ipc/ipc.cpp
  allocator/cuda_pool.cpp
  runtime/lifecycle.cpp
  runtime/sync.cpp
  common/common.cpp
)

add_library(memcuda STATIC ${SRCS})
target_include_directories(memcuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(memcuda PRIVATE CUDA::cudart Threads::Threads)

# Require redis++ (redis-plus-plus) and hiredis
find_path(REDISPP_INCLUDE redis++/redis++.h)
find_library(HIREDIS_LIB hiredis)

message(STATUS "Found redis++ include: ${REDISPP_INCLUDE}")
message(STATUS "Found hiredis lib: ${HIREDIS_LIB}")

target_include_directories(memcuda PRIVATE ${REDISPP_INCLUDE})
target_link_libraries(memcuda PRIVATE ${HIREDIS_LIB})
target_compile_definitions(memcuda PRIVATE USE_REDISPP=1)
