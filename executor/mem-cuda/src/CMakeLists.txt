cmake_minimum_required(VERSION 3.18)

# Core sources
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

# Redis client detection
if(USE_HIREDIS)
  find_library(HIREDIS_LIB hiredis)
  find_path(HIREDIS_INCLUDE hiredis/hiredis.h)
  if(HIREDIS_LIB AND HIREDIS_INCLUDE)
    message(STATUS "Found hiredis: ${HIREDIS_LIB}")
    target_compile_definitions(memcuda PRIVATE USE_HIREDIS=1)
    target_include_directories(memcuda PRIVATE ${HIREDIS_INCLUDE})
    target_link_libraries(memcuda PRIVATE ${HIREDIS_LIB})
  else()
    message(WARNING "hiredis not found; set USE_HIREDIS=OFF to disable or install hiredis")
  endif()
endif()

if(USE_REDISPP)
  # try detect redis++ header
  find_path(REDISPP_INCLUDE sw/redis++/redis++.h)
  if(REDISPP_INCLUDE)
    message(STATUS "Found redis++ headers: ${REDISPP_INCLUDE}")
    target_compile_definitions(memcuda PRIVATE USE_REDISPP=1)
    target_include_directories(memcuda PRIVATE ${REDISPP_INCLUDE})
    # redis++ depends on hiredis; ensure it's present
    if(NOT TARGET hiredis)
      if(HIREDIS_LIB)
        target_link_libraries(memcuda PRIVATE ${HIREDIS_LIB})
      endif()
    endif()
  else()
    message(WARNING "redis++ headers not found; set USE_REDISPP=OFF or install redis++")
  endif()
endif()

if(USE_RMM)
  message(STATUS "RMM adapter enabled (user must provide RMM include/link settings)")
  target_compile_definitions(memcuda PRIVATE USE_RMM=1)
endif()

target_include_directories(memcuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# src/CMakeLists.txt - build memcuda core library

cmake_minimum_required(VERSION 3.18)

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
target_link_libraries(memcuda PRIVATE CUDA::cudart pthread)

find_library(HIREDIS_LIB hiredis)
if(HIREDIS_LIB)
  target_link_libraries(memcuda PRIVATE ${HIREDIS_LIB})
endif()

if(USE_RMM)
  message(STATUS "RMM adapter enabled (user must provide RMM include/link settings)")
endif()
