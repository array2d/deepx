cmake_minimum_required(VERSION 3.15...3.29)
project(deepx-executor-mps LANGUAGES CXX OBJCXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)

# 仅支持 macOS
if(NOT APPLE)
    message(FATAL_ERROR "op-mem-mps 仅支持 macOS")
endif()

# 包含头文件目录
include_directories(src)

add_subdirectory(../cpp-common common)

# 源文件
file(GLOB_RECURSE DEEPX_SOURCES "src/deepx/*.cpp" "src/deepx/*.mm" "src/deepx/*.hpp" "src/deepx/*.h")
file(GLOB_RECURSE CLIENT_SOURCES "src/client/*.cpp" "src/client/*.mm")

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

find_package(yaml-cpp REQUIRED)
set(YAMLCPP_LIB "")
if (TARGET yaml-cpp::yaml-cpp)
  set(YAMLCPP_LIB yaml-cpp::yaml-cpp)
else()
  set(YAMLCPP_LIB yaml-cpp)
endif()

add_library(deepx_mps SHARED
    ${DEEPX_SOURCES}
)

# --- Offline compile Metal shaders into default.metallib ---
set(METAL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/deepx/tensorfunc)
set(METAL_SOURCES
  ${METAL_SRC_DIR}/elementwise_miaobyte.metal
)

set(METAL_AIR_DIR ${CMAKE_CURRENT_BINARY_DIR}/metal_air)
set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)

option(DEEPX_MPS_OFFLINE_METAL "Build .metallib at build time (requires Xcode/metal tools)" ON)

if (DEEPX_MPS_OFFLINE_METAL)
  execute_process(
    COMMAND xcrun -sdk macosx -find metal
    OUTPUT_VARIABLE METALC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE METALC_RV
    ERROR_QUIET
  )
  execute_process(
    COMMAND xcrun -sdk macosx -find metallib
    OUTPUT_VARIABLE METALLIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE METALLIB_RV
    ERROR_QUIET
  )

  if (METALC_RV EQUAL 0 AND METALLIB_RV EQUAL 0)
    add_custom_command(
      OUTPUT ${METAL_LIB}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${METAL_AIR_DIR}
      COMMAND ${METALC} -c ${METAL_SRC_DIR}/elementwise_miaobyte.metal -o ${METAL_AIR_DIR}/elementwise_miaobyte.air
      COMMAND ${METALLIB} ${METAL_AIR_DIR}/elementwise_miaobyte.air -o ${METAL_LIB}
      DEPENDS ${METAL_SOURCES}
      VERBATIM
    )

    add_custom_target(deepx_metal_kernels ALL DEPENDS ${METAL_LIB})
    add_dependencies(deepx_mps deepx_metal_kernels)
  else()
    message(WARNING "Metal offline tools not found (xcrun metal/metallib). Install Xcode or CLT, or set -DDEEPX_MPS_OFFLINE_METAL=OFF.")
  endif()
endif()
# ----------------------------------------------------------

target_link_libraries(deepx_mps
    PUBLIC
    deepx_common
    ${YAMLCPP_LIB}
    ${METAL_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

add_executable(${PROJECT_NAME} ${CLIENT_SOURCES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    deepx_mps
)

# 测试
add_subdirectory(test/tensorfunc)

