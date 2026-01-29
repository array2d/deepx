cmake_minimum_required(VERSION 3.15...3.29)
project(deepxcore LANGUAGES CXX ) 

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# 设置编译优化

set(CMAKE_BUILD_TYPE Debug)
# 设置 SIMD 编译选项
 
# 包含头文件目录
include_directories(src)

# 源文件

file(GLOB_RECURSE DEEPX_CORE_SOURCES "src/*.cpp")

add_library(deepxcore SHARED 
    ${DEEPX_CORE_SOURCES}
)

 
find_package(yaml-cpp REQUIRED)

set(YAMLCPP_LIB "")
if (TARGET yaml-cpp::yaml-cpp)
  set(YAMLCPP_LIB yaml-cpp::yaml-cpp)
else()
  set(YAMLCPP_LIB yaml-cpp)
endif()
  
target_link_libraries(deepxcore PUBLIC ${YAMLCPP_LIB})

target_include_directories(deepxcore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
 
 
add_subdirectory(test)
