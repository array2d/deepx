# tools/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)

# tools/CMakeLists.txt

add_executable(memcuda_ctl memcuda_ctl.cpp)
target_link_libraries(memcuda_ctl PRIVATE memcuda)

install(TARGETS memcuda_ctl DESTINATION bin)
