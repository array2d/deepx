# test/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)

# test/CMakeLists.txt

add_executable(ipc_demo ipc_demo.cpp)
target_link_libraries(ipc_demo PRIVATE memcuda)

add_executable(registry_demo registry_demo.cpp)
target_link_libraries(registry_demo PRIVATE memcuda)

add_executable(lifecycle_demo lifecycle_demo.cpp)
target_link_libraries(lifecycle_demo PRIVATE memcuda)

install(TARGETS ipc_demo registry_demo lifecycle_demo DESTINATION bin)
