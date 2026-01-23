cmake_minimum_required(VERSION 3.18)

# registry CMake - only used to install Lua scripts for runtime
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lua_scripts DESTINATION share/mem-cuda)
