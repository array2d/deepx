# src/registry/CMakeLists.txt

set(REGISTRY_SRCS
  redis_client.cpp
  registry.cpp
)

add_library(memcuda_registry STATIC ${REGISTRY_SRCS})
target_include_directories(memcuda_registry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(USE_HIREDIS)
  find_library(HIREDIS_LIB hiredis)
  if(HIREDIS_LIB)
    target_link_libraries(memcuda_registry PRIVATE ${HIREDIS_LIB})
    target_compile_definitions(memcuda_registry PRIVATE USE_HIREDIS=1)
  endif()
endif()

# Install lua scripts for runtime use
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lua_scripts DESTINATION share/mem-cuda)
# src/registry/CMakeLists.txt

set(REGISTRY_SRCS
  redis_client.cpp
  registry.cpp
)

add_library(memcuda_registry STATIC ${REGISTRY_SRCS})
target_include_directories(memcuda_registry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_library(HIREDIS_LIB hiredis)
if(HIREDIS_LIB)
  target_link_libraries(memcuda_registry PRIVATE ${HIREDIS_LIB})
endif()

# Install lua scripts for runtime use
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lua_scripts DESTINATION share/mem-cuda)
