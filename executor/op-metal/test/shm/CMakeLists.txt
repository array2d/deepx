cmake_minimum_required(VERSION 3.15)
find_library(METAL Metal REQUIRED)
find_library(FOUNDATION Foundation REQUIRED)

add_executable(test_shm_mtl_baseline test_shm_mtl_baseline.mm)
target_link_libraries(test_shm_mtl_baseline ${METAL} ${FOUNDATION})
set_target_properties(test_shm_mtl_baseline PROPERTIES
    CXX_STANDARD 17
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)

add_executable(test_shm_cross_fork test_shm_cross_fork.mm)
target_link_libraries(test_shm_cross_fork ${METAL} ${FOUNDATION})
set_target_properties(test_shm_cross_fork PROPERTIES
    CXX_STANDARD 17
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)

add_executable(test_cross_process test_cross_process.mm)
target_link_libraries(test_cross_process ${METAL} ${FOUNDATION})
set_target_properties(test_cross_process PROPERTIES
    CXX_STANDARD 17
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
)
