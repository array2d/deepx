add_executable(0_new 0_new.cpp)
target_link_libraries(0_new deepx CUDA::cudart)

add_executable(1_cublas_add 1_cublas_add.cpp)
target_link_libraries(1_cublas_add deepx CUDA::cudart)

add_executable(1_cublas_matmul 1_cublas_matmul.cpp)
target_link_libraries(1_cublas_matmul deepx CUDA::cudart)

add_executable(2_changeshape 2_changeshape.cpp)
target_link_libraries(2_changeshape deepx CUDA::cudart)