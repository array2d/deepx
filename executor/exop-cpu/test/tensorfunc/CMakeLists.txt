add_executable(1_shape 1_shape.cpp)
target_link_libraries(1_shape deepx_cpu)

add_executable(2_shape_combintion 2_shape_combintion.cpp)
target_link_libraries(2_shape_combintion deepx_cpu)

add_executable(2_tensor_new 2_tensor_new.cpp)
target_link_libraries(2_tensor_new deepx_cpu )

add_executable(2_tensor_range 2_tensor_range.cpp)
target_link_libraries(2_tensor_range deepx_cpu  )

add_executable(2_tensor_equal 2_tensor_equal.cpp)
target_link_libraries(2_tensor_equal deepx_cpu  )

add_executable(3_tensor_print 3_tensor_print.cpp)
target_link_libraries(3_tensor_print deepx_cpu  )


add_executable(4_tensor_matmul 4_tensor_matmul.cpp)
target_link_libraries(4_tensor_matmul deepx_cpu  )

add_executable(4_tensor_add 4_tensor_add.cpp)
target_link_libraries(4_tensor_add deepx_cpu   )

add_executable(4_tensor_sub 4_tensor_sub.cpp)
target_link_libraries(4_tensor_sub deepx_cpu  )

add_executable(4_tensor_mul 4_tensor_mul.cpp)
target_link_libraries(4_tensor_mul deepx_cpu  )

add_executable(4_tensor_max 4_tensor_max.cpp)
target_link_libraries(4_tensor_max deepx_cpu  )


add_executable(5_tensor_sum 5_tensor_sum.cpp)
target_link_libraries(5_tensor_sum deepx_cpu  )

add_executable(6_tensor_broadcast 6_tensor_broadcast.cpp)
target_link_libraries(6_tensor_broadcast deepx_cpu  )

add_executable(7_tensor_transpose 7_tensor_transpose.cpp)
target_link_libraries(7_tensor_transpose deepx_cpu  )

add_executable(8_tensor_concat 8_tensor_concat.cpp)
target_link_libraries(8_tensor_concat deepx_cpu  )