
add_executable(test_dtypes 0_dtypes.cpp)
target_link_libraries(test_dtypes deepxcore)

add_executable(test_tf  1_tf.cpp)
target_link_libraries(test_tf deepxcore)

add_executable(test_tfcheck  1_tfcheck.cpp)
target_link_libraries(test_tfcheck deepxcore)
 
add_executable(test_saveload  2_saveload.cpp)
target_link_libraries(test_saveload deepxcore)