add_executable(test_aruco testARUCO.cpp)

target_link_libraries(test_aruco  opencv_calib3d)
target_link_libraries(test_aruco  opencv_aruco)
