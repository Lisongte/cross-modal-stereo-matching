cmake_minimum_required(VERSION 3.16)
project(cross_modal_stereo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc calib3d)

add_executable(rectify_pair src/rectify_pair.cpp)
target_link_libraries(rectify_pair PRIVATE ${OpenCV_LIBS})

add_executable(stereo_match src/stereo_match.cpp)
target_link_libraries(stereo_match PRIVATE ${OpenCV_LIBS})
