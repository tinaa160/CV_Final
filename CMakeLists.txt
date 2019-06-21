set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(VERSION 2.8.3)
project(CV_final)

find_package(OpenCV REQUIRED)
add_executable(main src/main.cpp )
include_directories(${OpenCV_INCLUDE_DIRS} )
target_link_libraries(main ${OpenCV_LIBS} )




