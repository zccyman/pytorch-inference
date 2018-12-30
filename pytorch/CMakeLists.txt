cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(pytorch demo/main.cpp src/pytorch_interface.cpp)

target_link_libraries(pytorch "${TORCH_LIBRARIES}")
target_link_libraries(pytorch "${OpenCV_LIBS}")

set_property(TARGET pytorch PROPERTY CXX_STANDARD 11)
set_property(TARGET pytorch PROPERTY OUTPUT_NAME pytorch)
