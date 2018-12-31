cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch)

set(Torch_DIR "../../thirdparty/libtorch/share/cmake/Torch/")
set(Caffe2_DIR "../../thirdparty/libtorch/share/cmake/Caffe2/")
set(OpenCV_DIR "../../thirdparty/opencv/build/")

set(CMAKE_BUILD_TYPE "Release")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(../../thirdparty/CUDA/common/inc)
include_directories(../../pytorch/inc)

cuda_add_executable (pytorch
	../../pytorch/demo/main.cpp
	../../pytorch/inc/pytorch_softmax.hpp
	../../pytorch/src/pytorch_interface.cpp
	../../pytorch/src/pytorch_softmax.cpp
	../../pytorch/cu/pytorch_softmax.cu
)
#add_executable(pytorch demo/main.cpp src/pytorch_interface.cpp)

target_link_libraries(pytorch "${TORCH_LIBRARIES}")
target_link_libraries(pytorch "${OpenCV_LIBS}")

set_property(TARGET pytorch PROPERTY CXX_STANDARD 11)
set_property(TARGET pytorch PROPERTY OUTPUT_NAME pytorch)
