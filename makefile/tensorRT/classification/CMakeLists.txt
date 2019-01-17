cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(classification)

FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)

STRING (TOLOWER ${CMAKE_SYSTEM_NAME} LOWER_SYSTEM_NAME)
SET(CUDA_TARGET_INCLUDE  ${CUDA_TOOLKIT_ROOT_DIR}-${CUDA_VERSION}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-${LOWER_SYSTEM_NAME}/include)

include_directories(../../../thirdparty/TensorRT/include)
include_directories(../../../thirdparty/TensorRT/samples/common)
include_directories(../../../thirdparty/TensorRT/samples/common/windows)

link_directories(
    ../../../thirdparty/TensorRT/lib
)

#sampleOnnxMNIST
cuda_add_executable (sampleOnnxMNIST
	../../../tensorRT/classification/sampleOnnxMNIST.cpp
	../../../thirdparty/TensorRT/include/NvInfer.h
	../../../thirdparty/TensorRT/include/NvOnnxParser.h
)

target_link_libraries(sampleOnnxMNIST 
${CUDA_LIBRARIES}
nvinfer
nvinfer_plugin
nvonnxparser
nvparsers
)

#sampleMNIST
cuda_add_executable (sampleMNIST
	../../../tensorRT/classification/sampleMNIST.cpp
	../../../thirdparty/TensorRT/samples/common/windows/getopt.c
	../../../thirdparty/TensorRT/include/NvInfer.h
	../../../thirdparty/TensorRT/include/NvOnnxParser.h
)

target_link_libraries(sampleMNIST 
${CUDA_LIBRARIES}
nvinfer
nvinfer_plugin
nvonnxparser
nvparsers
)


