cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sampleFasterRCNN)

FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)

STRING (TOLOWER ${CMAKE_SYSTEM_NAME} LOWER_SYSTEM_NAME)
SET(CUDA_TARGET_INCLUDE  ${CUDA_TOOLKIT_ROOT_DIR}-${CUDA_VERSION}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-${LOWER_SYSTEM_NAME}/include)

include_directories(../../../thirdparty/TensorRT/include)
include_directories(../../../thirdparty/TensorRT/samples/common)

link_directories(
    ../../../thirdparty/TensorRT/lib
)

cuda_add_executable (sampleFasterRCNN
	../../../tensorRT/detection/sampleFasterRCNN.cpp
	../../../tensorRT/detection/factoryFasterRCNN.h
	../../../thirdparty/TensorRT/include/NvInfer.h
	../../../thirdparty/TensorRT/include/NvOnnxParser.h
)

target_link_libraries(sampleFasterRCNN 
${CUDA_LIBRARIES}
nvinfer
nvinfer_plugin
nvonnxparser
nvparsers
)


