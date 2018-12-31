cmake_minimum_required(VERSION 3.13)
SET(CUDA_VERSION 10.0)
FIND_PACKAGE(CUDA ${CUDA_VERSION} REQUIRED)
STRING (TOLOWER ${CMAKE_SYSTEM_NAME} LOWER_SYSTEM_NAME)
SET(CUDA_TARGET_INCLUDE  ${CUDA_TOOLKIT_ROOT_DIR}-${CUDA_VERSION}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-${LOWER_SYSTEM_NAME}/include)

include_directories(../../thirdparty/CUDA/common/inc)
include_directories(../../pytorch/inc)

cuda_add_executable (cuda_demo
	../../pytorch/cu/cuda_demo.cpp
	../../pytorch/cu/pytorch_softmax.cu
	../../pytorch/inc/pytorch_softmax.hpp
	../../pytorch/src/pytorch_softmax.cpp
)

TARGET_LINK_LIBRARIES(cuda_demo ${CUDA_LIBRARIES})


