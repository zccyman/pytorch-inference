//#include "pytorch_softmax.hpp"

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i,n)                      \
for(int i = blockIdx.x * blockDim.x + threadIdx.x; \
i < (n);                                           \
i +=blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) { \
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
  
//template <typename Dtype>
__global__ void softmax_exp0(float *a, int n, float *y) 
{
  CUDA_KERNEL_LOOP(i, n) {
    y[i] = exp(a[i]);
  }
}

const int blockSize = 1024;
__global__ void softmax_sum(const float *gArr, int arraySize, float *gOut) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    float sum = 0;
    for (int i = gthIdx; i < arraySize; i += gridSize)
        sum += gArr[i];
    __shared__ float shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = log(shArr[0]);
}

template <typename Dtype>
__global__ void softmax_exp1(Dtype *a, int n, Dtype *b, Dtype *y) {
  CUDA_KERNEL_LOOP(i, n) {
    y[i] = exp(a[i] - b[0]);
  }
}

void pytorch_gpu_softmax(float *array, int size){
	float *gpudata, *y, *b;
	
	cudaMalloc((void**)&gpudata, sizeof(float) * size);
    cudaMalloc((void**)&y, sizeof(float) * size);
	cudaMalloc((void**)&b, sizeof(float));
	
	cudaMemset(gpudata, 0, sizeof(float) * size);
	cudaMemset(y, 0, sizeof(float) * size);
	cudaMemset(b, 0, sizeof(float));
	
	cudaMemcpy(gpudata, array, sizeof(float) * size, cudaMemcpyHostToDevice);
	
	softmax_exp0 << < CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS >> > (gpudata, size, y);
	softmax_sum << < CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS >> > (y, size, b);
	softmax_exp1 << < CAFFE_GET_BLOCKS(1), CAFFE_CUDA_NUM_THREADS >> > (gpudata, size, b, y);
	
	cudaMemcpy(array, y, sizeof(float) * size, cudaMemcpyDeviceToHost);
	
	cudaFree(gpudata);
    cudaFree(y);
	cudaFree(b);
	
}

int main(int argc, char **argv) 
{
	std::vector<float> uniform;
    for (int i = 0; i < 3; i++) {
        uniform.push_back(float(i + 0.1f));
		std::cout << uniform[i] << std::endl;
    }
	std::cout << std::endl;

	pytorch_gpu_softmax(uniform.data(), 3);
	
	for (int i = 0; i < 3; i++) {
		std::cout << uniform[i] << std::endl;
	}

	return 0;
}