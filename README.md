# Serving PyTorch Models in C++ on Windows10 platforms

![Dynamic graph](https://github.com/zccyman/pytorch-inference/blob/master/examples/docs/pytorch-logo-dark.png)

## How to use

#### Prepare Data

	examples/data/train/

		- 0
		- 1
		  .
		  .
		  .
		- n

	examples/data/test/

		- 0
		- 1
		  .
		  .
		  .
		- n


#### Train Model
```
cd examples && python train.py
```

#### Transform Model
```
cd examples && python transform_model.py
```

#### Test Model
```
cd makefile/pytorch
mkdir build && cd build && cmake -A x64 ..

or

mkdir build && cd build && cmake -G "Visual Studio 15 2017 Win64" ..

set Command Arguments -> ..\..\..\examples\checkpoint ..\..\..\examples\images
set Environment -> path=%path%;../../../thirdparty/libtorch/lib;../../../thirdparty/opencv/build/x64/vc15/bin;
```	

#### Test CUDA Softmax
```
cd makefile/cuda
mkdir build && cd build && cmake -A x64 ..

or

mkdir build && cd build && cmake -G "Visual Studio 15 2017 Win64" ..
```	

#### Thirdparty

	thirdpart/
		- libtorch  
		- opencv 
		- CUDA
[here](https://pan.baidu.com/s/1r010qo44nF7BAWB04pGEtg#list/path=%2Fgithub%2Fpytorch-inference/thirdparty.zip)

## Environment

- Windows10
- VS2017
- CMake3.13
- CUDA10.0
- CUDNN7.3
- Pytorch1.0
- [Libtorch](https://download.pytorch.org/libtorch/cu100/libtorch-win-shared-with-deps-latest.zip)
- [OpenCV4.0.1](https://opencv.org/releases.html)

## Done

- train, transform and test pytorch model

- add cuda softmax module


## Todo

- build classification network

- compress pytorch model

- object detection pytorch inference using C++ on Window platforms


## Questions

- "torch.jit.trace" doesn’t support nn.DataParallel so far.

	
## Acknowledgement

- https://github.com/Wizaron/pytorch-cpp-inference

- https://zhuanlan.zhihu.com/p/52154049

- [Caffe](https://github.com/BVLC/caffe)

- [Pytorch](https://github.com/pytorch/pytorch)
