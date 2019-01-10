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

#### Inference onnx model
```
cd makefile/tensorRT/classification
mkdir build && cd build && cmake -A x64 ..

or

mkdir build && cd build && cmake -G "Visual Studio 15 2017 Win64" ..
set Environment -> path=%path%;../../../../thirdparty/TensorRT/lib;
```

#### Inference caffe model for faster-rcnn
```
cd makefile/tensorRT/detection
mkdir build && cd build && cmake -A x64 ..

or

mkdir build && cd build && cmake -G "Visual Studio 15 2017 Win64" ..
set Environment -> path=%path%;../../../../thirdparty/TensorRT/lib;
```
download [VGG16_faster_rcnn_final.caffemodel](https://github.com/Sephora-M/master-thesis/tree/master/py-faster-rcnn/data/faster_rcnn_models)

#### Thirdparty

	thirdparty/
		- libtorch  
		- opencv 
		- CUDA
		- TensorRT
		
download thirdparty from [here](https://pan.baidu.com/s/1NxACM1coAXthmXizXKyhow#list/path=%2Fgithub%2Fpublic%2Fpytorch-inference&parentPath=%2Fgithub/thirdparty.zip).

## Docker
```
docker pull zccyman/deepframe
nvidia-docker run -it --name=mydocker zccyman/deepframe /bin/bash
cd workspace && git clone https://github.com/zccyman/pytorch-inference.git
```

## Environment

- Windows10
- VS2017
- CMake3.13
- CUDA10.0
- CUDNN7.3
- Pyton3.5
- ONNX1.1.2
- TensorRT5.0.1
- Pytorch1.0
- [Libtorch](https://download.pytorch.org/libtorch/cu100/libtorch-win-shared-with-deps-latest.zip)
- [OpenCV4.0.1](https://opencv.org/releases.html)

## Done

- Support train and transform pytorch model

- Support multi-batch inference pytorch model in C++

- Support cpu and gpu softmax

- Support transform pytorch model to ONNX model, and inference onnx model using tensorRT

- Support inference caffe model for faster-rcnn using tensorRT

## Todo

- build classification network

- compress pytorch model

- object detection pytorch inference using C++ on Window platforms


## Notes

- "torch.jit.trace" doesn’t support nn.DataParallel so far.

- TensorRT doesn’t supports opset 7 above so far, but pyTorch ONNX exporter seems to export opset 9.

	
## Acknowledgement

- https://github.com/Wizaron/pytorch-cpp-inference

- https://zhuanlan.zhihu.com/p/52154049

- [Caffe](https://github.com/BVLC/caffe)

- [Pytorch](https://github.com/pytorch/pytorch)

- [TensorRT](https://developer.nvidia.com/tensorrt)
