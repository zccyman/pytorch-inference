#ifndef _PYTORCH_INTERFACE_H_
#define _PYTORCH_INTERFACE_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "torch/script.h"
#include "torch/torch.h"
#include "ATen/ATen.h"
#include "ATen/Type.h"


#define  JUST_ZERO_ONE_NORMALIZE  (0)

#ifdef PYTORCH_DLL_EXPORT
#define PYTORCH_API __declspec(dllexport)
#else
#define PYTORCH_API __declspec(dllimport)
#endif

typedef enum _DEVICE_TYPE_
{
   CPU = -1,
   GPU = 1
}DEVICE_TYPE;

PYTORCH_API int read_model(const std::string &model_file, const std::vector<int> &devices_id, int mode, \
	std::shared_ptr<torch::jit::script::Module> &model);
PYTORCH_API int predict(const std::vector<cv::Mat> &images, const std::shared_ptr<torch::jit::script::Module> &model, \
	int *tensor_size, int mode, std::vector<std::vector<float>> &probs);

#endif !_PYTORCH_INTERFACE_H_