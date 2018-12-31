#include "pytorch_interface.h"
#include "pytorch_softmax.hpp"

//preprocess
static int normalize(cv::Mat &image, const std::vector<float> &mean, const std::vector<float> &std)
{
	image = image / 255.0;

#if !(JUST_ZERO_ONE_NORMALIZE)
	std::vector<cv::Mat> img_channels(3);
	cv::split(image, img_channels);
	for (int ch = 0; ch < 3; ch++)
	{
		cv::Mat channel_mean(cv::Size(image.cols, image.rows), CV_32FC1, mean[ch]);
		cv::subtract(img_channels[ch], channel_mean, img_channels[ch]);
		img_channels[ch] /= std[ch];
	}
	cv::merge(img_channels, image);
#endif

	return 0;
}

static int preprocess(cv::Mat &image, int new_height, int new_width, \
	const std::vector<float> &mean, const std::vector<float> &std)
{
	if (1 == image.channels())
	{
		cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
	}
	else// Convert from BGR to RGB
	{
		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	}

	// Resize image
	cv::resize(image, image, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);

	image.convertTo(image, CV_32FC3);

	normalize(image, mean, std);

	return 0;
}//preprocess

int read_model(const std::string &model_file, const std::vector<int> &devices_id, int mode, std::shared_ptr<torch::jit::script::Module> &model)
{
	model = torch::jit::load(model_file);
	if (DEVICE_TYPE::GPU == mode)
	{
		model->to(at::kCUDA);
	}
	else
	{
		model->to(at::kCPU);
	}

	return 0;
}

int predict(const std::vector<cv::Mat> &images, const std::shared_ptr<torch::jit::script::Module> &model, \
	int *tensor_size, int mode, std::vector<std::vector<float>> &probs)
{
	for (int i = 0; i < images.size(); i++)
	{
		if (images[i].empty())
		{
			//LOG(INFO) << "WARNING: Cannot read image!" << std::endl;
			std::cout << "WARNING: Cannot read image!" << std::endl;
			return -1;
		}
	}

	//images_to_tensors
	int batch_size = images.size();
	static int n_channels = tensor_size[0];
	static int height = tensor_size[1];
	static int width = tensor_size[2];//resize 

	std::vector<int64_t> dims = { 1, height, width, n_channels };
	std::vector<int64_t> permute_dims = { 0, 3, 1, 2 };

	static std::vector<float> mean = { 0.485f, 0.456f, 0.406f};
	static std::vector<float> std = { 0.229f, 0.224f, 0.225f };

	std::vector<torch::Tensor> images_to_tensors;
	for (int i = 0; i < batch_size; i++)
	{
		cv::Mat image = images[i].clone();
		preprocess(image, height, width, mean, std);

		torch::Tensor image_to_tensor;
		torch::TensorOptions options(torch::kFloat32);//kUInt8 kFloat32 kFloat64
		image_to_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();

		image_to_tensor = image_to_tensor.permute(torch::IntList(permute_dims));
		image_to_tensor = image_to_tensor.toType(torch::kFloat32);
		images_to_tensors.push_back(image_to_tensor);
	}

	//inference
	torch::Tensor input_tensors = torch::cat(images_to_tensors, 0);
	std::vector<torch::jit::IValue> inputs;
	if (DEVICE_TYPE::GPU == mode)
	{
		inputs.push_back(input_tensors.to(at::kCUDA));
	}
	else
	{
		inputs.push_back(input_tensors.to(at::kCPU));
	}

	torch::Tensor results = model->forward(inputs).toTensor();
	//std::cout << "results: " << results << std::endl;
	if (DEVICE_TYPE::GPU == mode)
	{
		results = results.to(at::kCPU);//inference
	}
	//std::cout << results.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

	//softmax
	torch::IntList sizes = results.sizes();
	int n_samples = sizes[0];
	int n_classes = sizes[1];
	for (int nidx = 0; nidx < n_samples; nidx++)
	{
		std::vector<float> unnorm_probs(results.data<float>() + (nidx * n_classes),
			results.data<float>() + ((nidx + 1) * n_classes));

		//if (DEVICE_TYPE::GPU == mode)
		if (0)
		{
			pytorch_gpu_softmax(unnorm_probs);
			probs.push_back(unnorm_probs);
		}
		else
		{
			pytorch_cpu_softmax(unnorm_probs);
			probs.push_back(unnorm_probs);
		}

	}

	return 0;
}
