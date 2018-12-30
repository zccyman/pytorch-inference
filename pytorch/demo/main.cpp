#include "pytorch_interface.h"

#include "torch/script.h"
#include "torch/torch.h"

#include <iostream>
#include <memory>

int main(int argc, char **argv)
{
	std::string model_path = string(argv[1]);
	std::string image_path = string(argv[2]);
	
	int total_images = 1024;
	int batch_size = 32;
	int test_times = total_images / batch_size;

	int mode = DEVICE_TYPE::GPU;
	std::vector<int> devices_id;
	devices_id.push_back(0);

	std::shared_ptr<torch::jit::script::Module> model;
	read_model(model_path, devices_id, mode, model);

	//

	//

	cv::Mat image = cv::imread(image_path, 1);
	if (!image.empty())
	{
		std::cout << "read image success..." << std::endl;
		std::cout << "nchannels: " << image.channels() << std::endl;
	}

	clock_t start = clock();

	std::vector<std::vector<float>> probs;
	std::vector<cv::Mat> images;
	int tensor_size[3] = { 3, 224, 224 };
	for (int i = 0; i < test_times; i++)
	{
		images.resize(0);
		probs.resize(0);

		for (int j = 0; j < batch_size; j++)
		{
			images.push_back(image);
		}
		predict(images, model, tensor_size, mode, probs);
	}
	
	std::cout << "elapsed time: " << (clock() - start) / double(total_images) << " ms" << std::endl;

	//results of the last batch
	std::cout << "probs.size(): " << probs.size() << std::endl;
	for (int i = 0; i < probs.size(); i++)
	{
		auto ptr = std::max_element(probs[i].begin(), probs[i].end());
		float max_score = *ptr;
		int label = std::distance(probs[i].begin(), ptr);
		std::cout << "image_idx: " << i << " " \
			<< "max_score: " << max_score << " " \
			<< "label: " << label << std::endl;
	}
	
	return 0;
}
