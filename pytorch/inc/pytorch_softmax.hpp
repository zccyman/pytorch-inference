#ifndef _PYTORCH_SOFTMAX_HPP_
#define _PYTORCH_SOFTMAX_HPP_

#include <vector>

std::vector<float> softmax(std::vector<float> unnorm_probs);

template <typename Dtype>
int pytorch_gpu_softmax(Dtype *array, int size);

#endif !_PYTORCH_SOFTMAX_HPP_