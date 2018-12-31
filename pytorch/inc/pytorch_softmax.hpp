#ifndef _PYTORCH_SOFTMAX_HPP_
#define _PYTORCH_SOFTMAX_HPP_

#include <vector>

int pytorch_cpu_softmax(std::vector<float> &probs);

int pytorch_gpu_softmax(std::vector<float> &array);

#endif !_PYTORCH_SOFTMAX_HPP_