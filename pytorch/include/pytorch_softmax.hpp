#ifndef _PYTORCH_SOFTMAX_H_
#define _PYTORCH_SOFTMAX_H_

#include <vector>

std::vector<float> softmax(std::vector<float> unnorm_probs);

void pytorch_gpu_softmax(float *array, int size);

#endif !_PYTORCH_SOFTMAX_H_