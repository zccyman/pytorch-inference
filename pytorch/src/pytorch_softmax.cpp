#include "pytorch_softmax.hpp"

// Softmax
int pytorch_cpu_softmax(std::vector<float> &probs)
{
  // 1. Partition function
  float log_sum_of_exp_probs = 0.0f;
  for (auto& n : probs) 
  {
    log_sum_of_exp_probs += std::exp(n);
  }
  log_sum_of_exp_probs = std::log(log_sum_of_exp_probs);

  // 2. normalize
  for (int class_idx = 0; class_idx < probs.size(); class_idx++)
  {
	probs[class_idx] = std::exp(probs[class_idx] - log_sum_of_exp_probs);
  }

  return 0;
}
