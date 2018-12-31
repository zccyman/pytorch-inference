#include "pytorch_softmax.hpp"

// Softmax
std::vector<float> pytorch_cpu_softmax(std::vector<float> unnorm_probs)
{
  int n_classes = unnorm_probs.size();

  // 1. Partition function
  float log_sum_of_exp_unnorm_probs = 0.0f;
  for (auto& n : unnorm_probs) 
  {
    log_sum_of_exp_unnorm_probs += std::exp(n);
  }
  log_sum_of_exp_unnorm_probs = std::log(log_sum_of_exp_unnorm_probs);

  // 2. normalize
  std::vector<float> probs(n_classes);
  for (int class_idx = 0; class_idx != n_classes; class_idx++) 
  {
	probs[class_idx] = std::exp(unnorm_probs[class_idx] - log_sum_of_exp_unnorm_probs);
  }

  return probs;
}
