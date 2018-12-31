#include "pytorch_softmax.hpp"
#include <vector>
#include <iostream>

int main(int argc, char **argv) {
	std::vector<float> uniform;
    for (int i = 0; i < 3; i++) {
        uniform.push_back(float(i + 0.1f));
		std::cout << uniform[i] << std::endl;
    }
	std::cout << std::endl;

	pytorch_gpu_softmax(uniform.data(), 3);
	
	for (int i = 0; i < 3; i++) {
		std::cout << uniform[i] << std::endl;
	}

	return 0;
}