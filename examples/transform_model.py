#coding:utf-8
import torch
import torchvision
from functools import partial
import pickle

def transformModel_python2(model_file, inference_file, tensor_size):
	pickle.load = partial(pickle.load, encoding="latin1")
	pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
	
	model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)
	#print(model)
	#input("")
	model = model.cuda()
	model.eval()

	input_tensor = torch.rand(tensor_size).cuda()

	traced_script_module = torch.jit.trace(model, input_tensor)

	traced_script_module.save(inference_file)

def transformModel_python3(model_file, inference_file, tensor_size):
	model = torch.load(model_file)
	#print(model)
	#input("")
	model = model.to(torch.device("cuda"))
	model.eval()

	input_tensor = torch.rand(tensor_size).to(torch.device("cuda"))

	traced_script_module = torch.jit.trace(model, input_tensor)

	traced_script_module.save(inference_file)
	
if __name__ == "__main__":
	is_python3 = True
	model_file = "checkpoint/inference_temp.pth"
	inference_file = "checkpoint/inference.pth"
	tensor_size = [1, 3, 224, 224]
	if is_python3:
		transformModel_python3(model_file, inference_file, tensor_size)
	else:
		transformModel_python2(model_file, inference_file, tensor_size)
	
	

