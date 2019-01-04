#coding:utf-8
import torch
import torchvision
from torch.autograd import Variable
import torchvision.models as models
import torch.onnx
import onnx
from onnx import version_converter, helper

#from onnx_tf.backend import prepare
#from onnx_tf.frontend import tensorflow_graph_to_onnx_model
#import tensorflow as tf

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

def transformModel_onnx(model_file, inference_file, tensor_size):
	#model = torchvision.models.vgg16(pretrained=True)
	model = torch.load(model_file)
	model = model.cuda()

	dummy_input = Variable(torch.randn(1, 3, 224, 224)).cuda()

	#pytorch2onnx
	torch.onnx.export(model, dummy_input, inference_file, export_params=True, verbose=True, training=False)
	
	original_model = onnx.load(inference_file)#opset9

	onnx.checker.check_model(original_model)
	print('The model is checked!')

	#opset9 -> opset7
	converted_model = version_converter.convert_version(original_model, 7)
	onnx.checker.check_model(converted_model)
	print('The model is checked!')

	# #onnx2tensorflow
	# tf_rep = prepare(original_model)
	# tf_rep.export_graph('checkpoint/tf.pb')

	# #tensorflow2onnx
	# with tf.gfile.GFile("checkpoint/tf.pb", "rb") as f:
	# 	graph_def = tf.GraphDef()
	# 	graph_def.ParseFromString(f.read())
	# 	onnx_model = tensorflow_graph_to_onnx_model(graph_def,
	# 									"fc2/add",
	# 									opset=6)

	# 	file = open(inference_file, "wb")
	# 	file.write(onnx_model.SerializeToString())
	# 	file.close()


if __name__ == "__main__":
	is_python3 = True
	model_file = "checkpoint/inference_temp.pth"
	inference_file = "checkpoint/inference.pth"
	tensor_size = [1, 3, 224, 224]
	if is_python3:
		transformModel_python3(model_file, inference_file, tensor_size)
	else:
		transformModel_python2(model_file, inference_file, tensor_size)
	
	inference_file = "checkpoint/inference.onnx"	
	transformModel_onnx(model_file, inference_file, tensor_size)
	
	

