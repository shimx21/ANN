# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import OrderedDict
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum = 0.1):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.eps = 1e-5

		# Parameters
		self.weight = Parameter(torch.empty(num_features))
		self.bias = Parameter(torch.empty(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)

	def forward(self, input: torch.Tensor):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			batch_mean = input.mean(dim=(0,2,3))
			batch_var  = input.var(dim=(0,2,3))
			self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
			self.running_var  = self.running_var  * (1 - self.momentum) + batch_var * self.momentum
		else:
			batch_mean = self.running_mean
			batch_var  = self.running_var
		vsize = (1, self.num_features, 1, 1)
		return self.weight.view(vsize) * (input - batch_mean.view(vsize))/torch.sqrt(batch_var.view(vsize) + self.eps) + self.bias.view(vsize)
	# TODO END

class Dropout2d(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout2d, self).__init__()
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			mask = torch.bernoulli(torch.ones(input.shape[0:2]) * (1 - self.p)).view(input.shape[0],input.shape[1],1,1).to(input.device)
			return input * mask /(1 - self.p)
		return input
	# TODO END


class Model(nn.Module):
	class Settings:
		def __init__(
			self,
			conv_chnl = 64,
			conv_kern = 3,
			conv_stride = 1,
			conv_padd = 0,

			pool_kern = 2,
			pool_stride = 1,
			pool_padd = 0,

			momentum = 0.1,
			drop_rate = 0.5,

			disable_bn = True,
			disable_drop = True,
		):
			self.conv_chnl = conv_chnl
			self.conv_kern = conv_kern
			self.conv_stride = conv_stride
			self.conv_padd = conv_padd

			self.pool_kern = pool_kern
			self.pool_stride = pool_stride
			self.pool_padd = pool_padd

			self.momentum = momentum
			self.drop_rate = drop_rate

			self.disable_bn = disable_bn
			self.disable_drop = disable_drop

	def __init__(self, in_dim = [3, 32, 32], out_dim = 10, settings:list[Settings] = [Settings(), Settings()]):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		in_chnl = [in_dim[0]] + [s.conv_chnl for s in settings[:-1]]
		linear_in = in_dim[1:]
		for i in range(len(settings)):
			for k in [0, 1]:
				linear_in[k] += 2 * settings[i].conv_padd - settings[i].conv_kern
				linear_in[k] //= settings[i].conv_stride
				linear_in[k] += 1

				linear_in[k] += 2 * settings[i].pool_padd - settings[i].pool_kern
				linear_in[k] //= settings[i].pool_stride
				linear_in[k] += 1


		self.cnn = nn.Sequential(
			OrderedDict([(f"conv{i}", nn.Sequential(
				nn.Conv2d(
					in_chnl[i], 
					settings[i].conv_chnl, 
					settings[i].conv_kern, 
					settings[i].conv_stride,
					settings[i].conv_padd,
				),
				BatchNorm2d(
					settings[i].conv_chnl,
					settings[i].momentum,
				) if settings[i].disable_bn else Dropout2d(0),
				nn.ReLU(),
				Dropout2d(
					settings[i].drop_rate, 
					
				) if settings[i].disable_drop else Dropout2d(0),
				nn.MaxPool2d(
					settings[i].pool_kern, 
					settings[i].pool_stride,
					settings[i].pool_padd,
				)
			)) for i in range(len(settings))] +
			[
				("flatten", nn.Flatten(1)),
				("linear", nn.Linear(settings[-1].conv_chnl * linear_in[0] * linear_in[1], out_dim))
			])
		)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):	
		# TODO START
		# the 10-class prediction output is named as "logits"
		self.to(x.device)
		logits = self.cnn(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
