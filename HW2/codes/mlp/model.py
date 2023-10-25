# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum = 0.1):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.eps = 1e-5

		# Parameters
		self.weight = Parameter(torch.ones(num_features))
		self.bias = Parameter(torch.zeros(num_features))

		# Store the average mean and variance
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)
		
	def forward(self, input: torch.Tensor):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			batch_mean = input.mean(dim=0)
			batch_var  = input.var(dim=0)
			self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
			self.running_var  = self.running_var  * (1 - self.momentum) + batch_var * self.momentum
		else:
			batch_mean = self.running_mean
			batch_var  = self.running_var
		
		return self.weight * (input - batch_mean)/torch.sqrt(batch_var + self.eps) + self.bias
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		self.p = p

	def forward(self, input: torch.Tensor):
		# input: [batch_size, num_feature_map * height * width]
		if self.training:
			return input * torch.bernoulli(torch.full(input.size(), 1 - self.p))/(1 - self.p)
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5, in_dim = 3 * 32 * 32, out_dim = 10, hid_dim = 256):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
		self.mlp = nn.Sequential(
			nn.Linear(in_dim, hid_dim),
			BatchNorm1d(hid_dim),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.Linear(hid_dim, out_dim)
		)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.mlp(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
