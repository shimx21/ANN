# -*- coding: utf-8 -*-
import sys
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Model
from load_data import load_cifar_2d

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100,
	help='Batch size for mini-batch training and evaluating. Default: 100')
parser.add_argument('--num_epochs', type=int, default=20,
	help='Number of training epoch. Default: 20')
parser.add_argument('--learning_rate', type=float, default=1e-3,
	help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--drop_rate', type=float, default=0.5,
	help='Drop rate of the Dropout Layer. Default: 0.5')
parser.add_argument('--is_train', type=bool, default=True,
	help='True to train and False to inference. Default: True')
parser.add_argument('--data_dir', type=str, default='../cifar-10_data',
	help='Data directory. Default: ../cifar-10_data')
parser.add_argument('--train_dir', type=str, default='./train',
	help='Training directory for saving model. Default: ./train')
parser.add_argument('--inference_version', type=int, default=0,
	help='The version for inference. Set 0 to use latest checkpoint. Default: 0')
# Other 
parser.add_argument('--weight_decay', type=float, default=0.001,
	help='Weight Decay. Default = 0.001')
parser.add_argument('--hid_num', type=int, default=256,
	help='Number of nodes in hidden layer. Default = 256')
parser.add_argument('--momentum', type=float, nargs="+", default=[0.1, 0.1],
	help='Momentum of BatchNorm for calculating running. Default = [0.1, 0.1]')
parser.add_argument('--disable_bn', type=bool, default=False,
	help='True to disable BatchNorm layers. Default = False')
parser.add_argument('--disable_drop', type=bool, default=False,
	help='True to disable Dropout layers. Default = False')
# Wandb Support
parser.add_argument('--name', type=str, required=True,
	help='Name of this wandb run')
parser.add_argument('--wandb', type=bool, default=False,
	help='True to open wandb log')
args = parser.parse_args()
print(args)
if args.wandb:
	import wandb
	wandb.init(
		project="ANN-HW2",
		name=args.name,
		config={"command": sys.argv, **vars(args)}
	)

def shuffle(X, y, shuffle_parts):
	chunk_size = int(len(X) / shuffle_parts)
	shuffled_range = list(range(chunk_size))

	X_buffer = np.copy(X[0:chunk_size])
	y_buffer = np.copy(y[0:chunk_size])

	for k in range(shuffle_parts):
		np.random.shuffle(shuffled_range)
		for i in range(chunk_size):
			X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
			y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

		X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
		y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

	return X, y


def train_epoch(model, X, y, optimizer): # Training Process
	model.train()
	loss, acc = 0.0, 0.0
	st, ed, times = 0, args.batch_size, 0
	while st < len(X) and ed <= len(X):
		optimizer.zero_grad()
		X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
		loss_, acc_ = model(X_batch, y_batch)

		loss_.backward()
		optimizer.step()

		loss += loss_.cpu().data.numpy()
		acc += acc_.cpu().data.numpy()
		st, ed = ed, ed + args.batch_size
		times += 1
	loss /= times
	acc /= times
	return acc, loss


def valid_epoch(model, X, y): # Valid Process
	model.eval()
	loss, acc = 0.0, 0.0
	st, ed, times = 0, args.batch_size, 0
	while st < len(X) and ed <= len(X):
		X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
		loss_, acc_ = model(X_batch, y_batch)

		loss += loss_.cpu().data.numpy()
		acc += acc_.cpu().data.numpy()

		st, ed = ed, ed + args.batch_size
		times += 1
	loss /= times
	acc /= times
	return acc, loss


def inference(model, X): # Test Process
	model.eval()
	pred_ = model(torch.from_numpy(X).to(device))
	return pred_.cpu().data.numpy()


def deterministic(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
	deterministic(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if not os.path.exists(args.train_dir):
		os.mkdir(args.train_dir)
	if args.is_train:
		X_train, X_test, y_train, y_test = load_cifar_2d(args.data_dir)
		X_val, y_val = X_train[40000:], y_train[40000:]
		X_train, y_train = X_train[:40000], y_train[:40000]
		mlp_model = Model(
			drop_rate=args.drop_rate, 
			disable_bn=args.disable_bn, 
			disable_drop=args.disable_drop
		)
		mlp_model.to(device)
		print(mlp_model)
		
		params_num = sum(p.numel() for p in mlp_model.parameters())
		print(params_num)

		optimizer = optim.Adam(mlp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

		# model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
		# if os.path.exists(model_path):
		# 	mlp_model = torch.load(model_path)

		pre_losses = [1e18] * 3
		best_val_acc = 0.0
		for epoch in range(1, args.num_epochs+1):
			start_time = time.time()
			train_acc, train_loss = train_epoch(mlp_model, X_train, y_train, optimizer)
			X_train, y_train = shuffle(X_train, y_train, 1)

			val_acc, val_loss = valid_epoch(mlp_model, X_val, y_val)

			if val_acc >= best_val_acc:
				best_val_acc = val_acc
				best_epoch = epoch
				test_acc, test_loss = valid_epoch(mlp_model, X_test, y_test)
				# with open(os.path.join(args.train_dir, 'checkpoint_{}.pth.tar'.format(epoch)), 'wb') as fout:
				# 	torch.save(mlp_model, fout)
				# with open(os.path.join(args.train_dir, 'checkpoint_0.pth.tar'), 'wb') as fout:
				# 	torch.save(mlp_model, fout)

			epoch_time = time.time() - start_time
			print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
			print("  learning rate:                 " + str(optimizer.param_groups[0]['lr']))
			print("  training loss:                 " + str(train_loss))
			print("  training accuracy:             " + str(train_acc))
			print("  validation loss:               " + str(val_loss))
			print("  validation accuracy:           " + str(val_acc))
			print("  best epoch:                    " + str(best_epoch))
			print("  best validation accuracy:      " + str(best_val_acc))
			print("  test loss:                     " + str(test_loss))
			print("  test accuracy:                 " + str(test_acc))

			if args.wandb:
				wandb.log({
					"lr": optimizer.param_groups[0]['lr'],
					"train/loss": train_loss,
					"train/acc": train_acc,
					"val/loss": val_loss,
					"val/acc": val_acc
				}, step=epoch)
			
			if train_loss > max(pre_losses):
				for param_group in optimizer.param_groups:
					param_group['lr'] = param_group['lr'] * 0.9995
			pre_losses = pre_losses[1:] + [train_loss]

	else:
		mlp_model = Model(
			drop_rate=args.drop_rate,
			hid_dim=args.hid_num,
			disable_bn=args.disable_bn,
			disable_drop=args.disable_drop
		)
		mlp_model.to(device)
		model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
		if os.path.exists(model_path):
			mlp_model = torch.load(model_path)

		X_train, X_test, y_train, y_test = load_cifar_2d(args.data_dir)

		count = 0
		for i in range(len(X_test)):
			test_image = X_test[i].reshape((1, 3 * 32 * 32))
			result = inference(mlp_model, test_image)[0]
			if result == y_test[i]:
				count += 1
		print("test accuracy: {}".format(float(count) / len(X_test)))
