# ----------------------------------------
# Normalized Diversification
# NDiv loss implemented in Pytorch 
# ----------------------------------------
import numpy as np
import torch
import torch.nn.functional as F

def compute_pairwise_distance(x):
	''' computation of pairwise distance matrix
	---- Input
	- x: input tensor	(sample_number,2)
	---- Return
	- matrix: output matrix	torch.Tensor [sample_number,sample_number]
	'''
	y=x
	xx=torch.sum(torch.pow(x,2),dim=1)
	yy=torch.sum(torch.pow(y,2),dim=1)
	xy=torch.matmul(x,y.transpose(1,0))

	xx=xx.unsqueeze(0).expand_as(xy)
	yy=yy.unsqueeze(0).expand_as(xy)
	dist=xx.transpose(1,0)+yy-2*xy
	return dist

def compute_norm_pairwise_distance(x):
	''' computation of normalized pairwise distance matrix
	---- Input
	- x: input tensor	torch.Tensor (sample_number,2)
	---- Return
	- matrix: output matrix	torch.Tensor [sample_num, sample_num]
	'''
	x_pair_dist = compute_pairwise_distance(x)
	normalizer = torch.sum(x_pair_dist, dim = -1,keepdim=True)
	x_norm_pair_dist = x_pair_dist / (normalizer + 1e-12).detach()

	return x_norm_pair_dist

def NDiv_loss(x, y, alpha=1.5):
	''' NDiv loss function.
	---- Input
	- x: (sample_number,2)
	- y: (sample_number,3)
	- alpha: hyperparameter alpha in NDiv loss.
	---- Return
	- loss: normalized diversity loss.
	'''
	x=x.view(x.shape[0],2)
	y=y.view(y.shape[0],3)
	
	S = x.shape[0]
	y_norm_pair_dist = compute_norm_pairwise_distance(y)
	x_norm_pair_dist = compute_norm_pairwise_distance(x)
	
	ndiv_loss_matrix = F.relu(x_norm_pair_dist * alpha - y_norm_pair_dist)
	# print(x_norm_pair_dist)
	# print(y_norm_pair_dist)
	# exit()
	ndiv_loss = torch.sum(ndiv_loss_matrix) / (S * (S - 1))
	#exit()
	return ndiv_loss


def NDiv_loss_surface(x, y, alpha=0.8):
	''' NDiv loss function.
	---- Input
	- x: (sample_number,2)
	- y: (sample_number,3)
	- alpha: hyperparameter alpha in NDiv loss.
	---- Return
	- loss: normalized diversity loss.
	'''
	x=x.view(x.shape[0],2)
	y=y.view(y.shape[0],3)

	S = x.shape[0] # sample number
	y_norm_pair_dist = compute_norm_pairwise_distance(y)
	x_norm_pair_dist = compute_norm_pairwise_distance(x)

	ndiv_loss_matrix = F.relu(x_norm_pair_dist * alpha - y_norm_pair_dist)
	ndiv_loss = ndiv_loss_matrix.sum(-1).sum(-1) / (S * (S - 1))
	#exit()
	return ndiv_loss

