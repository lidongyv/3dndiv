# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-18 20:53:24  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-18 21:44:1

import numpy as np
import torch
import torch.nn.functional as F
import os
import time

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
	return torch.clamp(dist,min=1e-6)
	# if len(x.shape) == 2:
	# 	matrix = torch.norm(x[:,None,:] - x[None,:,:], p = 2, dim = 2)
	# elif len(x.shape) == 3:
	# 	matrix = torch.norm(x[:,:,None,:] - x[:,None,:,:], p = 2, dim = 3)
	# #be attention i do not use the norm
	# return matrix

def middle_position(i,j,size):
	#current is just the simplest version
	#u can try to add more middle steps then
	pi=np.array([i//size,i%size])
	pj=np.array([j//size,j%size])
	if pi[1]>pj[1]:
		pj+=pi
		pi=pj-pi
		pj=pj-pi
	if pi[0]>pj[0]:
		return pi[0]*size+pj[1]
	else:
		return pj[0]*size+pi[1]

def init_middle(x,size):
	pos=[]
	if os.path.exists('./auxiliary/sample/size%s.npy'%(str(size))):
		pos=np.load('./auxiliary/sample/size%s.npy'%(str(size)))
	else:
		for i in range(x.shape[0]):
			for j in range(x.shape[0]):
				middle=middle_position(i,j,size)
				pos.append([i,middle,j])
		np.save('./auxiliary/sample/size%s.npy'%(str(size)),pos)
	return np.array(pos)
def compute_norm_pairwise_distance(x,mid_pos):
	''' computation of normalized pairwise distance matrix
	---- Input
	- x: input tensor	torch.Tensor (sample_number,2)
	---- Return
	- matrix: output matrix	torch.Tensor [sample_num, sample_num]
	'''

	x_pair_dist = compute_pairwise_distance(x).view(-1)
	# size=np.sqrt(x.shape[0])
	# connection=torch.zeros_like(x_pair_dist)
	# only compute the pair on the grid, usless
	# for i in range(x.shape[0]):
	# 	for j in range(x.shape[0]):
	# 		if i//size==j//size or i%size==j%size:
	# 			connection=1
	# dist_straight=x_pair_dist*connection
	surface_dist=torch.zeros_like(x_pair_dist)
	#2s for this, too slow
	# for i in range(x.shape[0]):
	# 	for j in range(x.shape[0]):
	# 		middle=torch.tensor(middle_position(i,j,size)).to(x.device).long()
			# surface_dist[i,j]=x_pair_dist[i,middle]+x_pair_dist[middle,j]
	surface_dist=x_pair_dist[mid_pos[:,:2]]+x_pair_dist[mid_pos[:,1:]]
	normalizer = torch.sum(surface_dist, dim = -1,keepdim=True)
	x_norm_pair_dist = surface_dist / (normalizer + 1e-12).detach()


	# x_pair_dist = compute_pairwise_distance(x)
	# normalizer = torch.sum(x_pair_dist, dim = -1)
	# x_norm_pair_dist = x_pair_dist / (normalizer[...,None] + 1e-12).detach()

	return x_norm_pair_dist

def NDiv_loss_surface(x, y,mid_pos, alpha=1,mode=2):
	''' NDiv loss function.
	---- Input
	- x: (sample_number,2)
	#x is the 2d grid, the shortest path the min 2d
	- y: (sample_number,3)
	#y is the 3d points, the corresponidng to 2d is set by index
	- loss: normalized diversity loss.
	'''
	#original ndiv 0.00043s
	#costum ndiv   0.00063s
	#surface ndiv 2.3s
	#speed up 0.002s
	a=time.time()
	x=x.view(-1,2)
	y=y.view(-1,3)
	size=2/np.sqrt(x.shape[0])
	# mid_pos=init_middle(x,size)
	S = x.shape[0]
	x_norm_pair_dist = compute_norm_pairwise_distance(x,mid_pos)
	y_norm_pair_dist = compute_norm_pairwise_distance(y,mid_pos)
	
	if mode==-1:
		return 0*torch.mean(x)
	if mode==0:
		ndiv_loss_matrix = torch.abs(x_norm_pair_dist - y_norm_pair_dist)
	if mode==1:
		ndiv_loss_matrix = F.relu(y_norm_pair_dist-x_norm_pair_dist * alpha )
	if mode==2:
		ndiv_loss_matrix = F.relu(x_norm_pair_dist * alpha - y_norm_pair_dist)
	if mode==3:
		ndiv_loss_matrix =torch.clamp(torch.abs(x_norm_pair_dist - y_norm_pair_dist),min=0.1*size)
	if mode==4:
		ndiv_loss_matrix = F.relu(x_norm_pair_dist * alpha - y_norm_pair_dist)
	ndiv_loss = ndiv_loss_matrix.sum(-1).sum(-1) / (S * (S - 1))
	print(time.time()-a)
	return ndiv_loss

if __name__  == '__main__':
	x=torch.rand(100,2)
	y=torch.rand(100,3)
	loss=NDiv_loss_surface(x,y)
