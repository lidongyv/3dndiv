# -*- coding: utf-8 -*- 
#@Author: Lidong Yu   
#@Date: 2019-11-18 21:59:23  
#@Last Modified by: Lidong Yu  
#@Last Modified time: 2019-11-18 21:59:23

#this version is for the offline analyze of connection and shape

import numpy as np
import torch
import torch.nn.functional as F
import os
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

def compute_norm_pairwise_distance(x,dis,length,straight):
	''' computation of normalized pairwise distance matrix
	---- Input
	- x: input tensor	torch.Tensor (sample_number,2)
	---- Return
	- matrix: output matrix	torch.Tensor [sample_num, sample_num]
	'''
	x_pair_dist = compute_pairwise_distance(x)

	#direct distance only compute the points distance on the same row or colume
	x_pair_dist_direct = x_pair_dist*straight
	
	for i in range(x_pair_dist.shape[0]):
		for j in range(x_pair_dist.shape[1]):
			x_pair_dist[i,j]=torch.sum(x_pair_dist_direct[dis[i,j,:-1,0],dis[i,j,1:,1]])/length[i,j]
	normalizer = torch.sum(x_pair_dist, dim = -1,keepdim=True)
	x_norm_pair_dist = x_pair_dist / (normalizer + 1e-12).detach()

	return x_norm_pair_dist

def grid_dis(x1,y1,x2,y2,steps):
	if y1>y2:
		x1=x2+x1
		x2=x1-x2
		x1=x1-x2
		y1=y2+y1
		y2=y1-y2
		y1=y1-y2

	sequence=torch.ones(2*int((2+steps)/steps),2)
	sequence[:,0]=sequence[:,0]*x2
	sequence[:,1]=sequence[:,1]*y2
	if x1<=x2:
		x_range=torch.arange(x1,x2+1)
	else:
		x_range=torch.flip(torch.arange(x2,x1+1).view(1,-1),dims=[0,1]).view(-1)
	y_range=torch.arange(y1,y2+1)
	sequence[:x_range.shape[0],0]=x_range
	sequence[:x_range.shape[0],1]=y1
	sequence[x_range.shape[0]:x_range.shape[0]+y_range.shape[0],0]=x2
	sequence[x_range.shape[0]:x_range.shape[0]+y_range.shape[0],1]=y_range
	return sequence,x_range.shape[0]+y_range.shape[0]
def shortest_2d_sample(steps=0.1,device=0,straight=True):
	#x is a 2d grid with shape n*2
	#y is the output path with n*n*(k*2), k the the points from the shortest points
	#between i,j
	if os.path.exists(('./sample/%s.npy')%(str(steps))):
		dis=np.load(('./sample/%s.npy')%(str(steps)))
		dis=torch.from_numpy(dis).to(device)
	else:
		x=torch.arange(start=-1.0,end=1+steps,step=steps,device=device,requires_grad=False).view(-1,1)
		x=x.expand(x.shape[0],x.shape[0])
		y=x.transpose(1,0)
		#longest path on grid between diagnoal points have length with width+height
		dis=torch.zeros(x.shape[0],x.shape[1],x.shape[0],x.shape[1],x.shape[0]+x.shape[1],2)
		#whwh(w+h)*2
		dis_length=torch.zeros(x.shape[0],x.shape[1],x.shape[0],x.shape[1])
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				for m in range(x.shape[0]):
					for n in range(x.shape[1]):
						dis[i,j,m,n],dis_length[i,j]=grid_dis(i,j,m,n,steps)
		np.save('./sample/dis%s.npy'%(str(steps)),dis.cpu().data.numpy())
		np.save('./sample/len%s.npy'%(str(steps)),dis_length.cpu().data.numpy())
	if straight:
		dis_staright=torch.zeros(x.shape[0],x.shape[1],x.shape[0],x.shape[1])
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				for m in range(x.shape[0]):
					for n in range(x.shape[1]):
						if i-j==0 or m-n==0:
							dis_staright[i,j,m,n]=1
		dis=dis*dis_staright[...,None,None]
		dis_length=dis_length*dis_staright
	return dis,dis_length,dis_staright

def NDiv_loss_surface(x, y, alpha=0.8):
	''' NDiv loss function.
	---- Input
	- x: (sample_number,2)
	#x is the 2d grid, the shortest path the min 2d
	- y: (sample_number,3)
	#y is the 3d points, the corresponidng to 2d is set by index
	- loss: normalized diversity loss.
	'''

	dis,length,straight=shortest_2d_sample(steps=0.25,device=x.device)
	x=x.view(-1,2)
	y=y.view(-1,3)
	dis=dis.view(x.shape[0],y.shape[0],2*x.shape[0],2)
	length=length.view(x.shape[0],y.shape[0],1)
	straight=straight.view(x.shape[0],y.shape[0],1)
	S = x.shape[0]
	x_norm_pair_dist = compute_norm_pairwise_distance(x,dis,length,straight)
	y_norm_pair_dist = compute_norm_pairwise_distance(y)
	ndiv_loss_matrix = F.relu(x_norm_pair_dist * alpha - y_norm_pair_dist)
	ndiv_loss = ndiv_loss_matrix.sum(-1).sum(-1) / (S * (S - 1))

	return ndiv_loss

if __name__  == '__main__':
	a,b,c=shortest_2d_sample(steps=0.25,device=torch.ones(1).device)
