from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from my_utils import *
import visdom

class ToyData(data.Dataset):
	def __init__(self, train = True, npoints = 2500):

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

		self.transforms = transforms.Compose([
							 transforms.Resize(size =  224, interpolation = 2),
							 transforms.ToTensor(),
							 # normalize,
						])
		self.num=npoints
		self.mode=train

	def __getitem__(self, index):
		if self.mode:
			point_set=[]
			radius=3
			theta=np.repeat(np.linspace(0, 2.*np.pi, 180),360)
			phi=np.tile(np.linspace(0, 2.*np.pi, 360),180)
			number=self.num

			in_radius=0.5
			x=(radius+in_radius*np.cos(theta))*np.cos(phi)
			y=(radius+in_radius*np.cos(theta))*np.sin(phi)
			z=in_radius*np.sin(theta)
			point_set=np.array([x,y,z])
			point_set=torch.from_numpy(point_set).view(3,180*360).permute(1,0)/3.5
			np.random.seed(np.random.randint(1))
			index=np.random.randint(size=number,low=0,high=180*360)
			data = 0
			return data, point_set[index].float()
			# number=self.num
			# point_set=[]
			# x=np.array([np.random.rand(1000)*2-1,np.random.rand(1000)*2-1, \
			# 	np.random.rand(1000)*2-1,np.random.rand(1000)*2-1, 
			# 	np.ones(1000),-np.ones(1000)]).reshape(-1)
			# y=np.array([np.random.rand(1000)*2-1,np.random.rand(1000)*2-1, \
			# 	np.ones(1000),-np.ones(1000), \
			# 	np.random.rand(1000)*2-1,np.random.rand(1000)*2-1]).reshape(-1)
			# z=np.array([np.ones(1000),-np.ones(1000), \
			# 	np.random.rand(1000)*2-1,np.random.rand(1000)*2-1, \
			# 	np.random.rand(1000)*2-1,np.random.rand(1000)*2-1]).reshape(-1)
			# point_set=np.array([x,y,z])
			# point_set=torch.from_numpy(point_set).view(3,6000).permute(1,0)
			# index=np.random.randint(size=number,low=0,high=6000)
			# data = 0
			# return data, point_set[index].float()
		else:
			if index==0:
				radius=1
				point_set=[]
				theta=np.repeat(np.linspace(0, 2.*np.pi, 180),180)
				phi=np.tile(np.linspace(0, 2.*np.pi, 180),180)
				number=self.num
				x=(radius*np.cos(theta))*np.cos(phi)
				y=(radius*np.cos(theta))*np.sin(phi)
				z=radius*np.sin(theta)
				point_set=np.array([x,y,z])
				point_set=torch.from_numpy(point_set).view(3,180*180).permute(1,0)
				np.random.seed(np.random.randint(1))
				index=np.random.randint(size=number,low=0,high=180*180)
				data = 0
				return data, point_set[index].float()
			else:
				number=self.num
				point_set=[]
				x=np.array([np.random.rand(1000)*2-1,np.random.rand(1000)*2-1, \
					np.random.rand(1000)*2-1,np.random.rand(1000)*2-1, 
					np.ones(1000),-np.ones(1000)]).reshape(-1)
				y=np.array([np.random.rand(1000)*2-1,np.random.rand(1000)*2-1, \
					np.ones(1000),-np.ones(1000), \
					np.random.rand(1000)*2-1,np.random.rand(1000)*2-1]).reshape(-1)
				z=np.array([np.ones(1000),-np.ones(1000), \
					np.random.rand(1000)*2-1,np.random.rand(1000)*2-1, \
					np.random.rand(1000)*2-1,np.random.rand(1000)*2-1]).reshape(-1)
				point_set=np.array([x,y,z])
				point_set=torch.from_numpy(point_set).view(3,6000).permute(1,0)
				index=np.random.randint(size=number,low=0,high=6000)
				data = 0
				return data, point_set[index].float()

	def __len__(self):
		if self.mode:
			return 3200
		else:
			return 32

if __name__  == '__main__':
	data=  ToyData(npoints=2500,train=False)
	points=data.__getitem__(1)[1]
	print(points.shape)
	vis = visdom.Visdom(port = 8888, env='datatoy')
	vis.scatter(X = points,
			win = 'TRAIN_INPUT',
			opts = dict(
				title = "TRAIN_INPUT",
				markersize = 2,
				xtickmin=-1,
				xtickmax=1,
				xtickstep=0.5,
				ytickmin=-1,
				ytickmax=1,
				ytickstep=0.5,
				ztickmin=-1,
				ztickmax=1,
				ztickstep=0.5,
				),
			)