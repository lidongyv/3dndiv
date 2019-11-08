from __future__ import print_function
import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
from my_utils import *


class ShapeNet(data.Dataset):
	def __init__(self,root,npoint=2500):
		self.root=root
		self.npoints=npoint
		self.files=os.listdir(root)
		self.names=[]
		for i in self.files:
			if i.endswith('.ply'):
				self.names.append(i)
	def __getitem__(self, index):
		name=os.path.join(self.root,self.names[index])
		mystring = my_get_n_random_lines(name, n = self.npoints)
		point_set = np.loadtxt(mystring).astype(np.float32)
		point_set = point_set[:,0:3]
		point_set = torch.from_numpy(point_set)
		#10000,3
		# print(name)
		return point_set.contiguous(), name


	def __len__(self):
		return len(self.names)



if __name__  == '__main__':

	print('Testing Shapenet dataset')
	d  =  ShapeNet(class_choice =  None, balanced= False, train=True, npoints=2500)
	a = len(d)
	d  =  ShapeNet(class_choice =  None, balanced= False, train=False, npoints=2500)
	a = a + len(d)
	print(a)
	
