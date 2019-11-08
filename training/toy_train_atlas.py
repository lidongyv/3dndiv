from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append('./auxiliary/')
from toy_data import *
from model import *
from my_utils import *
from ply import *
import os
import json
import time, datetime
import visdom

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=1200, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 2500,  help='number of points')
parser.add_argument('--nb_primitives', type=int, default = 25,  help='number of primitives in the atlas')
parser.add_argument('--super_points', type=int, default = 2500,  help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="toyatlas"   ,  help='visdom environment')
parser.add_argument('--accelerated_chamfer', type=int, default =0   ,  help='use custom build accelarated chamfer')

opt = parser.parse_args()
print (opt)
# ========================================================== #

# =============DEFINE CHAMFER LOSS======================================== #
if opt.accelerated_chamfer:
	sys.path.append("./extension/")
	import dist_chamfer as ext
	distChamfer =  ext.chamferDist()

else:
	def pairwise_dist(x, y):
		xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
		rx = (xx.diag().unsqueeze(0).expand_as(xx))
		ry = (yy.diag().unsqueeze(0).expand_as(yy))
		P = (rx.t() + ry - 2*zz)
		return P


	def NN_loss(x, y, dim=0):
		dist = pairwise_dist(x, y)
		values, indices = dist.min(dim=dim)
		return values.mean()


	def distChamfer(a,b):
		x,y = a,b
		bs, num_points, points_dim = x.size()
		xx = torch.bmm(x, x.transpose(2,1))
		yy = torch.bmm(y, y.transpose(2,1))
		zz = torch.bmm(x, y.transpose(2,1))
		diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
		rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
		ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
		P = (rx.transpose(2,1) + ry - 2*zz)
		return torch.min(P, 1)[0], torch.min(P, 2)[0], torch.min(P, 1)[1], torch.min(P, 2)[1]
# ========================================================== #

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
vis = visdom.Visdom(port = 8888, env=opt.env)
now = datetime.datetime.now()
blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10
# ========================================================== #

dir_name=os.path.join('./toy')
if not os.path.isdir(dir_name):
	os.mkdir(dir_name)
# ===================CREATE DATASET================================= #
#Create train/test dataloader
dataset = ToyData(npoints = 2500)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
										  shuffle=True, num_workers=int(opt.workers))

print('training set', len(dataset))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
network1 = AE_AtlasNet_SPHERE(num_points = 2500, nb_primitives = 1)
network2 = AE_AtlasNet(num_points = 2500, nb_primitives = 1)
network3 = AE_AtlasNet(num_points = 2500, nb_primitives = 25)

network1.cuda() #put network on GPU
network1.apply(weights_init) #initialization of the weight
network2.cuda() #put network on GPU
network2.apply(weights_init) #initialization of the weight
network3.cuda() #put network on GPU
network3.apply(weights_init) #initialization of the weight
if opt.model != '':
	network1.load_state_dict(torch.load(opt.model))
	print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001 #learning rate
optimizer1 = optim.Adam(network1.parameters(), lr = lrate)
optimizer2 = optim.Adam(network2.parameters(), lr = lrate)
optimizer3 = optim.Adam(network3.parameters(), lr = lrate)

labels_generated_points = torch.Tensor(range(1, (opt.nb_primitives+1)*(opt.num_points//opt.nb_primitives)+1)).view(opt.num_points//opt.nb_primitives,(opt.nb_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.nb_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
# ========================================================== #
save_data=[]
# =============start of the learning loop ======================================== #
for epoch in range(opt.nepoch):
	#TRAIN MODE
	network1.train()
	network2.train()
	network3.train()
	# learning rate schedule
	if epoch==100:
		optimizer1 = optim.Adam(network1.parameters(), lr = lrate/10.0)
		optimizer2 = optim.Adam(network2.parameters(), lr = lrate/10.0)
		optimizer3 = optim.Adam(network3.parameters(), lr = lrate/10.0)
	for i, data in enumerate(dataloader, 0):
		optimizer1.zero_grad()
		optimizer2.zero_grad()
		optimizer3.zero_grad()
		img, points= data
		points = points.transpose(2,1).contiguous()
		points = points.cuda()
		#SUPER_RESOLUTION optionally reduce the size of the points fed to PointNet
		points = points[:,:,:opt.super_points].contiguous()
		#END SUPER RESOLUTION
		pointsReconstructed1  = network1(points) #forward pass
		pointsReconstructed2  = network2(points) #forward pass
		pointsReconstructed3  = network3(points) #forward pass
		dist1, dist2,_,_ = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed1) #loss function
		loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
		loss_net.backward()
		optimizer1.step() #gradient update
		dist1, dist2,_,_ = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed2) #loss function
		loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
		loss_net.backward()
		optimizer2.step() #gradient update
		dist1, dist2,_,_ = distChamfer(points.transpose(2,1).contiguous(), pointsReconstructed3) #loss function
		loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
		loss_net.backward()
		optimizer3.step() #gradient update


		save_data.append([points.transpose(2,1).contiguous()[0].data.cpu().numpy(),pointsReconstructed1[0].data.cpu().numpy(),pointsReconstructed2[0].data.cpu().numpy() \
			,pointsReconstructed3[0].data.cpu().numpy()])
		# VIZUALIZE
		if i%1 <= 0:
			vis.scatter(X = points.transpose(2,1).contiguous()[0].data.cpu(),
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
			vis.scatter(X = pointsReconstructed1[0].data.cpu(),
					win = 'TRAIN_INPUT_RECONSTRUCTED1',
					opts = dict(
						title="Atlas_Sphere",
						markersize=2,
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
			vis.scatter(X = pointsReconstructed2[0].data.cpu(),
					win = 'TRAIN_INPUT_RECONSTRUCTED2',
					opts = dict(
						title="Atlas_Patch",
						markersize=2,
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
			vis.scatter(X = pointsReconstructed3[0].data.cpu(),
					win = 'TRAIN_INPUT_RECONSTRUCTED3',
					opts = dict(
						title="Atlas_Patch_25",
						markersize=2,
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
		print('[%d: %d/%d] train loss:  %f ' %(epoch, i, len_dataset/32, loss_net.item()))

	#save last network
	print('saving net...')
	torch.save(network1.state_dict(), '%s/network1.pth' % (dir_name))
	torch.save(network2.state_dict(), '%s/network2.pth' % (dir_name))
	torch.save(network2.state_dict(), '%s/network3.pth' % (dir_name))
	np.save('%s/atlas_data' % (dir_name),np.array(save_data))
