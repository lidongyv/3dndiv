from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append('./auxiliary/')
from dataset_nd import *
from toy_data import *
from ndmodel import *
from my_utils import *
from ply import *
from surface_loss import *
import os
import json
import time, datetime
import visdom

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=1200, help='number of epochs to train for')
parser.add_argument('--model', type=str, default = '',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 100,  help='number of points')
parser.add_argument('--nb_primitives', type=int, default = 1,  help='number of primitives in the atlas')
parser.add_argument('--super_points', type=int, default = 100,  help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default ="surface"   ,  help='visdom environment')
parser.add_argument('--accelerated_chamfer', type=int, default =0   ,  help='use custom build accelarated chamfer')
parser.add_argument('--ngpu', type=int, default = 1,  help='number of gpus')
opt = parser.parse_args()
print (opt)
# ========================================================== #


def custom_dischamfer(x,y):
	xx=torch.sum(torch.pow(x,2),dim=1)
	yy=torch.sum(torch.pow(y,2),dim=1)
	xy=torch.matmul(x,y.transpose(1,0))
	
	xx=xx.unsqueeze(1).expand_as(xy)
	yy=yy.unsqueeze(0).expand_as(xy)
	dist=torch.sqrt(xx+yy-2*xy+1e-6)
	return torch.min(dist,dim=0)[0],torch.min(dist,dim=1)[0]

# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
vis = visdom.Visdom(env=opt.env)
dir_name =  os.path.join('chamfer')
if not os.path.exists(dir_name):
	os.mkdir(dir_name)

opt.manualSeed = random.randint(1, 10000) # fix seed
opt.manualSeed = 7269 # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10
# ========================================================== #

# ===================CREATE DATASET================================= #
#Create train/test dataloader
dataset = ShapeNet(root='./data/test',npoint=10000)
#dataset = ToyData(npoints = 2500)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
network = ND_Map(num_points = opt.num_points, nb_primitives = opt.nb_primitives)
network=torch.nn.DataParallel(network,device_ids=range(opt.ngpu))
network.cuda() #put network on GPU
network.apply(weights_init) #initialization of the weight

if opt.model != '':
	network.load_state_dict(torch.load(opt.model))
	print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001 #learning rate
optimizer = optim.Adam(network.parameters(), lr = lrate)
# ========================================================== #


#initialize learning curve on visdom, and color for each primitive in visdom display
train_curve = []

# ========================================================== #

# =============start of the learning loop ======================================== #
#TRAIN MODE
network.train()
grid_x=torch.arange(50).unsqueeze(0).expand(50,50).contiguous()
grid_y=grid_x.transpose(1,0).contiguous()
grid=torch.cat([grid_x.view(-1,1),grid_y.view(-1,1)],dim=-1).cuda().float()
grid=(grid-25)/25
color_g=torch.ceil(((grid.data.cpu()+1)/2)*255).long()
color_g=torch.cat([color_g,torch.ones_like(color_g[:,:1])*133],dim=1)
color_g=color_g.data.numpy()

# learning rate schedule
for i, data in enumerate(dataloader, 0):
	loss_net=torch.ones(1).cuda()
	step=-1
	points, file_name = data
	file_name=file_name[0].split('.points.ply')[0].split('/')[-1]

	# points=data[1]
	points = points.cuda().contiguous().squeeze()
	#10000,3
	recons_dense=[]
	recons_sparse=[]
	loss=[]
	alpha=1.5
	vis.scatter(X = points.data.cpu(),
			win = 'dense_points',
			opts = dict(
				title = "dense_points",
				markersize = 3,
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
	target_points=points
	s_x=torch.arange(int(np.sqrt(opt.num_points))).unsqueeze(0). \
	expand(np.sqrt(opt.num_points).astype(np.int),np.sqrt(opt.num_points). \
	astype(np.int)).contiguous()
	s_y=s_x.transpose(1,0).contiguous()
	sample_points=torch.cat([s_x.view(-1,1),s_y.view(-1,1)],dim=-1).cuda().float()
	sample_points=(sample_points-np.sqrt(opt.num_points)/2)/(np.sqrt(opt.num_points)/2)
	color=torch.ceil(((sample_points.data.cpu()+1)/2)*255).long()
	color=torch.cat([color,color[:,:1]],dim=1)
	color=color.data.numpy()
	color_t=torch.ceil(((target_points.data.cpu()+1)/2)*255).long().data.numpy()
	size=2/np.sqrt(opt.num_points)
	mid_pos=init_middle(sample_points,size)

	while(loss_net.item()>5*1e-3 and step<1e4):
		#optimize each object
		optimizer.zero_grad()
		#END SUPER RESOLUTION
		pointsReconstructed  = network(sample_points) 
		dist1, dist2= custom_dischamfer(target_points, pointsReconstructed) #loss function
		#dist1, dist2,_,_ = distChamfer(target_points, pointsReconstructed) #loss function
		loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
		# loss_ndiv=loss_net
		loss_ndiv=NDiv_loss_surface(sample_points,pointsReconstructed,mid_pos, \
			alpha=alpha,mode=0)
		loss_train=loss_net+loss_ndiv
		loss_train.backward()
		optimizer.step()
		step+=1
		# alpha=torch.max(loss_net.detach()/loss_ndiv.detach(),100)
		vis.line(
			X=step*torch.ones(1).cpu(),
			Y=loss_net.item()*torch.ones(1).cpu(),
			win='chamfer',
			update='append',
			opts=dict(xlabel='minibatches',
						ylabel='chamfer Loss',
						title='chamfer Loss',
						legend=['chamfer Loss'])
		
			)
		vis.line(
			X=step*torch.ones(1).cpu(),
			Y=loss_ndiv.item()*torch.ones(1).cpu(),
			win='ndiv',
			update='append',
			opts=dict(xlabel='minibatches',
						ylabel='ndiv Loss',
						title='ndiv Loss',
						legend=['ndiv Loss'])
		
			)
		loss.append([loss_net.item(),loss_ndiv.item()])
		# VIZUALIZE
		if step%100 ==0:
			
			vis.scatter(X = target_points.data.cpu(),
					win = 'TRAIN_Target',
					opts = dict(
						title = "TRAIN_Target",
						markersize = 3,
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
			vis.scatter(X = pointsReconstructed.data.cpu(),
					win = 'TRAIN_INPUT_RECONSTRUCTED',
					opts = dict(
						markercolor=color,
						title="TRAIN_INPUT_RECONSTRUCTED",
						markersize = 3,
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
			vis.scatter(X = sample_points.data.cpu().numpy(),
					win = 'TRAIN_INPUT',
					opts = dict(
						markercolor=color,
						title="TRAIN_INPUT",
						markersize = 4,
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
			with torch.no_grad():
				gridReconstructed  = network(grid)
			vis.scatter(X = gridReconstructed.data.cpu(),
					win = 'grid_RECONSTRUCTED',
					opts = dict(
						markercolor=color_g,
						title="grid_RECONSTRUCTED",
						markersize = 3,
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
			vis.scatter(X = grid.data.cpu().numpy(),
					win = 'grid_INPUT',
					opts = dict(
						markercolor=color_g,
						title="grid_INPUT",
						markersize = 4,
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

			recons_dense.append(np.concatenate([ \
				# target_points.data.cpu().numpy(), \
				# pointsReconstructed.data.cpu().numpy(), \
				# sample_points.data.cpu().numpy(), \
				gridReconstructed.data.cpu().numpy(), \
				# grid.data.cpu().numpy() \
			],axis=1))
			recons_sparse.append(np.concatenate([ \
				pointsReconstructed.data.cpu().numpy(), \
			],axis=1))
		print('[object id:%d,step: %d] chamfer loss: %f loss_ndiv: %f' %(i,step, loss_net.item(),loss_ndiv.item()))
	
	#save last network
	print('saving net...')
	torch.save({'state':network.state_dict(),'steps':step}, './results/surface/%s.pth' % (file_name))
	np.save('./results/surface/dense_%s.npy'%(file_name),np.array(recons_dense))
	np.save('./results/surface/sparse_%s.npy'%(file_name),np.array(recons_sparse))
	np.save('./results/surface/loss_%s.npy'%(file_name),np.array(loss))
	print(file_name)
