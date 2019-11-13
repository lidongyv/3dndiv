from __future__ import print_function

import argparse
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('./auxiliary/')
from dataset_nd import *
from toy_data import *
from ndmodel import *
from my_utils import *
import os
import visdom

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=1200, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--num_points', type=int, default=25, help='number of points')
parser.add_argument('--nb_primitives', type=int, default=1, help='number of primitives in the atlas')
parser.add_argument('--super_points', type=int, default=2500,
                    help='number of input points to pointNet, not used by default')
parser.add_argument('--env', type=str, default="chamfer", help='visdom environment')
parser.add_argument('--accelerated_chamfer', type=int, default=0, help='use custom build accelarated chamfer')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus')
opt = parser.parse_args()
print(opt)


# ========================================================== #
def pair_dist_matrix(x):
    return torch.norm(x[None, :, :] - x[:, None, :], p=2, dim=2)


def compute_pair_distance(x):
    x_pair_dist_matrix = pair_dist_matrix(x)
    x_pair_dist_sum = torch.sum(x_pair_dist_matrix, dim=1)
    x_pair_dist_normalized = x_pair_dist_matrix / x_pair_dist_sum[..., None].detach()
    return x_pair_dist_normalized


def compute_pairwise_divergence(space2d, space3d):
    space2d_dist_matrix = compute_pair_distance(space2d)
    space3d_dist_matrix = compute_pair_distance(space3d)
    div = F.relu(space2d_dist_matrix * 0.8 - space3d_dist_matrix)
    return div.sum()


def custom_dischamfer(x, y):
    xx = torch.sum(torch.pow(x, 2), dim=1)
    yy = torch.sum(torch.pow(y, 2), dim=1)
    xy = torch.matmul(x, y.transpose(1, 0))

    xx = xx.unsqueeze(0).expand_as(xy)
    yy = yy.unsqueeze(0).expand_as(xy)
    dist = xx.transpose(1, 0) + yy - 2 * xy
    return torch.min(dist, dim=0)[0], torch.min(dist, dim=1)[0]


def yifan_dischamfer(a, b):
    # when a and b are in different size
    dist_ab = torch.norm((a[:, None, :] - b), p=2, dim=2)
    dist_ba = torch.norm((b[:, None, :] - a), p=2, dim=2)
    return torch.min(dist_ab, dim=1)[0], torch.min(dist_ba, dim=1)[0]


# =============DEFINE stuff for logs ======================================== #
# Launch visdom for visualization
# vis = visdom.Visdom(env=opt.env)
vis = visdom.Visdom()
dir_name = os.path.join('chamfer')
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

opt.manualSeed = random.randint(1, 10000)  # fix seed
opt.manualSeed = 7269  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
best_val_loss = 10
# ========================================================== #

# ===================CREATE DATASET================================= #
# Create train/test dataloader
dataset = ShapeNet(root='./data/test', npoint=10000)
# dataset = ToyData(npoints = 2500)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
network = ND_Map(num_points=opt.num_points, nb_primitives=opt.nb_primitives)
network = torch.nn.DataParallel(network, device_ids=range(opt.ngpu))
network.cuda()  # put network on GPU
network.apply(weights_init)  # initialization of the weight

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)
# ========================================================== #


# initialize learning curve on visdom, and color for each primitive in visdom display
train_curve = []

# ========================================================== #

# =============start of the learning loop ======================================== #
# TRAIN MODE
network.train()
grid_x = torch.arange(100).unsqueeze(0).expand(100, 100).contiguous()
grid_y = grid_x.transpose(1, 0).contiguous()
grid = torch.cat([grid_x.view(-1, 1), grid_y.view(-1, 1)], dim=-1).cuda().float()
grid = (grid - 50) / 50
color_g = torch.ceil(((grid.data.cpu() + 1) / 2) * 255).long()
color_g = torch.cat([color_g, torch.ones_like(color_g[:, :1]) * 133], dim=1)
color_g = color_g.data.numpy()

input_grid_x = torch.arange(5).unsqueeze(0).expand(5, 5).contiguous()
input_grid_y = input_grid_x.transpose(1, 0).contiguous()
input_grid = torch.cat([input_grid_x.view(-1, 1), input_grid_y.view(-1, 1)], dim=-1).cuda().float()
input_grid = (input_grid - 2.5) / 2.5
color = torch.ceil(((input_grid.data.cpu() + 1) / 2) * 255).long()
color = torch.cat([color, torch.ones_like(color[:, :1]) * 133], dim=1)
color = color.data.numpy()
# learning rate schedule
for i, data in enumerate(dataloader, 0):
    loss_net = torch.ones(1).cuda()
    step = -1
    points, file_name = data
    file_name = file_name[0].split('.points.ply')[0].split('/')[-1]

    # points=data[1]
    points = points.cuda().contiguous().squeeze()  # points.size: [1,10000,3]
    recons = []
    while (loss_net.item() > 5 * 1e-3):
        # sample_index = np.random.randint(low=0, high=points.shape[0], size=opt.num_points)
        # target_points = points[sample_index, :]
        # target_points = points # yfwu: do not do downsample on the original shapes
        # sample_points = torch.rand_like(target_points[:, 0:2]) * 2 - 1
        target_points = points

        # color = torch.ceil(((sample_points.data.cpu() + 1) / 2) * 255).long()
        # color = torch.cat([color, color[:, :1]], dim=1)
        # color = color.data.numpy()
        color_t = torch.ceil(((target_points.data.cpu() + 1) / 2) * 255).long().data.numpy()

        # optimize each object
        optimizer.zero_grad()
        # END SUPER RESOLUTION
        pointsReconstructed = network(input_grid)  # 2500,3
        dist1, dist2 = yifan_dischamfer(target_points, pointsReconstructed)  # loss function
        # dist1, dist2,_,_ = distChamfer(target_points, pointsReconstructed) #loss function
        loss_net = (torch.mean(dist1)) + (torch.mean(dist2))
        # loss_net = (torch.mean(dist1)) + (torch.mean(dist2)) + 0.01*compute_pairwise_divergence(input_grid, pointsReconstructed)
        loss_net.backward()
        optimizer.step()
        step += 1
        # VIZUALIZE
        if step % 100 <= 0:
            vis.scatter(X=points.data.cpu(),
                        win='Target',
                        opts=dict(
                            title="Target",
                            markersize=3,
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
            # vis.scatter(X=target_points.data.cpu(),
            #            win='TRAIN_Target',
            #            opts=dict(
            #                title="TRAIN_Target",
            #                markersize=3,
            #                xtickmin=-1,
            #                xtickmax=1,
            #                xtickstep=0.5,
            #                ytickmin=-1,
            #                ytickmax=1,
            #                ytickstep=0.5,
            #                ztickmin=-1,
            #                ztickmax=1,
            #                ztickstep=0.5,
            #            ),
            #            )
            vis.scatter(X=pointsReconstructed.data.cpu(),
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            markercolor=color,
                            title="TRAIN_INPUT_RECONSTRUCTED",
                            markersize=3,
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
            vis.scatter(X=input_grid.data.cpu().numpy(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            markercolor=color,
                            title="TRAIN_INPUT",
                            markersize=4,
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
                gridReconstructed = network(grid)
            vis.scatter(X=gridReconstructed.data.cpu(),
                        win='grid_RECONSTRUCTED',
                        opts=dict(
                            markercolor=color_g,
                            title="grid_RECONSTRUCTED",
                            markersize=3,
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
            vis.scatter(X=grid.data.cpu().numpy(),
                        win='grid_INPUT',
                        opts=dict(
                            markercolor=color_g,
                            title="grid_INPUT",
                            markersize=4,
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

        print('[object id:%d,step: %d] train loss: %f' % (i, step, loss_net.item()))

    # save last network
    print('saving net...')
    torch.save({'state': network.state_dict(), 'steps': step}, './chamfer/%s.pth' % (file_name))
    np.save('./results/chamfer/%s.npy' % (file_name), np.array(recons))
    print(file_name)
