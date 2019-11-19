from __future__ import print_function

import argparse
import sys

import torch.optim as optim

sys.path.append('./auxiliary/')
from auxiliary.dataset_nd import *
from auxiliary.ndmodel import *
from auxiliary.my_utils import *
import os
import visdom

# =============PARAMETERS======================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--num_points', type=int, default=25, help='number of input points')
parser.add_argument('--lrate', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.8, help='alpha in ndiv loss')
parser.add_argument('--env', type=str, default='chamfer_ndiv_random_25')
parser.add_argument('--fixed', action='store_true', help='input fixed grid rather than random sampled points')
parser.add_argument('--ndiv', action='store_true', help='use ndiv loss')
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
dir_name = os.path.join('./results/%s' % opt.env)
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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
print('training set', len(dataset))
len_dataset = len(dataset)
# ========================================================== #

# ===================CREATE network================================= #
network = ND_Map(num_points=opt.num_points, nb_primitives=1)
network = torch.nn.DataParallel(network, device_ids=[0])
network.cuda()  # put network on GPU
network.apply(weights_init)  # initialization of the weight

# if opt.model != '':
#    network.load_state_dict(torch.load(opt.model))
#    print(" Previous weight loaded ")
# ========================================================== #

# ===================CREATE optimizer================================= #
optimizer = optim.Adam(network.parameters(), lr=opt.lrate)

# =============start of the learning loop ======================================== #
# TRAIN MODE
network.train()
sqrt_num_points = int(np.sqrt(opt.num_points))
grid_x = torch.arange(100).unsqueeze(0).expand(100, 100).contiguous()
grid_y = grid_x.transpose(1, 0).contiguous()
grid_test = torch.cat([grid_x.view(-1, 1), grid_y.view(-1, 1)], dim=-1).cuda().float()
grid_test = (grid_test / (grid_test.max() - grid_test.min())) * 2 - 1
color_2d_test = torch.ceil(((grid_test.data.cpu() + 1) / 2) * 255).long()
color_2d_test = torch.cat([color_2d_test, torch.ones_like(color_2d_test[:, :1]) * 133], dim=1)
color_2d_test = color_2d_test.data.numpy()

grid_x_train = torch.arange(sqrt_num_points).unsqueeze(0).expand(sqrt_num_points, sqrt_num_points).contiguous()
grid_y_train = grid_x_train.transpose(1, 0).contiguous()
grid_train = torch.cat([grid_x_train.view(-1, 1), grid_y_train.view(-1, 1)], dim=-1).cuda().float()
grid_train = (grid_train / (grid_train.max() - grid_train.min())) * 2 - 1
color_2d_train = torch.ceil(((grid_train.data.cpu() + 1) / 2) * 255).long()
color_2d_train = torch.cat([color_2d_train, torch.ones_like(color_2d_train[:, :1]) * 133], dim=1)
color_2d_train = color_2d_train.data.numpy()

# learning rate schedule
for i, data in enumerate(dataloader, 0):
    loss_net = torch.ones(1).cuda()
    step = -1
    points, file_name = data
    file_name = file_name[0].split('.points.ply')[0].split('/')[-1]
    vis = visdom.Visdom(env=(opt.env + '_' + file_name))

    # points=data[1]
    points = points.cuda().contiguous().squeeze()  # points.size: [1,10000,3]
    recons = []
    best_model = network.state_dict()
    best_step = step
    best_loss = loss_net
    while (step <= opt.nepoch):
        target_points = points
        color_3d_target = torch.ceil(((target_points.data.cpu() + 1) / 2) * 255).long().data.numpy()

        if opt.fixed:
            input = grid_train
        else:
            sample_index = np.random.randint(low=0, high=points.shape[0], size=opt.num_points)
            sample_points = torch.rand_like(points[sample_index, :][:, 0:2]) * 2 - 1
            input = sample_points
            color_2d_train = torch.ceil(((sample_points.data.cpu() + 1) / 2) * 255).long()
            color_2d_train = torch.cat([color_2d_train, color_2d_train[:, :1]], dim=1)
            color_2d_train = color_2d_train.data.numpy()

        # optimize each object
        optimizer.zero_grad()
        # END SUPER RESOLUTION
        pointsReconstructed = network(input)  # 2500,3
        dist1, dist2 = yifan_dischamfer(target_points, pointsReconstructed)
        dist1 = torch.mean(dist1)
        dist2 = torch.mean(dist2)
        chamfer = dist1 + dist2
        ndiv = 0.01 * compute_pairwise_divergence(grid_train, pointsReconstructed)

        if opt.ndiv:
            loss_net = chamfer + ndiv
        else:
            loss_net = chamfer

        if loss_net < best_loss:
            best_model = network.state_dict()
            best_step = step

        loss_net.backward()
        optimizer.step()

        step += 1

        vis.line(
            X=step * torch.ones(1).cpu(),
            Y=dist1.item() * torch.ones(1).cpu(),
            win='loss_chamfer_train_target',
            update='append',
            opts=dict(xlabel='iters',
                      ylabel='loss_chamfer_target_recon',
                      title='loss_chamfer_train_target',
                      legend=['loss_chamfer_train_target'])

        )
        vis.line(
            X=step * torch.ones(1).cpu(),
            Y=dist2.item() * torch.ones(1).cpu(),
            win='loss_chamfer_train_recon',
            update='append',
            opts=dict(xlabel='iters',
                      ylabel='loss_chamfer_recon_target',
                      title='loss_chamfer_train_recon',
                      legend=['loss_chamfer_train_recon'])
        )
        vis.line(
            X=step * torch.ones(1).cpu(),
            Y=chamfer.item() * torch.ones(1).cpu(),
            win='loss_chamfer_total',
            update='append',
            opts=dict(xlabel='iters',
                      ylabel='loss_chamfer_total',
                      title='loss_chamfer_total',
                      legend=['loss_chamfer_total'])
        )

        if opt.ndiv:
            vis.line(
                X=step * torch.ones(1).cpu(),
                Y=ndiv.item() * torch.ones(1).cpu(),
                win='loss_ndiv*0.01',
                update='append',
                opts=dict(xlabel='iters',
                          ylabel='loss_ndiv*0.01',
                          title='loss_ndiv*0.01',
                          legend=['loss_ndiv*0.01'])
            )
        vis.line(
            X=step * torch.ones(1).cpu(),
            Y=loss_net.item() * torch.ones(1).cpu(),
            win='loss_total',
            update='append',
            opts=dict(xlabel='minibatches',
                      ylabel='loss_total',
                      title='loss_total',
                      legend=['loss_total'])
        )

        # VIZUALIZE
        if step % 100 <= 0:
            vis.scatter(X=target_points.data.cpu(),
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
            vis.scatter(X=input.data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            markercolor=color_2d_train,
                            title="INPUT",
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
            vis.scatter(X=pointsReconstructed.data.cpu(),
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            markercolor=color_2d_train,
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
            with torch.no_grad():
                gridReconstructed = network(grid_test)
            vis.scatter(X=gridReconstructed.data.cpu(),
                        win='grid_RECONSTRUCTED',
                        opts=dict(
                            markercolor=color_2d_test,
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
            vis.scatter(X=grid_test.data.cpu().numpy(),
                        win='grid_INPUT',
                        opts=dict(
                            markercolor=color_2d_test,
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
            recons.append(np.concatenate([gridReconstructed.data.cpu().numpy(), grid_test.data.cpu().numpy()], axis=1))
            print(
                '[object id:%d,step: %d] train loss: %f   chamfer_ab: %f   chamfer_ba: %f    chamfer_total: %f    ndiv: %f   ' % (
                    i, step, loss_net.item(), dist1.item(), dist2.item(), chamfer.item(), ndiv.item()))

    # save last network
    print('saving net...')
    torch.save({'state': best_model, 'steps': best_step}, './results/%s/%s.pth' % (opt.env, file_name))
    np.save('./results/%s/%s.npy' % (opt.env, file_name), np.array(recons))
    print('Training finished: ', file_name)
    print('----------------------------------')
