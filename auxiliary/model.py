from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

#UTILITIES
class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans


        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class PointNetfeatNormal(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(PointNetfeatNormal, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans


        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

#OUR METHOD
import resnet

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class SVR_AtlasNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 5, pretrained_encoder = False, cuda=True):
        super(SVR_AtlasNet, self).__init__()
        self.usecuda = cuda
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.usecuda:
                rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            else:
                rand_grid = Variable(torch.FloatTensor(grid[i]))

            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

class AE_AtlasNet(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

#TEST with spheric noise
class AE_AtlasNet_SPHERE(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_SPHERE, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 3 +self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x.size(0),grid.size(1), grid.size(2)).contiguous())
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = grid.expand(x.size(0),grid.size(1), grid.size(2)).contiguous()
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

class SVR_AtlasNet_SPHERE(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1, pretrained_encoder=False):
        super(SVR_AtlasNet_SPHERE, self).__init__()
        self.num_points = num_points
        self.pretrained_encoder = pretrained_encoder

        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 3 +self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def forward(self, x):
        x = x[:,:3,:,:].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = x[:,:3,:,:].contiguous()

        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = Variable(grid.expand(x.size(0),grid.size(1), grid.size(2)).contiguous())
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0,self.nb_primitives):
            grid = grid.contiguous().unsqueeze(0)
            grid = grid.expand(x.size(0),grid.size(1), grid.size(2)).contiguous()
            # print(grid.sizegrid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), grid.size(2)).contiguous()
            y = torch.cat( (grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()


#BASELINE
class PointDecoder(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)
        self.fc1 = nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = nn.Linear(self.bottleneck_size, bottleneck_size//2)
        self.fc3 = nn.Linear(bottleneck_size//2, bottleneck_size//4)
        self.fc4 = nn.Linear(bottleneck_size//4, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points).transpose(1,2).contiguous()
        return x



class PointDecoderNormal(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(PointDecoderNormal, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)
        self.fc1 = nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = nn.Linear(self.bottleneck_size, bottleneck_size//2)
        self.fc3 = nn.Linear(bottleneck_size//2, bottleneck_size//4)
        self.fc4 = nn.Linear(bottleneck_size//4, self.num_points * 6)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 6, self.num_points).transpose(1,2).contiguous()
        return x


class AE_Baseline(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(AE_Baseline, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = PointDecoder(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AE_Baseline_normal(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024):
        super(AE_Baseline_normal, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = nn.Sequential(
        PointNetfeatNormal(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = PointDecoderNormal(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SVR_Baseline(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, pretrained_encoder=False):
        super(SVR_Baseline, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.pretrained_encoder = pretrained_encoder
        self.encoder = resnet.resnet18(pretrained=self.pretrained_encoder,  num_classes=bottleneck_size)
        self.decoder = PointDecoder(num_points = num_points, bottleneck_size = self.bottleneck_size)

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()

        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    # print('testing our method...')
    # sim_data = Variable(torch.rand(1, 3, 400, 400))
    # model = PointNetAE_RNN_grid2mesh()
    # model.cuda()
    # out = model(sim_data.cuda())
    # print(out.size())

    # print('testing baseline...')
    # sim_data = Variable(torch.rand(1, 3, 400, 400))
    # model = PointNetAEBottleneck()
    # model.cuda()
    # out = model(sim_data.cuda())
    # print(out.size())

    print('testing PointSenGet...')
    sim_data = Variable(torch.rand(1, 4, 192, 256))
    model = Hourglass()
    # model.cuda()
    # out = model(sim_data.cuda())
    out = model(sim_data)
    print(out.size())



class AE_AtlasNet_toy(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_toy, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.nb_primitives)])
        self.decoder2=nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size*2) for i in range(0,self.nb_primitives)])

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives)).cuda(x.device)
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(grid[i]))
            rand_grid = rand_grid.transpose(0,1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0),rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_gan(self,x):
        x = self.encoder(x).detach()
        x=torch.randn_like(x).cuda(x.device)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives)).cuda(x.device)
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
    def forward_inference_gan_div(self,x):
        x = self.encoder(x).detach()
        x=torch.randn_like(x).cuda(x.device)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives)).cuda(x.device)
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return x,torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
    def forward_inference_cgan(self, x):
        x = self.encoder(x)
        latent=torch.randn_like(x).cuda(x.device)
        x=torch.cat([x,latent],dim=1)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives)).cuda(x.device)
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder2[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
    def forward_inference_cgan_div(self, x):
        x = self.encoder(x)
        latent=torch.randn_like(x).cuda(x.device)
        x=torch.cat([x,latent],dim=1)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.nb_primitives)).cuda(x.device)
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder2[i](y))
        return latent,torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
#TEST with spheric noise
class AE_AtlasNet_SPHERE_toy(nn.Module):
    def __init__(self, num_points = 2048, bottleneck_size = 1024, nb_primitives = 1):
        super(AE_AtlasNet_SPHERE_toy, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat(num_points, global_feat=True, trans = False),
        nn.Linear(1024, self.bottleneck_size),
        nn.BatchNorm1d(self.bottleneck_size),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 3 +self.bottleneck_size) for i in range(0,self.nb_primitives)])
        self.decoder2 = nn.ModuleList([PointGenCon(bottleneck_size = 3 +self.bottleneck_size*2) for i in range(0,self.nb_primitives)])

    def forward(self, x):
        x = self.encoder(x)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)).cuda(x.device) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()

    def forward_inference_gan(self, x):
        x = self.encoder(x).detach()
        x=torch.randn_like(x).cuda(x.device)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)).cuda(x.device) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
    def forward_inference_gan_div(self, x):
        x = self.encoder(x).detach()
        x=torch.randn_like(x).cuda(x.device)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)).cuda(x.device) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return x,torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
    def forward_inference_cgan(self, x):
        x = self.encoder(x)
        latent=torch.randn_like(x).cuda(x.device)
        x=torch.cat([x,latent],dim=1)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)).cuda(x.device) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder2[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
        x=torch.randn_like(x).cuda(x.device)

    def forward_inference_cgan(self, x):
        x = self.encoder(x)
        latent=torch.randn_like(x).cuda(x.device)
        x=torch.cat([x,latent],dim=1)
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),3,self.num_points//self.nb_primitives)).cuda(x.device) #sample points randomly
            rand_grid.data.normal_(0,1)
            rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid**2, dim=1, keepdim=True)).expand(x.size(0),3,self.num_points//self.nb_primitives)
            #assert a number of things like norm/visdom... then copy to other functions
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder2[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()
        x=torch.randn_like(x).cuda(x.device)


class Discriminator(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(Discriminator, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.trans = trans
        self.discriminator = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 1),
        )
        self.sigmoid=torch.nn.Sigmoid()
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x=self.discriminator(x)
        x=self.sigmoid(x)
        return x


class MeshMap(nn.Module):
    def __init__(self):

        super(MeshMap, self).__init__()
        self.e21 = nn.Linear(2,2*4)
        self.ne21=nn.LayerNorm(2*4)
        self.e22 = nn.Linear(2*4,2*16)
        self.ne22=nn.LayerNorm(2*16)
        self.e23 = nn.Linear(2*16,2*64)
        self.ne23=nn.LayerNorm(2*64)
        self.encoder_2d=nn.Sequential(self.e21,self.ne21,self.e22,self.ne22,self.e23,self.ne23)

        self.e31=nn.Linear(2*64,3*64)
        self.ne31=nn.LayerNorm(3*64)
        self.e32=nn.Linear(3*64,3*128)
        self.ne32=nn.LayerNorm(3*128)
        self.e33=nn.Linear(3*128,3*256)
        self.ne33=nn.LayerNorm(3*256)
        self.encode_3d=nn.Sequential(self.e31,self.ne31,self.e32,self.ne32,self.e33,self.ne33)
        # self.d21=nn.Linear(2*64,2*16)
        # self.nd21=nn.LayerNorm(2*16)
        # self.d22=nn.Linear(2*16,2*4)
        # self.nd22=nn.LayerNorm(2*4)
        # self.d23=nn.Linear(2*4,2)
        # self.decoder_2d=nn.Sequential(self.d21,self.nd21,self.d22,self.nd22,self.d23)
        self.d31=nn.Linear(3*256,3*64)
        self.nd31=nn.LayerNorm(3*64)
        self.d32=nn.Linear(3*64,3*16)
        self.nd32=nn.LayerNorm(3*16)
        self.d33=nn.Linear(3*16,3)
        self.decoder_3d=nn.Sequential(self.d31,self.nd31,self.d32,self.nd32,self.d33)
    def forward(self, x):
        batchsize = x.size()[0]
        encode_2d=self.encoder_2d(x)
        # decoder_2d=self.decoder_2d(encode_2d)
        encode_3d=self.encode_3d(encode_2d)
        decode_3d=self.decoder_3d(encode_3d)
        return torch.tanh(decode_3d)


class ND_Map(nn.Module):
    def __init__(self, num_points = 2500, nb_primitives = 1):
        super(ND_Map, self).__init__()
        self.num_points = num_points
        self.nb_primitives = nb_primitives
        self.decoder =nn.ModuleList([MeshMap() for i in range(0,self.nb_primitives)])
        # nn.ModuleList([MeshMap() for i in range(0,self.nb_primitives)])


    def forward(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            outs.append(self.decoder[i](x))
        return torch.cat(outs,0).contiguous()
