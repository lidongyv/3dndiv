from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 1027):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.gn1 = torch.nn.GroupNorm(self.bottleneck_size,self.bottleneck_size)
        self.gn2 = torch.nn.GroupNorm(self.bottleneck_size//2,self.bottleneck_size//2)
        self.gn3 = torch.nn.GroupNorm(self.bottleneck_size//4,self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.leaky_relu(self.gn1(self.conv1(x)))
        x = F.leaky_relu(self.gn2(self.conv2(x)))
        x = F.leaky_relu(self.gn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, num_points = 2500, global_feat = True, trans = False):
        super(Discriminator, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.relu=torch.nn.LeakyReLU()
        self.gn1 = torch.nn.GroupNorm(64,64)
        self.gn2 = torch.nn.GroupNorm(128,128)
        self.gn3 = torch.nn.GroupNorm(1024,1024)
        self.feature=nn.Sequential(self.conv1,self.gn1,self.relu,self.conv2,self.relu,self.gn2,self.conv3)

        self.posterior=PointGenCon()
        self.trans = trans
        self.discriminator = nn.Sequential(
        nn.Linear(1024, 512),
        nn.GroupNorm(1,512),
        nn.LeakyReLU(0.1),
        nn.Linear(512, 128),
        nn.GroupNorm(1,128),
        nn.LeakyReLU(0.1),
        nn.Linear(128, 64),
        nn.Linear(64, 1),
        )
        self.sigmoid=torch.nn.Sigmoid()
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    def forward(self, x):
        input=x
        batchsize = x.size()[0]
        x = self.feature(x)

        x_f,_ = torch.max(x, 2)
        x_d = x_f.view(batchsize, 1024)
        x_d=self.discriminator(x_d)
        x_d=self.sigmoid(x_d)
        x_r=x_f.view(batchsize,1024,1)
        grid=torch.rand_like(input)*2-1
        x_r=torch.cat([x_r.expand(batchsize,1024,input.shape[-1]),grid],dim=1)
        x_r=self.posterior(x_r)
        x_r=x_r.transpose(2,1)
        return x_d,x_r


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
