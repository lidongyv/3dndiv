from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from ndiv_pytorch import NDiv_loss
import visdom
import numpy as np
from mutils import *
from torch import autograd
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='lsun',help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='./',help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./log_inverse_train', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', default=111,type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass
BATCH_SIZE=opt.batchSize
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(914)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),

                               ]))
    nc=3
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),

                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),

                           ]))
    nc=3

elif opt.dataset == 'mnist':
        opt.imageSize=64
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),

                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

vis = visdom.Visdom(env=opt.dataset)


image_window = vis.image(
    np.random.rand(64, 64),
    opts=dict(title='image!', caption='image.'),
)
loss_real_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='Loss real',
                                 title='Loss real',
                                 legend=['Loss']))
loss_g_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='Loss g',
                                 title='Training LossG',
                                 legend=['LossG']))
loss_fake_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatgches',
                                 ylabel='Loss fake',
                                 title='loss fake',
                                 legend=['Loss fake']))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0,1e-6)
        #torch.nn.init.kaiming_normal_(m.weight)
        #m.weight.data=m.weight.data/opt.batchSize
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 1e-6)
        m.bias.data.fill_(0)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_size = nz
        self.img_channels = nc

        self.fc = nn.Sequential(Linear(self.latent_size,256),Linear(256,512))

        self.up1 = deconv2DBatchNormRelu(512, 512, 4, stride=1,group_dim=32)
        self.up1_1 = conv2DBatchNormRelu(512, 512, 3,padding=1,group_dim=32)
        #4
        self.up3 = deconv2DBatchNormRelu(512, 256, 4, stride=2,padding=1,group_dim=1)
        self.up3_1 = conv2DBatchNormRelu(256, 256, 3,padding=1,group_dim=1)
        #8
        self.up4 = deconv2DBatchNormRelu(256, 128, 4, stride=2,padding=1,group_dim=1)
        self.up4_1 = conv2DBatchNormRelu(128, 128, 3,padding=1,group_dim=1)
        #16
        self.up5 = deconv2DBatchNormRelu(128, 64, 4, stride=2,padding=1,group_dim=1)
        self.up5_1 = conv2DBatchNormRelu(64, 64, 3,padding=1,group_dim=1)
        #32
        self.up6 =  nn.ConvTranspose2d(    64,      nc, 4, 2, 1, bias=False)

        self.transform=conv2DBatchNormRelu(64,2,3,padding=1,group_dim=1)

        self.sigmoid=nn.Sigmoid()
        self.generate=nn.Sequential(self.up1,self.up3,self.up4,self.up5)

    def forward(self, input):
        input=input.view(input.shape[0],-1)
        x=self.fc(input)
        x = x.unsqueeze(-1).unsqueeze(-1)
        output = self.generate(x)
        grid=self.sigmoid(self.transform(output))
        grid=grid.permute(0,2,3,1)
        output=torch.nn.functional.grid_sample(output,grid)
        output=self.sigmoid(self.up6(output))
        return output



netG = Generator()
#netG.apply(weights_init)
netG=torch.nn.DataParallel(netG,device_ids=range(ngpu))
netG.cuda(0)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.latent_size = nz
        #self.img_size = img_size
        self.img_channels = nc

        self.conv0 = conv2D(self.img_channels, 32, 3, stride=1,padding=1)
        self.conv0_1 = conv2D(32, 32, 3, stride=1,padding=1)
        #32*64*64
        self.conv1 = conv2DGroupNormRelu(32, 64, 3, stride=2,padding=(1,0,1,0),group_dim=64)
        #(64+2-4)/2+1=32
        self.conv1_1 = conv2DGroupNormRelu(64, 64, 3,stride=1,padding=1,group_dim=64)
        self.conv2 = conv2DGroupNormRelu(64, 128, 3, stride=2,padding=(1,0,1,0),group_dim=128)
        #32+2-4/2+1=16
        self.conv2_1 = conv2DGroupNormRelu(128, 128, 3, stride=1,padding=1,group_dim=128)
        self.conv3 = conv2DGroupNormRelu(128, 256, 3, stride=2,padding=(1,0,1,0),group_dim=ndf)
        #16+2-4/2+1=8
        self.conv3_1 = conv2DGroupNormRelu(256, 256, 3, stride=1,padding=1,group_dim=ndf)
        self.conv4 = conv2DGroupNormRelu(256, 512, 3, stride=2,padding=(1,0,1,0),group_dim=ndf)
        self.conv4_1 = conv2DGroupNormRelu(512, 512, 3, stride=1,padding=1,group_dim=ndf)
        self.conv5 = conv2DGroupNormRelu(512, 1024, 4, stride=1,padding=0,group_dim=1)
        self.conv6=conv2DGroupNormRelu(1024, 512, 1, stride=1,padding=0,group_dim=1)
        self.conv7=conv2D(512, 256, 1, stride=1,padding=0)
        self.conv8=conv2D(256, 1, 1, stride=1,padding=0)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.LeakyReLU(0.1)
        self.decoder = nn.Sequential(self.conv0,self.conv0_1,self.conv1,self.conv1_1,self.conv2,self.conv2_1,self.conv3,self.conv3_1,self.conv4,self.conv4_1, \
           self.conv5,self.conv6,self.conv7,self.conv8,self.sigmoid)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                #m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                #m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                #m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    def forward(self, input):

        output = self.decoder(input)
        #print(output.shape)
        #output=self.discrim(output.view(input.shape[0],-1))
        return output.view(-1, 1,1,1).squeeze()


netD = Discriminator()

netD=torch.nn.DataParallel(netD,device_ids=range(ngpu))
netD.cuda(0)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#torch.autograd.set_detect_anomaly(True)
count=0

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        print('[%d/%d][%d/%d] real: %.4f fake: %.4f fake_real: %.4f'
              % (epoch, opt.niter, i, len(dataloader),
             errD_real.item(),errD_fake.item(),errG.item()))
        image = fake[0,...].data.cpu().numpy().astype('float32')
        image = np.reshape(image, [nc, 64, 64])
        vis.image(
            image,
            opts=dict(title='image!', caption='image.'),
            win=image_window,
        )
        vis.line(
            X=torch.ones(1).cpu() * count,
            Y=errD_real.item() * torch.ones(1).cpu(),
            win=loss_real_window,
            update='append')
        vis.line(
            X=torch.ones(1).cpu() * count,
            Y=errG.item() * torch.ones(1).cpu(),
            win=loss_g_window,
            update='append')
        vis.line(
            X=torch.ones(1).cpu() * count,
            Y=errD_fake.item() * torch.ones(1).cpu(),
            win=loss_fake_window,
            update='append')
        count+=1
    vutils.save_image(real_cpu,
            '%s/real_samples.png' % opt.outf,
            normalize=False)
    fake = netG(fixed_noise)
    vutils.save_image(fake.detach(),
            '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
            normalize=False)
    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))