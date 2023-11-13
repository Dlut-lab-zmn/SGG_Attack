# Copyright (c) Mn,Zhao. All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
###########################
# Generator: Resnet
###########################

# To control feature map in generator
ngf = 8

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, data_dim='high', eps=1.0, evaluate=False):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.data_dim = data_dim
        self.init_perturbation = Parameter(torch.randn(1,3,2048,2048) ,requires_grad=True)#torch.zeros(1,3,2048,2048) - 1
        self.init_position = Parameter(torch.randn(1,1,2048,2048),requires_grad=True)
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
            

        # Input size = 3, n/4, n/4
        self.upsampl_inf1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl_inf2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf_inf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )
        

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)
        
        self.eps = eps
        self.evaluate = evaluate

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.upsampl_inf2(x)
        x = self.blockf_inf(x)

        #x_out = torch.clamp(x_inf + input, min=0, max=1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual



if __name__ == '__main__':
    netG = GeneratorResnet(data_dim='low')
    test_sample = torch.rand(1, 3, 64, 57)
    print('Generator output:', netG(test_sample).size())
    print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad))
