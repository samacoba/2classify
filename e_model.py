# coding: utf-8
# pytorchモデル用スクリプト

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Conv(nn.Module):
    def __init__(self, in_ch = 3):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch, 32, 4, 2, 1)
        self.conv2=nn.Conv2d(32, 64, 4, 2, 1)
        self.norm2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64, 128, 4, 2, 1)
        self.norm3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128, 256, 4, 2, 1)
        self.norm4=nn.BatchNorm2d(256)

    def forward(self, x):
        # convolution
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)), negative_slope=0.2)
        #h4 = F.relu(self.norm4(self.conv4(h3)))
        return h4


class DeConv(nn.Module):
    def __init__(self, out_ch = 3):
        super().__init__()
        self.deconv1=nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dnorm1=nn.BatchNorm2d(128)
        self.deconv2=nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dnorm2=nn.BatchNorm2d(64)
        self.deconv3=nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dnorm3=nn.BatchNorm2d(32)
        self.deconv4=nn.ConvTranspose2d(32, out_ch, 4, 2, 1)

    def forward(self, x):
        # deconvolution
        dh1 = F.leaky_relu(self.dnorm1(self.deconv1(x)), negative_slope=0.2)
        dh2 = F.leaky_relu(self.dnorm2(self.deconv2(dh1)), negative_slope=0.2)
        dh3 = F.leaky_relu(self.dnorm3(self.deconv3(dh2)), negative_slope=0.2)
        #y = F.tanh(self.deconv4(dh3))
        y = self.deconv4(dh3)

        return y


class Discriminator(nn.Module):
    def __init__(self, in_ch = 3):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch, 32, 4, 2, 1)
        self.conv2=nn.Conv2d(32, 64, 4, 2, 1)
        self.norm2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64, 128, 4, 2, 1)
        self.norm3=nn.BatchNorm2d(128)
        self.conv4=nn.Conv2d(128, 64, 4, 2, 1)
        self.norm4=nn.BatchNorm2d(64)
        self.conv5=nn.Conv2d(64, 1, 4)

    def forward(self, x):
        # convolution
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.norm2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.norm3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.norm4(self.conv4(h3)), negative_slope=0.2)
        y = self.conv5(h4)
        return y, [h2, h3, h4]


