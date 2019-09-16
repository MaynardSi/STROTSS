from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as f

###### MODEL FILE #######

class TransformationNetwork (nn.Module):
    def __init__ (self):
        super().__init__()

        self.dims = 2

        kernel = 3

        channel_base = 32

        layers = []

        ##################### DOWNSAMPLE #####################
        # layers.append(ConvLayer(3, 32, kernel_size=9, stride=1))
        # layers.append(nn.BatchNorm2d(32))
        # layers.append(nn.ReLU(inplace=True))

        # layers.append(ConvLayer(32, 64, kernel_size=3, stride=2))
        # layers.append(nn.BatchNorm2d(64))
        # layers.append(nn.ReLU(inplace=True))

        # layers.append(ConvLayer(64, 128, kernel_size=3, stride=2))
        # layers.append(nn.BatchNorm2d(128))
        # layers.append(nn.ReLU(inplace=True))


        layers.append(nn.ReflectionPad2d(kernel*3 // 2))
        layers.append(nn.Conv2d(3, channel_base, kernel*3, stride=1, padding=0))
        layers.append(nn.InstanceNorm2d(32, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ReflectionPad2d(kernel // 2))
        layers.append(nn.Conv2d(32, 64, kernel, stride=2, padding=0))
        layers.append(nn.InstanceNorm2d(64, affine=True))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.ReflectionPad2d(kernel // 2))
        layers.append(nn.Conv2d(64, 128, kernel, stride=2, padding=0))
        layers.append(nn.InstanceNorm2d(128, affine=True))
        layers.append(nn.ReLU(inplace=True))

        ##################### RESIDUAL BLOCKS #####################
        layers.append(ResBlock(128))
        layers.append(ResBlock(128))
        layers.append(ResBlock(128))
        layers.append(ResBlock(128))
        layers.append(ResBlock(128))

        ##################### UPSAMPLE #####################
    
        #layers.append(nn.ReflectionPad2d(kernel))
        layers.append(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(64, affine=True))
        layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose2d(64, 32, 4, stride= 2, padding=1))
        layers.append(nn.InstanceNorm2d(32, affine=True))
        layers.append(nn.ReLU(True))

        layers.append(nn.ReflectionPad2d(kernel*3 // 2))
        layers.append(nn.Conv2d(32, 3, kernel*3, stride=1, padding=0))

        # Non-linearities
        self.relu = torch.nn.ReLU()


        ###################################################

        self.trans_net = nn.Sequential(*layers)

    def forward(self, x):
        # for layer in self.trans_net:
        #     x = layer(x)
        #     print(x.size())
        return self.trans_net(x)

class ResBlock (nn.Module):
    def __init__ (self, in_channels):
        super(ResBlock,self).__init__()

        kernel = 3

        layers = []
        
        layers.append(nn.ReflectionPad2d(kernel // 2))
        layers.append(nn.Conv2d(in_channels, in_channels, kernel, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ReflectionPad2d(kernel // 2))
        layers.append(nn.Conv2d(in_channels, in_channels, kernel, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(in_channels))

        self.resblock = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblock(x)
