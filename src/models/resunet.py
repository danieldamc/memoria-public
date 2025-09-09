
import math
import torch
from torch import nn

from ..blocks.muti_res_conv_block import Variant1

class Down(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 num_convs:int=2,
                 kernel_size:int=3,
                 skip_kernel_size:int=3,
                 dropout:int=None,
                 activation:nn.Module=nn.ReLU,
                 normalization:nn.Module=nn.BatchNorm2d,
                 activation_kwargs={},
                 normalization_kwargs={}):
        super(Down, self).__init__()

        if dropout is not None:
            self.p = dropout
            self.dropout = nn.Dropout(p=self.p)
        else:
            self.dropout = nn.Identity()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = Variant1(in_channels=in_channels,
                                    out_channels=out_channels,
                                    num_convs=num_convs,
                                    kernel_size=kernel_size,
                                    skip_kernel_size=skip_kernel_size,
                                    activation=activation,
                                    normalization=normalization,
                                    activation_kwargs=activation_kwargs,
                                    normalization_kwargs=normalization_kwargs)

    def forward(self, x):
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


class Up(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 num_convs:int=2,
                 kernel_size:int=3,
                 skip_kernel_size:int=1,
                 bilinear:bool=False,
                 activation:nn.Module=nn.ReLU,
                 normalization:nn.Module=nn.BatchNorm2d,
                 activation_kwargs={},
                 normalization_kwargs={}):
        super(Up, self).__init__()

        if in_channels == out_channels:
            traspose_channels = in_channels
            in_conv_channels = in_channels * 2
            out_conv_channels = out_channels
        else:
            traspose_channels = in_channels // 2
            in_conv_channels = in_channels
            out_conv_channels = out_channels

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, traspose_channels, kernel_size=2, stride=2)

        self.conv = Variant1(in_channels=in_conv_channels,
                             out_channels=out_conv_channels,
                             num_convs=num_convs,
                             kernel_size=kernel_size,
                             skip_kernel_size=skip_kernel_size,
                             activation=activation,
                             normalization=normalization,
                             activation_kwargs=activation_kwargs,
                             normalization_kwargs=normalization_kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        # print(f"after up and concat shape: {x.shape}")
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResUnet(nn.Module):
    def __init__(self, in_channels=3,
                 out_channels=1,
                 num_convs=2,
                 kernel_size=3,
                 skip_kernel_size=1,
                 dropout=None,
                 activation=nn.ReLU,
                 normalization=nn.BatchNorm2d,
                 activation_kwargs={},
                 normalization_kwargs={},
                 bilinear=False,
                 features=[64, 128, 256, 512, 1024]):
        super(ResUnet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.bilinear = bilinear

        self.encoder.append(Variant1(in_channels=in_channels,
                                     out_channels=features[0],
                                     num_convs=num_convs,
                                     kernel_size=kernel_size,
                                     skip_kernel_size=skip_kernel_size,
                                     activation=activation,
                                     normalization=normalization,
                                     activation_kwargs=activation_kwargs,
                                     normalization_kwargs={'num_features': features[0]}))

        for i in range(1, len(features)):
            self.encoder.append(Down(in_channels=features[i-1],
                                     out_channels=features[i],
                                     num_convs=num_convs,
                                     kernel_size=kernel_size,
                                     skip_kernel_size=skip_kernel_size,
                                     dropout=dropout,
                                     activation=activation,
                                     normalization=normalization,
                                     activation_kwargs=activation_kwargs,
                                     normalization_kwargs={'num_features': features[i]}))
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(Up(in_channels=features[i],
                                   out_channels=features[i-1],
                                   num_convs=num_convs,
                                   kernel_size=kernel_size,
                                   skip_kernel_size=skip_kernel_size,
                                   bilinear=self.bilinear,
                                   activation=activation,
                                   normalization=normalization,
                                   activation_kwargs=activation_kwargs,
                                   normalization_kwargs={'num_features': features[i-1]}))

        self.outconv = OutConv(in_channels=features[0], out_channels=out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.decoder, skips):
            x = up(x, skip)

        return self.sigmoid(self.outconv(x))