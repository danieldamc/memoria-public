import math
from torch import nn
from ..blocks.conv_block import ConvBlock

# CHANGE NAME TO THE CLASS
class Variant1(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 mid_channels=None, 
                 num_convs=2, 
                 kernel_size=3,
                 skip_kernel_size=1,
                 activation=nn.ReLU, 
                 normalization=nn.BatchNorm2d,
                 activation_kwargs={},
                 normalization_kwargs={}):
        super(Variant1, self).__init__()

        assert skip_kernel_size % 2 == 1, "skip_kernel_size must be odd"

        if not mid_channels:
            mid_channels = out_channels

        self.convs = nn.ModuleList()
        self.convs.append(ConvBlock(in_channels=in_channels, 
                                    out_channels=mid_channels, 
                                    kernel_size=kernel_size, 
                                    padding=1, 
                                    activation=activation, 
                                    normalization=normalization,
                                    activation_kwargs=activation_kwargs,
                                    normalization_kwargs=normalization_kwargs
                                    ))

        if num_convs > 2:
            for i in range(num_convs - 2):
                if i == 0:
                    self.convs.append(ConvBlock(in_channels=mid_channels, 
                                                out_channels=out_channels, 
                                                kernel_size=kernel_size, 
                                                padding=1, 
                                                activation=activation, 
                                                normalization=normalization,
                                                activation_kwargs=activation_kwargs,
                                                normalization_kwargs=normalization_kwargs
                                                ))
                else:
                    self.convs.append(ConvBlock(in_channels=out_channels, 
                                                out_channels=out_channels, 
                                                kernel_size=kernel_size, 
                                                padding=1, 
                                                activation=activation, 
                                                normalization=normalization,
                                                activation_kwargs=activation_kwargs,
                                                normalization_kwargs=normalization_kwargs))


        self.convs.append(ConvBlock(in_channels=out_channels, 
                                    out_channels=out_channels, 
                                    kernel_size=kernel_size, 
                                    padding=1, 
                                    activation=None, 
                                    normalization=normalization,
                                    activation_kwargs=None,
                                    normalization_kwargs=normalization_kwargs))
        
        self.skip_conv = ConvBlock(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=skip_kernel_size, 
                                   padding=math.floor(skip_kernel_size//2), 
                                   activation=None, 
                                   normalization=normalization,
                                   activation_kwargs=None,
                                   normalization_kwargs=normalization_kwargs)

        self.final_activation = activation(activation_kwargs) if activation is not None else nn.Identity()

    def forward(self, x):
        s = self.skip_conv(x)
        
        for conv in self.convs:
            x = conv(x)
        out = self.final_activation(x + s)

        return out