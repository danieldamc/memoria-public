from torch import nn
from ..blocks.conv_block import ConvBlock

class PatchGan(nn.Module):
    def __init__(self, 
                 input_channels:int,
                 num_filters:int=64,
                 num_layers:int=3):
        super(PatchGan, self).__init__()
        sequence = [ConvBlock(in_channels=input_channels, 
                              out_channels=num_filters, 
                              kernel_size=4, 
                              padding=1, 
                              stride=2,
                              normalization=None,
                              activation=nn.LeakyReLU, 
                              activation_kwargs={"negative_slope": 0.2})]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [ConvBlock(in_channels=num_filters * nf_mult_prev, 
                                   out_channels=num_filters * nf_mult, 
                                   kernel_size=4, 
                                   padding=1, 
                                   stride=2,
                                   normalization=nn.BatchNorm2d,
                                   activation=nn.LeakyReLU,  
                                   normalization_kwargs={"num_features": num_filters * nf_mult},
                                   activation_kwargs={"negative_slope": 0.2})]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        
        sequence += [ConvBlock(in_channels=num_filters * nf_mult_prev, 
                              out_channels=num_filters * nf_mult, 
                              kernel_size=4, 
                              padding=1, 
                              stride=1,
                              normalization=nn.BatchNorm2d,
                              activation=nn.LeakyReLU,  
                              normalization_kwargs={"num_features": num_filters * nf_mult},
                              activation_kwargs={"negative_slope": 0.2})]

        sequence += [nn.Conv2d(in_channels=num_filters * nf_mult, 
                              out_channels=1, 
                              kernel_size=4, 
                              padding=1, 
                              stride=1)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)