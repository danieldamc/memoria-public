from torch import nn
from ..blocks.conv_block import ConvBlock

class BasicBlock(nn.Module):
    r""" Basic Residual Block

        See the paper:
        `Deep Residual Learning for Image Recognition`_
        by He et al. (2015) for more details.

        Args:
            channels (int): Number of input/output channels
            kernel_size (int): Kernel size of the convolution
            activation (torch.nn.Module): Activation function
            norm_layer (torch.nn.Module): Normalization layer function

        Shape:
            - Input: :math:`(N, C, H, W)`
            - Output: :math:`(N, C, H, W)`

        Examples:
            >>> x = torch.randn(1, 3, 256, 256)
            >>> basic_block = BasicBlock(channels=3, kernel_size=3, activation=nn.ReLU, norm_layer=nn.BatchNorm2d)
            >>> output = basic_block(x)
        
        .._Deep Residual Learning for Image Recognition
            https://arxiv.org/abs/1512.03385
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        
        if in_channels != out_channels:
            self.skip_conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, activation=None, norm_layer=norm_layer)
        else:
            self.skip_conv = nn.Identity()
        
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, activation=activation, norm_layer=norm_layer)
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, activation=None, norm_layer=norm_layer)

        self.activation = activation()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        out += self.skip_conv(x)

        out = self.activation(out)
        return out

class BottleNeckBlock(nn.Module):
    def __init__(self, channels, contraction=4, kernel_size=3, activation=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(BottleNeckBlock, self).__init__()
        
        assert channels % contraction == 0, "Channels must be divisible by contraction"

        self.mid_channels = channels // contraction

        self.conv1 = ConvBlock(in_channels=channels, out_channels=self.mid_channels, kernel_size=1, activation=activation, norm_layer=norm_layer)
        self.conv2 = ConvBlock(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=kernel_size, activation=activation, norm_layer=norm_layer)
        self.conv3 = ConvBlock(in_channels=self.mid_channels, out_channels=channels, kernel_size=1, activation=None, norm_layer=norm_layer)

        self.activation = activation()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += x
        out = self.activation(out)

        return out