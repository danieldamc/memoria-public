import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    r"""Double Conv Block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            mid_channels (int): Number of middle channels
            kernel_size (int): Kernel size of the convolution
            bias (bool): If True, adds a learnable bias to the output
            activation (torch.nn.Module): Activation function
            norm_layer (torch.nn.Module): Normalization layer function
            initialization (torch.nn.Module): Initialization function

        Shape:
            - Input: :math:`(N, C_{in}, H, W)`
            - Output: :math:`(N, C_{out}, H, W)`

        Examples::
            >>> x = torch.randn(1, 3, 256, 256)
            >>> double_conv = DoubleConv(in_channels=3, out_channels=64, mid_channels=None, kernel_size=3, bias=True, activation=F.tanh, norm_layer=nn.InstanceNorm2d, initialization=None)
            >>> output = double_conv(x)
        """
    def __init__(self, in_channels=3, out_channels=64, mid_channels=None, kernel_size=3, bias=True, activation=nn.ReLU, norm_layer=nn.BatchNorm2d, initialization=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=bias)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=bias)

        if initialization is not None:
            initialization(self.conv1.weight)
            initialization(self.conv2.weight)
            initialization(self.conv1x1.weight)
        
        if norm_layer is not None:
            self.bn1 = norm_layer(out_channels)
            self.bn2 = norm_layer(out_channels)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.Identity()


    def forward(self, x):
        c1 = self.activation(self.bn1(self.conv1(x)))
        s = self.conv1x1(x)
        c2 = self.bn2(self.conv2(c1))
        out = self.activation(c2 + s)
        return out