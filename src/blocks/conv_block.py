from torch import nn

class ConvBlock(nn.Module):
    r""" Convolutional Block

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size of the convolution
            padding (int): Padding size of the convolution
            activation (torch.nn.Module): Activation function
            norm_layer (torch.nn.Module): Normalization layer function

        Shape:
            - Input: :math:`(N, C_{in}, H, W)`
            - Output: :math:`(N, C_{out}, H, W)`    

        Examples:
            >>> x = torch.randn(1, 3, 256, 256)
            >>> conv_block = ConvBlock(in_channels=3, out_channels=64, kernel_size=3, padding=1, activation=nn.ReLU, norm_layer=nn.BatchNorm2d)
            >>> output = conv_block(x)
    """
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:int=3, 
                 padding:int=0, 
                 stride:int=1,
                 activation:nn.Module=nn.ReLU, 
                 normalization:nn.Module=nn.BatchNorm2d, 
                 activation_kwargs={}, 
                 normalization_kwargs={}):
        super(ConvBlock, self).__init__()

        #assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert kernel_size > 0, "kernel_size must be greater than 0"
        assert in_channels > 0, "in_channels must be greater than 0"
        assert out_channels > 0, "out_channels must be greater than 0"
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.norm = normalization(**normalization_kwargs) if normalization is not None else nn.Identity()
        self.acti = activation(**activation_kwargs) if activation is not None else nn.Identity()

    def forward(self, x):
        return self.acti(self.norm(self.conv(x)))