

from torch import nn





class ConvBlock_Basic(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1.
    Padding ensures output size is same as input size.
    ReLU after first two Conv2d layers.

    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock_Basic, self).__init__()
        layers = [
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)





class ConvBlock_Init(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU only after the first Conv2d layers
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock_Init, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)





class ConvBlock_Out(nn.Module):
    """
    Two Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    ReLU only after the first Conv2d layers
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock_Out, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)





class ResidualBlock(nn.Module):
    """
    inputs > W + b > sigma > ... > W + b > sigma > X
       |___ > Y
    out = X+Y
    """

    def __init__(self, units, activation, name="residual_block", **kwargs):
        super(ResidualBlock, self).__init__()
        self._units = units
        self._activation = activation
        self._layers = nn.ModuleList(
            [nn.Linear(units[i], units[i]) for i in range(len(units))]
        )


    def forward(self, inputs):
        residual = inputs
        for i, h_i in enumerate(self._layers):
            inputs = self._activation(h_i(inputs))
        residual = residual + inputs
        return residual






