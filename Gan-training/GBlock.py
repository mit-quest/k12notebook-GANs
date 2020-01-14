class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size, stride,
                                       padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(True)

    def forward(self, input):
        """Forward method of GBlock.

        By default, this block increases the spatial resolution by 2:

            input:  [batch_size, in_channels, H, W]
            output: [batch_size, out_channels, 2*H, 2*W]
        """

        x = self.conv(input)
        x = self.bn(x)
        out = self.act(x)
        return out

    
class Generator(nn.Module):
    """DCGAN Generator."""

    # Maps output resoluton to number of GBlocks.
    res2blocks = {32: 3, 64: 4, 128: 5, 256: 6, 512: 7}

    def __init__(self, dim_z=100, resolution=64, G_ch=64, block=GBlock, init='N02'):
        super().__init__()

        self.G_ch = G_ch
        self.init = init
        self.dim_z = dim_z

        self.num_blocks = self.res2blocks[resolution]
        
        # Determine number of channels at each layer.
        # of the form G_ch * [..., 32, 16, 8, 4, 2, 1]
        self.ch_nums = [G_ch * (2**i) for i in range(self.num_blocks, 0, -1)]
        
        # Input layer: latent z is fed into convolution.
        self.input = block(dim_z, self.ch_nums[0], kernel_size=4, stride=1, padding=0)
        
        # Build our GBlocks 
        self.GBlocks = nn.Sequential(*[
            block(in_c, out_c)
            for in_c, out_c in zip(self.ch_nums, self.ch_nums[1:])
        ])
        
        # Final output layer produces RGB image with shape [3, resolution, resolution]
        self.out = nn.ConvTranspose2d(self.ch_nums[-1], 3, 4, 2, 1)  # RGB image has 3 channels
        self.tanh = nn.Tanh()                                        # "Squashes" out to be in range[-1, 1]
        
        self.init_weights()
        
    def forward(self, x):
        
        x = x.view(x.size(0), -1, 1, 1)
        x = self.input(x)
        
        x = self.GBlocks(x)
        x = self.out(x)
        return self.tanh(x)
    
    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)