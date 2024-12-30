import torch
from torch import nn
from torch.nn import functional as F

        
class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv3d(inplanes, outplanes, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm3d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        fx = self.conv(x)
        
        if self.isn is not None:
            fx = self.isn(fx)
            
        fx = self.lrelu(fx)
        return fx
    
    
    
class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator"""
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(2, 128, norm=False, 
                                 kernel_size = (2,3,3), 
                                 stride= (1,1,1), 
                                 padding=(1,1,1)) # 8x64x64
        self.block2 = BasicBlock(128, 256, 
                                 kernel_size = (2,4,4), 
                                 stride= (2,2,2), 
                                 padding=(0,1,1)) # 4x32x32
        self.block3 = BasicBlock(256, 512, 
                                 kernel_size = (2,4,4), 
                                 stride= (2,2,2), 
                                 padding=(0,1,1)) # 2x16x16
        self.block4 = BasicBlock(512, 1024, 
                                kernel_size = (2,3,3), 
                                 stride= (1,1,1), 
                                 padding=(1,1,1)) # 2x16x16
        self.block5 = nn.Conv3d(1024, 1, 
                                 kernel_size = (2,3,3), 
                                 stride= (1,1,1), 
                                 padding=(1,1,1)) # 2x16x16
        
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        # blocks forward
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        fx = nn.Sigmoid()(fx)
        
        return fx