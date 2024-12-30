import torch
from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv3d(inplanes, outplanes, kernel_size, stride, padding)
        
        self.bn=None
        if norm:
            self.bn = nn.BatchNorm3d(outplanes)
        
    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)
        
        if self.bn is not None:
            fx = self.bn(fx)
            
        return fx

    
class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose3d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(outplanes)       
        
        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout3d(p=0.5, inplace=True)
            
    def forward(self, x):
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)
            
        return fx
    
class UnetGenerator(nn.Module):
    """Unet-like Encoder-Decoder model"""
    def __init__(self,print_featuremap_size = False):
        super().__init__()
        self.print_featuremap_size = print_featuremap_size
        self.encoder1 = nn.Conv3d(1, 16, 
                                  kernel_size=(3,7,7), 
                                  stride=1, 
                                  padding=(1,3,3)) # 8,64,64
        self.encoder2 = EncoderBlock(16, 32, 
                                     kernel_size=(3,7,7), 
                                     stride=1, 
                                     padding=(1,3,3)) # 8,64,64
        self.encoder3 = EncoderBlock(32, 64, 
                                     kernel_size=(3,7,7), 
                                     stride=1, 
                                     padding=(1,3,3)) # 8,64,64
        self.encoder4 = EncoderBlock(64, 128, 
                                     kernel_size=(3,7,7), 
                                     stride=1, 
                                     padding=(1,3,3)) # 8,64,64
        self.encoder5 = EncoderBlock(128, 256, 
                                     kernel_size=(2,4,4), 
                                     stride=(2,2,2), 
                                     padding=(0,1,1)) # 4,32,32
        self.encoder6 = EncoderBlock(256, 256, 
                                      kernel_size=(2,4,4), 
                                     stride=(2,2,2), 
                                     padding=(0,1,1)) # 2,16,16
        self.encoder7 = EncoderBlock(256, 256, 
                                     kernel_size=(2,4,4), 
                                     stride=(2,2,2), 
                                     padding=(0,1,1)) # 1,4,4
        self.encoder8 = EncoderBlock(256, 256, 
                                     norm=False, 
                                     kernel_size=(1,3,3), 
                                     stride=1,
                                     padding=(0,1,1)) # 1,4,4
        
        self.decoder8 = DecoderBlock(256, 256, dropout=True, 
                                     kernel_size=(1,3,3), 
                                     stride=1, 
                                     padding=(0,1,1))
        self.decoder7 = DecoderBlock(2*256, 256, dropout=True, 
                                     kernel_size=(2,4,4), 
                                     stride=(2,2,2), 
                                     padding=(0,1,1))
        self.decoder6 = DecoderBlock(2*256, 256, dropout=True, 
                                     kernel_size=(2,4,4), 
                                     stride=(2,2,2), 
                                     padding=(0,1,1))
        self.decoder5 = DecoderBlock(2*256, 128, 
                                     kernel_size=(2,4,4), 
                                     stride=(2,2,2), 
                                     padding=(0,1,1))
        self.decoder4 = DecoderBlock(2*128, 64, 
                                     kernel_size=(3,7,7), 
                                     stride=1, padding=(1,3,3))
        self.decoder3 = DecoderBlock(2*64, 32, 
                                     kernel_size=(3,7,7), 
                                     stride=1, padding=(1,3,3))
        self.decoder2 = DecoderBlock(2*32, 16, 
                                     kernel_size=(3,7,7), 
                                     stride=1, padding=(1,3,3))
        self.decoder1 = nn.ConvTranspose3d(2*16, 1, 
                                        kernel_size=(3,7,7), 
                                        stride=1, padding=(1,3,3))
        
    def forward(self, x):
        
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        output = self.decoder1(d2)
        # output = torch.tanh(d1)

        # if self.print_featuremap_size:
        #     print(f"Input: {x.size()}")
        #     print(f"After encoder1: {e1.size()}")
        #     print(f"After encoder2: {e2.size()}")
        #     print(f"After encoder3: {e3.size()}")
        #     print(f"After encoder4: {e4.size()}")
        #     print(f"After encoder5: {e5.size()}")
        #     print(f"After encoder6: {e6.size()}")
        #     print(f"After encoder7: {e7.size()}")
        #     print(f"After encoder8: {e8.size()}")
        #     print(f"After decoder8: {d8.size()}")
        #     print(f"After skip connection (decoder8 + e7): {d8.size()}")
        #     print(f"After decoder7: {d7.size()}")
        #     print(f"After skip connection (decoder7 + e6): {d7.size()}")
            
        #     print(f"After decoder6: {d6.size()}")
        #     print(f"After skip connection (decoder6 + e5): {d6.size()}")
        #     print(f"After decoder5: {d5.size()}")
        #     print(f"After skip connection (decoder5 + e4): {d5.size()}")
        #     print(f"After decoder4: {d4.size()}")
        #     print(f"After skip connection (decoder4 + e3): {d4.size()}")
        #     print(f"After decoder3: {d3.size()}")
        #     print(f"After skip connection (decoder3 + e2): {d3.size()}")
        #     print(f"After decoder2: {d2.size()}")
        #     print(f"After skip connection (decoder2 + e1): {d2.size()}")
        #     print(f"After decoder1: {d1.size()}")
        #     print(f"Output: {output.size()}")
        
        return output

        # return torch.tanh(d1)