""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from .full_freq_att import  FullSpectralAttentionLayer


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True,select_strategy='mean'):
        super(UNet, self).__init__()

        self.name = "FSAUNet"
        self.select_strategy=select_strategy
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512,1024)
        self.down5 = Down(1024, 2048 // factor)
        self.drop = DropOut(2048,2048)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        #加深模块
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128// factor, bilinear)
        self.up5 = Up(128,64,bilinear)
        self.outc = OutConv(64, n_classes)

        ####################################FSA Attention Module##################################################
        self.att1 = FullSpectralAttentionLayer(512,7,7,select_strategy=self.select_strategy)
        self.att2 = FullSpectralAttentionLayer(256,14,14,select_strategy=self.select_strategy)
        self.att3 = FullSpectralAttentionLayer(128,28,28,select_strategy=self.select_strategy)
        self.att4 = FullSpectralAttentionLayer(64,56,56,select_strategy=self.select_strategy)
        self.att5 = FullSpectralAttentionLayer(64, 112, 112,select_strategy=self.select_strategy)
        #########################################################################################################



    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x6 = self.drop(x6)

        x = self.up1(x6, x5) #512 7 7
        x = self.att1(x)
        x = self.up2(x, x4) # 256 14 14
        x = self.att2(x)
        x = self.up3(x, x3) #128 28 28
        x = self.att3(x)
        x = self.up4(x, x2) #64 56 56
        x = self.att4(x)
        x = self.up5(x, x1) #64 112 112
        x = self.att5(x)
        logits = self.outc(x)



        return logits
if __name__=='__main__':
    torch.backends.cudnn.enabled = False
    net1=UNet(3,1)

    input_img=torch.zeros((4,3,112,112))

    device='cuda'
    net1=net1.to(device)

    x=input_img.to(device)
    net1.eval()

    out1=net1(x)

    print(out1.shape)

