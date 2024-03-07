import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net_tiny(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_tiny, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 进一步减少通道数
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=4)
        self.Conv2 = conv_block(ch_in=4, ch_out=8)
        self.Conv3 = conv_block(ch_in=8, ch_out=16)
        self.Conv4 = conv_block(ch_in=16, ch_out=32)
        self.Conv5 = conv_block(ch_in=32, ch_out=64)

        self.Up5 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv5 = conv_block(ch_in=64, ch_out=32)

        self.Up4 = up_conv(ch_in=32, ch_out=16)
        self.Up_conv4 = conv_block(ch_in=32, ch_out=16)

        self.Up3 = up_conv(ch_in=16, ch_out=8)
        self.Up_conv3 = conv_block(ch_in=16, ch_out=8)

        self.Up2 = up_conv(ch_in=8, ch_out=4)
        self.Up_conv2 = conv_block(ch_in=8, ch_out=4)

        self.Conv_1x1 = nn.Conv2d(4, output_ch, kernel_size=1, stride=1, padding=0)
        # self.sig = nn.Sigmoid()

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        # d1 = self.sig(d1) # limit to 0-1
        return d1
