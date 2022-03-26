import torch
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import Upsample, make_layer


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.body(x)
        return out + x


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class TDBlock(nn.Module):

    def __init__(self, channels, kSize=3):
        super(TDBlock, self).__init__()
        self.r1 = ResidualBlock(channels, channels)
        self.r2 = ResidualBlock(channels * 2, channels * 2)
        self.r3 = ResidualBlock(channels * 4, channels * 4)
        self.g = nn.Conv2d(channels * 8, channels, 3, 1, 1, 1)
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        r1 = self.r1(x)

        c1 = torch.cat([x, r1], dim=1)
        r2 = self.r2(c1)

        c2 = torch.cat([c1, r2], dim=1)
        r3 = self.r3(c2)

        g = self.g(torch.cat([c2, r3], dim=1))
        out = self.ca(g)
        return out + x


class TDGROUP(nn.Module):

    def __init__(self, channels, kSize=3):
        super(TDGROUP, self).__init__()
        self.b1 = TDBlock(channels)
        self.b2 = TDBlock(channels)
        self.b3 = TDBlock(channels)

        self.g1 = nn.Conv2d(channels * 2, channels, 3, 1, 1, 1)
        self.g2 = nn.Conv2d(channels * 2, channels, 3, 1, 1, 1)
        self.g3 = nn.Conv2d(channels * 2, channels, 3, 1, 1, 1)

    def forward(self, x):
        b1 = self.b1(x)

        b2 = self.b2(self.g1(torch.cat([x, b1], dim=1)))

        b3 = self.b3(self.g2(torch.cat([b1, b2], dim=1)))

        out = self.g3(torch.cat([b2, b3], dim=1))
        return out + x


class UAttention(nn.Module):

    def __init__(self, num_feat):
        super(UAttention, self).__init__()

        self.ua1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.ua2 = nn.Sequential(
            nn.MaxPool2d(2, 2), nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1), nn.ReLU(True))
        self.ua3 = nn.Sequential(
            nn.MaxPool2d(2, 2), nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(num_feat * 4, num_feat * 4, 3, 1, 1), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.ua4 = nn.Sequential(
            nn.Conv2d(num_feat * 6, num_feat * 2, 3, 1, 1), nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1),
            nn.ReLU(True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.ua5 = nn.Sequential(
            nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1), nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        res1 = self.ua1(x)
        res2 = self.ua2(res1)
        res = self.ua3(res2)
        res = self.ua4(torch.cat([res2, res], dim=1))
        res = self.ua5(torch.cat([res1, res], dim=1))
        return res * x


@ARCH_REGISTRY.register()
class TD(nn.Module):
    """Residual Channel Attention Networks.

    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
    Ref git repo: https://github.com/yulunzhang/RCAN.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_group=10,
                 num_block=16,
                 squeeze_factor=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(TD, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.g1 = nn.Sequential(TDGROUP(num_feat), TDGROUP(num_feat))
        self.g2 = nn.Sequential(TDGROUP(num_feat), TDGROUP(num_feat))
        self.g3 = nn.Sequential(TDGROUP(num_feat), TDGROUP(num_feat))

        self.conv_after_body = nn.Conv2d(num_feat * 3, num_feat * 3, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat * 3)
        self.conv_last = nn.Conv2d(num_feat * 3, num_out_ch, 3, 1, 1)
        self.ua = UAttention(num_feat=num_feat * 3)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)

        g1 = self.g1(x)
        g2 = self.g2(g1)
        res = self.g3(g2)

        res = self.ua(torch.cat([g1, g2, res], 1))
        res = res + torch.cat([x, x, x], 1)
        res = self.conv_after_body(res)

        x = self.conv_last(self.upsample(res))
        x = x / self.img_range + self.mean

        return x
