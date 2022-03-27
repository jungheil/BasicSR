import torch
from torch import nn as nn
import math

from basicsr.utils.registry import ARCH_REGISTRY


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class TDB_Residual(nn.Module):

    def __init__(self, num_feat, k_size=3):
        super(TDB_Residual, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, k_size, padding=(k_size - 1) // 2, stride=1), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, k_size, padding=(k_size - 1) // 2, stride=1), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.body(x)
        return torch.cat([x, out + x], 1)


class CALayer(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(CALayer, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class TDB(nn.Module):

    def __init__(self, num_feat, n_res, squeeze_factor=16):
        super(TDB, self).__init__()
        self.body = nn.Sequential(*[TDB_Residual(num_feat * 2**i) for i in range(n_res)])
        self.llf = nn.Conv2d(num_feat * 2**n_res, num_feat, 1, padding=0, stride=1)
        self.ca = CALayer(num_feat, squeeze_factor)
        self.g = nn.Sequential(nn.Conv2d(num_feat * 2, num_feat, 1, padding=0, stride=1))

    def forward(self, x):
        out = self.body(x)
        out = self.llf(out)
        out = self.ca(out)
        return self.g(torch.cat([x, out + x], 1))


class TDG(nn.Module):

    def __init__(self, num_feat, n_res, n_block, squeeze_factor=16):
        super(TDG, self).__init__()
        self.body = nn.Sequential(*[TDB(num_feat, n_res, squeeze_factor) for i in range(n_block)])

    def forward(self, x):
        return self.body(x) + x


class UAD(nn.Sequential):

    def __init__(self, num_feat, step, k_size=3):
        c = num_feat * 2**step
        m = [
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c, c * 2, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.ReLU(inplace=True)
        ]
        super(UAD, self).__init__(*m)


class UAU(nn.Sequential):

    def __init__(self, num_feat, step, k_size=3):
        c = num_feat * 2**step
        m = [
            nn.Conv2d(c * 3, c, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(c, c, k_size, padding=(k_size - 1) // 2, stride=1),
            nn.ReLU(inplace=True)
        ]
        super(UAU, self).__init__(*m)


class UALayer(nn.Module):

    def __init__(self, num_feat, n_step=3):
        super(UALayer, self).__init__()

        self.n_step = n_step
        self.head = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1))
        self.uad = nn.ModuleList()
        self.us = nn.ModuleList()
        self.uau = nn.ModuleList()
        for i in range(n_step):
            self.uad.append(UAD(num_feat, i))
            self.us.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            self.uau.append(UAU(num_feat, i))
        self.tail = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        out = self.head(x)

        res = [out]
        for i in range(self.n_step - 1):
            out = self.uad[i](out)
            res.append(out)
        out = self.uad[self.n_step - 1](out)
        for i in range(self.n_step)[::-1]:
            out = self.us[i](out)
            out = self.uau[i](torch.cat([out, res[i]], 1))
        return out * x


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
                 num_dis=3,
                 num_group=2,
                 num_block=3,
                 num_res=3,
                 squeeze_factor=16,
                 ua_step=3,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(TD, self).__init__()

        self.num_dis = num_dis
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.sfe = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.dis = nn.ModuleList()
        for i in range(num_dis):
            self.dis.append(
                nn.Sequential(*[TDG(num_feat, num_res, num_block, squeeze_factor) for i in range(num_group)]))

        self.ua = UALayer(num_feat * 3, ua_step)
        self.conv_after_body = nn.Conv2d(num_feat * 3, num_feat * 3, 3, 1, 1)
        self.upsample = Upsample(upscale, num_feat * 3)
        self.tail = nn.Conv2d(num_feat * 3, num_out_ch, 3, 1, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range
        x = self.head(x)
        out = self.sfe(x)

        d_out = []
        for i in range(self.num_dis):
            out = self.dis[i](out)
            d_out.append(out)

        out = self.ua(torch.cat(d_out, 1))
        out = out + torch.cat([x, x, x], 1)
        out = self.conv_after_body(out)

        x = self.tail(self.upsample(out))
        x = x / self.img_range + self.mean

        return x
