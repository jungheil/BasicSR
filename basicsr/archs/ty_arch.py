import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

torch.backends.cudnn.benchmark = True


class Upsampler(nn.Sequential):

    def __init__(self, num_feat, scale, kernel_size=7):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(
                    nn.Conv2d(num_feat, 4 * num_feat, kernel_size, 1, (kernel_size - 1) // 2, padding_mode='replicate'))
                m.append(nn.PixelShuffle(2))

        elif scale == 3:
            m.append(
                nn.Conv2d(num_feat, 9 * num_feat, kernel_size, 1, (kernel_size - 1) // 2, padding_mode='replicate'))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CSA(nn.Module):

    def __init__(self, num_feat, reduction_ratio=16, kernel_size=7):
        super(CSA, self).__init__()
        self.channel_attention = ChannelAttention(num_feat, reduction_ratio)
        self.spatial_attention = SpatialAttention(num_feat, kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class SpatialAttention(nn.Module):

    def __init__(self, num_feat, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=num_feat,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            padding_mode='replicate')

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        out = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(out)
        # conv = conv.repeat(1, x.size()[1] // 8, 1, 1)
        out = torch.sigmoid(out)
        return x * out

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class ChannelAttention(nn.Module):

    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        r_feat = max(8, num_feat // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_feat, r_feat, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(r_feat, num_feat, bias=True))

    def forward(self, x):
        b, c, _, _ = x.size()
        a = self.avg_pool(x).view(b, c)
        m = self.max_pool(x).view(b, c)
        y = (self.fc(a) + self.fc(m)).view(b, c, 1, 1)
        y = torch.sigmoid(y) * 2
        return x * y.expand_as(x)


class Block(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, res_scale=1, k_size=3, dilation=1, conv='D'):
        super(Block, self).__init__()
        self.res_scale = res_scale
        padding = int((k_size - 1) / 2) * dilation

        if conv == 'N':
            self.body = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, k_size, 1, padding, dilation, padding_mode='replicate'),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
            )
        elif conv == 'S':
            self.body = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, k_size, 1, padding, dilation, padding_mode='replicate'), nn.SiLU())
        elif conv == 'T':
            self.body = nn.Sequential(
                nn.Conv2d(
                    num_feat,
                    num_feat,
                    k_size,
                    1,
                    padding,
                    dilation,
                    groups=num_feat,
                    bias=False,
                    padding_mode='replicate'),
                nn.Conv2d(num_feat, num_feat * 4, 1, 1, 0),
                nn.SiLU(),
                nn.Conv2d(num_feat * 4, num_feat, 1, 1, 0),
            )
        elif conv == 'D':
            self.body = nn.Sequential(
                nn.Conv2d(
                    num_feat,
                    num_feat,
                    k_size,
                    1,
                    padding,
                    dilation,
                    groups=num_feat,
                    bias=False,
                    padding_mode='replicate'),
                nn.Conv2d(num_feat, num_feat, 1, 1, 0),
                nn.SiLU(),
            )
        else:
            raise RuntimeError('conv?')

    def forward(self, x):
        res = self.body(x)
        return res * self.res_scale + x


class IMDModule(nn.Module):

    def __init__(self, num_feat, num_sp_feat, num_block=4, k_size=3):
        super(IMDModule, self).__init__()
        self.b = nn.ModuleList([Block(num_feat, k_size=k_size) for _ in range(num_block)])
        self.sp_fuse = nn.ModuleList([nn.Conv2d(num_feat, num_sp_feat, 1, 1, 0) for _ in range(num_block)])

        self.attention = CSA(num_sp_feat * num_block)

        self.sp_tail = nn.Conv2d(num_sp_feat * num_block, num_feat, 1, 1, 0)

    def forward(self, x):
        sp = []
        out = x
        for i in range(self.b.__len__()):
            out = self.b[i](out)
            sp.append(self.sp_fuse[i](out))

        sp = self.attention(torch.cat(sp, dim=1))
        sp = self.sp_tail(sp)
        out = out + x
        return (out, sp)


@ARCH_REGISTRY.register()
class TY(nn.Module):

    def __init__(
            self,
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_sp_feat=16,
            num_modules=6,
            num_block=4,
            upscale=2,
            img_range=255.0,
            rgb_mean=(0.33024667, 0.41541553, 0.42345934),
    ):
        super(TY, self).__init__()
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.img_range = img_range

        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, padding_mode='replicate')

        self.M = nn.ModuleList()
        for _ in range(num_modules):
            self.M.append(IMDModule(num_feat, num_sp_feat, num_block, 3))

        self.ca = ChannelAttention(num_feat * num_modules)

        self.fuse = nn.Sequential(
            nn.Conv2d(num_feat * num_modules, num_feat, 1, 1, 0),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, padding_mode='replicate'),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.upsampler = Upsampler(num_feat, upscale, 7)
        self.tail = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, padding_mode='replicate')

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.fea_conv(x)
        out = x
        sp = []
        for m in self.M:
            out, s = m(out)
            sp.append(s)

        out = self.ca(torch.cat(sp, dim=1))
        out = self.fuse(out) + x

        out = self.upsampler(out)
        out = self.tail(out)
        out = out / self.img_range + self.mean

        return out
