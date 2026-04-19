import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_pad_layer(pad: str, pad_size: int):
    if pad == 'reflection':
        return nn.ReflectionPad2d(pad_size)
    elif pad == 'replication':
        return nn.ReplicationPad2d(pad_size)
    elif pad == 'zero':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported pad mode: {pad}")


def conv3x3(in_ch, out_ch, pad='zero', bias=True, stride=1):
    layers = []
    if pad in ['reflection', 'replication']:
        layers.append(_get_pad_layer(pad, 1))
        padding = 0
    else:
        padding = 1
    layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=bias))
    return nn.Sequential(*layers)


def conv1x1(in_ch, out_ch, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)


def act(act_fun='LeakyReLU'):
    if act_fun == 'LeakyReLU':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_fun == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_fun in ['none', None]:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported act_fun: {act_fun}")


def conv_block(in_ch, out_ch, pad='zero', norm_layer=nn.BatchNorm2d,
               act_fun='LeakyReLU', need_bias=True):
    layers = [
        conv3x3(in_ch, out_ch, pad=pad, bias=need_bias, stride=1),
        norm_layer(out_ch) if norm_layer is not None else nn.Identity(),
        act(act_fun),
        conv3x3(out_ch, out_ch, pad=pad, bias=need_bias, stride=1),
        norm_layer(out_ch) if norm_layer is not None else nn.Identity(),
        act(act_fun),
    ]
    return nn.Sequential(*layers)


class UpFirstBlock(nn.Module):
    """
    先卷积，再上采样
    """
    def __init__(self, in_ch, out_ch, upsample_mode='bilinear',
                 pad='zero', norm_layer=nn.BatchNorm2d,
                 act_fun='LeakyReLU', need_bias=True):
        super().__init__()
        self.conv = conv_block(
            in_ch, out_ch,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        up_kwargs = {}
        if upsample_mode == 'bilinear':
            up_kwargs['align_corners'] = False

        self.up = nn.Upsample(scale_factor=2, mode=upsample_mode, **up_kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class DownBlockNoSkip(nn.Module):
    def __init__(self, in_ch, out_ch,
                 pad='reflection',
                 norm_layer=nn.BatchNorm2d,
                 act_fun='LeakyReLU',
                 need_bias=True):
        super().__init__()

        self.down = conv3x3(in_ch, in_ch, pad=pad, bias=need_bias, stride=2)

        self.conv = conv_block(
            in_ch, out_ch,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

class DownFuseBlock(nn.Module):
    """
    先把高分辨率特征下采样回来，再与浅层 skip 融合
    """
    def __init__(self, in_ch, skip_ch, out_ch,
                 pad='zero', norm_layer=nn.BatchNorm2d,
                 act_fun='LeakyReLU', need_bias=True):
        super().__init__()
        self.down = conv3x3(in_ch, in_ch, pad=pad, bias=need_bias, stride=2)
        self.fuse = conv_block(
            in_ch + skip_ch, out_ch,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

    def forward(self, x_high, skip):
        x = self.down(x_high)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = x
        x = self.fuse(x)
        return x


class NNet2(nn.Module):
    """
    对应 UNet2 的 up-first 版本：
      input -> stem(128)
            -> up_block(64) -> high-res bottleneck(64)
            -> down_fuse(skip with stem) -> final
    """
    def __init__(
        self,
        num_input_channels=32,
        num_output_channels=3,
        channels=[128, 64],
        upsample_mode='bilinear',
        pad='reflection',
        norm_layer=nn.BatchNorm2d,
        act_fun='LeakyReLU',
        need_sigmoid=True,
        need_bias=True
    ):
        super().__init__()
        assert len(channels) == 2, "channels should be [128, 64]"
        c1, c2 = channels  # c1=128, c2=64

        # stem at original resolution
        self.stem = conv_block(
            num_input_channels, c1,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        # up-first branch: 128 -> 64, then x2 spatial size
        self.up_block = UpFirstBlock(
            c1, c2,
            upsample_mode=upsample_mode,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        # bottleneck at high resolution
        self.bottleneck = conv_block(
            c2, c2,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        # downsample back and fuse with shallow skip
        self.down_fuse = DownFuseBlock(
            in_ch=c2,
            skip_ch=c1,
            out_ch=c1,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        self.final = nn.Sequential(
            conv1x1(c1, num_output_channels, bias=need_bias),
            nn.Sigmoid() if need_sigmoid else nn.Identity()
        )

    def forward(self, x):
        skip = self.stem(x)          # [B,128,H,W]
        x = self.up_block(skip)      # [B,64,2H,2W]
        x = self.bottleneck(x)       # [B,64,2H,2W]
        x = self.down_fuse(x, skip)  # [B,128,H,W]
        out = self.final(x)          # [B,3,H,W]
        return out

class NNet2_NoSkip(nn.Module):
    def __init__(
        self,
        num_input_channels=32,
        num_output_channels=3,
        channels=[128, 64],
        upsample_mode='bilinear',
        pad='reflection',
        norm_layer=nn.BatchNorm2d,
        act_fun='LeakyReLU',
        need_sigmoid=True,
        need_bias=True
    ):
        super().__init__()
        assert len(channels) == 2, "channels should be [128, 64]"
        c1, c2 = channels

        self.stem = conv_block(
            num_input_channels, c1,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        self.up_block = UpFirstBlock(
            c1, c2,
            upsample_mode=upsample_mode,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        self.bottleneck = conv_block(
            c2, c2,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        self.down_block = DownBlockNoSkip(
            in_ch=c2,
            out_ch=c1,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        self.final = nn.Sequential(
            nn.Conv2d(c1, num_output_channels, kernel_size=1, stride=1, padding=0, bias=need_bias),
            nn.Sigmoid() if need_sigmoid else nn.Identity()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.up_block(x)
        x = self.bottleneck(x)
        x = self.down_block(x)
        return self.final(x)