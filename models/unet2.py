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


def _get_activation(act_fun='LeakyReLU'):
    if act_fun == 'LeakyReLU':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_fun == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_fun in ['none', None]:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported act_fun: {act_fun}")


def _conv3x3(in_ch, out_ch, pad='zero', need_bias=True):
    layers = []
    if pad in ['reflection', 'replication']:
        layers.append(_get_pad_layer(pad, 1))
        padding = 0
    else:
        padding = 1

    layers.append(
        nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3, stride=1, padding=padding, bias=need_bias
        )
    )
    return nn.Sequential(*layers)


class DoubleConv(nn.Module):
    """
    标准 U-Net 的双卷积块:
    Conv -> Norm -> Act -> Conv -> Norm -> Act
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        pad='zero',
        norm_layer=nn.BatchNorm2d,
        act_fun='LeakyReLU',
        need_bias=True
    ):
        super().__init__()

        layers = []

        # conv 1
        layers.append(_conv3x3(in_ch, out_ch, pad=pad, need_bias=need_bias))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        layers.append(_get_activation(act_fun))

        # conv 2
        layers.append(_conv3x3(out_ch, out_ch, pad=pad, need_bias=need_bias))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        layers.append(_get_activation(act_fun))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    """
    下采样模块，不改变通道数或按 stride conv 保持同通道数下采样
    """
    def __init__(self, ch, downsample_mode='maxpool', pad='zero', need_bias=True):
        super().__init__()

        if downsample_mode == 'maxpool':
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_mode == 'avgpool':
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        elif downsample_mode == 'stride':
            if pad in ['reflection', 'replication']:
                self.down = nn.Sequential(
                    _get_pad_layer(pad, 1),
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=0, bias=need_bias)
                )
            else:
                self.down = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1, bias=need_bias)
        else:
            raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")

    def forward(self, x):
        return self.down(x)


class EncoderBlock(nn.Module):
    """
    一个标准 encoder block:
    DoubleConv -> Downsample
    返回:
        feat: skip feature
        down: downsampled feature
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        pad='zero',
        norm_layer=nn.BatchNorm2d,
        act_fun='LeakyReLU',
        need_bias=True,
        downsample_mode='maxpool'
    ):
        super().__init__()
        self.conv = DoubleConv(
            in_ch=in_ch,
            out_ch=out_ch,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )
        self.down = Downsample(
            ch=out_ch,
            downsample_mode=downsample_mode,
            pad=pad,
            need_bias=need_bias
        )

    def forward(self, x):
        feat = self.conv(x)
        down = self.down(feat)
        return feat, down


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        pad='zero',
        norm_layer=nn.BatchNorm2d,
        act_fun='LeakyReLU',
        need_bias=True
    ):
        super().__init__()
        self.conv = DoubleConv(
            in_ch=in_ch,
            out_ch=out_ch,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """
    一个标准 decoder/up block:
    Upsample -> concat(skip) -> DoubleConv
    """
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        upsample_mode='bilinear',
        pad='zero',
        norm_layer=nn.BatchNorm2d,
        act_fun='LeakyReLU',
        need_bias=True
    ):
        super().__init__()

        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=2, stride=2, bias=need_bias
            )
        elif upsample_mode in ['nearest', 'bilinear']:
            up_kwargs = {}
            if upsample_mode == 'bilinear':
                up_kwargs['align_corners'] = False

            if pad in ['reflection', 'replication']:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode=upsample_mode, **up_kwargs),
                    _get_pad_layer(pad, 1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0, bias=need_bias)
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode=upsample_mode, **up_kwargs),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=need_bias)
                )
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

        self.conv = DoubleConv(
            in_ch=out_ch + skip_ch,
            out_ch=out_ch,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

    def forward(self, x, skip, skip_gain=1.0):
        x = self.up(x)

        # 尺寸对齐，避免奇数尺寸输入时 mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        skip = skip_gain * skip
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet2(nn.Module):
    """
    标准可配置 U-Net

    参数说明：
    - encoder_channels: encoder 每层输出通道，如 [32, 64, 128, 256]
      表示：
        enc1: in -> 32
        enc2: 32 -> 64
        enc3: 64 -> 128
        enc4: 128 -> 256
      bottleneck 默认用最后一层通道数的 2 倍，也可手动指定 bottleneck_channels

    - decoder_channels:
      若为 None，则默认使用 encoder_channels[::-1]
      例如 encoder_channels=[32,64,128,256]
      则 decoder_channels=[256,128,64,32]

    - bottleneck_channels:
      若为 None，则默认 2 * encoder_channels[-1]
    """
    def __init__(
        self,
        num_input_channels=3,
        num_output_channels=3,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=None,
        bottleneck_channels=None,
        upsample_mode='bilinear',
        downsample_mode='maxpool',
        pad='zero',
        norm_layer=nn.BatchNorm2d,
        act_fun='LeakyReLU',
        need_sigmoid=True,
        need_bias=True,
        skip_gain=1.0
    ):
        super().__init__()

        self.skip_gain = float(skip_gain)
        self.encoder_channels = list(encoder_channels)

        if len(self.encoder_channels) < 1:
            raise ValueError("encoder_channels must contain at least one element")

        if bottleneck_channels is None:
            bottleneck_channels = 2 * self.encoder_channels[-1]
        self.bottleneck_channels = bottleneck_channels

        if decoder_channels is None:
            decoder_channels = self.encoder_channels[::-1]
        self.decoder_channels = list(decoder_channels)

        if len(self.decoder_channels) != len(self.encoder_channels):
            raise ValueError(
                f"decoder_channels length ({len(self.decoder_channels)}) must equal "
                f"encoder_channels length ({len(self.encoder_channels)})"
            )

        # -------- Encoder --------
        self.enc_blocks = nn.ModuleList()

        prev_ch = num_input_channels
        for out_ch in self.encoder_channels:
            self.enc_blocks.append(
                EncoderBlock(
                    in_ch=prev_ch,
                    out_ch=out_ch,
                    pad=pad,
                    norm_layer=norm_layer,
                    act_fun=act_fun,
                    need_bias=need_bias,
                    downsample_mode=downsample_mode
                )
            )
            prev_ch = out_ch

        # -------- Bottleneck --------
        self.bottleneck = Bottleneck(
            in_ch=self.encoder_channels[-1],
            out_ch=self.bottleneck_channels,
            pad=pad,
            norm_layer=norm_layer,
            act_fun=act_fun,
            need_bias=need_bias
        )

        # -------- Decoder --------
        self.up_blocks = nn.ModuleList()
        curr_in = self.bottleneck_channels
        skip_channels = self.encoder_channels[::-1]

        for skip_ch, out_ch in zip(skip_channels, self.decoder_channels):
            self.up_blocks.append(
                UpBlock(
                    in_ch=curr_in,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    upsample_mode=upsample_mode,
                    pad=pad,
                    norm_layer=norm_layer,
                    act_fun=act_fun,
                    need_bias=need_bias
                )
            )
            curr_in = out_ch

        # -------- Final --------
        self.final = nn.Conv2d(curr_in, num_output_channels, kernel_size=1, bias=need_bias)
        if need_sigmoid:
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, x):
        skips = []

        # encoder
        for enc in self.enc_blocks:
            feat, x = enc(x)
            skips.append(feat)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        skips = skips[::-1]
        for up, skip in zip(self.up_blocks, skips):
            x = up(x, skip, skip_gain=self.skip_gain)

        return self.final(x)


# -------------------------
# 兼容你原先调用风格的工厂函数
# -------------------------
def build_unet(
    num_input_channels=3,
    num_output_channels=3,
    feature_scale=4,
    more_layers=0,
    concat_x=False,   # 这里保留参数名，仅为兼容旧接口；标准 U-Net 里未使用
    upsample_mode='bilinear',
    downsample_mode='maxpool',
    pad='zero',
    norm_layer=nn.BatchNorm2d,
    act_fun='LeakyReLU',
    need_sigmoid=True,
    need_bias=True,
    skip_gain=1.0,
    encoder_channels=None,
    decoder_channels=None,
    bottleneck_channels=None
):
    """
    一个方便的构造函数。
    你可以直接指定 encoder_channels；
    如果不指定，就按经典 U-Net 通道给默认值。
    """
    if encoder_channels is None:
        base = max(4, 64 // feature_scale)
        encoder_channels = [base, base * 2, base * 4, base * 8]

        # 若想更深，可继续加层
        for _ in range(more_layers):
            encoder_channels.append(encoder_channels[-1])

    net = UNet(
        num_input_channels=num_input_channels,
        num_output_channels=num_output_channels,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        bottleneck_channels=bottleneck_channels,
        upsample_mode=upsample_mode,
        downsample_mode=downsample_mode,
        pad=pad,
        norm_layer=norm_layer,
        act_fun=act_fun,
        need_sigmoid=need_sigmoid,
        need_bias=need_bias,
        skip_gain=skip_gain
    )
    return net


# -------------------------
# 简单测试
# -------------------------
if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)

    net = UNet(
        num_input_channels=3,
        num_output_channels=3,
        encoder_channels=(32, 64, 128, 256),
        decoder_channels=None,          # 默认对称 [256,128,64,32]
        bottleneck_channels=512,
        upsample_mode='bilinear',
        downsample_mode='maxpool',
        pad='zero',
        norm_layer=nn.BatchNorm2d,
        act_fun='none',                 # 这里现在可以安全用 none
        need_sigmoid=True,
        need_bias=True,
        skip_gain=1.0
    )

    y = net(x)
    print("input :", x.shape)
    print("output:", y.shape)