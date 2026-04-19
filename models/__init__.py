from .skip import skip, skip_inverse
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet
from .unet2 import UNet2
from .nnet2 import NNet2, NNet2_NoSkip
import torch 
import torch.nn as nn

def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    if NET_TYPE == 'ResNet':
        # TODO
        net = ResNet(input_depth, 3, 10, 16, 1, nn.BatchNorm2d, False)
    elif NET_TYPE == 'skip':
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
    elif NET_TYPE == 'skip_inverse':
        net = skip_inverse(input_depth, n_channels, num_channels_down=[skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up=[skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip=[skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11,
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
    elif NET_TYPE == 'skip_noupdssample':
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios = [32, 16, 8, 4, 2, 1], fill_noise=False,pad=pad)

    elif NET_TYPE =='UNet':
        net = UNet(num_input_channels=input_depth, num_output_channels=3, 
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)
    elif NET_TYPE == 'UNet2':
        enc_ch = [skip_n33d] * num_scales if isinstance(skip_n33d, int) else skip_n33d
        dec_ch = None if skip_n33u is None else ([skip_n33u] * num_scales if isinstance(skip_n33u, int) else skip_n33u)

        # 这里借用 skip_n11 作为 skip_gain
        skip_gain = float(skip_n11) if isinstance(skip_n11, (int, float)) else 1.0

        net = UNet2(
            num_input_channels=input_depth,
            num_output_channels=n_channels,
            encoder_channels=enc_ch,
            decoder_channels=dec_ch,
            upsample_mode=upsample_mode,
            downsample_mode=downsample_mode,   
            pad=pad,
            norm_layer=nn.BatchNorm2d,
            act_fun=act_fun,
            need_sigmoid=True,
            need_bias=True,
            skip_gain=skip_gain
        )
    elif NET_TYPE == 'NNet2':
        net = NNet2(num_input_channels=input_depth, num_output_channels=3, channels=[128, 64],
            upsample_mode=upsample_mode,
            pad=pad,
            norm_layer=nn.BatchNorm2d,
            act_fun=act_fun,
            need_sigmoid=True,
            need_bias=True)

    elif NET_TYPE == 'NNet2Noskip':
        net = NNet2_NoSkip(
            num_input_channels=input_depth,
            num_output_channels=3,
            channels=[128, 64],
            upsample_mode=upsample_mode,
            pad=pad,
            norm_layer=nn.BatchNorm2d,
            act_fun=act_fun,
            need_sigmoid=True,
            need_bias=True
        )

    elif NET_TYPE == 'identity':
        assert input_depth == 3
        net = nn.Sequential()
    elif NET_TYPE == 'MLP':
        net = PixelMLP(in_dim=input_depth)
    else:
        assert False

    return net


class PixelMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, depth=4, out_dim=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        out = self.net(x)

        out = out.view(B, H, W, 3).permute(0, 3, 1, 2)
        return out