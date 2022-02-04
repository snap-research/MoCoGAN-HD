import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        which_conv=nn.Conv2d,
        wide=True,
        preactivation=False,
        activation=None,
        downsample=None,
    ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        self.conv1 = self.which_conv(in_channels=self.in_channels,
                                     out_channels=self.hidden_channels)
        self.conv1 = spectral_norm(self.conv1)
        self.conv2 = self.which_conv(in_channels=self.hidden_channels,
                                     out_channels=self.out_channels)
        self.conv2 = spectral_norm(self.conv2)

        self.learnable_sc = True if (
            in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     padding=0)
            self.conv_sc = spectral_norm(self.conv_sc)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


def D_arch(ch=32):
    arch = {}
    arch[256] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 8, 16]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
        'downsample': [True] * 6 + [False],
        'resolution': [128, 64, 32, 16, 8, 4, 4],
    }
    arch[128] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8, 16]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 16]],
        'downsample': [True] * 5 + [False],
        'resolution': [64, 32, 16, 8, 4, 4],
    }
    arch[64] = {
        'in_channels': [3] + [ch * item for item in [1, 2, 4, 8]],
        'out_channels': [item * ch for item in [1, 2, 4, 8, 16]],
        'downsample': [True] * 4 + [False],
        'resolution': [32, 16, 8, 4, 4],
    }
    arch[32] = {
        'in_channels': [3] + [item * ch for item in [4, 4, 4]],
        'out_channels': [item * ch for item in [4, 4, 4, 4]],
        'downsample': [True, True, False, False],
        'resolution': [16, 16, 16, 16],
    }
    return arch


class Discriminator(nn.Module):
    def __init__(self,
                 D_ch=96,
                 D_wide=True,
                 resolution=128,
                 D_activation=nn.ReLU(inplace=False),
                 output_dim=1,
                 proj_dim=256,
                 D_init='ortho',
                 skip_init=False,
                 D_param='SN'):
        super(Discriminator, self).__init__()
        self.ch = D_ch
        self.D_wide = D_wide
        self.activation = D_activation
        self.init = D_init

        self.arch = D_arch(self.ch)[resolution]

        self.which_conv = functools.partial(nn.Conv2d,
                                            kernel_size=3,
                                            padding=1)

        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if
                                   self.arch['downsample'][index] else None))
            ]]
        self.blocks = nn.ModuleList(
            [nn.ModuleList(block) for block in self.blocks])

        self.proj0 = spectral_norm(
            nn.Linear(self.arch['out_channels'][-1], proj_dim))
        self.proj1 = spectral_norm(nn.Linear(proj_dim, proj_dim))
        self.proj2 = spectral_norm(nn.Linear(proj_dim, proj_dim))

        self.linear = spectral_norm(nn.Linear(proj_dim, output_dim))

        if not skip_init:
            self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum(
                    [p.data.nelement() for p in module.parameters()])
        print('Param count for D'
              's initialized parameters: %d' % self.param_count)

    def forward(self, x, proj_only=False):
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        h = torch.sum(self.activation(h), [2, 3])
        h = self.activation(self.proj0(h))
        out = self.linear(h)

        proj_head = self.proj2(self.activation(self.proj1(h)))

        if proj_only:
            return proj_head
        return out, proj_head
