"""
Heavily relies on code from https://github.com/rosinality/glow-pytorch, with changes in order to have easier
manipulation of the layers and models, while also making the code more readible
"""
import torch
from torch import nn
from typing import Tuple
from .flow_layers import FlowModule, ZeroConv2d, _diff_clamp, FlowSequential, InvConv2D, Squeeze, Shuffle

T = torch.Tensor


def _get_MLP(in_features: int, out_features: int, hidden_width: int=64, depth: int=1, zero: bool=False):
    """
    Helper function that creates an MLP for use in various modules
    :param in_features: number of features to use
    :param out_features: number of output features
    :param hidden_width: width of the hidden layers in the MLP
    :param depth: number of hidden layers in the MLP
    :param zero: whether to initialize the MLP as the zero function
    :return: a torch.nn Module representing the wanted MLP
    """
    layers = [nn.Linear(in_features, hidden_width), nn.ReLU(inplace=True)]
    for i in range(depth):
        layers.append(nn.Linear(hidden_width, hidden_width))
        if zero: layers[-1].weight.data.zero_()
        layers[-1].bias.data.zero_()
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_width, out_features))
    if zero: layers[-1].weight.data.zero_()
    layers[-1].bias.data.zero_()
    return nn.Sequential(*layers)


def _cond_block(n_channels, cond_features, hidden_width, affine, n_flows, squeeze: bool = True, shuffle: bool = False):
    """
    Create conditional Glow block
    :param n_channels: number of input channels
    :param cond_features: number of conditional features
    :param hidden_width: hidden width used in the affine injectors and coupling layers
    :param affine: whether the transformation should be affine or just linear
    :param n_flows: number of flows in the block
    :param squeeze: whether to squeeze the input at the start of the block or not
    :param shuffle: a bool indicating whether to shuffle instead of using the invertible 2D convolutions
    :return: a flow module that represents a conditional Glow's flow block
    """
    def flow():
        return FlowSequential(
            InvConv2D(in_channel=n_channels) if not shuffle else Shuffle(dim=0),
            AffineInjector(n_channels=n_channels, cond_channels=cond_features, hidden_width=hidden_width,
                           affine=affine, same_size=False),
            CondAffineCoupling(n_channels=n_channels, cond_features=cond_features, affine=affine,
                               hidden_width=hidden_width, same_size=False)
        )

    bl = [flow() for _ in range(n_flows)]
    if squeeze: bl = [Squeeze()] + bl
    return FlowSequential(*bl)


class CondAffineCoupling(FlowModule):

    def __init__(self, n_channels: int, cond_features: int, affine: bool=True, hidden_width: int=512,
                 same_size: bool=True):
        """
        Implementation of the conditional affine coupling.

        The only difference between the original affine coupling and the conditional version is that in the conditional
        version, channels with the values of the features (expanded to same shape as input) are added to the input.

        :param n_channels: number of input channels in the data
        :param cond_features: number of features in conditional input
        :param affine: a boolean indicating whether to use an affine transformation or not
        :param hidden_width: the width that should be used for the hidden layer in the convolution
        :param same_size: a boolean indicating whether the conditional parameters have the same spatial dimensions as
                          the input variables (as used in SRFlow) or not
        """
        super(CondAffineCoupling, self).__init__()

        self.affine = affine
        self.same = same_size

        # define network for affine coupling
        self.CNN = nn.Sequential(
            nn.Conv2d(n_channels//2 + cond_features, hidden_width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_width, hidden_width, 1, padding=0),
            nn.ReLU(inplace=True),
            ZeroConv2d(hidden_width, n_channels if self.affine else n_channels//2),
        )

        self.CNN[0].weight.data.normal_(0, .05)
        self.CNN[0].bias.data.zero_()

        self.CNN[2].weight.data.normal_(0, .05)
        self.CNN[2].bias.data.zero_()

    def _make_input(self, x: T, cond: T):
        """
        helper method to append the conditional channels to the input
        :param x: the input data tensor, with shape [N, n_channels, h, w]
        :param cond: the conditional features, with shape [N, k_features]
        :return: the features appended to the input, a tensor with shape [N, n_channels+n_features, h, w]
        """
        if not self.same:
            ones = torch.ones(cond.shape[0], cond.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            return torch.cat([x, ones*cond[:, :, None, None]], dim=1)
        else: return torch.cat([x, cond], dim=1)

    def forward(self, x: T, cond: T=None) -> Tuple[T, T]:
        in_a, in_b = x.chunk(2, 1)
        net_out = self.CNN(self._make_input(in_a, cond))

        if self.affine:
            # if the transformation is affine, divide to translation and scale
            log_s, t = net_out.chunk(2, 1)
            # pass scale through a sigmoid for numerical stability, add 2 so initially acts as identity
            s = _diff_clamp(torch.sigmoid(log_s + 2), min=1e-3, max=1e3)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        else:
            out_b = in_b + net_out
            logdet = torch.tensor(0, device=x.device, dtype=x.dtype)

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, y: T, cond=None) -> T:
        out_a, out_b = y.chunk(2, 1)
        net_out = self.CNN(self._make_input(out_a, cond))

        if self.affine:
            log_s, t = net_out.chunk(2, 1)
            s = _diff_clamp(torch.sigmoid(log_s + 2), min=1e-3, max=1e3)
            in_b = out_b / s - t
        else: in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class AffineInjector(FlowModule):

    def __init__(self, n_channels: int, cond_channels: int, hidden_width: int=512, affine: bool=True,
                 same_size: bool=True):
        """
        Implementation of the Affine Injector Flow module used in SRFlow, allowing for simple conditional operations.
        The affine injector is a learned, conditional, element wise affine transformation of the inputs.

        :param n_channels: number of channels of the input variable
        :param cond_channels: number of channels the conditional variables have; note, it is assumed that the
                              conditional variables have the same spatial dimensions as the inputs
        :param hidden_width: number of channels in the hidden layer of the transformation
        :param affine: a boolean indicating whether the transformation should be affine or just translational
        :param same_size: a boolean indicating whether the conditional parameters have the same spatial dimensions as
                          the input variables (as used in SRFlow) or not
        """
        super(AffineInjector, self).__init__()
        self.affine = affine
        self.same = same_size

        # define network for affine coupling
        self.CNN = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_width, hidden_width, 1, padding=0),
            nn.ReLU(inplace=True),
            ZeroConv2d(hidden_width, 2*n_channels if self.affine else n_channels),
        )

        self.CNN[0].weight.data.normal_(0, .05)
        self.CNN[0].bias.data.zero_()

        self.CNN[2].weight.data.normal_(0, .05)
        self.CNN[2].bias.data.zero_()

    def _make_input(self, x: T, cond: T):
        """
        helper method to append the conditional channels to the input
        :param x: the input data tensor, with shape [N, n_channels, h, w]
        :param cond: the conditional features, with shape [N, k_features]
        :return: the features appended to the input, a tensor with shape [N, n_channels+n_features, h, w]
        """
        if not self.same:
            ones = torch.ones(cond.shape[0], cond.shape[1], x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            return ones*cond[:, :, None, None]
        else: return cond

    def forward(self, x: T, cond: T=None) -> Tuple[T, T]:
        bias = self.CNN(self._make_input(x, cond))
        if self.affine:
            bias, log_scale = bias.chunk(2, dim=1)
            log_scale = _diff_clamp(log_scale, min=-5, max=5)
        else: log_scale = torch.zeros_like(bias)
        return x*torch.exp(log_scale) + bias, torch.sum(log_scale.reshape(log_scale.shape[0], -1), dim=1)

    def reverse(self, y: T, cond: T=None) -> T:
        bias = self.CNN(self._make_input(y, cond))
        if self.affine:
            bias, log_scale = bias.chunk(2, dim=1)
            log_scale = _diff_clamp(log_scale, min=-5, max=5)
        else: log_scale = torch.zeros_like(bias)
        return torch.exp(-log_scale)*(y - bias)