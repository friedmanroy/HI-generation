"""
Heavily relies on code from https://github.com/rosinality/glow-pytorch, with changes in order to have easier
manipulation of the layers and models, while also making the code more readible
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from .flow_layers import FlowModule, ZeroConv2d, _random_LU, FlowSequential, Squeeze, _diff_clamp

T = torch.Tensor


def _get_MLP(in_features: int, out_features: int, hidden_width: int=64, depth: int=1, zero: bool=False):
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


class CondActNorm(FlowModule):

    def __init__(self, n_channels: int, cond_features: int, width: int=32, depth: int=1):
        super().__init__()
        self.register_module('loc', _get_MLP(cond_features, n_channels, hidden_width=width, depth=depth, zero=True))
        self.register_module('scale', _get_MLP(cond_features, n_channels, hidden_width=width, depth=depth))

    def forward(self, x: T, cond: T=None) -> Tuple[T, T]:
        _, _, height, width = x.shape

        scale = self.scale(cond)*.1
        log_abs = torch.log(torch.abs(scale))
        logdet = height * width * torch.sum(log_abs, dim=1)
        return scale * (x + self.loc(cond)), logdet

    def reverse(self, y: T, cond: T=None) -> T:
        return y * 10 / self.scale(cond) - self.loc(cond)


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


class CondInvConv2D(FlowModule):

    def __init__(self, n_channels: int, cond_features: int, hidden_width: int=128, depth: int=1):
        super(CondInvConv2D, self).__init__()
        w_p, w_l, w_s, w_u, u_mask, l_mask = _random_LU(n_channels)
        self.n_channels = n_channels
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.register_module('w_l', _get_MLP(cond_features, n_channels*n_channels,
                                             hidden_width=hidden_width, depth=depth))
        self.register_module('w_u', _get_MLP(cond_features, n_channels * n_channels,
                                             hidden_width=hidden_width, depth=depth))
        self.register_module('w_s', _get_MLP(cond_features, n_channels,
                                             hidden_width=hidden_width, depth=depth))


    def calc_weight(self, cond: T):
        w_l = self.w_l(cond).reshape(cond.shape[0], self.n_channels, self.n_channels)
        w_u = self.w_u(cond).reshape(cond.shape[0], self.n_channels, self.n_channels)
        w_s = self.w_s(cond)
        weight = (
                self.w_p
                @ (w_l * self.l_mask + self.l_eye)
                @ ((w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(w_s)))
        )

        return weight, w_s

    def forward(self, x: T, cond: T=None) -> Tuple[T, T]:
        _, _, height, width = x.shape

        weight, w_s = self.calc_weight(cond)

        out = (weight[:, None]@x.transpose(dim0=1, dim1=2)).transpose(dim0=1, dim1=2)
        logdet = height * width * torch.sum(w_s, dim=-1)

        return out, logdet

    def reverse(self, y: T, cond: T=None) -> T:
        weight, _ = self.calc_weight(cond)
        out = (weight[:, None]@y.transpose(dim0=1, dim1=2)).transpose(dim0=1, dim1=2)
        return out


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


class CondGlowFlow(FlowModule):

    def __init__(self, n_channels: int, cond_features: int, affine: bool=True, hidden_layer: int=512):
        """
        A single flow block used in HI-generation
        :param n_channels: the number of input (and output) channels
        :param transf: the transformation to use in the affine coupling layer; if None is supplied, the same
                       transformation that was used in HI-generation is utilized
        :param affine: a boolean indication whether to use an affine or additive layer
        :param hidden_layer: the number of channels in the hidden layer - only used if no transform is supplied
        """
        super(CondGlowFlow, self).__init__()
        self.register_module('flow',
                             FlowSequential(CondActNorm(n_channels=n_channels, cond_features=cond_features),
                                            CondInvConv2D(n_channels=n_channels, cond_features=cond_features),
                                            CondAffineCoupling(n_channels=n_channels, cond_features=cond_features,
                                                               affine=affine, hidden_width=hidden_layer))
                             )

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        return self.flow.forward(x)

    def reverse(self, y: T, cond=None) -> T:
        return self.flow.reverse(y)


class CondGlowBlock(FlowModule):

    def __init__(self, n_channels: int, cond_features: int, n_flows: int=None, squeeze: bool=True,
                 hidden_layer: int=512, affine: bool=True):
        """
        A block in the HI-generation architecture
        :param n_channels: the number of channels in the input
        :param flows: either a list of flows or a singular flow that will be used in the block
        :param n_flows: if a single flow was supplied for the "flows" argument, a number should be supplied here to
                        control the number of instances of the same flow should be used in the block
        :param squeeze: a boolean indicating whether the input should be squeezed as mentioned in section 3.6 of RealNVP
        :param hidden_layer: the number of channels in the hidden layer - only used if no Flow is supplied
        :param affine: whether to use affine transformations - only used if no Flow is supplied
        """
        super(CondGlowBlock, self).__init__()
        chans = 4*n_channels if squeeze else n_channels
        layers = [CondGlowFlow(n_channels=chans, cond_features=cond_features, affine=affine, hidden_layer=hidden_layer)
                  for _ in range(n_flows)]
        layers = FlowSequential(Squeeze(), *layers) if squeeze else FlowSequential(*layers)
        self.register_module('block', layers)

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        return self.block.forward(x)

    def reverse(self, y: T, cond=None) -> T:
        return self.block.reverse(y)
