"""
Heavily relies on code from https://github.com/rosinality/glow-pytorch, with changes in order to have easier
manipulation of the layers and models, while also making the code more readible
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple
import numpy as np
from scipy import linalg as la

T = torch.Tensor


def _zero_clamp(mat: T, min: float=1e-3, max: float=1e3):
    """
    Clamp values of matrix so that they are not too close to 0
    :param mat: the matrix whose values should be clamped
    :return: the clamped matrix
    """
    return mat + (torch.clamp(torch.abs(mat), min=min, max=max)*torch.sign(mat) - mat).detach()


def _diff_clamp(x: T, min: float, max: float):
    return x + (torch.clamp(x, min=min, max=max) - x).detach()


def _np2tp(*x: np.ndarray): return tuple([torch.from_numpy(a).float() for a in x])


def _random_LU(n_channels: int):
    """
    Generates a random LU decomposition for the invertible 1x1 convolution used by Glow

    :param n_channels: number of channels the input has
    :return: random decomposition needed in order to define the 1x1 convolution used by Glow
    """
    # sample a random weight matrix
    weight = np.random.randn(n_channels, n_channels)

    # get PLU decomposition using scipy
    q, _ = la.qr(weight)
    w_p, w_l, w_u = la.lu(q.astype(np.float32))

    # extract relevant matrices
    w_s = np.diag(w_u)
    w_u = np.triu(w_u, 1)
    u_mask = np.triu(np.ones_like(w_u), 1)
    l_mask = u_mask.T

    return _np2tp(w_p.copy(), w_l.copy(), w_s.copy(), w_u.copy(), u_mask.copy(), l_mask.copy())


class FlowModule(nn.Module):
    """
    Basic interface for a Flow module, used to inform design choices throughout this implementation of Glow.

    In this implementation, the forward flow represents the direction from data to noise (forward because this is the
    direction used during training), while the reverse flow represents the direction from noise to data.
    """
    def __init__(self): super(FlowModule, self).__init__()

    def forward(self, x: T, cond: T=None) -> Tuple[T, T]: raise NotImplementedError

    def reverse(self, y: T, cond: T=None) -> T: raise NotImplementedError


class Split(nn.Module):
    """
    Implementation of a split module, used to split tensors to two, along a particular dimension.
    """

    def __init__(self, dim: int=0):
        """
        :param dim: the dimension along which to split or concatenate tensors; batch dimension is not considered, so
                    as an example, to split along the channel dimension, dim=0 should be supplied
        """
        super(Split, self).__init__()
        self.dim = dim+1

    def forward(self, x: T) -> Tuple[T, T]:
        x, z = x.chunk(2, dim=self.dim)
        return x, z

    def reverse(self, y: T, z: T) -> T:
        return torch.cat([y, z], dim=self.dim)


class ActNorm(FlowModule):
    """
    Implementation of the ActNorm Flow module. This module represents a learnable affine transformation, for each
    channel of the input.
    """

    def __init__(self, n_channels: int):
        """
        :param n_channels: number of input channels of the data
        """
        super(ActNorm, self).__init__()

        self.loc = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, n_channels, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input: T):
        """
        Initializes the values of the ActNorm transformation by setting them so that the inputted batch is standardized.
        In other words, by setting loc=-mean(batch) and scale=1/std(batch).

        :param input: a batch of data
        """
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        _, _, height, width = x.shape

        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        log_abs = torch.log(torch.abs(self.scale))

        logdet = height * width * torch.sum(log_abs)
        return self.scale * (x + self.loc), logdet

    def reverse(self, y: T, cond: T=None) -> T:
        return y / self.scale - self.loc


class ZeroConv2d(nn.Module):
    """
    Helper module to define a zero-initialized convolutional layer, used by Glow in the AffineCoupling layers
    """

    def __init__(self, in_channel, out_channel):
        super(ZeroConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(FlowModule):
    """
    Implementation of the AffineCoupling Flow module used in Glow.
    """

    def __init__(self, n_channels: int, affine: bool=True, hidden_width: int=512):
        """
        :param n_channels: number of input channels
        :param affine: whether the transformation should be affine or only linear
        :param hidden_width: the width that should be used for the hidden layers of the transformation
        """
        super(AffineCoupling, self).__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(n_channels // 2, hidden_width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_width, hidden_width, 3, padding=1),
            nn.ReLU(inplace=True),
            ZeroConv2d(hidden_width, n_channels if self.affine else n_channels // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        in_a, in_b = x.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = torch.tensor(0)

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, y: T, cond=None) -> T:
        out_a, out_b = y.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class InvConv2D(FlowModule):

    def __init__(self, in_channel):
        super().__init__()

        # weight = np.random.randn(in_channel, in_channel)
        # q, _ = la.qr(weight)
        # w_p, w_l, w_u = la.lu(q.astype(np.float32))
        # w_s = np.diag(w_u)
        # w_u = np.triu(w_u, 1)
        # u_mask = np.triu(np.ones_like(w_u), 1)
        # l_mask = u_mask.T
        #
        # w_p = torch.from_numpy(w_p)
        # w_l = torch.from_numpy(w_l)
        # w_s = torch.from_numpy(w_s)
        # w_u = torch.from_numpy(w_u)

        w_p, w_l, w_s, w_u, u_mask, l_mask = _random_LU(in_channel)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        _, _, height, width = x.shape

        weight = self.calc_weight()

        out = F.conv2d(x, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def reverse(self, y: T, cond=None) -> T:
        weight = self.calc_weight()

        return F.conv2d(y, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class Shuffle(FlowModule):

    def __init__(self, dim: int=0):
        super(Shuffle, self).__init__()
        self.perm = None
        self.inv_perm = None
        self.register_buffer('ldet', torch.tensor(0))
        self.dim = dim

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        if self.perm is None:
            self.perm = torch.randperm(x.shape[self.dim+1]).tolist()
            self.inv_perm = np.argsort(self.perm)
        return x.transpose(dim0=0, dim1=self.dim+1)[self.perm].transpose(dim0=self.dim+1, dim1=0), self.ldet

    def reverse(self, y: T, cond=None) -> T:
        return y.transpose(dim0=0, dim1=self.dim+1)[self.inv_perm].transpose(dim0=self.dim+1, dim1=0)


class Squeeze(FlowModule):
    def __init__(self, squeeze_amnt: int=2):
        super(Squeeze, self).__init__()
        self.amnt = squeeze_amnt
        self.register_buffer('ldet', torch.tensor(0))

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        b_size, n_channel, height, width = x.shape
        squeezed = x.view(b_size, n_channel, height//self.amnt, self.amnt, width//self.amnt, self.amnt)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel*self.amnt*self.amnt, height//self.amnt, width//self.amnt)
        return out, self.ldet

    def reverse(self, y: T, cond=None) -> T:
        b_size, n_channel, height, width = y.shape

        unsqueezed = y.view(b_size, n_channel//(self.amnt*self.amnt), self.amnt, self.amnt, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel//(self.amnt*self.amnt), height * self.amnt, width * self.amnt
        )
        return unsqueezed


class FlowSequential(FlowModule):

    def __init__(self, *modules):
        super(FlowSequential, self).__init__()
        self.register_module('layers', nn.ModuleList(modules))
        self.register_buffer('ldet', torch.zeros(1))

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        ldet = self.ldet
        for mod in self.layers:
            x, l = mod.forward(x, cond=cond)
            ldet = ldet + l
        return x, ldet

    def reverse(self, y: T, cond=None) -> T:
        for mod in self.layers[::-1]:
            y = mod.reverse(y, cond=cond)
        return y


class GlowFlow(FlowModule):

    def __init__(self, n_channels: int, affine: bool=True, hidden_layer: int=512):
        """
        A single flow block used in HI-generation
        :param n_channels: the number of input (and output) channels
        :param transf: the transformation to use in the affine coupling layer; if None is supplied, the same
                       transformation that was used in HI-generation is utilized
        :param affine: a boolean indication whether to use an affine or additive layer
        :param hidden_layer: the number of channels in the hidden layer - only used if no transform is supplied
        """
        super(GlowFlow, self).__init__()
        self.register_module('flow',
                             FlowSequential(ActNorm(n_channels=n_channels), InvConv2D(in_channel=n_channels),
                                            AffineCoupling(n_channels=n_channels, affine=affine,
                                                           hidden_width=hidden_layer))
                             )

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        return self.flow.forward(x)

    def reverse(self, y: T, cond=None) -> T:
        return self.flow.reverse(y)


class GlowBlock(FlowModule):

    def __init__(self, n_channels: int, n_flows: int=None, squeeze: bool=True, hidden_layer: int=512,
                 affine: bool=True):
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
        super(GlowBlock, self).__init__()
        chans = 4*n_channels if squeeze else n_channels
        layers = [GlowFlow(n_channels=chans, affine=affine, hidden_layer=hidden_layer) for _ in range(n_flows)]
        layers = FlowSequential(Squeeze(), *layers) if squeeze else FlowSequential(*layers)
        self.register_module('block', layers)

    def forward(self, x: T, cond=None) -> Tuple[T, T]:
        return self.block.forward(x)

    def reverse(self, y: T, cond=None) -> T:
        return self.block.reverse(y)
