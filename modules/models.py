import torch
import numpy as np
from torch import nn
from .flow_layers import GlowBlock, ActNorm, InvConv2D, FlowSequential, Squeeze, Shuffle
from .conditional_layers import _cond_block
from .priors import SplitGaussianPrior, GaussianPrior, CondGaussianPrior, SplitCondPrior
from typing import List, Tuple

T = torch.Tensor


class Glow(nn.Module):

    def __init__(self, n_channels: int, n_flows: int, n_blocks: int, temperature: float = 0.8, affine: bool = True,
                 hidden_width: int = 512, learn_priors: bool = False):
        super(Glow, self).__init__()
        self.n_flows, self.n_blocks = n_flows, n_blocks
        self.temperature = temperature
        self.affine = affine
        self.hidden_width = hidden_width
        self.priors = nn.ModuleList()
        self.in_shape, self.latent_shapes = None, None

        for i in range(self.n_blocks):
            bl = GlowBlock(n_channels=n_channels, n_flows=self.n_flows, affine=self.affine,
                           hidden_layer=self.hidden_width)
            self.priors.append(SplitGaussianPrior(bl, learn_params=learn_priors, temperature=temperature))
            n_channels = 2 * n_channels
        bl = GlowBlock(n_channels=n_channels, n_flows=self.n_flows, affine=self.affine,
                       hidden_layer=self.hidden_width, squeeze=False)
        self.priors.append(GaussianPrior(bl, learn_params=False, temperature=temperature))

    def forward(self, x: T) -> Tuple[List[T], T]:
        # save input shape
        if self.in_shape is None: self.in_shape = x.shape[1:]
        shapes = []

        # iterate over all the priors
        zs = []
        log_prob = torch.tensor(0)
        for pr in self.priors[:-1]:
            x, z, lp = pr.forward(x)
            zs.append(z)
            log_prob = log_prob + lp
            if self.latent_shapes is None: shapes.append(list(z.shape[1:]))

        z, lp = self.priors[-1].forward(x)
        zs.append(z)
        log_prob = log_prob + lp

        # save shapes of the latent variables, in order to be able to easily sample new points
        if self.latent_shapes is None:
            shapes.append(list(x.shape[1:]))
            self.latent_shapes = shapes

        # return latent variables and the calculated log-probability
        return zs, log_prob

    @torch.no_grad()
    def reverse(self, zs: list = None):
        x = self.priors[-1].reverse(zs[-1] if zs else None)
        for i, pr in enumerate(self.priors[::-1][1:]):
            x = pr.reverse(x, zs[-i - 2] if zs else None)
        return x

    @torch.no_grad()
    def sample_latent(self, N: int, device='cpu'):
        return [self.temperature * torch.randn(N, *shp, device=device) for shp in self.latent_shapes]


class CondGlow(nn.Module):

    def __init__(self, n_channels: int, cond_features: int, n_flows: int, n_blocks: int, temperature: float = 1,
                 affine: bool = True, hidden_width: int = 512, learn_priors: bool = False, start: bool = False,
                 cond_priors: bool = False, input_size: int = None):
        """
        Creates the conditional Glow model used for HIGlow
        :param n_channels: number of channels in the input images
        :param cond_features: number of conditional features to use
        :param n_flows: number of flows in each cGlow block
        :param n_blocks: number of blocks to use; at most log2(size) where size is the smallest spatial dimension
        :param temperature: the temperature used in the prior (should usually just be set to 1)
        :param affine: bool indicating whether to use affine coupling layers or just linear coupling layers
        :param hidden_width: width of hidden layers in the affine coupling and affine injector layers
        :param learn_priors: bool indicating whether the prior's mean and variance should be learned
        :param start: bool indicating whether the first layer in the model should be an actnorm layer or not
        :param cond_priors: bool indicating whether to use conditional priors or not
        :param input_size: if cond_priors is True, the size of the input must be defined to properly work
        """
        super(CondGlow, self).__init__()
        self.n_flows, self.n_blocks = n_flows, n_blocks
        self.cond_features = cond_features
        self.temperature = temperature
        self.affine = affine
        self.hidden_width = hidden_width
        self.priors = nn.ModuleList()
        self.in_shape, self.latent_shapes = None, None
        self.start = start

        if self.start: self.strt = FlowSequential(ActNorm(n_channels))
        if cond_priors: input_size = input_size*input_size//4
        n_channels = 4 * n_channels
        for i in range(self.n_blocks):
            # define conditional block
            bl = _cond_block(n_channels, cond_features, hidden_width, affine, n_flows)

            # add conditional priors
            if cond_priors:
                self.priors.append(SplitCondPrior(bl, cond_priors, n_channels*input_size, hidden_width, temperature))
            # use Gaussian prior
            else: self.priors.append(SplitGaussianPrior(bl, learn_params=learn_priors, temperature=temperature))

            # after each block, input is squeezed and split, so channels increase by a factor of 2
            n_channels = 2 * n_channels
            if cond_priors: input_size = input_size//4

            # add final conditional block
        bl = _cond_block(n_channels // 4, cond_features, hidden_width, affine, n_flows, squeeze=False)

        if cond_priors:
            self.priors.append(
                FlowSequential(bl, CondGaussianPrior(cond_features, n_channels*input_size, hidden_width, temperature))
            )
        else: self.priors.append(GaussianPrior(bl, learn_params=learn_priors, temperature=temperature))

    def forward(self, x: T, cond: T) -> Tuple[List[T], T]:
        # save input shape
        if self.in_shape is None: self.in_shape = x.shape[1:]
        shapes = []

        log_prob = 0
        if self.start: x, log_prob = self.strt.forward(x, cond)

        # iterate over all the priors
        zs = []
        for pr in self.priors[:-1]:
            x, z, lp = pr.forward(x, cond)
            zs.append(z)
            log_prob = log_prob + lp
            if self.latent_shapes is None: shapes.append(list(z.shape[1:]))

        z, lp = self.priors[-1].forward(x, cond)
        zs.append(z)
        log_prob = log_prob + lp

        # save shapes of the latent variables, in order to be able to easily sample new points
        if self.latent_shapes is None:
            shapes.append(list(x.shape[1:]))
            self.latent_shapes = shapes

        # return latent variables and the calculated log-probability
        return zs, log_prob

    @torch.no_grad()
    def reverse(self, zs: list = None, cond: T = None, clip_val: int = None):
        if zs is None: zs = self.sample_latent(N=cond.shape[0])
        x = self.priors[-1].reverse(zs[-1], cond=cond)
        for i, pr in enumerate(self.priors[::-1][1:]):
            x = pr.reverse(x, zs[-i - 2] if zs else None, cond=cond)
            if clip_val is not None: x = torch.clamp(x, min=-clip_val, max=clip_val)
        if self.start: x = self.strt.reverse(x, cond)
        if clip_val is not None: x = torch.clamp(x, min=-clip_val, max=clip_val)
        return x

    @torch.no_grad()
    def sample_latent(self, N: int):
        return [self.temperature * torch.randn(N, *shp, device=list(self.parameters())[0].device) for shp in
                self.latent_shapes]

    @torch.no_grad()
    def sample(self, cond: T, N: int, clip_val: int = None):
        return self.reverse(self.sample_latent(N), cond=cond, clip_val=clip_val)
