import torch
import numpy as np
from torch import nn
from .flow_layers import GlowBlock, ActNorm, InvConv2D, FlowSequential, Squeeze, Shuffle
from .conditional_layers import CondGlowBlock, CondAffineCoupling, AffineInjector
from .priors import SplitGaussianPrior, GaussianPrior, PPCAPrior, SplitOp
from typing import Union, List, Tuple

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


def _cond_block(n_channels, cond_features, hidden_width, affine, n_flows, squeeze: bool = True, shuffle: bool = False):
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


class CondGlow(nn.Module):

    def __init__(self, n_channels: int, cond_features: int, n_flows: int, n_blocks: int, temperature: float = 1,
                 affine: bool = True, hidden_width: int = 512, learn_priors: bool = False, start: bool = False):
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
        n_channels = 4 * n_channels
        for i in range(self.n_blocks):
            bl = _cond_block(n_channels, cond_features, hidden_width, affine, n_flows)
            self.priors.append(SplitGaussianPrior(bl, learn_params=learn_priors, temperature=temperature))
            n_channels = 2 * n_channels
        bl = _cond_block(n_channels // 4, cond_features, hidden_width, affine, n_flows, squeeze=False)
        self.priors.append(GaussianPrior(bl, learn_params=learn_priors, temperature=temperature))

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
