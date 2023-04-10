import torch
import numpy as np
from torch import nn
from .flow_layers import GlowBlock, ActNorm, InvConv2D, FlowSequential, Squeeze, Shuffle
from .conditional_layers import _cond_block
from .priors import SplitGaussianPrior, GaussianPrior, CondGaussianPrior, SplitCondPrior
from typing import List, Tuple, Union

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
                 cond_priors: bool = False, input_size: int = None, cond_hidden: int = None, add_actnorm: bool = False,
                 clamp_val: float = 1e-3):
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
        :param cond_hidden: width of hidden layer in conditional priors; if None, the same width as in the rest of
                            the model is used
        :param add_actnorm: a bool indicating whether to add an actnorm at beginning of each block or not
        :param clamp_val: clamping value for sigmoid functions
        """
        super(CondGlow, self).__init__()
        self.n_flows, self.n_blocks = n_flows, n_blocks
        self.cond_features = cond_features
        self.temperature = temperature
        self.affine = affine
        self.hidden_width = hidden_width
        self.priors = nn.ModuleList()
        self.in_shape, self.latent_shapes, self.latent_len = None, None, None
        self.start = start
        cond_hidden = cond_hidden if cond_hidden is not None else hidden_width

        if self.start: self.strt = FlowSequential(ActNorm(n_channels))
        if cond_priors: input_size = input_size*input_size//4
        n_channels = 4 * n_channels
        for i in range(self.n_blocks):
            # define conditional block
            bl = _cond_block(n_channels, cond_features, hidden_width, affine, n_flows,
                             add_actnorm=add_actnorm, clamp_val=clamp_val)

            # add conditional priors
            if cond_priors:
                self.priors.append(SplitCondPrior(bl, cond_features, input_size*n_channels, cond_hidden, temperature))
            # use Gaussian prior
            else: self.priors.append(SplitGaussianPrior(bl, learn_params=learn_priors, temperature=temperature))

            # after each block, input is squeezed and split, so channels increase by a factor of 2
            n_channels = 2 * n_channels
            if cond_priors: input_size = input_size//4

        # add final conditional block
        bl = _cond_block(n_channels//4, cond_features, hidden_width, affine, n_flows, squeeze=False,
                         add_actnorm=add_actnorm, clamp_val=clamp_val)

        if cond_priors:
            self.priors.append(
                FlowSequential(bl, CondGaussianPrior(cond_features, n_channels//4, cond_hidden, temperature))
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

    def reverse(self, zs: Union[list, T] = None, cond: T = None, clip_val: int = None):
        if self.latent_len is None: self.latent_len = np.sum([np.prod(sh) for sh in self.latent_shapes])

        if zs is None: zs = self.sample_latent(N=cond.shape[0])
        elif torch.is_tensor(zs): zs = self.latent_to_list(zs)
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

    def latent_to_list(self, z: T):
        assert z.shape[1] == self.latent_len
        ret = []
        sh = 0
        for shape in self.latent_shapes:
            ret.append(z[:, sh:np.prod(shape)+sh].reshape(-1, *shape))
            sh += np.sum(shape)
        return ret

    def list_to_latent(self, z: list):
        N = z[0].shape[0]
        return torch.concat([a.reshape(N, -1) for a in z], dim=1)

    @torch.no_grad()
    def sample(self, cond: T, N: int, clip_val: int = None):
        return self.reverse(self.sample_latent(N), cond=cond, clip_val=clip_val)

    def latent_ll(self, z: Union[list, T], cond: T=None):
        if torch.is_tensor(z): z = self.latent_to_list(z)
        N = z[0].shape[0]
        ll = 0
        for i, pr in enumerate(self.priors):
            ll = ll -.5*torch.sum((((z[i] - pr.mean)**2)*torch.exp(-pr.log_var)).reshape(N, -1), dim=1) \
                 -.5*torch.sum(pr.log_var)
        return ll