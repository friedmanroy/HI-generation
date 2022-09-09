import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, List, Tuple
from .flow_layers import Split, FlowModule, T, _diff_clamp
from .conditional_layers import _get_MLP
import numpy as np


class GaussianPrior(nn.Module):

    def __init__(self, flow: FlowModule, learn_params: bool=False, temperature: float=1.):
        """
        Represents a diagonal Gaussian prior over a flow module
        :param flow: the flow module that should be wrapped by the prior
        :param learn_params: a boolean controlling whether the mean and diagonal variance of the prior should be learned
        :param temperature: the sampling temperature to use for sampling/reverse operations
        """
        super(GaussianPrior, self).__init__()
        if learn_params:
            self.mean = nn.Parameter(torch.zeros(1))
            self.log_var = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('mean', torch.zeros(1))
            self.register_buffer('log_var', torch.zeros(1))
        self.register_module('flow', flow)
        self.shape = None
        self.temp = temperature

    def _initialize(self, x: T, cond: T=None):
        """
        Initializes the parameters of the prior for further learning
        :param inp: the input used in order to get the shapes needed
        """
        with torch.no_grad(): z, _ = self.flow.forward(x, cond=cond)
        self.shape = list(z.shape[1:])
        self.mean.data = torch.zeros_like(z[0])
        self.log_var.data = torch.zeros_like(z[0])

    def _logp(self, z: T):
        m = (z - self.mean[None])*torch.sqrt(torch.exp(-self.log_var[None]))
        logp = -0.5*torch.sum(m.reshape(m.shape[0], -1)**2, dim=1) \
               -0.5*np.prod(self.shape)*np.log(2*np.pi) \
               -0.5*torch.sum(self.log_var)
        return logp

    def forward(self, x: T, cond: T=None):
        if self.shape is None: self._initialize(x, cond=cond)
        z, logdet = self.flow.forward(x, cond=cond)
        return z, self._logp(z) + logdet

    def reverse(self, z: T=None, cond: T=None, N: int=1):
        if z is None:
            z = self.mean[None] + \
                self.temp*torch.sqrt(torch.exp(self.log_var))*\
                torch.randn(N, *self.shape, dtype=self.mean.dtype, device=self.mean.device)
        return self.flow.reverse(z, cond=cond)

    def likelihood(self, x: T) -> T:
        _, ll = self.forward(x)
        return ll

    def sample(self, N: int=1, cond: T=None) -> T:
        return self.reverse(N=N, cond=cond)


class CondGaussianPrior(FlowModule):

    def __init__(self, n_conds: int, out_size: int, hidden_width: int=64, temperature: float=1.):
        """

        :param n_conds:
        :param out_size:
        :param hidden_width:
        :param temperature:
        """
        super(CondGaussianPrior, self).__init__()
        self.register_module('mean', _get_MLP(n_conds, out_size, hidden_width, zero=True))
        self.register_module('log_var', _get_MLP(n_conds, out_size, hidden_width, zero=True))
        self.shape = None
        self.temp = temperature

    def _initialize(self, x: T, cond: T):
        """
        Initializes the parameters of the prior for further learning
        :param x: the input used in order to get the shapes needed
        """
        with torch.no_grad(): z, _ = self.flow.forward(x, cond=cond)
        self.shape = list(z.shape[1:])

    def _logp(self, z: T, cond: T):
        mean = self.mean(cond).reshape(-1, self.shape)
        log_var = _diff_clamp(self.log_var(cond), -3, 5)
        m = (z - mean)*torch.sqrt(torch.exp(-log_var.reshape(-1, self.shape)))
        logp = -0.5*torch.sum(m.reshape(m.shape[0], -1)**2, dim=1) \
               -0.5*np.prod(self.shape)*np.log(2*np.pi) \
               -0.5*torch.sum(log_var, dim=1)
        return logp

    def forward(self, x: T, cond: T=None):
        if self.shape is None: self._initialize(x, cond=cond)
        return x, self._logp(x, cond)

    def reverse(self, y: T=None, cond: T=None):
        return y

    def likelihood(self, x: T, cond: T) -> T:
        _, ll = self.forward(x, cond)
        return ll

    def sample(self, N: int=1, cond: T=None) -> T:
        mean = self.mean(cond).reshape(-1, self.shape)
        log_var = _diff_clamp(self.log_var(cond), -3, 5).reshape(-1, self.shape)
        z = mean + \
            self.temp * torch.sqrt(torch.exp(log_var)) * \
            torch.randn(N, *self.shape, dtype=mean.dtype, device=mean.device)
        return self.reverse(z, cond=cond)


class SplitGaussianPrior(nn.Module):

    def __init__(self, flow: FlowModule, learn_params: bool=False, temperature: float=1.):
        super(SplitGaussianPrior, self).__init__()
        if learn_params:
            self.mean = nn.Parameter(torch.zeros(1))
            self.log_var = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('mean', torch.zeros(1))
            self.register_buffer('log_var', torch.zeros(1))
        self.register_module('flow', flow)
        self.shape = None
        self.temp = temperature
        self.split = Split()

    def _initialize(self, x: T):
        self.shape = list(x.shape[1:])
        self.mean.data = torch.zeros_like(x[0])
        self.log_var.data = torch.zeros_like(x[0])

    def _logp(self, z: T):
        m = (z - self.mean[None]) * torch.sqrt(torch.exp(-self.log_var[None]))
        logp = - 0.5 * torch.sum(m.reshape(m.shape[0], -1) ** 2, dim=1) \
               - 0.5 * np.prod(self.shape) * np.log(2 * np.pi) \
               - 0.5 * torch.sum(self.log_var)
        return logp

    def forward(self, x: T, cond: T=None) -> Tuple[T, T, T]:
        x, logdet = self.flow.forward(x, cond=cond)
        x, z = self.split.forward(x)
        if self.shape is None: self._initialize(z)
        return x, z, self._logp(z) + logdet

    def reverse(self, x: T, z: T=None, cond: T=None, N: int=None) -> T:
        if z is None:
            z = self.mean[None] + \
                self.temp * torch.sqrt(torch.exp(self.log_var)) * \
                torch.randn(N, *self.shape, dtype=self.mean.dtype, device=self.mean.device)
        return self.flow.reverse(self.split.reverse(x, z), cond=cond)

    def likelihood(self, x: T, cond: T=None) -> Tuple[T, T]:
        x, _, ll = self.forward(x, cond=cond)
        return x, ll

    def sample(self, x: T, N: int=1, cond: T=None) -> T:
        return self.reverse(x=x, cond=cond, N=N)


class SplitCondPrior(nn.Module):

    def __init__(self, flow: FlowModule, n_conds: int, out_size: int, hidden_width: int=64,
                 temperature: float=1.):
        super(SplitCondPrior, self).__init__()
        self.register_module('prior', CondGaussianPrior(n_conds, out_size, hidden_width=hidden_width,
                                                        temperature=temperature))
        self.register_module('split', SplitOp(flow))

    def forward(self, x: T, cond: T=None) -> Tuple[T, T, T]:
        x, logdet = self.flow.forward(x, cond=cond)
        x, z = self.split.forward(x)
        if self.shape is None: self._initialize(z)
        return x, z, self.prior.likelihood(z, cond=cond) + logdet

    def reverse(self, x: T, z: T=None, cond: T=None) -> T:
        return self.flow.reverse(self.split.reverse(x, z), cond=cond)

    def likelihood(self, x: T, cond: T=None) -> Tuple[T, T]:
        x, z, log_det = self.split.forward(x, cond=cond)
        return self.prior.likelihood(x, cond=cond) + log_det

    def sample(self, x: T, N: int=1, cond: T=None) -> T:
        z = self.prior.sample(N=N, cond=cond)
        return self.reverse(x=x, z=z, cond=cond)


class SplitOp(nn.Module):

    def __init__(self, flow: FlowModule):
        super(SplitOp, self).__init__()
        self.register_module('flow', flow)
        self.shape = None
        self.split = Split()

    def _initialize(self, x: T):
        self.shape = list(x.shape[1:])

    def forward(self, x: T, cond: T=None) -> Tuple[T, T, T]:
        x, logdet = self.flow.forward(x, cond=cond)
        x, z = self.split.forward(x)
        if self.shape is None: self._initialize(z)
        return x, z, logdet

    def reverse(self, x: T, z: T, cond: T=None) -> T:
        return self.flow.reverse(self.split.reverse(x, z), cond=cond)


class PPCAPrior(nn.Module):

    def __init__(self):
        super(PPCAPrior, self).__init__()
        self.mean = nn.Parameter(torch.zeros(1))
        self.W = nn.Parameter(torch.zeros(1))
        self.log_phi = nn.Parameter(torch.tensor(0.))
        self.shape = None

    def _initialize(self, x: T, z: T):
        self.shape = list(x.shape[1:])
        self.lat_dim = int(z.shape[1])
        self.dim = int(np.prod(list(x.shape[1:])))
        self.mean.data = torch.zeros_like(x[0])
        self.W.data = .01*torch.randn_like(x[0].flatten())[:, None]@torch.randn_like(z[0])[None, :]

    def _phi(self):
        return _diff_clamp(self.log_phi, -5, 3)

    def _post(self, z: T, M: T, m: T):
        m = z-torch.linalg.solve(M, (m@self.W).T).T
        mahala = torch.sum(m*(m@M), dim=-1)*torch.exp(-self._phi())
        det = self.lat_dim*self._phi() - torch.logdet(M)
        return -.5*(mahala + det + self.lat_dim*np.log(2*np.pi))

    def _evidence(self, M: T, m: T) -> T:
        Wm = m@self.W
        mahala = (
                         torch.sum(m*m, dim=1) -
                         torch.sum(Wm * torch.linalg.solve(M, Wm.T).T, dim=1)
                 )*torch.exp(-self._phi())
        det = (self.dim - self.lat_dim)*self._phi() + torch.logdet(M)
        return -.5*(mahala + det + self.dim*np.log(2*np.pi))

    def _logp(self, x: T, z: T):
        m = (x-self.mean[None]).reshape(x.shape[0], self.dim)
        M = self.W.T@self.W + torch.eye(self.lat_dim)*torch.exp(self._phi())
        if z is not None: return self._evidence(M, m) + self._post(z, M, m)
        return self._evidence(M, m)

    def forward(self, x: T, z: T=None) -> Tuple[T, T]:
        if self.shape is None: self._initialize(x, z)
        return self._logp(x, z)

    def posterior(self, x: T) -> Tuple[T, T]:
        M = self.W.T @ self.W + torch.eye(self.lat_dim, device=self.W.device)*torch.exp(self._phi())
        mean = torch.linalg.solve(M, self.W.T@(x.reshape(x.shape[0], -1) - self.mean[None]).T)
        prec = M*torch.exp(-self._phi())
        return mean, prec

    def sample(self, z: T=None, N: int=1) -> T:
        if z is None: z = torch.randn(N, self.lat_dim, dtype=self.mean.dtype, device=self.mean.device)
        x = (self.W@z.T).T.reshape(z.shape[0], *self.shape)
        return x + self.mean[None] + torch.randn_like(x)*torch.exp(self._phi()/2)
