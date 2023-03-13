import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip
import csv


def load_HI_data(r_path: str= '', return_funcs: bool=False, test_confs: int=3, n_params: int=2,
                 sigmoid_normalization: bool=True):
    """
    Loads and normalizes the HI data
    :param r_path: root path for HI data
    :param return_funcs: a flag indicating whether the normalization functions should be returned or not
    :param test_confs: a flag indicating whether the test should be returned
    :param n_params: number of parameters to return from the simulations
    :param sigmoid_normalization: a flag indicating whether the normalization should pass through a sigmoid function
    :return:
    """

    # ---------------------------------------------------------------------- Load HI maps and transform
    data = np.load(r_path + 'Images_HI_IllustrisTNG_z=5.99.npy')

    # preprocessing of the data
    data = np.log10(data)

    # turn into a torch tensor with a channel dimension
    data = torch.from_numpy(data)[:, None].float()

    # ---------------------------------------------------------------------- Define data normalization transforms
    # if sigmoid normalization, standardize data
    if sigmoid_normalization: dmean, dstd = torch.mean(data), torch.std(data)
    # if not sigmoid, make sure normalization is so that the data is between 0 and 1
    else: dmean, dstd = torch.min(data), (torch.max(data) - torch.min(data))

    def norm_func(d: torch.Tensor) -> torch.Tensor:
        d = (d - dmean) / dstd
        if sigmoid_normalization:
            d = 2 / (1+torch.exp(-d)) - 1
        return d

    def ret_func(d: torch.Tensor) -> torch.Tensor:
        if sigmoid_normalization:
            d = (torch.clamp(d, -.999, .999) + 1)/2
            d = torch.log(d/(1-d))
        return d*dstd + dmean

    data_funcs = tuple([norm_func, ret_func])
    data = norm_func(data)

    # ---------------------------------------------------------------------- Load parameter values
    sim_params = []
    param_names = [
        r'$\Omega_m$',
        r'$\sigma_8$',
        r'$A_{SN1}$',
        r'$A_{AGN1}$',
        r'$A_{SN2}$',
        r'$A_{AGN2}$',
    ]
    with open(r_path + 'CAMELs_params.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for i, row in enumerate(csv_reader):
            if i > 0:
                r = [float(it) for it in row[1:-1]]
                for _ in range(15): sim_params.append(r)

    conds = torch.from_numpy(np.array(sim_params)).float()

    # use only the number of simulation parameters that are needed (2 for cosmological)
    conds = conds[:, :n_params]
    param_names = param_names[:n_params]

    # ---------------------------------------------------------------------- Define parameters normalization transforms
    cmin, cmax = conds.min(dim=0)[0], conds.max(dim=0)[0]

    def cnorm_func(c: torch.Tensor) -> torch.Tensor: return 2 * (c - cmin[None]) / (cmax - cmin)[None] - 1

    def cret_func(n: torch.Tensor) -> torch.Tensor: return (cmax - cmin)[None]*(n+1)/2 + cmin[None]

    cond_funcs = tuple([cnorm_func, cret_func])
    conds = cnorm_func(conds)

    # ---------------------------------------------------------------------- Split into train and test
    train_inds = [True if (i + 1) % 15 <= 15 - test_confs else False for i in range(data.shape[0])]
    test_inds = [not ind for ind in train_inds]

    dtrain, dtest = data[train_inds], data[test_inds]
    ctrain, ctest = conds[train_inds], conds[test_inds]

    if return_funcs:
        return (dtrain, ctrain), (dtest, ctest), param_names, data_funcs, cond_funcs
    return (dtrain, ctrain), (dtest, ctest)


def HI_dataset(r_path: str='', train: bool=True, sigmoid_normalization: bool=False):
    (dtrain, ctrain), (dtest, ctest) = load_HI_data(r_path, sigmoid_normalization=sigmoid_normalization)
    return TensorDataset(dtrain, ctrain) if train else TensorDataset(dtest, ctest)


class CyclicShiftTransform:

    def __init__(self, image_size: int):
        self.size = image_size

    def __call__(self, sample):
        image, label = sample
        shiftx, shifty = np.random.choice(self.size-1, 2)
        return torch.roll(image, shifts=[shiftx, shifty], dims=[-2, -1]), label


class RandomFlip:

    def __init__(self, dim: int=0):
        self.dim = dim

    def __call__(self, sample):
        image, label = sample
        if np.random.rand() > .5: image = torch.flip(image, dims=[self.dim+1])
        return image, label


HI_transform = Compose([
    CyclicShiftTransform(64),
    RandomFlip(1),
    RandomFlip(2),
])


class EarlyStop:

    def __init__(self, test_loader: DataLoader, validate_every: int=5, patience: int=5, ):
        self.loader = test_loader
        self.counter = 0
        self.epoch = 0
        self.losses = []
        self.epochs = []
        self.validate_every = validate_every
        self.patience = patience
        self.stop = False

    def __call__(self, loss_func):
        self.epoch += 1

        if not self.epoch%self.validate_every:
            loss = []
            with torch.set_grad_enabled(False):
                for d in self.loader: loss.append(loss_func(d).item())
            loss = np.mean(loss)
            self.losses.append(loss)
            self.epochs.append(self.epoch)

            if self.losses[-1] > self.losses[-2]: self.counter += 1
            else: self.counter = 0
            if self.counter > self.patience: self.stop = True

        return self.stop


