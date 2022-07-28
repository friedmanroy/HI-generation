from modules import Glow
import numpy as np
import torch
from tqdm import trange
from torch.optim import Adam
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from eval_utils import save_images
from pathlib import Path


load_ep = 0


# ------------------------------------------------------------------------ define hyperparameters
# optimizer hyperparameters
lr = 1e-3
epochs = 600
decay_mult = .994  # multiplies the learning rate by this decay each epoch, lazy workaround for weight decay
batch_size = 100

# model hyperparameters
params = {
    'n_channels': 1,
    'n_blocks': 3,
    'temperature': 1,
    'n_flows': 10,
    'affine': True,
    'hidden_width': 16,
    'learn_priors': True
}

# ------------------------------------------------------------------------ define paths
# the following is the root path where the model checkpoints and samples throughout training will be saved
path = f'HI_maps/quant_blocks={params["n_blocks"]}_flows={params["n_flows"]}_' \
       f'hidden={params["hidden_width"]}_batch={batch_size}/'

sample_path = path + 'samples/'
Path(sample_path).mkdir(exist_ok=True, parents=True)
check_path = path + 'checkpoints/'
Path(check_path).mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------ load data
data = np.load('Images_HI_IllustrisTNG_z=5.99.npy')
data = np.log10(data)
data = (data - np.mean(data))/np.std(data)

# push data through sigmoid for better dynamic range, and extend range to (-1, 1)
data = 2/(1+np.exp(-data)) - 1

# turn into a torch tensor with a channel dimension
data = torch.from_numpy(data)[:, None].float()

# ------------------------------------------------------------------------ batching function
N = data.shape[0]
n_batches = N//batch_size


def make_dataset():
    """
    Function to create batches of normal data - if using a torch dataset, there is no reason to use this
    :return: a list of batches that should be used during training
    """
    dataset = data[np.random.permutation(N)]
    return [dataset[i:i+batch_size] for i in range(0, N, batch_size)]


# ------------------------------------------------------------------------ define model, optimizer, and learning params
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Glow(**params)
model, data = model.to(device), data.to(device)
with torch.no_grad(): model.forward(data[:batch_size])

optim = Adam(model.parameters(), lr=lr)

# ------------------------------------------------------------------------ load pretrained model
# load_ep = 125
# model.load_state_dict(torch.load(check_path + f'model_{load_ep:04}.pt', map_location=torch.device(device)))
# optim.load_state_dict(torch.load(check_path + f'optim_{load_ep:04}.pt', map_location=torch.device(device)))
# print(f'{optim.param_groups[0]["lr"]:.2E}')

# ------------------------------------------------------------------------ train model
losses = []
pbar = trange(epochs, ascii=True)
for i in pbar:
    model.zero_grad()
    data = data[np.random.permutation(N)]
    dataset = make_dataset()
    ep_loss = []
    for j, batch in enumerate(dataset):
        # randomly mirror the data to add some augmentation (essentially to NX4)
        if np.random.rand() > .5: batch = torch.flip(batch, dims=[-2])
        if np.random.rand() > .5: batch = torch.flip(batch, dims=[-1])

        # model forward and backward
        z, log_p = model.forward(batch)
        loss = -torch.mean(log_p)
        loss.backward()

        # clip gradients (for good measure)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optim.step()

        ep_loss.append(loss.item())
        pbar.set_postfix_str(f'epoch {i+load_ep}; {j}/{n_batches}; '
                             f'step loss: {ep_loss[-1]:.2f}, avg.: {np.mean(ep_loss):.2f}, '
                             f'lr: {optim.param_groups[0]["lr"]:.2E}')

        # plot samples, plot loss and make checkpoints
        if not j%(n_batches//4) and j>0:
            it = i*n_batches + j + 1

            # generate samples
            samps = np.clip(model.reverse(
                model.sample_latent(N=16, device=device)
            ).cpu().detach().numpy()[:, 0], -1, 1)/2 + 1

            # save samples as a grid of images
            save_images([samp for samp in samps], save_p=sample_path + f'{i+load_ep+1:04}-{j+1:04}.png')
            plt.close('all')

            # save grid of real images
            save_images([samp[0].cpu().numpy() for samp in data[:16]], save_p=sample_path + f'real.png')
            plt.close('all')

            # save a graph of the negative log-likelihood as a function of epochs
            if load_ep+i>0:
                plt.figure()
                plt.plot(np.arange(len(losses[1:]))+load_ep, losses[1:], lw=2)
                plt.xlabel('epoch')
                plt.ylabel('NLL')
                plt.savefig(path + 'loss.png')

    losses.append(np.mean(ep_loss))

    # decay learning rate
    optim.param_groups[0]['lr'] = optim.param_groups[0]['lr']*decay_mult

    # create checkpoint for optimizer and model
    if not (i+load_ep+1)%10:
        torch.save(model.state_dict(), check_path + f'model_{i+load_ep+1:04}.pt')
        torch.save(optim.state_dict(), check_path + f'optim_{i+load_ep+1:04}.pt')
