from modules.models import CondGlow
import numpy as np
import torch
from tqdm import trange
from torch.optim import Adam
from matplotlib import pyplot as plt
from pathlib import Path
from eval_utils import save_images


def data_augment(batch):
    if np.random.rand() < .5: batch = torch.flip(batch, dims=[-2])
    if np.random.rand() < .5: batch = torch.flip(batch, dims=[-1])

    if np.random.rand() < .25: batch = torch.rot90(batch, k=1, dims=[-2, -1])
    elif np.random.rand() < .25: batch = torch.rot90(batch, k=2, dims=[-2, -1])
    elif np.random.rand() < .25: batch = torch.rot90(batch, k=3, dims=[-2, -1])

    return batch


load_ep = 0

# ------------------------------------------------------------------------ define hyperparameters
# optimizer hyperparameters
lr = 1e-4                   # learning rate used to train model
epochs = 600                # number of epochs to train model
decay_mult = .995           # decay rate of learning rate (set to 1 for no decay)
batch_size = 16             # batch size to use while training

n_conds = 2                 # number of conditional parameters
augment = True              # whether to augment the data with flips and rotations

# model hyperparameters
params = {
    'n_channels': 1,        # number of channels in the input
    'n_blocks': 6,          # number of blocks in the Glow architecture (basically, floor(log_2(image_size)))
    'n_flows': 20,          # number of flows in each level of the Glow architecture
    'affine': True,         # whether the flow steps are affine or not (should usually be true)
    'hidden_width': 32,     # defines the complexity of each of the flow steps
    'learn_priors': True,   # whether to train a Gaussian for each latent variable or not
    'cond_features': n_conds
}

# ------------------------------------------------------------------------ define paths
r_path = ''
path = r_path +\
       f'condHI_maps/wtest_{"aug_" if augment else ""}blocks={params["n_blocks"]}_flows={params["n_flows"]}_' \
       f'hidden={params["hidden_width"]}_batch={batch_size}_nconds={n_conds}/'

sample_path = path + 'samples/'
Path(sample_path).mkdir(exist_ok=True, parents=True)
check_path = path + 'checkpoints/'
Path(check_path).mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------ load data
print('Loading data...', flush=True)
data = np.load(r_path + 'Images_HI_IllustrisTNG_z=5.99.npy')

# preprocessing of the data
data = np.log10(data)
data = (data - np.mean(data))/np.std(data)

# push data through sigmoid for better dynamic range
data = 2/(1+np.exp(-data)) - 1

# turn into a torch tensor with a channel dimension
data = torch.from_numpy(data)[:, None].float()
print('Data loaded.\n\n', flush=True)

# ------------------------------------------------------------------------ load params
import csv

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
    line_count = 0
    for i, row in enumerate(csv_reader):
        if i>0:
            r = [float(it) for it in row[1:-1]]
            for _ in range(15): sim_params.append(r)

conds = torch.from_numpy(np.array(sim_params)).float()
conds = 2*(conds - conds.min(dim=0)[0][None])/(conds.max(dim=0)[0] - conds.min(dim=0)[0])[None] - 1

conds = conds[:, :n_conds]

# ------------------------------------------------------------------------ split training from rest of data
# get indices (leave out every 15-th sample)
inds = [i for i in range(data.shape[0]) if (i+1)%15 <= 12]

data = data[inds]
conds = conds[inds]
print(f'Training on data with shape {list(data.shape)}')

# ------------------------------------------------------------------------ batching function
N = data.shape[0]
n_batches = N//batch_size


def make_dataset():
    inds = np.random.permutation(N)
    dataset = data[inds]
    pars = conds[inds]
    return [(dataset[i:i+batch_size], pars[i:i+batch_size]) for i in range(0, N, batch_size)]


# ------------------------------------------------------------------------ define model, optimizer, and learning params
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CondGlow(**params)
with torch.no_grad(): model.forward(data[:batch_size], conds[:batch_size])

optim = Adam(model.parameters(), lr=lr)

# ------------------------------------------------------------------------ define model, optimizer, and learning params
# load_ep = 350
# model.load_state_dict(torch.load(check_path + f'model_{load_ep:04}.pt', map_location=torch.device(device)))
# optim.load_state_dict(torch.load(check_path + f'optim_{load_ep:04}.pt', map_location=torch.device(device)))
# print(f'Learning rate is: {optim.param_groups[0]["lr"]:.2E}')

model, data, conds = model.to(device), data.to(device), conds.to(device)

# ------------------------------------------------------------------------ train model
losses = []
pbar = trange(epochs, ascii=True)
for i in pbar:
    dataset = make_dataset()
    ep_loss = []
    for j, (batch, cond) in enumerate(dataset):
        if augment: batch = data_augment(batch)

        # model forward and backward
        optim.zero_grad()
        z, log_p = model.forward(batch, cond)
        loss = -torch.mean(log_p)/(batch.shape[-3]*batch.shape[-2]*batch.shape[-1])
        loss.backward()

        # clip gradients for good measure
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optim.step()

        ep_loss.append(loss.item())
        if not j%100: pbar.set_postfix_str(f'epoch {i+load_ep}; {j}/{n_batches}; '
                                           f'step loss: {ep_loss[-1]:.2f}, avg.: {np.mean(ep_loss):.2f}, '
                                           f'lr: {optim.param_groups[0]["lr"]:.2E}')

        # plot samples
        if not j%(n_batches//(4 if j<5 else 2)):
            it = i*n_batches + j + 1

            inds = np.random.choice(conds.shape[0], 16)
            cs = conds[inds]
            samps = model.sample(N=16, cond=cs, clip_val=500).cpu().detach().numpy()[:, 0]
            samps = .5*np.clip(samps, -1, 1) + 1

            save_images([samp for samp in samps], save_p=sample_path + f'{i+load_ep+1:04}-{j+1:04}.png')
            plt.close('all')
            save_images([samp[0].cpu().numpy() for samp in data[inds]], save_p=sample_path + f'real.png')
            plt.close('all')

    # plot loss over epochs
    losses.append(np.mean(ep_loss))
    if load_ep+i>0:
        plt.figure()
        plt.plot(np.arange(len(losses[1:]))+load_ep, losses[1:], lw=2)
        plt.xlabel('epoch')
        plt.ylabel('NLL')
        plt.savefig(path + 'loss.png')

    # save checkpoints
    if not (i+load_ep+1)%(10 if i+load_ep < 50 else 50):
        torch.save(model.state_dict(), check_path + f'model_{i+load_ep+1:04}.pt')
        torch.save(optim.state_dict(), check_path + f'optim_{i+load_ep+1:04}.pt')

    # update learning rate
    optim.param_groups[0]['lr'] = max(optim.param_groups[0]['lr']*decay_mult, 1e-5)
