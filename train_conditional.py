from modules.models import CondGlow
import numpy as np
import torch
from tqdm import trange
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from pathlib import Path
from eval_utils import save_images
from HI_utils import HI_dataset, HI_transform, EarlyStop
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-b', type=int, default=6, help='number of blocks to use (default=6)')
parser.add_argument('-f', type=int, default=10, help='number of flows per block to use (default=10)')
parser.add_argument('-a', type=int, default=1, help='0 for no augment, else augment (defualt=1)')
parser.add_argument('-w', type=int, default=16, help='hidden width to use (defualt=16)')
parser.add_argument('-e', type=int, default=600, help='number of epochs (defualt=600)')
parser.add_argument('-bs', type=int, default=64, help='batch size to use for training (defualt=64)')

args = parser.parse_args()

load_ep = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------------------------------------------------ define hyperparameters
# optimizer hyperparameters
lr = 3e-4
epochs = args.e
decay_mult = .995
batch_size = args.bs
n_conds = 2
augment = args.a != 0

# model hyperparameters
params = {
    'n_channels': 1,
    'n_blocks': args.b,
    'temperature': 1,
    'n_flows': args.f,
    'affine': True,
    'hidden_width': args.w,
    'learn_priors': True,
    'cond_features': n_conds
}

# ------------------------------------------------------------------------ define paths
r_path = ''
path = r_path +\
       f'condHI_maps/{"aug_" if augment else ""}blocks={params["n_blocks"]}_flows={params["n_flows"]}_' \
       f'hidden={params["hidden_width"]}_batch={batch_size}_nconds={n_conds}/'

sample_path = path + 'samples/'
Path(sample_path).mkdir(exist_ok=True, parents=True)
check_path = path + 'checkpoints/'
Path(check_path).mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------------ load data
train_dataset = HI_dataset(r_path, train=True)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=2 if torch.cuda.is_available() else 0,
                    pin_memory=torch.cuda.is_available(),
                    )
n_batches = len(loader)

test_dataset = HI_dataset(r_path, train=False)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                         num_workers=2 if torch.cuda.is_available() else 0,
                         pin_memory=torch.cuda.is_available(),
                         )

# ------------------------------------------------------------------------ define model, optimizer, and learning params
model = CondGlow(**params)
tmp_imgs, tmp_conds = next(iter(loader))
with torch.no_grad(): model.forward(tmp_imgs, tmp_conds)

optim = AdamW(model.parameters(), lr=lr)
lr_scheduler = ReduceLROnPlateau(optim, patience=5, factor=.2)

# ------------------------------------------------------------------------ load pretrained model
# load_ep = 350
# model.load_state_dict(torch.load(check_path + f'model_{load_ep:04}.pt', map_location=torch.device(device)))

# ------------------------------------------------------------------------ define loss function and early stopper
model = model.to(device)
early_stop = EarlyStop(test_loader, validate_every=5, patience=5)


def loss_func(point: tuple):
    imgs, labels = point
    _, log_p = model.forward(imgs.to(device), labels.to(device))
    return -torch.mean(log_p)/(imgs.shape[-3]*imgs.shape[-2]*imgs.shape[-1])

# ------------------------------------------------------------------------ train model
losses = []
pbar = trange(epochs, ascii=True)
for i in pbar:
    ep_loss = []
    for j, batch in enumerate(loader):
        if augment: batch = HI_transform(batch)

        # model forward and backward
        optim.zero_grad(set_to_none=True)
        loss = loss_func(batch)
        loss.backward()

        # clip gradients for good measure
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optim.step()

        ep_loss.append(loss.item())
        if not j%100: pbar.set_postfix_str(f'epoch {i+load_ep}; {j}/{len(loader)}; '
                                           f'step loss: {ep_loss[-1]:.2f}, avg.: {np.mean(ep_loss):.2f}, '
                                           f'lr: {optim.param_groups[0]["lr"]:.2E}')

        # plot conditional samples
        if not j%(n_batches//(5 if i < 5 else 2)):
            samp, cond = batch
            it = i*n_batches + j + 1

            cs = cond[:16]
            samps = model.sample(N=16, cond=cs.to(device), clip_val=500).cpu().detach().numpy()[:, 0]
            samps = .5*np.clip(samps, -1, 1) + 1

            save_images([samp for samp in samps], save_p=sample_path + f'{i+load_ep+1:04}-{j+1:04}.png')
            plt.close('all')
            save_images([s[0].cpu().numpy() for s in samp[:16]], save_p=sample_path + f'real.png')
            plt.close('all')

    # plot train and test loss over epochs
    losses.append(np.mean(ep_loss))
    plt.figure()
    plt.plot(np.arange(len(losses[1:]))+1, losses[1:], lw=2, label='train')
    plt.plot(early_stop.epochs, early_stop.losses, lw=2, label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('NLL')
    plt.savefig(path + 'loss.png')

    # early stopping
    if early_stop(loss_func, losses[-1]):
        torch.save(model.state_dict(), check_path + f'model_stopped.pt')
        print('Stopped early!')
        break

    # save checkpoints
    if not (i+load_ep+1)%(10 if i+load_ep < 50 else 50):
        torch.save(model.state_dict(), check_path + f'model_{i+load_ep+1:04}.pt')

    # reduce learning rate if average loss is not decreasing
    lr_scheduler.step(np.mean(losses[-5:]) if i > 10 else losses[-1])

