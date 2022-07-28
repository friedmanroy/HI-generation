from modules import GlowFlow, FlowSequential, AffineCoupling, Shuffle, ActNorm, GaussianPrior, InvConv2D
import numpy as np
import torch
from tqdm import trange
from torch.optim import Adam
from matplotlib import pyplot as plt
import matplotlib.animation as manimation


# ------------------------------------------------------------------------ generate data
N = 500
revs = .8
angs = np.linspace(0, 2*np.pi*revs, N)
r = np.linspace(.1, 1, N)

data = np.concatenate([(r*np.cos(angs))[:, None], (r*np.sin(angs))[:, None]], axis=1)
# data = np.concatenate([np.linspace(-1, 1, N)[:, None], np.sin(1*np.pi*np.linspace(-1, 1, N))[:, None]], axis=1)
data = data + .05*np.random.randn(*data.shape)
data = (data - np.mean(data, axis=0)[None, :])/np.std(data, axis=0)[None, :]

xlims = [np.min(data[:, 0]) - .25, np.max(data[:, 0]) + .25]
ylims = [np.min(data[:, 1]) - .25, np.max(data[:, 1]) + .25]

# ------------------------------------------------------------------------ batch data
batch_size = 25


def make_dataset():
    dataset = data[np.random.permutation(N)]
    return [torch.from_numpy(dataset[i:i+batch_size][:, :, None, None]).float() for i in range(0, N, batch_size)]


# ------------------------------------------------------------------------ define model
affine, hidden = True, 4
layers = []
for i in range(20):
    layers.append(FlowSequential(
        AffineCoupling(n_channels=2, affine=affine, hidden_width=hidden),
        # LinearTransf(),
        InvConv2D(in_channel=2),
        ActNorm(n_channels=2),
    ))
flow = FlowSequential(*layers)
model = GaussianPrior(flow, learn_params=False, temperature=1)

# ------------------------------------------------------------------------ define model
lr = 1e-3
optim = Adam(model.parameters(), lr=lr)

# ------------------------------------------------------------------------ setup animation
FFMpegWriter = manimation.writers['pillow']
metadata = dict(title='flow fitting process', artist='')
writer = FFMpegWriter(fps=20, metadata=metadata)

z_samples = torch.randn(N, 2, 1, 1)


def make_frame(i):
    plt.clf()
    plt.text(xlims[0] * .98, ylims[1] * .98, f"iter {i}/{epochs*(N//batch_size)}",
             horizontalalignment='left', verticalalignment='top')
    plt.scatter(data[:, 0], data[:, 1], 30, alpha=.5, label='real')
    samples = model.reverse(z_samples).detach().cpu().numpy()
    plt.scatter(samples[:, 0], samples[:, 1], 30, alpha=.5, label='sampled')
    plt.xlim(xlims)
    plt.ylim(ylims)
    # plt.axis('off')


# ------------------------------------------------------------------------ train model
epochs = 500
losses = []
pbar = trange(epochs)

with torch.no_grad():
    model.forward(torch.from_numpy(data[:batch_size][:, :, None, None]).float())
    model.zero_grad()

fig = plt.figure()
make_frame(0)
with writer.saving(fig, 'sample_spiral/fitting.gif', 100):

    for i in pbar:
        model.zero_grad()
        data = data[np.random.permutation(N)]
        dataset = make_dataset()
        ep_loss = []
        for j, batch in enumerate(dataset):
            z, log_p = model.forward(batch)
            loss = -torch.mean(log_p)
            loss.backward()
            optim.step()
            ep_loss.append(loss.item())
            pbar.set_postfix_str(f'step loss: {ep_loss[-1]:.2f}, avg.: {np.mean(ep_loss):.2f}, '
                                 f'log-lr: {np.log10(optim.param_groups[0]["lr"]):.2f}')

            if not (i*(N//batch_size) + j)%25:
                make_frame(i*(N//batch_size) + j)
                writer.grab_frame()
        losses.append(np.mean(ep_loss))
        optim.param_groups[0]['lr'] = max(optim.param_groups[0]['lr']*.99, 1e-5)

plt.figure()
plt.plot(losses[1:], lw=2)
plt.xlabel('epoch')
plt.ylabel('NLL')
plt.show()
