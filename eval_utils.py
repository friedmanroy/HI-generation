import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import skew as skewness, kurtosis


def _power_spectrum(map: np.ndarray, size: int=64, dl: float=25/64):
    FT_box  = np.fft.fftn(map, norm='ortho') # norm is important to be ortho, otherwise normalization is wrong!
    k       = 2*np.pi*np.fft.fftfreq(size, dl)
    pk      = np.zeros(size)
    count   = np.zeros(size)    # to count powers falling in each bin
    dk      = 2*np.pi/(size * dl) # dk is the bin size
    for i in range(size):
        for j in range(size):
            kbar      = np.sqrt(k[i]**2.0 + k[j]**2.0) # get the k_bar in 3D
            t         = int(round(kbar/dk)) # find the corresponding bin number
            count[t] += 1.0
            pk[t]    += FT_box[i,j]*np.conj(FT_box[i,j])
    pk       /= count # average each bin
    pk       *= (dl)**2.0
    dk        = np.arange(float(size)) * dk
    return  dk[1:], pk[1:]


def power_spectrum(maps: np.ndarray, size: int=64, dl: float=25/64):
    ds, ps = [], []
    for map in maps:
        d, p = _power_spectrum(map, size, dl)
        ds.append(d)
        ps.append(p)
    return np.stack(ds), np.stack(ps)


def wasserstein(samps1: np.ndarray, samps2: np.ndarray, n_projections: int=256):
    # if more than one dimension, use sliced-Wasserstein distance
    if samps1.ndim > 1:
        proj = np.random.randn(samps1.shape[1], n_projections)
        proj = proj/np.linalg.norm(proj, axis=0)[None, :]
        proj1, proj2 = np.sort(samps1@proj, axis=0), np.sort(samps2@proj, axis=0)
        dist = np.sum(np.abs(proj1 - proj2), axis=0)
        return np.mean(dist)
    # if only 1D data, use the 1D Wasserstein sample based distance metric
    return np.sum(np.abs(np.sort(samps1) - np.sort(samps2)))


def R2(pred: np.ndarray, true: np.ndarray):
    residuals = np.sum((pred - true)**2)
    variance = np.sum((pred - np.mean(pred))**2)
    return 1 - residuals/variance


def compare_spectrums(true_maps: np.ndarray, samples: np.ndarray, size: int=64, dl: float=25/64,
                      type: str= 'mean', fontsize: int=18, loglog: bool=True,
                      semilogx: bool=False, semilogy: bool=False, save_p: str=None, title: str='',
                      compare_to: list=None, legend: bool=True, equal: bool=False, small: bool=False):
    font = {'family': 'serif',
            'color': 'black',
            'size': fontsize,
            }

    true_freqs, true_pows = power_spectrum(true_maps, size=size, dl=dl)
    samp_freqs, samp_pows = power_spectrum(samples, size=size, dl=dl)

    comp_freqs = None
    if compare_to is not None: comp_freqs = [power_spectrum(c[0], size=size, dl=dl)[1] for c in compare_to]
    plots, ylabel = [], ''

    if type.lower() == 'mean':
        mu_true, mu_samp = np.mean(true_pows, axis=0), np.mean(samp_pows, axis=0)
        plots = [(mu_true, 'CAMELS', 'o'), (mu_samp, 'HIGlow', '')]

        if compare_to is not None:
            plots += [(np.mean(comp_freqs[i], axis=0), compare_to[i][1]) for i in range(len(compare_to))]

        ylabel = r'$\mu$(P)'

    elif type.lower() == 'std':
        std_true, std_samp = np.std(true_pows, axis=0), np.std(samp_pows, axis=0)
        plots = [(std_true, 'CAMELS', 'o'), (std_samp, 'HIGlow', '')]

        if compare_to is not None:
            plots += [(np.std(comp_freqs[i], axis=0), compare_to[i][1]) for i in range(len(compare_to))]

        ylabel = r'$\sigma$(P)'

    elif type.lower() == 'skew':
        skew_true, skew_samp = skewness(true_pows, axis=0), skewness(samp_pows, axis=0)
        plots = [(skew_true, 'CAMELS', 'o'), (skew_samp, 'HIGlow', '')]

        if compare_to is not None:
            plots += [(skewness(comp_freqs[i], axis=0), compare_to[i][1]) for i in range(len(compare_to))]

        ylabel = r'skew(P)'

    elif type.lower() == 'kurtosis':
        kurt_true, kurt_samp = kurtosis(true_pows, axis=0), kurtosis(samp_pows, axis=0)
        plots = [(kurt_true, 'CAMELS'), (kurt_samp, 'HIGlow')]

        if compare_to is not None:
            plots += [(kurtosis(comp_freqs[i], axis=0), compare_to[i][1]) for i in range(len(compare_to))]

        ylabel = r'kurtosis(P)'

    elif type.lower() in ['wasserstein', 'w1']:
        dists = np.array([wasserstein(true_pows[:, i], samp_pows[:, i]) for i in range(samp_pows.shape[1])])
        plots = [(dists, (None if compare_to is None else 'sim. vs HIGlow'))]

        if compare_to is not None:
            for i, cf in enumerate(comp_freqs):
                plots.append(
                    (
                        np.array([wasserstein(true_pows[:, i], cf[:, i]) for i in range(true_pows.shape[1])]),
                        f'sim. vs {compare_to[i][1]}'
                    )
                )

        ylabel = r'$\mathcal{W}_1$(P)'

    else: raise NotImplemented

    if equal: plt.figure(dpi=100, figsize=(4.5, 4.5))
    elif small: plt.figure(dpi=100, figsize=(3.5, 4.5))
    else: plt.figure(dpi=100)
    label = None

    # make plots
    for (plot, label, marker) in plots:
        plt.plot(true_freqs[0], plot, lw=3, alpha=.75, label=label, marker=marker)

    # add frequency label
    plt.xlabel('k [h/Mpc]', fontdict=font)
    plt.ylabel(ylabel, fontdict=font)

    # make log, if needed
    if loglog: plt.loglog()
    elif semilogx: plt.semilogx()
    elif semilogy: plt.semilogy()

    # add labels
    if label is not None and legend: plt.legend(fontsize=fontsize)

    # further cosmetics
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim([samp_freqs[0][0]-samp_freqs[0][0]/10,
              samp_freqs[0][~np.isnan(true_pows[0])][-1]+samp_freqs[0][~np.isnan(true_pows[0])][-1]/10])
    plt.title(title, fontdict=font)
    plt.tight_layout()

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
        spine.set_alpha(.7)

    if save_p is None: plt.show()
    else: plt.savefig(save_p)


def save_images(images, save_p: str=None, cols=None, title: str=None, dpi: int=150):
    """
    Save a list of images in a single figure with matplotlib. This only really works well with a relatively small number
    of images, such as 16 - for any more, consider using torchvision's tile.

    :param images: list of numpy arrays, each representing an image
    :param save_p: the path where the image should be saved.
    :param cols: number of columns the figure should have. If None is given, then this is chosen so that the resulting
                 aspect ratio will be as close to square as possible
    :param title: the title to give the plot
    """
    N = len(images)
    if cols is None:
        cols = int(N//(np.where((N % np.arange(1, np.floor(np.sqrt(N) + 1))) == 0)[0] + 1)[-1])

    fig = plt.figure(figsize=(cols, int(np.ceil(N / float(cols)))), dpi=dpi)
    gs = GridSpec(cols, int(np.ceil(N / float(cols))))
    gs.update(wspace=.1, hspace=.1)

    for n, image in enumerate(images):
        a = plt.subplot(gs[n])
        a.imshow(image)
        a.axis('off')

    fig.set_size_inches(np.array(fig.get_size_inches()) * N / 6)
    if title is not None: plt.title(title)
    gs.tight_layout(fig)
    if save_p is None: plt.show()
    else: plt.savefig(save_p)
