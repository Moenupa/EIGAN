import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def plot_mean_std(fake_distribution: np.ndarray,
                  real_distribution: np.ndarray,
                  cord: np.ndarray):
    d = {
        'fake mean':    fake_distribution.mean(axis=0),
        'real mean':    real_distribution.mean(axis=0),
        'fake std':     fake_distribution.std(axis=0),
        'real std':     real_distribution.std(axis=0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for _ax, (_title, _c) in zip(axes, d.items()):
        _fig = _ax.scatter(
            cord[:, 0], cord[:, 1],
            vmin=-1, vmax=1,
            c=_c, cmap='RdBu'
        )
        _ax.set_title(_title)
        plt.colorbar(_fig, ax=_ax)


def plot_distribution(fake: torch.Tensor, real: torch.Tensor, noise: torch.Tensor, model):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax0, ax1, ax2 = axes.flatten()
    sns.histplot(fake[:, 0], color='b', label='fake', ax=ax0)
    sns.histplot(real[:, 0], color='r', label='real', ax=ax0)

    with torch.no_grad():
        model.eval()
        _fake = model.disc(fake).numpy().flatten()
        _real = model.disc(real).numpy().flatten()
        _random = model.disc(noise).numpy().flatten()

    sns.histplot(_fake, color='b', label='fake', ax=ax1)
    sns.histplot(_real, color='r', label='real', ax=ax1)
    sns.histplot(_random, color='g', label='noise', ax=ax1)
    plt.legend()

    sns.histplot((real.mean(dim=0)-fake.mean(dim=0))
                 .numpy().flatten(), ax=ax2)

    ax0.set_title('Distribution on Feature 0')
    ax1.set_title('Discriminator Output')
    ax2.set_title('Mean Difference Real-Fake')
