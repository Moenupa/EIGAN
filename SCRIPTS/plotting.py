import numpy as np
import pandas as pd
import torch


import seaborn as sns
import matplotlib.pyplot as plt


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
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    ax0, ax1, ax2, *_ = axes.flatten()
    sns.histplot(real[:, 0], color='r', label='real', ax=ax0)
    sns.histplot(fake[:, 0], color='b', label='fake', ax=ax0)

    with torch.no_grad():
        model.eval()
        _fake = model.disc(fake).numpy().flatten()
        _real = model.disc(real).numpy().flatten()
        _random = model.disc(noise).numpy().flatten()

    sns.histplot(_fake, color='b', label='fake', ax=ax1)
    sns.histplot(_real, color='r', label='real', ax=ax1)
    sns.histplot(_random, color='g', label='noise', ax=ax1)
    ax1.legend()

    diff = real - fake
    sns.histplot(diff.mean(dim=0).numpy().flatten(), ax=ax2)
    ax2.set_xlim(-1, 1)

    ax0.set_title('Distribution on Feature 0')
    ax1.set_title('Discriminator Output')
    ax2.set_title('Mean Difference Real-Fake')

    # 7 points centered around point indexed 1333, obtained by:
    # point = cord[1333]
    # neighbors = np.argsort(np.linalg.norm(cord - point, axis=1))[:7]
    # cord[neighbors], neighbors
    # 1333 because I like the number
    sample_id = np.array([1333, 2344, 1034, 2319,  734,  555,  531])
    d = {}
    for i_feat in sample_id[:3]:
        d[f'real_{i_feat}'] = real[:, i_feat].numpy().flatten()
        d[f'fake_{i_feat}'] = fake[:, i_feat].numpy().flatten()
    sample = pd.DataFrame(d)
    sns.pairplot(sample,
                 kind='kde')


def plot_slice(fake, real, cord: np.ndarray, n: int = 1):
    """
    plot n random slices of real vs fake data
    """
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    n, *_ = real.shape

    for ax0, ax1 in axes:
        idx = np.random.randint(0, n)
        ax0.scatter(cord[:, 0], cord[:, 1], c=real[idx, :],
                    vmin=-1, vmax=1, cmap='RdBu', s=2)
        ax1.scatter(cord[:, 0], cord[:, 1], c=fake[idx, :],
                    vmin=-1, vmax=1, cmap='RdBu', s=2)
        ax0.axis('off')
        ax1.axis('off')

    col1, col2 = axes[0]
    col1.set_title('Real')
    col2.set_title('Fake')
