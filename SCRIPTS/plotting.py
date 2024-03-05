import numpy as np
# import seaborn as sns
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
