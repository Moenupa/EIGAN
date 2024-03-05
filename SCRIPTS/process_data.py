import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os.path
import pickle


class Africa_Whole_Flat(Dataset):
    """
    Custom Data set for pytorch loading
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample_np = self.data[idx, :]
        sample = torch.from_numpy(sample_np)
        return sample


class AfricaWholeFlatDataset(Dataset):
    """
    Custom Data set for pytorch loading, the right way
    """

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = torch.from_numpy(data)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]


class MinMaxScaler():
    """
    A data scaler that scale data to given range and scale back to initial range
    """

    def __init__(self, range: tuple[int] = (-1, 1)):
        self.min = np.empty(0)
        self.max = np.empty(0)
        self.diff = np.empty(0)
        self.range = range

    def fit(self, data: np.ndarray):
        """
        Fit the scaler to data and cache the min max value of data for transform and future inversion
        :param data: np.ndarray, expected arrangement: row as instances of data and column as features of data
        """
        self.min = np.min(data, axis=0)
        max = np.max(data, axis=0)
        self.diff = max - self.min

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the input data to range, using previous fitted min and max
        :param data: np.ndarray, expected arrangement: row as instances of data and column as features of data
        :param range: default to (-1,1)
        :return: data scaled to the range of input or (-1,1) by default
        """
        range_min, range_max = self.range
        data_std = (data - self.min) / self.diff
        data_scaled = data_std * (range_max - range_min) + range_min
        return data_scaled

    def inverse_transform(self, data_scaled: np.ndarray) -> np.ndarray:
        """
        Transform the scaled data back to original data range
        :param data_scaled: np.ndarray, expected arrangement: row as instances of data and column as features of data
        :return: data, at original scale
        """
        range_min = self.range[0]
        range_max = self.range[1]
        data_std = (data_scaled - range_min) / (range_max - range_min)
        data = data_std * self.diff + self.min
        return data


def slice_dataset(data: np.ndarray, slice_ratio=None):
    """
    partition data at different ratio cutoff, e.g. first 20%, first 40%...
    """
    if slice_ratio is None:
        slice_ratio = [0.2, 0.4, 0.6, 0.8, 1]
    length, dim = data.shape
    data_slices = []
    for ratio in slice_ratio:
        end_point = int(np.floor(ratio*length))
        data_slices.append(data[:end_point, :])
    return data_slices


def load_csv_with_cache(path: str) -> np.ndarray:
    cache = f'{path}.pkl'

    if os.path.exists(cache):
        with open(cache, 'rb') as handler:
            return pickle.load(handler)

    data = np.genfromtxt(path, delimiter=',', skip_header=True)
    with open(cache, 'wb') as handler:
        pickle.dump(data, handler)
    return data


if __name__ == "__main__":

    data = np.genfromtxt("./data/R30P.csv", delimiter=',', skip_header=True)
    slices = slice_dataset(data)
    ratio = [0.2, 0.4, 0.6, 0.8, 1]
    for idx, slice in enumerate(slices):
        file_name = "./data/slices/R30P_{:1.0f%}.csv".format(ratio[idx])
        np.savetxt(file_name, slice, delimiter=',')
