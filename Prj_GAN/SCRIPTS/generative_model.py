"""
Title: WGAN Architecture and Training Program
Author: Enting Zhou
Date: 06/15/2022
Availability: https://github.com/ETZET/MCMC_GAN
"""
from dis import dis
import os.path

import torch
from torch import nn
from torch.autograd import Variable
from torch import autograd
from process_data import Africa_Whole_Flat, MinMaxScaler
from torch.utils.data import DataLoader
import numpy as np
import time
import wandb
import matplotlib
from scipy.stats import wasserstein_distance as EMD

# matplotlib.use('Agg')


class WGAN_SIMPLE(nn.Module):
    """
    Generative Model Architecture

    Model Architecture cited from Scheiter, M., Valentine, A., Sambridge, M., 2022. Upscaling
    and downscaling Monte Carlo ensembles with generative models, Geophys. J. Int., ggac100.

    This model use gradient penalty to enforce 1-Lipschitz constraint instead of Weight Clipping in the original paper.
    Citation: Gulrajani, Ahmed & Arjovsky. Improved training of wasserstein gans. Adv. Neural Inf. Process. Syst.
    """

    def __init__(self, ndim, nhid=300, nlatent=100, device="cpu"):
        """
        :param ndim: Number of feature in input data
        :param nhid: Number of hidden units per layer
        :param device: device on which a torch.Tensor is or will be allocated
        :param gen: Generator that consist of four layers of dropout layers with linear output
        :param disc: Discriminator that consist of four layers of dropout layers with linear output
        """
        super().__init__()

        self.ndim = ndim
        self.nlatent = nlatent
        self.device = device
        self.scaler = None

        self.gen = nn.Sequential(
            nn.Linear(self.nlatent, nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid, nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid, nhid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(nhid, ndim),
        )

        self.disc = nn.Sequential(
            nn.Linear(self.ndim, nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid, nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid, nhid),
            nn.LeakyReLU(0.1),
            nn.Linear(nhid, 1),
        )

        self.gen.apply(init_weights)
        self.disc.apply(init_weights)

        self.gen.to(device)
        self.disc.to(device)

    def normalize(self, data):
        self.scaler = MinMaxScaler()
        self.scaler.fit(data)
        return self.scaler.transform(data)

    def optimize(self, data, output_path, batch_size=128, use_wandb=False, lr=1e-4,
                 beta1=0.5, lambda_term=10, epochs=200, kkd=1, kkg=1, device="cpu"):

        # normalizae data to (-1,1) range
        data = self.normalize(data)
        # construct dataset and dataloader for batch training
        map_dataset = Africa_Whole_Flat(data)
        dataloader = DataLoader(map_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=1)

        # Your Code Goes Here
        
        # Your Code Ends Here

    def calculate_gradient_penalty(self, real_images, fake_images, lambda_term):
        batch_size = real_images.shape[0]
        eta = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
        eta = eta.expand(batch_size, real_images.size(1)).to(self.device)

        interpolated = eta * real_images + \
            ((1 - eta) * fake_images).to(self.device)

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.disc(interpolated.float())

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                      prob_interpolated.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * lambda_term
        return grad_penalty

    def load(self, checkpoint):
        self.load_state_dict(checkpoint["model_state_dict"])
        self.scaler = checkpoint["scaler"]

    def generate(self, num=50000) -> np.ndarray:
        fake_data = np.zeros((num, self.ndim))
        # if num is divisible by 100, generate by batch, else generate one by one
        if num % 100 == 0:
            for i in range(num // 100):
                l, r = 100 * i, 100 * (i+1)
                fake_data[l:r, :] = \
                    self.gen(torch.randn(100, self.nlatent,
                             device=self.device)).cpu().detach().numpy()
        else:
            for i in range(num):
                fake_data[i, :] = self.gen(torch.randn(
                    1, self.nlatent, device=self.device)).cpu().detach().numpy()
        # scaler the data back to original range
        fake_data_scaled = self.scaler.inverse_transform(fake_data)
        del fake_data
        return fake_data_scaled


def eval(model, data):
    dim = data.shape[1]

    # generate fake data using Generator
    fake_data = np.zeros((50000, dim))
    for i in range(500):
        left_idx = 100 * i
        right_idx = 100 * (i+1)
        with torch.no_grad():
            fake_batch = model.gen(torch.randn(
                100, model.nlatent, device=model.device)).cpu().detach().numpy()
        fake_data[left_idx:right_idx, :] = fake_batch
    # compare mean
    real_avg = np.mean(data, axis=0)
    fake_avg = np.mean(fake_data, axis=0)
    avg_diff_pixel = np.sum(np.absolute(real_avg-fake_avg))/dim
    # compare std
    real_std = np.std(data, axis=0)
    fake_std = np.std(fake_data, axis=0)
    std_diff_pixel = np.sum(np.absolute(real_std-fake_std))/dim
    # calculate EMD distance
    distance = np.zeros(dim)
    for i in range(dim):
        distance[i] = EMD(data[:, i], fake_data[:, i])
    emd_dist_pixel = np.sum(distance)/dim

    return avg_diff_pixel, std_diff_pixel, emd_dist_pixel


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
