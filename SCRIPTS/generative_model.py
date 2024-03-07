"""
Title: WGAN Architecture and Training Program
Author: Enting Zhou
Date: 06/15/2022
Availability: https://github.com/ETZET/MCMC_GAN
"""
# from dis import dis
import os.path

from process_data import AfricaWholeFlatDataset

import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import Adam
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
from scipy.stats import wasserstein_distance as EMD


class WGAN_SIMPLE(Module):
    """
    Generative Model Architecture

    Model Architecture cited from Scheiter, M., Valentine, A., Sambridge, M., 2022. Upscaling
    and downscaling Monte Carlo ensembles with generative models, Geophys. J. Int., ggac100.

    This model use gradient penalty to enforce 1-Lipschitz constraint instead of Weight Clipping in the original paper.
    Citation: Gulrajani, Ahmed & Arjovsky. Improved training of wasserstein gans. Adv. Neural Inf. Process. Syst.
    """

    def __init__(self, ndim: int = 2382, nhid: int = 300, nlatent: int = 100, device="cpu"):
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
            nn.Linear(ndim, nhid),
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
        
    def sample_z(self, *size, is_normalized: bool = False) -> torch.Tensor:
        if is_normalized:
            return torch.randn(*size, device=self.device)
    
        return torch.FloatTensor(*size).uniform_(-1, 1).to(self.device)

    def optimize(self, normalized_data: np.ndarray, output_path, batch_size=128, use_wandb=False, lr=1e-4,
                 betas=(0.5, 0.999), lambda_term=10, epochs: int = 200, kkd: int = 1, kkg: int = 1, device="cpu"):

        # construct dataset and dataloader for batch training
        map_dataset = AfricaWholeFlatDataset(normalized_data)
        dataloader = DataLoader(map_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=8)

        # Your Code Goes Here
        optimizer_gen = Adam(self.gen.parameters(), lr=lr, betas=betas)
        optimizer_disc = Adam(self.disc.parameters(), lr=lr, betas=betas)

        disc_fake: torch.Tensor
        disc_real: torch.Tensor

        self.train()
        for epoch in tqdm(range(epochs)):

            for batch in dataloader:
                batch: torch.Tensor = batch.to(device).float()
                size = batch.size(0)

                # update disc, lock gen to save computation
                for _ in range(kkd):
                    optimizer_disc.zero_grad()
                    optimizer_gen.zero_grad()

                    noise = self.sample_z(size, self.nlatent)
                    fake_batch = self.gen(noise).detach()

                    disc_fake = self.disc(fake_batch)
                    disc_real = self.disc(batch)
                    gp = self.calculate_gradient_penalty(batch, fake_batch)

                    neg_wd = disc_fake.mean() - disc_real.mean()
                    disc_loss = neg_wd + lambda_term * gp
                    disc_loss.backward()

                    optimizer_disc.step()

                # update gen, lock disc
                for _ in range(kkg):
                    optimizer_disc.zero_grad()
                    optimizer_gen.zero_grad()

                    noise = self.sample_z(size, self.nlatent)
                    fake_batch = self.gen(noise)

                    disc_fake = self.disc(fake_batch)

                    gen_loss = -disc_fake.mean()
                    gen_loss.backward()

                    optimizer_gen.step()

                if use_wandb:
                    wandb.log({"disc_loss": disc_loss,
                              "gen_loss": gen_loss,
                               "ewd": -neg_wd})

            if (epoch + 1) % 10 == 0:
                avg, std, emd = eval_model(self, normalized_data)
                if use_wandb:
                    wandb.log({"avg": avg, 'std': std, 'emd': emd})
                
                if os.path.exists(output_path):
                    torch.save(self, f'{output_path}/{epoch}.pt')
        # Your Code Ends Here

    def calculate_gradient_penalty(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
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
                        ** 2).mean()
        return grad_penalty

    def load(self, checkpoint):
        self.load_state_dict(checkpoint["model_state_dict"])

    def generate(self, num: int = 50000, normed: bool = False) -> np.ndarray:
        fake_data = np.zeros((num, self.ndim))
        # if num is divisible by 100, generate by batch, else generate one by one
        if num % 100 == 0:
            for i in range(num // 100):
                l, r = 100 * i, 100 * (i+1)
                fake_data[l:r, :] = \
                    self.gen(self.sample_z(100, self.nlatent,
                             is_normalized=normed)).cpu().detach().numpy()
        else:
            for i in range(num):
                fake_data[i, :] = self.gen(self.sample_z(
                    1, self.nlatent)).cpu().detach().numpy()
        return fake_data


class Generator(Module):
    def __init__(self, ndim: int = 2382, nhid: int = 300, nlatent: int = 100) -> None:
        super().__init__()

        self.gen = nn.Sequential(
            spectral_norm(nn.Linear(nlatent, nhid)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            spectral_norm(nn.Linear(nhid, nhid)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            spectral_norm(nn.Linear(nhid, nhid)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            spectral_norm(nn.Linear(nhid, ndim)),
        )

        self.gen.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.gen(z)


class Discriminator(Module):
    def __init__(self, ndim: int = 2382, nhid: int = 300) -> None:
        super().__init__()

        self.disc = (nn.Sequential(
            spectral_norm(nn.Linear(ndim, nhid)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(nhid, nhid)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(nhid, nhid)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(nhid, 1)),
        ))

        self.disc.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x)


class WGAN_SN(Module):
    def __init__(self, ndim: int = 2382, nhid: int = 300, nlatent: int = 100, device="cpu") -> None:
        super().__init__()

        self.ndim = ndim
        self.nlatent = nlatent
        self.device = device

        self.gen = Generator(ndim, nhid, nlatent)
        self.disc = Discriminator(ndim, nhid)

        self.gen.to(device)
        self.disc.to(device)

    def sample_z(self, *size, is_normalized: bool = False) -> torch.Tensor:
        if is_normalized:
            return torch.randn(*size, device=self.device)

        return torch.FloatTensor(*size).uniform_(-1, 1).to(self.device)

    def optimize(self, normalized_data: np.ndarray, output_path: str, batch_size=128, use_wandb=False, lr=1e-4,
                 betas=(0.5, 0.999), lambda_term=10, epochs: int = 200, kkd: int = 1, kkg: int = 1):
        map_dataset = AfricaWholeFlatDataset(normalized_data)
        dataloader = DataLoader(map_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=8)

        # Your Code Goes Here
        optimizer_gen = Adam(self.gen.parameters(), lr=lr, betas=betas)
        optimizer_disc = Adam(self.disc.parameters(), lr=lr, betas=betas)

        disc_fake: torch.Tensor
        disc_real: torch.Tensor

        self.train()
        for epoch in tqdm(range(epochs)):

            for batch in dataloader:
                batch: torch.Tensor = batch.to(self.device).float()
                size = batch.size(0)

                # update disc, lock gen to save computation
                for _ in range(kkd):
                    optimizer_disc.zero_grad()
                    optimizer_gen.zero_grad()

                    noise = self.sample_z(size, self.nlatent)
                    fake_batch = self.gen(noise).detach()

                    disc_fake = self.disc(fake_batch)
                    disc_real = self.disc(batch)

                    disc_loss = disc_fake.mean() - disc_real.mean()
                    disc_loss.backward()

                    optimizer_disc.step()

                # update gen, lock disc
                for _ in range(kkg):
                    optimizer_disc.zero_grad()
                    optimizer_gen.zero_grad()

                    noise = self.sample_z(size, self.nlatent)
                    fake_batch = self.gen(noise)

                    disc_fake = self.disc(fake_batch)

                    gen_loss = -disc_fake.mean()
                    gen_loss.backward()

                    optimizer_gen.step()

                if use_wandb:
                    wandb.log({"disc_loss": disc_loss,
                              "gen_loss": gen_loss,
                               "ewd": -disc_loss})

            if (epoch + 1) % 10 == 0:
                if use_wandb:
                    avg, std, emd = eval_model(self, normalized_data)
                    wandb.log({"avg": avg, 'std': std, 'emd': emd})

                if os.path.exists(output_path):
                    torch.save({'gen': self.gen, 'disc': self.disc},
                               f'{output_path}/{epoch}.pt')


def eval_model(model: WGAN_SIMPLE, data: np.ndarray, normed: bool = False):
    dim = data.shape[1]

    # generate fake data using Generator
    fake_data = np.zeros((50000, dim))
    for i in range(500):
        l, r = 100 * i, 100 * (i+1)
        with torch.no_grad():
            fake_batch = \
                model.gen(
                    model.sample_z(100, model.nlatent, is_normalized=normed)
                ).cpu().detach().numpy()
        fake_data[l:r, :] = fake_batch
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


if __name__ == '__main__':
    # test
    model = WGAN_SN()
    print(model.gen)
    print(model.disc)