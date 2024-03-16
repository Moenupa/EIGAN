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
from torch.optim import AdamW
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
import wandb
from scipy.stats import wasserstein_distance as EMD


def sample_z(*size: int, device: str = 'cpu', uniform: bool = False) -> torch.Tensor:
    if uniform:
        return torch.FloatTensor(*size).uniform_(-1, 1).to(device)

    return torch.randn(*size, device=device)


class WGAN_SIMPLE(Module):
    """
    Generative Model Architecture

    Model Architecture cited from Scheiter, M., Valentine, A., Sambridge, M., 2022. Upscaling
    and downscaling Monte Carlo ensembles with generative models, Geophys. J. Int., ggac100.

    This model use gradient penalty to enforce 1-Lipschitz constraint instead of Weight Clipping in the original paper.
    Citation: Gulrajani, Ahmed & Arjovsky. Improved training of wasserstein gans. Adv. Neural Inf. Process. Syst.
    """

    def __init__(self, ndim: int = 2382, nhid: int = 300, nlatent: int = 100, uniform_z: bool = True, device="cpu"):
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
        self.uniform_z = uniform_z

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

    def optimize(self, normalized_data: np.ndarray, output_path, args, lambda_term=10):

        # construct dataset and dataloader for batch training
        map_dataset = AfricaWholeFlatDataset(normalized_data)
        dataloader = DataLoader(map_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=8)

        # Your Code Goes Here
        betas = (args.beta1, args.beta2)
        optimizer_gen = AdamW(self.gen.parameters(), lr=args.g_lr, betas=betas)
        optimizer_disc = AdamW(self.disc.parameters(), lr=args.d_lr, betas=betas)

        disc_fake: torch.Tensor
        disc_real: torch.Tensor

        self.train()
        for epoch in trange(args.epochs):

            for batch in dataloader:
                batch: torch.Tensor = batch.to(self.device).float()
                size = batch.size(0)

                # update disc, lock gen to save computation
                for _ in range(args.kkd):
                    optimizer_disc.zero_grad()
                    optimizer_gen.zero_grad()

                    noise = sample_z(size, self.nlatent,
                                     device=self.device, uniform=self.uniform_z)
                    fake_batch = self.gen(noise).detach()

                    disc_fake = self.disc(fake_batch)
                    disc_real = self.disc(batch)
                    gp = self.calculate_gradient_penalty(batch, fake_batch)

                    neg_wd = disc_fake.mean() - disc_real.mean()
                    disc_loss = neg_wd + lambda_term * gp
                    disc_loss.backward()

                    optimizer_disc.step()

                # update gen, lock disc
                for _ in range(args.kkg):
                    optimizer_disc.zero_grad()
                    optimizer_gen.zero_grad()

                    noise = sample_z(size, self.nlatent,
                                     device=self.device, uniform=self.uniform_z)
                    fake_batch = self.gen(noise)

                    disc_fake = self.disc(fake_batch)

                    gen_loss = -disc_fake.mean()
                    gen_loss.backward()

                    optimizer_gen.step()

                if args.use_wandb:
                    wandb.log({"disc_loss": disc_loss,
                              "gen_loss": gen_loss,
                               "loss": -neg_wd})

            if (epoch + 1) % 10 == 0:
                if args.use_wandb:
                    avg, std, emd = eval_model(self, normalized_data)
                    wandb.log({"avg": avg, 'std': std, 'emd': emd})

                if os.path.exists(output_path):
                    torch.save(self.state_dict(), f'{output_path}/{epoch}.pt')
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

    def generate(self, num: int = 15400) -> np.ndarray:
        fake_data = np.zeros((num, self.ndim))
        # if num is divisible by 100, generate by batch, else generate one by one
        if num % 100 == 0:
            for l in range(0, num, 100):
                fake_data[l:l+100, :] = self.gen(
                    sample_z(100, self.nlatent,
                             device=self.device,
                             uniform=self.uniform_z)
                ).cpu().detach().numpy()
        else:
            for i in range(num):
                fake_data[i, :] = self.gen(
                    sample_z(1, self.nlatent,
                             device=self.device,
                             uniform=self.uniform_z)
                ).cpu().detach().numpy()
        return fake_data


def eval_model(model: WGAN_SIMPLE, data: np.ndarray):
    datapoints, n_feat = data.shape

    # generate fake data using Generator
    fake_data = model.generate(datapoints)
    # compare mean
    real_avg = np.mean(data, axis=0)
    fake_avg = np.mean(fake_data, axis=0)
    avg_diff_pixel = np.sum(np.absolute(real_avg-fake_avg)) / n_feat
    # compare std
    real_std = np.std(data, axis=0)
    fake_std = np.std(fake_data, axis=0)
    std_diff_pixel = np.sum(np.absolute(real_std-fake_std)) / n_feat
    # calculate EMD distance
    distance = np.zeros(n_feat)
    for i in range(n_feat):
        distance[i] = EMD(data[:, i], fake_data[:, i])
    emd_dist_pixel = np.sum(distance) / n_feat

    return avg_diff_pixel, std_diff_pixel, emd_dist_pixel


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    # test
    pass
