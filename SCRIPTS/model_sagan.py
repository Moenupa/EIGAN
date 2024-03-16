# from dis import dis
import os.path

from process_data import AfricaWholeFlatDataset
from generative_model import sample_z, eval_model

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


class SAWGAN(Module):
    """self attention wgan"""

    def __init__(self, n_feat: int = 2382, z_dim: int = 16, hid_dim: int = 16, uniform_z: bool = True, device="cpu") -> None:
        super().__init__()

        self.n_feat = n_feat
        self.z_dim = z_dim
        self.device = device
        self.uniform_z = uniform_z

        self.gen = Generator(n_feat, z_dim, hid_dim)
        self.disc = Discriminator(n_feat, hid_dim)

        self.gen.to(device)
        self.disc.to(device)

    def optimize(self, normalized_data: np.ndarray, output_path: str, batch_size=128, use_wandb=False, lr=2e-4,
                 betas=(0.5, 0.999), epochs: int = 200, kkd: int = 1, kkg: int = 1):
        map_dataset = AfricaWholeFlatDataset(normalized_data)
        dataloader = DataLoader(map_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=8)

        # Your Code Goes Here
        optimizer_gen = Adam(self.gen.parameters(), lr=lr, betas=betas)
        optimizer_disc = Adam(self.disc.parameters(), lr=lr, betas=betas)

        disc_fake: torch.Tensor
        disc_real: torch.Tensor

        self.train()
        for epoch in range(epochs):

            for batch in tqdm(dataloader):
                batch: torch.Tensor = batch.to(self.device).float()
                size = batch.size(0)
                batch = batch.view(size, 1, self.n_feat)

                # update disc, lock gen to save computation
                for _ in range(kkd):
                    optimizer_disc.zero_grad()

                    noise = sample_z(size, self.z_dim, self.n_feat,
                                     device=self.device, uniform=self.uniform_z)
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

                    noise = sample_z(size, self.z_dim, self.n_feat,
                                     device=self.device, uniform=self.uniform_z)
                    fake_batch = self.gen(noise)

                    disc_fake = self.disc(fake_batch)

                    gen_loss = -disc_fake.mean()
                    gen_loss.backward()

                    optimizer_gen.step()

                if use_wandb:
                    wandb.log({"disc_loss": disc_loss,
                              "gen_loss": gen_loss,
                               "loss": -disc_loss})

            if (epoch + 1) % 10 == 0:
                if use_wandb:
                    avg, std, emd = eval_model(self, normalized_data)
                    wandb.log({"avg": avg, 'std': std, 'emd': emd})

                if os.path.exists(output_path):
                    torch.save(self.state_dict(),
                               f'{output_path}/{epoch}.pt')

    def generate(self, n: int = 15400) -> np.ndarray:
        fake_data = np.zeros((n, self.n_feat))
        # if num is divisible by 100, generate by batch, else generate one by one
        for l in range(0, n, 100):
            fake_data[l:l+100, :] = self.gen(
                sample_z(n, self.z_dim, self.n_feat,
                         device=self.device,
                         uniform=self.uniform_z)
            ).cpu().detach().numpy()
        return fake_data


class SelfAttn(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()

        self.q = spectral_norm(nn.Conv1d(in_channels=in_dim,
                                         out_channels=in_dim // 8,
                                         kernel_size=1))
        self.k = spectral_norm(nn.Conv1d(in_channels=in_dim,
                                         out_channels=in_dim // 8,
                                         kernel_size=1))
        self.v = spectral_norm(nn.Conv1d(in_channels=in_dim,
                                         out_channels=in_dim,
                                         kernel_size=1))
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
            inputs :
                x : input feature maps (B, C, N)
            returns :
                out : self attention value + input feature 
                attention: (B, N, N)
        """
        B, C, N = x.shape

        # Q, K are (B, C, N)
        Q = self.q(x).view(B, -1, N).permute(0, 2, 1)
        K = self.k(x).view(B, -1, N)
        V = self.v(x).view(B, -1, N)

        attention = self.softmax(torch.bmm(Q, K))
        # attention = self.softmax(torch.einsum('biu,bjv->bvu', Q, K))

        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.view(B, C, N)

        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):

    def __init__(self, n_feat: int = 2382, z_dim=16, hid_dim=16):
        super().__init__()

        self.l1 = nn.Sequential(
            spectral_norm(nn.Conv1d(z_dim, hid_dim, kernel_size=1)),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU()
        )
        self.attn = SelfAttn(hid_dim)

        self.l2 = nn.Sequential(
            spectral_norm(nn.Conv1d(hid_dim, 1, kernel_size=1)),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )

    def forward(self, z: torch.Tensor):
        # z as a (batch, z_dim, n_feat) tensor
        assert z.size(1) == 16, f'z.size(1)={z.size(1)}'

        x = self.l1(z)
        assert x.size(1) == 16, f'x.size(1)={x.size(1)}'
        x, p = self.attn(x)
        x = self.l2(x)
        assert x.size(1) == 1, f'x.size(1)={x.size(1)}'

        return x  # , p


class Discriminator(nn.Module):

    def __init__(self, n_feat: int = 2382, hid_dim: int = 16):
        super().__init__()

        self.l1 = nn.Sequential(
            spectral_norm(nn.Conv1d(1, hid_dim, kernel_size=1)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        self.attn = SelfAttn(hid_dim)

        self.l2 = nn.Sequential(
            spectral_norm(nn.Conv1d(hid_dim, hid_dim, kernel_size=1)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )

        self.fc = nn.Linear(n_feat * hid_dim, 1)

    def forward(self, x: torch.Tensor):
        x = self.l1(x)
        x, p = self.attn(x)
        x = self.l2(x)
        x = x.flatten(1, -1)
        x = self.fc(x)

        return x  # , p
