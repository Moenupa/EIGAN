# from dis import dis
import os.path

from process_data import AfricaWholeFlatDataset
from generative_model import sample_z, eval_model

import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm, trange
import wandb
from scipy.stats import wasserstein_distance as EMD


class SAWGAN(Module):
    """self attention wgan"""

    def __init__(self, n_feat: int = 2382, hid_dim: int = 64, uniform_z: bool = True, device="cpu") -> None:
        super().__init__()

        self.n_feat = n_feat
        self.device = device
        self.uniform_z = uniform_z

        self.gen = Generator(n_feat, hid_dim)
        self.disc = Discriminator(n_feat, hid_dim)

        self.gen.to(device)
        self.disc.to(device)

        self.gen.apply(self.weights_init)
        self.disc.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def optimize(self, normalized_data: np.ndarray, output_path: str, args):
        map_dataset = AfricaWholeFlatDataset(normalized_data)
        map_dataset.data = map_dataset.data.view(-1, 1, self.n_feat)
        dataloader = DataLoader(map_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=8)

        # Your Code Goes Here
        betas = (args.beta1, args.beta2)
        optimizer_gen = AdamW(self.gen.parameters(), lr=args.g_lr, betas=betas)
        optimizer_disc = AdamW(self.disc.parameters(),
                               lr=args.d_lr, betas=betas)

        disc_fake: torch.Tensor
        disc_real: torch.Tensor

        self.train()
        for epoch in trange(args.epochs):

            for batch in dataloader:
                batch: torch.Tensor = batch.to(self.device).float()
                size = batch.size(0)
                # print(torch.cuda.memory_allocated())

                # update disc, lock gen to save computation
                for _ in range(args.kkd):
                    optimizer_disc.zero_grad()

                    noise = sample_z(size, 1, self.n_feat,
                                     device=self.device, uniform=self.uniform_z)
                    fake_batch = self.gen(noise).detach()

                    disc_fake = self.disc(fake_batch)
                    disc_real = self.disc(batch)

                    disc_loss = disc_fake.mean() - disc_real.mean()
                    disc_loss.backward()

                    optimizer_disc.step()

                # update gen, lock disc
                for _ in range(args.kkg):
                    optimizer_disc.zero_grad()
                    optimizer_gen.zero_grad()

                    noise = sample_z(size, 1, self.n_feat,
                                     device=self.device, uniform=self.uniform_z)
                    fake_batch = self.gen(noise)

                    disc_fake = self.disc(fake_batch)

                    gen_loss = -disc_fake.mean()
                    gen_loss.backward()

                    optimizer_gen.step()

                if args.use_wandb:
                    wandb.log({"disc_loss": disc_loss,
                              "gen_loss": gen_loss,
                               "loss": -disc_loss})

            if (epoch + 1) % 1 == 0:
                if args.use_wandb:
                    avg, std, emd = eval_model(self, normalized_data)
                    wandb.log({"avg": avg, 'std': std, 'emd': emd})

                if os.path.exists(output_path):
                    torch.save(self.state_dict(),
                               f'{output_path}/{epoch}.pt')

    def generate(self, n: int = 15400) -> np.ndarray:
        fake_data = np.zeros((n, self.n_feat))
        # if num is divisible by 100, generate by batch, else generate one by one
        for l in range(0, n, 100):
            noise = sample_z(100, 1, self.n_feat,
                             device=self.device,
                             uniform=self.uniform_z)
            fake_data[l:l+100, :] = \
                self.gen(noise).cpu().detach().numpy().squeeze(axis=1)
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
        self.gamma = nn.Parameter(torch.tensor(0.01))

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

    def __init__(self, n_feat: int = 2382, hid_dim: int = 8):
        super().__init__()

        self.l1 = nn.Sequential(
            spectral_norm(nn.Conv1d(1, hid_dim, kernel_size=1)),
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

        x = self.l1(z)
        x, p = self.attn(x)
        x = self.l2(x)

        return x  # , p


class Discriminator(nn.Module):

    def __init__(self, n_feat: int = 2382, hid_dim: int = 8):
        super().__init__()

        self.l1 = nn.Sequential(
            spectral_norm(nn.Conv1d(1, hid_dim, kernel_size=1)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4)
        )
        self.attn = SelfAttn(hid_dim)

        self.fc = nn.Linear(n_feat * hid_dim, 1)

    def forward(self, x: torch.Tensor):
        x = self.l1(x)
        x, p = self.attn(x)
        x = x.flatten(1, -1)
        x = self.fc(x)

        return x  # , p
