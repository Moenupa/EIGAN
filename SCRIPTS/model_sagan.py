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


class SAWGAN(Module):
    """self attention wgan"""

    def __init__(self, n_feat: int = 2382, d_hid: int = 400, d_z: int = 100,
                 uniform_z: bool = True, device="cpu") -> None:
        super().__init__()

        self.n_feat = n_feat
        self.d_z = d_z
        self.device = device
        self.uniform_z = uniform_z

        self.gen = Generator(d_z, n_feat, d_hid)
        self.disc = Discriminator(n_feat, d_hid)

        self.gen.to(device)
        self.disc.to(device)

        self.gen.apply(self.weights_init)
        self.disc.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def optimize(self, normalized_data: np.ndarray, output_path: str, args):
        map_dataset = AfricaWholeFlatDataset(normalized_data)
        map_dataset.data = map_dataset.data.view(1, -1, self.n_feat)
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

                    noise = sample_z(1, size, self.d_z,
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

                    noise = sample_z(1, size, self.d_z,
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
            noise = sample_z(1, 100, self.d_z,
                             device=self.device,
                             uniform=self.uniform_z)
            fake_data[l:l+100, :] = \
                self.gen(noise).cpu().detach().numpy().squeeze()
        return fake_data


class MultiHeadLinear(nn.Module):
    def __init__(self, d_in: int, heads: int, d_k: int, bias: bool, sn: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, heads * d_k, bias=bias)
        if sn:
            self.linear = spectral_norm(self.linear)
        self.heads = heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x).view(*x.shape[:-1], self.heads, self.d_k)
        return x


class SelfAttn(nn.Module):
    def __init__(self, d_in: int, d_out: int = 400, num_heads: int = 8,
                 dropout: float = 0.2, bias: bool = True):
        super().__init__()
        assert d_out % num_heads == 0, \
            f'multi-head division err: d_out {d_out} % num_heads {num_heads} != 0'

        self.d_in = d_in
        self.d_out = d_out
        self.d_k = d_out // num_heads
        self.num_heads = num_heads

        self.q = MultiHeadLinear(d_in, num_heads, self.d_k, bias=bias)
        self.k = MultiHeadLinear(d_in, num_heads, self.d_k, bias=bias)
        self.v = MultiHeadLinear(d_in, num_heads, self.d_k, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.fc = spectral_norm(nn.Linear(d_out, d_out))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        """
            inputs :
                x : input feature maps (C, B, N)
            returns :
                out : self attention value + input feature 
                attention: (B, N, N)
        """
        C, B, N = x.shape

        # Q, K are (C, B, d_k, heads)
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        attention = self.softmax(torch.einsum('ibhd,jbhd->ijbh', Q, K)
                                 / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)))
        attention = self.attn_dropout(attention)

        out = torch.einsum('ijbh,jbhd->ibhd', attention, V)
        out = out.view(C, B, -1)
        out = self.fc(out)
        return out


class Generator(nn.Module):

    def __init__(self, d_z: int, d_out: int, d_hid: int = 400):
        super().__init__()

        self.model = nn.Sequential(
            spectral_norm(nn.Linear(d_z, d_hid)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            SelfAttn(d_hid, d_hid),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            spectral_norm(nn.Linear(d_hid, d_out)),
        )

    def forward(self, z: torch.Tensor):
        # z as a (batch, z_dim) tensor
        x = self.model(z)
        return x


class Discriminator(nn.Module):

    def __init__(self, d_in: int, d_hid: int = 400):
        super().__init__()

        self.model = nn.Sequential(
            SelfAttn(d_in, d_hid),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(d_hid, d_hid)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(d_hid, d_hid)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Linear(d_hid, 1))
        )

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        return x
