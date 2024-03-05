from generative_model import WGAN_SIMPLE
from process_data import load_csv_with_cache, MinMaxScaler

import argparse
from glob import glob

import numpy as np
import torch
import wandb


DATA_ROOT = './DATA'
MODEL_ROOT = './MODEL'
data_paths, model_paths = glob(f'{DATA_ROOT}/*'), glob(f'{MODEL_ROOT}/*')
assert data_paths
assert model_paths


def train(args=None):
    data = load_csv_with_cache(
        f'{DATA_ROOT}/Rayleigh_P30_downsampled_flat.csv')

    scaler = MinMaxScaler()
    scaler.fit(data)
    normed_data = scaler.transform(data)

    if args.use_wandb:
        wandb.init(project="EIGAN", config=args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WGAN_SIMPLE(ndim=2382, device=device)
    model.optimize(normed_data, MODEL_ROOT, use_wandb=args.use_wandb,
                   lr=args.lr, batch_size=args.batch_size,
                   betas=(args.beta1, args.beta2),
                   epochs=args.epochs, device=device)

    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,
                        default=200, help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=2e-4, help="learning rate for the optimizer")
    parser.add_argument("-b", "--batch-size", type=int,
                        default=128, help="batch size for the optimizer")
    parser.add_argument("--beta1", type=float,
                        default=0.9, help="beta1 in Adam optimizer")
    parser.add_argument("--beta2", type=float,
                        default=0.999, help="beta2 in Adam optimizer")
    parser.add_argument("--use-wandb", type=bool,
                        default=False, help="whether to use wandb or not")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
