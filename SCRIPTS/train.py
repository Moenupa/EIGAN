from generative_model import WGAN_SIMPLE
from model_sagan import SAWGAN
from process_data import load_csv_with_cache, MinMaxScaler

import argparse
import os
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
    model = SAWGAN(device=device)
    exp_root = f'{MODEL_ROOT}/{args.model}_lr{args.lr:.0e}_{args.beta1}_b{args.batch_size}'
    os.makedirs(exp_root)

    model.optimize(normed_data, exp_root, use_wandb=args.use_wandb,
                   lr=args.lr, batch_size=args.batch_size,
                   betas=(args.beta1, args.beta2),
                   epochs=args.epochs)

    return model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="WGAN", help="model name")
    parser.add_argument("--epochs", type=int,
                        default=200, help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=2e-4, help="learning rate for the optimizer")
    parser.add_argument("-b", "--batch-size", type=int,
                        default=128, help="batch size for the optimizer")
    parser.add_argument("-z", "--uniform-noise", type=bool,
                        default=True, help="whether to use uniform noise, otherwise normalized noise")
    parser.add_argument("--beta1", type=float,
                        default=0.5, help="beta1 in Adam optimizer")
    parser.add_argument("--beta2", type=float,
                        default=0.999, help="beta2 in Adam optimizer")
    parser.add_argument("--use-wandb", type=bool,
                        default=False, help="whether to use wandb or not")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    train(args)
