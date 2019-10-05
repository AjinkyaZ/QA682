import argparse
import json
from pprint import pprint
from random import seed, shuffle
from time import time

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable as var
from torchviz import make_dot
from tqdm import tqdm

import _pickle as pkl
import wandb
from models import *
from utils import *

wandb.init(project="qa682")


def train(model, X, y, optimizer, epoch, loss_fn):
    model.train()

    bs = model.batch_size
    n_batches = len(y)//bs
    print(f"Batch size: {bs}, Batches: {n_batches}")

    tloss = 0.0
    tic_train = time()
    for bindex, i in enumerate(range(n_batches)):
        model.init_params(bs)
        Xb = torch.LongTensor(X[i:i+bs])
        yb = torch.LongTensor(y[i:i+bs])

        if torch.cuda.is_available():
            Xb = Xb.cuda()
            yb = yb.cuda()

        pred = model(Xb)

        # make_dot(pred)

        start_loss = loss_fn(pred[:, :model.output_size], yb[:, 0])
        end_loss = loss_fn(pred[:, model.output_size:], yb[:, 1])
        tbloss = start_loss + end_loss
        tloss += tbloss.item()

        print(f"batch {bindex} : {tbloss.item():0.6f}")
        tbloss.backward()
        optimizer.step()
        optimizer.zero_grad()

    toc_train = time()
    print(f"Epoch {epoch} took: {toc_train-tic_train:0.3f}s")
    print(f"Epoch training loss: {tloss:0.6f}")

    return tloss


def validate(model, X, y, epoch, loss_fn):
    model.eval()

    bs = model.batch_size
    n_batches = len(y)//bs

    vloss = 0.0
    tic_val = time()
    with torch.no_grad():
        for bindex, i in enumerate(range(n_batches)):
            model.init_params(bs)
            Xb = torch.LongTensor(X[i:i+bs])
            yb = torch.LongTensor(y[i:i+bs])
            if torch.cuda.is_available():
                Xb = Xb.cuda()
                yb = yb.cuda()

            val_pred = model(Xb)

            start_loss = loss_fn(
                val_pred[:, :model.output_size], yb[:, 0])
            end_loss = loss_fn(
                val_pred[:, model.output_size:], yb[:, 1])
            vbloss = start_loss + end_loss
            vloss += vbloss.item()
    toc_val = time()
    print(f"Epoch validation loss: {vloss:0.6f}")
    return vloss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nt', '--num_train', default=1024, type=int)
    parser.add_argument('-nv', '--num_val', default=256, type=int)
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-e',  '--epochs', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float)
    parser.add_argument('-o',  '--optimizer', default='Adam')
    parser.add_argument('-nl', '--n_layers', default=1, type=int)
    parser.add_argument('-s',  '--save_every', default=5, type=int)
    parser.add_argument('-hs', '--hidden_size', default=64, type=int)
    parser.add_argument('-ld', '--linear_dropout', default=0.3, type=float)
    parser.add_argument('-ls', '--seq_dropout', default=0.0, type=float)
    parser.add_argument('-mf', '--model_file',
                        default='./checkpoint.pth.tar', type=str)

    args = parser.parse_args()
    print("Using config:")
    for arg, val in vars(args).items():
        print(arg, val)

    glove = setup_glove()
    # print(glove.vectors.size())
    VOCAB_SIZE = glove.vectors.size()[0]
    with open('../data/data.json', 'r') as f:
        data = json.load(f)

    seed(1)
    zipped_data = list(zip(data['X_train'], data['y_train']))
    shuffle(zipped_data)
    data['X_train'], data['y_train'] = list(zip(*zipped_data))

    idxs_train, padlens_train, X_train, y_train = make_data(
        data['X_train'], data['y_train'], args.num_train, glove)
    idxs_val, padlens_val, X_val, y_val = make_data(
        data['X_val'], data['y_val'], args.num_val, glove)
    # print(len(X_train), len(y_train), len(X_val), len(y_val))

    conf = {"vocab": glove.vectors,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "cell_type": "GRU",
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "opt": args.optimizer,
            "n_layers": args.n_layers,
            "linear_dropout": args.linear_dropout,
            "seq_dropout": args.seq_dropout,
            "save_every": args.save_every}

    print()
    print("Constructing model...")
    model = ModelV2(conf)
    model.init_params(model.batch_size)
    print(model)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")

    wandb.watch(model)
    wandb.config.update(args)
    wandb.config.n_params = n_params

    if torch.cuda.is_available():
        model = model.cuda()

    # v_preds, losses, vlosses = model.fit((X_train, y_train), (X_val, y_val))

    optimizer = model.opt(model.parameters(), model.lr)

    epochs_train = args.epochs

    # ignore padding index
    nll_loss = nn.NLLLoss(ignore_index=glove.stoi['<pad>'])
    tlosses = []
    vlosses = []
    best_vloss = float("inf")

    print("Training model...")
    tic = time()

    for epoch in range(epochs_train):
        tic_train = time()
        print(f"Epoch {epoch}")

        # train model

        tloss = train(model, X_train, y_train, optimizer, epoch, nll_loss)
        tlosses.append(tloss)

        # validate model
        vloss = validate(model, X_val, y_val, epoch, nll_loss)
        vlosses.append(vloss)

        wandb.log({"train_loss": tloss,
                   "val_loss": vloss
                   }, step=epoch)
        if vloss < best_vloss:
            is_best = True
            best_vloss = vloss
        else:
            is_best = False

        if epoch % args.save_every == 0:
            print(f"Saving model to {args.model_file}")
            checkpoint = {'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'training_loss': tloss,
                          'validation_loss': vloss,
                          'model_init_config': conf}
            save_checkpoint(checkpoint, is_best,
                            args.model_file)
        print()

    toc = time()
    print(f"{epochs_train} epochs took {toc-tic} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(tlosses))), tlosses, label='train')
    plt.plot(list(range(len(vlosses))), vlosses, label='val')
    plt.legend()
    plt.savefig(f"run.png")


if __name__ == "__main__":
    main()
