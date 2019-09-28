import argparse
import json
from pprint import pprint
from random import seed, shuffle
from time import time

import matplotlib.pyplot as plt
from tqdm import tqdm

import _pickle as pkl
from models import *
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nt', '--num_train', default=1024, type=int)
    parser.add_argument('-nv', '--num_val', default=256, type=int)
    parser.add_argument('-bs', '--batch_size', default=32, type=int)
    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float)
    parser.add_argument('-o', '--optimizer', default='Adam')
    parser.add_argument('-nl', '--n_layers', default=1, type=int)
    parser.add_argument('-s', '--save_every', default=5, type=int)
    parser.add_argument('-hs', '--hidden_size', default=64, type=int)
    parser.add_argument('-ld', '--linear_dropout', default=0.3, type=float)
    parser.add_argument('-ls', '--seq_dropout', default=0.0, type=float)

    args = parser.parse_args()
    print("Using config:")
    for arg, val in vars(args).items():
        print(arg, val)

    glove = setup_glove()
    # print(glove.vectors.size())
    VOCAB_SIZE = glove.vectors.size()[0]
    with open('../data/data.json', 'r') as f:
        data = json.load(f)

    """
    idx = 5
    example_X = (data['X_train'][idx])
    example_y = (data['y_train'][idx])

    print("ID:", example_X[0])
    print("Context:", example_X[1])
    print("Question:", example_X[2])
    print("Answer Span:", example_y)
    s, e = example_y
    assert example_X[1][s:e] == example_X[3]
    print("Answer:", example_X[3])

    print("=="*30)
    """

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
    print(model)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")

    if torch.cuda.is_available():
        model = model.cuda()

    model_name = f"{type(model).__name__}" \
        f"_D{args.num_train}_B{model.batch_size}_" \
        f"E{model.epochs}_H{model.hidden_size}_LR{model.lr}_O{model.opt_name}"
    print(f"Model file: {model_name}\n")

    print("Training model...")
    tic = time()
    v_preds, losses, vlosses = model.fit((X_train, y_train), (X_val, y_val))
    toc = time()
    print(f"took {toc-tic} seconds")
    torch.save(model, f"../evaluation/models/{model_name}")

    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(losses))), losses, label='train')
    plt.plot(list(range(len(vlosses))), vlosses, label='val')
    plt.legend()
    plt.savefig(f"run.png")

    model = torch.load(f"../evaluation/models/{model_name}")
    print(f"Saved to: {model_name}")


if __name__ == "__main__":
    main()
