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
    glove = setup_glove()
    print(glove.vectors.size())
    VOCAB_SIZE = glove.vectors.size()[0]
    with open('../data/data.json', 'r') as f:
        data = json.load(f)

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

    seed(1)
    zipped_data = list(zip(data['X_train'], data['y_train']))
    shuffle(zipped_data)
    data['X_train'], data['y_train'] = list(zip(*zipped_data))

    num_train = 128
    num_val = 32

    idxs_train, padlens_train, X_train, y_train = make_data(
        data['X_train'], data['y_train'], num_train, glove)
    idxs_val, padlens_val, X_val, y_val = make_data(
        data['X_val'], data['y_val'], num_val, glove)
    print(len(X_train), len(y_train), len(X_val), len(y_val))

    conf = {"vocab": glove.vectors,
            "learning_rate": 5e-3,
            "epochs": 1,
            "cell_type": "GRU",
            "hidden_size": 50,
            "batch_size": 32,
            "opt": "Adamax",
            "n_layers": 1,
            "linear_dropout": 0.3,
            "seq_dropout": 0.0,
            "save_every": 5}

    model = ModelV2(conf)
    print(model)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params}")
    if torch.cuda.is_available():
        model = model.cuda()

    opt = conf["opt"]
    model_name = f"{type(model).__name__}_D{num_train}_B{model.batch_size}_" \
        f"E{model.epochs}_H{model.hidden_size}_LR{model.lr}_O{opt}"
    print(model_name)

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
