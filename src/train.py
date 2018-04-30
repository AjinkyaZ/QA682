from utils import *
from models import *

from tqdm import tqdm
from pprint import pprint
import json
import _pickle as pkl
from time import time
from random import shuffle, seed

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
print("Answer:", example_X[3])

_, p, X, y = make_data([example_X], [example_y], 1, glove)
print(y)
print(get_answer_span(y[0], p[0], example_X[1]))
print("===========")

seed(1)
zipped_data = list(zip(data['X_train'], data['y_train']))
shuffle(zipped_data)
data['X_train'], data['y_train'] = list(zip(*zipped_data))

num_train = 32
num_val = 32

idxs_train, padlens_train, X_train, y_train = make_data(data['X_train'], data['y_train'], num_train, glove)
idxs_val, padlens_val, X_val, y_val = make_data(data['X_val'], data['y_val'], num_val, glove)
print(len(X_train), len(y_train), len(X_val), len(y_val))

conf = {"vocab": glove.vectors,
        "learning_rate": 0.01,
        "epochs": 10,
        "hidden_size": 50,
        "batch_size": 16,
        "opt": "Adamax",
        "n_layers": 1,
        "linear_dropout": 0.4,
        "seq_dropout": 0.3,
        "save_every": 5}

model = ModelV2(conf)
print(model)
if torch.cuda.is_available():
    model = model.cuda()
model_name = "%s_D%s_B%s_E%s_H%s_LR%s_O%s"%(type(model).__name__, num_train, model.batch_size, model.epochs, model.hidden_size, model.lr, conf["opt"])
print(model_name)

tic = time()
v_preds, losses, vlosses = model.fit((X_train, y_train), (X_val, y_val))
toc = time()
print("took", toc-tic, "seconds")
torch.save(model, '../evaluation/models/%s'%model_name)
import matplotlib.pyplot as plt
plt.plot(list(range(len(losses))), losses, label='train')
plt.plot(list(range(len(vlosses))), vlosses, label='val')
plt.legend()
plt.show()

model = torch.load('../evaluation/models/%s'%model_name)
print("Saved to: %s"%model_name)