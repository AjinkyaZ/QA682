from utils import *

from tqdm import tqdm
from pprint import pprint
import json
import _pickle as pkl
from time import time

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

_, X, y = make_data([example_X], [example_y], 1, glove)
print(len(tokenize(example_X[1])))
print(get_answer_span(y[0], example_X[1]))
num_ex_train = 512
num_ex_val = 64
idxs_train, X_train, y_train = make_data(data['X_train'], data['y_train'], num_ex_train, glove)
idxs_val, X_val, y_val = make_data(data['X_val'], data['y_val'], num_ex_val, glove)
print(len(X_train), len(y_train), len(X_val), len(y_val))

from models import *

conf = {"vocab": glove.vectors,
        "learning_rate": 0.4,
        "epochs": 5,
        "hidden_size": 50,
        "batch_size": 64,
        "opt": "Adamax",
        "n_layers": 1}
model = ModelV2(conf)
print(model)
model_name = "%s_D%s_B%s_E%s_H%s_LR%s_O%s"%(type(model).__name__, num_ex_train, model.batch_size, model.epochs, model.hidden_size, model.lr, conf["opt"])
print(model_name)

tic = time()
v_preds, losses, vlosses = model.fit((X_train, y_train), (X_val, y_val))
toc = time()
print("took", toc-tic, "seconds")
torch.save(model, '../evaluation/models/%s'%model_name)
import matplotlib.pyplot as plt
plt.plot(list(range(len(losses))), losses)
plt.plot(list(range(len(vlosses))), vlosses)
plt.show()

model = torch.load('../evaluation/models/%s'%model_name)
print(model)
print(model_name)