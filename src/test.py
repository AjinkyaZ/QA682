from utils import *

import torch
from torch.autograd import Variable as var
from tqdm import tqdm
from pprint import pprint
import json
import _pickle as pkl
from time import time
import json
import gc

glove = setup_glove()
print(glove.vectors.size())
VOCAB_SIZE = glove.vectors.size()[0]
with open('../data/data.json', 'r') as f:
    data = json.load(f)

model_name = "ModelV2_D512_B32_E5_H32_LR0.01_OAdamax"
model = torch.load('../evaluation/models/%s'%model_name)
print(model)
print(model_name)


#bs = int(model_name.split("_")[2][1:])
bs = 64
print("batch size", bs)

num_test = 128
idxs_test, padlens_test, X_test, y_test = make_data(data['X_test'], data['y_test'], num_test, glove)

dev_results = {}

def switch_idxs(pred):
    if pred[0]>pred[1]:
        pred[0], pred[1] = pred[1], pred[0]
    return pred

bs = 32
print("Test data size:", num_test)
for bindex,  i in tqdm(enumerate(range(0, len(y_test)-bs+1, bs))):
    print("batch:", bindex)
    # model.init_params(bs)
    Xb = torch.LongTensor(X_test[i:i+bs])
    yb = var(torch.LongTensor(y_test[i:i+bs]))
    
    pred = model.predict(Xb).data.tolist()
    pred = list(map(switch_idxs, pred))
    qids = [data['X_test'][j][0] for j in idxs_test[i:i+bs]]
    test_paras = [data['X_test'][j][1] for j in idxs_test[i:i+bs]]
    pads = padlens_test[i:i+bs]
    answers = list(map(get_answer_span, pred, pads, test_paras))
    batch_results = list(zip(qids, answers))
    # print(len(dict(batch_results)))
    dev_results.update(dict(batch_results))

num_processed = len(dev_results)
print("processed:", num_processed)
dev_results['version'] = '1.1'
fname = 'run_%s_test%s.json'%(model_name, num_test)
with open('../data/%s'%fname, 'w') as f:
    print("saving to %s"%fname)
    json.dump(dev_results, f)