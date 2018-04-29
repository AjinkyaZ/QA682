from utils import *

import torch
from torch.autograd import Variable as var
from tqdm import tqdm
from pprint import pprint
import json
import _pickle as pkl
from time import time
import json

glove = setup_glove()
print(glove.vectors.size())
VOCAB_SIZE = glove.vectors.size()[0]
with open('../data/data.json', 'r') as f:
    data = json.load(f)

model_name = "ModelV2_D1280_B64_E20_H50_LR0.4_OAdamax"
model = torch.load('../evaluation/models/%s'%model_name)
print(model)
print(model_name)


#bs = int(model_name.split("_")[2][1:])
bs = 64
print("batch size", bs)

num_test = 512
idxs_test, X_test, y_test = make_data(data['X_test'], data['y_test'], num_test, glove)

dev_results = {}

print("Test data size:", num_test)
for bindex,  i in tqdm(enumerate(range(0, len(y_test)-bs+1, bs))):
    print("batch:", bindex)
    #model.init_params(bs)
    Xb = torch.LongTensor(X_test[i:i+bs])
    yb = var(torch.LongTensor(y_test[i:i+bs]))
    pred = model.predict(Xb).data.tolist()
    print(pred)
    test_paras = [data['X_test'][j][1] for j in idxs_test]
    qids = [data['X_test'][j][0] for j in idxs_test]
    answers = list(map(get_answer_span, pred, test_paras))
    batch_results = list(zip(qids, answers))
    dev_results.update(dict(batch_results))

num_processed = len(dev_results)
print("processed:", num_processed)
dev_results['version'] = '1.1'
fname = 'run_%s_test%s.json'%(model_name, num_processed)
with open('../data/%s'%fname, 'w') as f:
    json.dump(dev_results, f)