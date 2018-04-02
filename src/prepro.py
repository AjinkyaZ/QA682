from tqdm import tqdm
from pprint import pprint
import json
from sklearn.cross_validation import train_test_split
from random import random
import numpy as np
import _pickle as pkl
from collections import Counter

with open('../data/train-v1.1.json', 'r') as f:
    train_data = json.load(f)
with open('../data/dev-v1.1.json', 'r') as f:
    test_data = json.load(f) # use as test

data_struct = {'train_X': [], 'train_y': [], 'test_X': [], 'test_y': []} # structured data
split_ratio = 0.2
data_tokens = []

for dataset in ['train', 'test']:
    if dataset == 'train':
        data = train_data
    else:
        data = test_data
    for d in tqdm(data['data']):
        paras = d['paragraphs']
        for para in paras:
            context = para['context']
            data_tokens.extend(context.lower().strip().split(" "))
            qas = para['qas']
            for qa in qas:
                q_text = qa['question']
                data_tokens.extend(q_text.lower().strip().split(" "))
                for a in qa['answers']:
                    start, ans_text = a['answer_start'], a['text']
                    X_q = (context, q_text, ans_text)
                    ans = (start, start+len(ans_text))
                    data_tokens.extend(ans_text.lower().strip().split(" "))
                    data_struct[dataset+'_X'].append(X_q)
                    data_struct[dataset+'_y'].append(ans)

def clean(token):
    cleaned_token = token.strip(".,?!-:;'()[]\"`")
    if cleaned_token[-2:] == "'s":
        cleaned_token = cleaned_token[:-2]
    if cleaned_token[-2:] == "'t":
        cleaned_token = cleaned_token[:-2]+'t'
    return cleaned_token
data_tokens = [clean(i.lower()) for i in data_tokens]
data_tokens = [i for i in data_tokens if len(i)>0]
data_vocab = Counter(data_tokens)

raw_data = {'X_train': None,
        'y_train': None,
        'X_val': None,
        'y_val': None,
        'X_test': data_struct['test_X'],
        'y_test': data_struct['test_y']}
raw_data['X_train'], raw_data['X_val'], raw_data['y_train'], raw_data['y_val'] \
     = train_test_split(data_struct['train_X'], data_struct['train_y'], test_size=0.25)
for k, v in raw_data.items():
    print(k, len(v))
with open('../data/data.json', 'w') as f:
    json.dump(raw_data, f)


with open('../data/glove/glove.6B.50d.pkl', 'rb') as f:
    glove_dict = pkl.load(f)
print(len(glove_dict))

oov = []
for w, c in data_vocab.most_common():
    if w not in glove_dict:
        oov.append((w, c))
pprint(sorted(oov, key=lambda x:-x[1])[:10])


def get_glove_vec(token, dim=50):
    if token in glove_dict:
        return glove_dict[token]
    else:
        return [-1]*dim # unknown/OOV token
    
def make_features(context, ques):
    context_words = [clean(w) for w in context.lower().split(" ")][:200]
    ques_words = [clean(w) for w in ques.lower().split(" ")][:20]
    
    dim = 50
    context_vec = [get_glove_vec(w, dim) for w in context_words]
    ques_vec = [get_glove_vec(w) for w in ques_words]
    
    # zero pad shorter sequences
    if len(context_vec)<200:
        padding_len = 200 - len(context_vec)
        padding_zeros = [[0]*dim]*padding_len
        padding_zeros.extend(context_vec)
        context_vec = padding_zeros
    if len(ques_vec)<20:
        padding_len = 20 - len(ques_vec)
        padding_zeros = [[0]*dim]*padding_len
        padding_zeros.extend(ques_vec)
        ques_vec = padding_zeros

    context_vec = np.array(context_vec)
    ques_vec = np.array(ques_vec)
    return (context_vec, ques_vec)


data = {'X_train':[], 'y_train': [], 'X_test': [], 'y_test': [], 'X_val': [], 'y_test': []}
for k, d in raw_data.items():
    print(k)
    if 'y_' in k:
        data[k] = np.array(d)
        np.save('./features/%s'%(k), data[k])
        continue
    for (c, q, _) in d:
        data[k].append(make_features(c, q))
    data[k] = np.array(data[k])
    np.save('./features/%s'%(k), data[k])        