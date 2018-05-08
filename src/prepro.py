from utils import clean

import _pickle as pkl
from collections import Counter
import json
import numpy as np
from pprint import pprint
from random import random
from sklearn.cross_validation import train_test_split
from tqdm import tqdm


def main():
    with open('../data/train-v1.1.json', 'r') as f:
        train_data = json.load(f)
    with open('../data/dev-v1.1.json', 'r') as f:
        test_data = json.load(f)  # use as test

    data_struct = {'train_X': [], 'train_y': [],
                   'test_X': [], 'test_y': []}  # structured data
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

    data_tokens = [clean(i.lower()) for i in data_tokens]
    data_tokens = [i for i in data_tokens if len(i) > 0]
    data_vocab = Counter(data_tokens)

    raw_data = {'X_train': None,
                'y_train': None,
                'X_val': None,
                'y_val': None,
                'X_test': data_struct['test_X'],
                'y_test': data_struct['test_y']}
    raw_data['X_train'], raw_data['X_val'], raw_data['y_train'], raw_data['y_val'] = train_test_split(
        data_struct['train_X'], data_struct['train_y'], test_size=0.25)
    for k, v in raw_data.items():
        print(k, len(v))
    with open('../data/data.json', 'w') as f:
        json.dump(raw_data, f)


if __name__ == "__main__":
    main()
