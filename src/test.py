import argparse
import json
from pprint import pprint
from time import time

import torch
from torch.autograd import Variable as var
from tqdm import tqdm

import _pickle as pkl
from models import *
from utils import *


def switch_idxs(pred):
    if pred[0] > pred[1]:
        pred[0], pred[1] = pred[1], pred[0]
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nt', '--num_test', default=1024, type=int)
    parser.add_argument('-mf', '--model_file',
                        default='../data/checkpoint.pth.tar', type=str)

    args = parser.parse_args()
    print("Using config:")
    for arg, val in vars(args).items():
        print(arg, val)

    glove = setup_glove()
    # print(glove.vectors.size())
    VOCAB_SIZE = glove.vectors.size()[0]
    with open('../data/data.json', 'r') as f:
        data = json.load(f)

    model_path = args.model_file
    print()
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(f"{model_path}")
    model = ModelV2(checkpoint['model_init_config'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(model)

    bs = model.batch_size

    idxs_test, padlens_test, X_test, y_test = make_data(
        data['X_test'], data['y_test'], args.num_test, glove)

    dev_results = {}

    # pprint(data['X_test'][idxs_test[0]])

    print("Test data size:", args.num_test)
    n_batches = args.num_test//bs
    print(f"Batch size: {bs}, Batches: {n_batches}")
    print()

    model.eval()
    with torch.no_grad():
        for bindex,  i in enumerate(range(0, len(y_test)-bs+1, bs)):
            print("batch:", bindex)
            model.init_params(bs)

            Xb = torch.LongTensor(X_test[i:i+bs])
            yb = var(torch.LongTensor(y_test[i:i+bs]))
            if torch.cuda.is_available():
                Xb = Xb.cuda()
                yb = yb.cuda()

            pred = model.predict(Xb).data.tolist()
            pred = list(map(switch_idxs, pred))
            qids = [data['X_test'][j][0] for j in idxs_test[i:i+bs]]
            test_paras = [data['X_test'][j][1] for j in idxs_test[i:i+bs]]
            pads = padlens_test[i:i+bs]
            answers = list(map(get_answer_span, pred, pads, test_paras))
            # pprint(answers)
            batch_results = list(zip(qids, answers))
            # print(len(dict(batch_results)))
            dev_results.update(dict(batch_results))

    num_processed = len(dev_results)
    # print("processed:", num_processed)
    dev_results['version'] = '1.1'

    fname = f"run_test{args.num_test}.json"
    with open(f"../data/{fname}", 'w') as f:
        print(f"saving to {fname}")
        json.dump(dev_results, f)


if __name__ == "__main__":
    main()
