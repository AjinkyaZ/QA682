from utils.dataset import DataSet
from tqdm import tqdm
from collections import defaultdict
from nltk import word_tokenize
from nltk.corpus import stopwords
from pprint import pprint
import _pickle as pkl


d = DataSet(path='../data/train')

stops = set(stopwords.words('english'))
puncts = set([',', '.', ' ', ':', ';', '-', '?', '!', '$', '@', '#'
              '\'', '\'\'' , '’', '"', '”', '“', '`', '``', '‘',
              '(', ')', '{', '}', '[', ']', 
              '\'s', 'n\'t'])
ignore = stops.union(puncts)
vocab = defaultdict(int)

print("Reading bodies...")
for i in tqdm(d.articles.values()):
    wi = word_tokenize(i.lower())
    for w in wi:
        if w in ignore: continue
        vocab[w] += 1

print("Reading headlines...")
for i in tqdm(d.stances):
    hi = i['Headline']
    wi = word_tokenize(hi.lower())
    for w in wi:
        if w in ignore: continue
        vocab[w] += 1

MAX_SIZE = 1000
print("Generating vocabulary - top %s words"%MAX_SIZE)
sorted_vocab = [(w, vocab[w]) for w in sorted(list(vocab.keys()), key=lambda x: -vocab[x])][:MAX_SIZE]

with open('../data/vocab_counts_1000.pkl', 'wb') as f:
    pkl.dump(sorted_vocab, f)