import os
import pickle

import torch
import numpy as np


def check_cache(fname):
    path = os.path.join('cache', fname + '.pkl')
    return os.path.exists(path)


def load_cache(fname):
    path = os.path.join('cache', fname + '.pkl')
    f = open(path, 'rb')
    return pickle.load(f)


def save_cache(obj, fname):
    path = os.path.join('cache', fname + '.pkl')
    f = open(path, 'wb')
    pickle.dump(obj, f)


def load_file(path, max_len):
    with open(path, 'r') as f:
        sents = [s.lower() for s in f.read().strip().split() if \
                 len(s.strip()) > 0]
        if max_len > 0:
            sents = sents[:max_len]
    return sents


def create_pretrained(vocab):
    np.random.seed(1)
    glove_words, glove_vecs = load_cache('glove')
    embeddings = np.zeros((vocab.size(), 300))
    for i in range(vocab.size()):
        word = vocab.decoding[i]
        if word in glove_words:
            embeddings[i,:] = glove_vecs[glove_words[word]]
        else:
            embeddings[i,:] = np.random.uniform(-1.0, 1.0, size=(300,))
    embeddings = torch.from_numpy(embeddings).float()
    return embeddings
