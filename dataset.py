import os
import collections

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from utils import check_cache, load_cache, save_cache, load_file


def load_politics(
    corpus_path,
    metadata_path,
    clip_count,
    max_len,
    batch_size,
    vocab=None
):
    if vocab is None:
        vocab = build_politics_vocab(corpus_path, metadata_path, clip_count)
    ds = PoliticsDataset(vocab, corpus_path, metadata_path, max_len)
    return DataLoader(ds, batch_size, shuffle=True)


def build_politics_vocab(corpus_path, metadata_path, clip_count, verbose=True):
    corpus = []
    df = pd.read_csv(metadata_path, dtype={'docid':str})
    for i, row in df.iterrows():
        if verbose:
            if i % 1000 == 0:
                print(f'building vocab: {i} / {len(df)}')
        path = os.path.join(corpus_path, f"{row['docid']}.txt")
        corpus.append(load_file(path, -1))
    vocab = Vocab(corpus, clip_count)
    if verbose:
        print(f'vocab size: {len(vocab)}')
    return Vocab(corpus, clip_count)


class Vocab:

    def __init__(self, corpus, clip_count):
        self.words = self._init(corpus, clip_count)
        self.encoding = {w:i for i,w in enumerate(sorted(self.words), 1)}
        self.decoding = {i:w for i,w in enumerate(sorted(self.words), 1)}

        # special tokens
        self._register_special('<pad>', 0)
    
    def __len__(self):
        assert len(self.encoding) == len(self.decoding)
        return len(self.encoding)
    
    def _init(self, corpus, clip_count):
        counter = collections.Counter()
        for sample in corpus:
            counter.update(sample)
        
        if clip_count > 0:
            for key in list(counter.keys()):
                if counter[key] < clip_count:
                    counter.pop(key)
        
        # special tokens
        # counter['<sos>'] = 0
        # counter['<eos>'] = 0
        counter['<unk>'] = 0

        return counter
    
    def _register_special(self, token, idx):
        self.encoding[token] = idx
        self.decoding[idx] = token

    def size(self):
        assert len(self.encoding) == len(self.decoding)
        return len(self.encoding)


class PoliticsDataset(Dataset):

    def __init__(self, vocab, corpus_path, metadata_path, max_len):
        self.corpus_path = corpus_path
        self.df = pd.read_csv(metadata_path, dtype={'docid': str})
        self.vocab = vocab
        self.max_len = max_len
        self.multi = 'multi' in metadata_path
        self.cache = {}

        self.unlabeled = 'coca_unlabeled' in corpus_path
        if self.unlabeled:
            self.df.sort_values('year', ascending=False, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
    
    def encode(self, sample):
        enc = self.vocab.encoding
        return np.array([enc.get(w, enc['<unk>']) for w in sample])
    
    def decode(self, sample):
        dec = self.vocab.decoding
        return np.array([dec.get(i, '<unk>') for i in sample])
    
    def pad(self, sample):
        return np.pad(sample, (0, self.max_len - len(sample)), 'constant')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        docid, label = entry['docid'], entry['label']
        if idx in self.cache:
            return self.cache[idx]
        path = os.path.join(self.corpus_path, '{}.txt'.format(docid))
        words = load_file(path, self.max_len)
        # encode doc
        x = self.pad(self.encode(words))
        if self.multi:
            label = eval(label)
            if 3 in label:
                label.remove(3)
            base = np.zeros(3)  # three labels
            if len(label) > 0:
                base[label] = 1
            y = base.astype(int)
        else:
            y = (np.array([0, 1]) == label).astype(int)
        # cache
        x = torch.from_numpy(x).long()
        y = torch.from_numpy(y).float()
        self.cache[idx] = (x, y)
        return (x, y)
