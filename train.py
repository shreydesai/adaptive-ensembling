import os
import sys
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np

from models import CNN, AdaptiveCNN
from utils import load_cache, create_pretrained
from dataset import load_politics


class Logger:

    def __init__(self, name):
        self.name = name
        self.data = collections.defaultdict(list)
    
    def log(self, k, v):
        self.data[k].append(v)


def train_se(epoch):
    for i, (a, b) in enumerate(zip(src_train_loader, trg_train_loader)):
        student.train()
        teacher.train()

        src_inputs, src_labels = a
        trg_inputs, _, _ = b

        s_optimizer.zero_grad()
        if args.cuda:
            src_inputs = src_inputs.cuda()
            src_labels = src_labels.cuda()
            trg_inputs = trg_inputs.cuda()
        
        # cross-entropy loss
        ce_out = student(src_inputs)
        ce_loss = cross_entropy_loss(ce_out, src_labels)
        
        s_trg_out = student(trg_inputs)
        with torch.no_grad():
            t_trg_out = teacher(trg_inputs)
        trg_loss = consistency_loss(s_trg_out, t_trg_out)
        
        # train student
        loss = ce_loss + trg_loss
        loss.backward()
        s_optimizer.step()

        # train teacher
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            if t_p.requires_grad:
                param = args.alpha * t_p.data + (1. - args.alpha) * s_p.data
                t_p.data.copy_(param)
        
        # logging
        logger.log('t-ce', ce_loss.item())
        logger.log('t-trg', trg_loss.item())


def train_ae(epoch):
    for i, (a, b) in enumerate(zip(src_train_loader, trg_train_loader)):
        student.train()
        teacher.train()

        src_inputs, src_labels = a
        trg_inputs, _ = b

        s_optimizer.zero_grad()
        t_optimizer.zero_grad()
        if args.cuda:
            src_inputs = src_inputs.cuda()
            src_labels = src_labels.cuda()
            trg_inputs = trg_inputs.cuda()
            trg_times = trg_times.cuda()
        
        # cross-entropy loss
        ce_out = student(src_inputs)
        ce_loss = cross_entropy_loss(ce_out, src_labels)

        # consistency loss (src)
        with torch.no_grad():
            t_src_out = teacher(src_inputs, student)
        src_loss = consistency_loss(ce_out, t_src_out)
        
        # consistency loss (trg)
        s_trg_out = student(trg_inputs)
        t_trg_out = teacher(trg_inputs, student)
        trg_loss = consistency_loss(s_trg_out, t_trg_out)
        
        loss = ce_loss + trg_loss + src_loss
        
        # train student
        loss.backward()
        s_optimizer.step()
        t_optimizer.step()
        
        # logging
        logger.log('t-ce', ce_loss.item())
        logger.log('t-trg', trg_loss.item())
        if args.train_src:
            logger.log('t-src', src_loss.item())


def dev(sample=False):
    model = teacher
    if not all([args.train_src, args.train_trg]):
        model = student
    model.eval()
    dev_loss = 0.
    for i, (inputs, labels) in enumerate(trg_dev_loader):
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            logits = model(inputs)
        loss = cross_entropy_loss(logits, labels)
        dev_loss += loss.item()
        if sample:
            logger.log('v-ce', loss.item())
            return loss.item()
    dev_loss /= len(trg_dev_loader)
    return dev_los


def test():
    teacher.load_state_dict(torch.load(f'teacher_{args.name}.th'))
    teacher.eval()
    ds = test_loader.dataset
    y_true, y_pred = [], []
    for i in range(len(ds)):
        x, y = ds.__getitem__(i)
        if args.cuda:
            x = inputs.cuda()
        x = x[None, :]  # [1, T]
        with torch.no_grad():
            pred = teacher(x).squeeze()
        pred = pred.cpu().numpy()
        y_true.append(np.argmax(y))
        y_pred.append(np.argmax(pred))
    return classification_report(y_true, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--train-src', action='store_true')
    parser.add_argument('--train-trg', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--clip-count', type=int)
    parser.add_argument('--clip-grad', type=float)
    parser.add_argument('--input-size', type=int)
    parser.add_argument('--output-size', type=int)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--max-len', type=int)
    parser.add_argument('--kernel-size', type=int)
    parser.add_argument('--interval', type=int)
    parser.add_argument('--cuda', type=bool)
    args = parser.parse_args()
    print(args)

    train_meta =  # fill in
    train_csv =  # fill in
    coca_unlabeled_train_meta =  # fill in
    coca_unlabeled_train_csv =  # fill in
    coca_labeled_dev_meta =  # fill in
    coca_labeled_dev_csv =  # fill in
    coca_test_meta =  # fill in
    coca_test_csv =  # fill in

    src_train_loader = load_politics(
        train_meta,
        train_csv,
        args.clip_count,
        args.max_len,
        args.batch_size,
    )
    vocab = src_train_loader.dataset.vocab
    trg_train_loader = load_politics(
        coca_unlabeled_train_meta,
        coca_unlabeled_train_csv,
        args.clip_count,
        args.max_len,
        args.batch_size,
        vocab,
    )
    trg_dev_loader = load_politics(
        coca_labeled_dev_meta,
        coca_labeled_dev_csv,
        args.clip_count,
        args.max_len,
        args.batch_size,
        vocab,
    )
    test_loader = load_politics(
        coca_test_meta,
        coca_test_csv,
        args.clip_count,
        args.max_len,
        args.batch_size,
        vocab,
    )

    # pretrained embeddings
    pretrained = create_pretrained(vocab)

    # models
    params = [
        args.input_size,
        vocab.size(),
        args.hidden_size,
        args.output_size,
        args.max_len,
        args.kernel_size,
        args.dropout,
        pretrained
    ]

    if args.ensemble:
        student = AdaptiveCNN(*params)
        teacher = AdaptiveCNN(*params)
        train_fx = train_ae
        for n, p in teacher.named_parameters():
            if not n.endswith('_consts'):
                p.requires_grad_(False)
            if args.verbose and p.requires_grad:
                print('training', n)
    else:
        student = CNN(*params)
        teacher = CNN(*params)
        train_fx = train_se
    
    if args.cuda:
        student = student.cuda()
        teacher = teacher.cuda()

    print(student)
    print(
        'params:',
        sum([p.numel() for p in student.parameters() if p.requires_grad])
    )
    
    s_optimizer = optim.Adam(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    t_optimizer = optim.Adam(
        teacher.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    cross_entropy_loss = nn.BCEWithLogitsLoss(reduction='sum')
    consistency_loss = nn.MSELoss(reduction='sum')
    logger = Logger(args.name)

    best_loss = np.inf

    for epoch in range(1, args.epochs + 1):
        train_fx(epoch)
        loss = dev(epoch)
        if loss < best_loss:
            best_loss = loss
            torch.save(student.state_dict(), f'student_{args.name}.th')
            torch.save(teacher.state_dict(), f'teacher_{args.name}.th')
            print('* saved')

    report = test()
    logger.log('test', report)
    print(report)
    pickle.dump(logger.data, open(f'log_{args.name}.pkl', 'wb'))

    # clean up
    os.remove(f'student_{args.name}.th')
    os.remove(f'teacher_{args.name}.th')
