# -*- coding:utf-8 -*-
from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools

import utils.utils as utils
from utils.tran_ner import *
import utils.evaluate as evaluate

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with based-transition-ner')
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    parser.add_argument('--emb_file', default='./data/sskip.100.vectors',
                        help='path to pre-trained embedding')
    parser.add_argument('--pretrained_dim', type=int, default=100, help='pretrained word embedding dimension')
    parser.add_argument('--train_file', default='./data/train.parser', help='path to training file')
    parser.add_argument('--dev_file', default='./data/dev.parser', help='path to development file')
    parser.add_argument('--test_file', default='./data/test.parser', help='path to test file')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (1)')
    parser.add_argument('--hidden_dim', type=int, default=100, help='hidden dimension')
    parser.add_argument('--lstm_input_dim', type=int, default=100, help='lstm_input dimension')
    parser.add_argument('--action_dim', type=int, default=20, help='action dimension')
    parser.add_argument('--rel_dim', type=int, default=20, help='relation dimension')
    parser.add_argument('--drop_out', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--singleton_ratio', type=float, default=0.2, help='singleton ratio')
    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='path to checkpoint prefix')
    parser.add_argument('--epoch', type=int, default=200, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch idx')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--layers', type=int, default=1, help='number of lstm layers')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.001, help='decay ratio of learning rate')
    parser.add_argument('--fine_tune', action='store_false', help='fine tune pre-trained embedding dictionary')
    parser.add_argument('--load_check_point', default='', help='path of checkpoint')
    parser.add_argument('--load_opt', action='store_true', help='load optimizer from ')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer method')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='grad clip at')
    parser.add_argument('--mini_count', type=float, default=5, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--least_iters', type=int, default=50, help='at least train how many epochs before stop')
    parser.add_argument('--shrink_embedding', action='store_true',
                        help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    args = parser.parse_args()

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    print('setting:')
    print(args)

    # load corpus
    print('loading corpus')
    with codecs.open(args.train_file, 'r', 'utf-8') as f:
        lines = f.readlines()
    with codecs.open(args.dev_file, 'r', 'utf-8') as f:
        dev_lines = f.readlines()
    with codecs.open(args.test_file, 'r', 'utf-8') as f:
        test_lines = f.readlines()

    # converting format
    dev_features, dev_labels, dev_actions = utils.read_corpus(dev_lines)
    test_features, test_labels, test_actions = utils.read_corpus(test_lines)

    if args.load_check_point:
        if os.path.isfile(args.load_check_point):
            print("loading checkpoint: '{}'".format(args.load_check_point))
            checkpoint_file = torch.load(args.load_check_point)
            args.start_epoch = checkpoint_file['epoch']
            f_map = checkpoint_file['f_map']
            a_map = checkpoint_file['a_map']
            train_features, train_labels = utils.read_corpus(lines)
        else:
            print("no checkpoint found at: '{}'".format(args.load_check_point))
    else:
        print('constructing coding table')
        # converting format
        train_features, train_labels, train_actions, f_map, l_map, a_map, singletons = utils.generate_corpus(lines, if_shrink_feature=True, thresholds=0)

        f_set = {v for v in f_map}
        f_map = utils.shrink_features(f_map, train_features, args.mini_count)


        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features),f_set)  # Add word in dev and in test into feature_map
        #print("dt_f_set size: '{}'".format(len(dt_f_set)))
        dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features), dt_f_set)
        #print("dt_f_set size: '{}'".format(len(dt_f_set)))


        if not args.rand_embedding:
            print("feature size: '{}'".format(len(f_map)))
            print('loading embedding')
            if args.fine_tune:  # which means does not do fine-tune
                f_map = {'<eof>': 0}
            f_map, embedding_tensor, in_doc_words = utils.load_embedding_wlm(args.emb_file, ' ', f_map, dt_f_set,args.caseless,args.unk, args.pretrained_dim, shrink_to_corpus=args.shrink_embedding)
            print("embedding size: '{}'".format(len(f_map)))


        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels))
        l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels), l_set)
        for label in l_set:
            if label not in l_map:
                l_map[label] = len(l_map)

    # construct dataset
    dataset = utils.construct_dataset(train_features, train_labels, train_actions, f_map, l_map, a_map, singletons, args.singleton_ratio, args.caseless)
    dev_dataset = utils.construct_dataset(dev_features, dev_labels, dev_actions, f_map, l_map, a_map, singletons, args.singleton_ratio, args.caseless)
    test_dataset = utils.construct_dataset(test_features, test_labels, test_actions, f_map, l_map, a_map, singletons, args.singleton_ratio, args.caseless)

    dataset_loader = [torch.utils.data.DataLoader(tup, args.batch_size, shuffle=True, drop_last=False) for tup in dataset]
    dev_dataset_loader = [torch.utils.data.DataLoader(tup, args.batch_size, shuffle=False, drop_last=False) for tup in dev_dataset]
    test_dataset_loader = [torch.utils.data.DataLoader(tup, args.batch_size, shuffle=False, drop_last=False) for tup in test_dataset]


    # build model
    print('building model')

    ner_model = TRAN_NER(f_map, a_map, len(f_map), len(a_map), args.pretrained_dim, args.hidden_dim, args.action_dim, args.rel_dim, args.layers, args.drop_out)

    if args.load_check_point:
        ner_model.load_state_dict(checkpoint_file['state_dict'])#state_dict=?
    else:
        if not args.rand_embedding:
            ner_model.load_pretrained_embedding(embedding_tensor)
        print('random initialization')
        ner_model.rand_init(init_embedding=args.rand_embedding)

    if args.update == 'sgd':
        optimizer = optim.SGD(ner_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.update == 'adam':
        optimizer = optim.Adam(ner_model.parameters(), lr=args.lr)


    if args.load_check_point and args.load_opt:
        optimizer.load_state_dict(checkpoint_file['optimizer'])


    if args.gpu >= 0:
        if_cuda = True
        print('device: ' + str(args.gpu))
        torch.cuda.set_device(args.gpu)
        ner_model.cuda()

    else:
        if_cuda = False
    '''
    if args.load_check_point:
        dev_f1, dev_acc = eval_batch(ner_model, dev_dataset_loader, pack, l_map)
        test_f1, test_acc = eval_batch(ner_model, test_dataset_loader, pack, l_map)
        print('(checkpoint: dev F1 = %.4f, dev acc = %.4f, F1 on test = %.4f, acc on test= %.4f)' %
              (dev_f1,
               dev_acc,
               test_f1,
               test_acc))
    '''

    tot_length = sum(map(lambda t: len(t), dataset_loader))
    best_f1 = float('-inf')
    #best_acc = float('-inf')
    track_list = list()
    start_time = time.time()
    epoch_list = range(args.start_epoch, args.start_epoch + args.epoch)
    patience_count = 0

    #label2id = evaluate.construct_label2id(a_map)

    for epoch_idx, args.start_epoch in enumerate(epoch_list):

        epoch_loss = 0
        ner_model.train()

        for feature, label, action in tqdm(
                itertools.chain.from_iterable(dataset_loader), mininterval=2,
                desc=' - Tot it %d (epoch %d)' % (tot_length, args.start_epoch), leave=False, file=sys.stdout):

            fea_v, label_v, action_v = utils.repack_vb(if_cuda, feature, label, action)
            ner_model.zero_grad()
            loss,_,_ = ner_model.forward(fea_v, action_v,if_cuda)
            loss.backward()
            nn.utils.clip_grad_norm(ner_model.parameters(), args.clip_grad)
            optimizer.step()
            epoch_loss += utils.to_scalar(loss)

        # update lr
        utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

        # average
        epoch_loss /= tot_length

        dev_f1, dev_pre, dev_rec = evaluate.calc_f1_score1(ner_model, dev_dataset_loader, a_map, if_cuda)

        #dev_pre, dev_rec, dev_f1 = evaluate.calc_f1_score(ner_model, dev_dataset_loader, a_map, label2id, if_cuda)

        if dev_f1 > best_f1:
            patience_count = 0
            best_f1 = dev_f1

            test_f1, test_pre, test_rec = evaluate.calc_f1_score1(ner_model, test_dataset_loader, a_map, if_cuda)
            #test_pre, test_rec, test_f1 = evaluate.calc_f1_score(ner_model, test_dataset_loader, a_map, label2id, if_cuda)

            track_list.append(
                {'loss': epoch_loss, 'dev_f1': dev_f1, 'test_f1': test_f1})

            print(
                '(loss: %.4f, epoch: %d, dev F1 = %.4f, dev pre = %.4f, dev rec = %.4f, F1 on test = %.4f, pre on test = %.4f, rec on test = %.4f), saving...' %
                (epoch_loss,
                 args.start_epoch,
                 dev_f1,
                 dev_pre,
                 dev_rec,
                 test_f1,
                 test_pre,
                 test_rec))

            try:
                utils.save_checkpoint({
                    'epoch': args.start_epoch,
                    'state_dict': ner_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'f_map': f_map,
                    'a_map': a_map,
                }, {'track_list': track_list,
                    'args': vars(args)
                    }, args.checkpoint + 'stack_lstm_transition_ner')
            except Exception as inst:
                print(inst)

        else:
            patience_count += 1
            print('(loss: %.4f, epoch: %d, dev F1 = %.4f)' %
                  (epoch_loss,
                   args.start_epoch,
                   dev_f1))
            track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1})

        print('epoch: ' + str(args.start_epoch) + '\t in ' + str(args.epoch) + ' take: ' + str(
            time.time() - start_time) + ' s')

        if patience_count >= args.patience and args.start_epoch >= args.least_iters:
            break


    # print best
    eprint(
            args.checkpoint + ' dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f \n' % (
            dev_f1, dev_rec, dev_pre, test_f1, test_rec, test_pre))

    # printing summary
    print('setting:')
    print(args)





