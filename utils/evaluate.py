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

def compute_ents(sen_len, actions, id2action):
    rels = ["ERROR" for i in range(sen_len)]
    bufferi = list()
    stacki = list()
    outputi = list()
    bufferi.append(-999)
    stacki.append(-999)
    outputi.append(-999)
    for idx in range(sen_len):
        bufferi.append(sen_len - idx - 1)
    for k,v in enumerate(actions):
        action_str = id2action[v]
        if action_str.startswith("S"):
            assert (len(bufferi) > 1)
            stacki.append(bufferi.pop())
        elif action_str.startswith("R"):
            assert (len(stacki) > 1)
            while len(stacki)>1:
                rels[stacki[-1]] = action_str
                outputi.append(stacki.pop())
        elif action_str.startswith("O"):
            assert (len(bufferi) > 1)
            outputi.append(bufferi[-1])
            rels[bufferi.pop()] = "0"
    assert (len(bufferi) == 1)
    return rels

def construct_label2id(action2id):
    label2id = dict()
    for k, v in action2id.items():
        if str(k).startswith("R"):
            label2id[k] = len(label2id)
    return label2id


def calc_f1_score(ner_model, dataset_loader, action2id, label2id, if_cuda):
    id2action = {v: k for k, v in action2id.items()}
    for feature, label, action in itertools.chain.from_iterable(dataset_loader):
        fea_v, label_v, action_v = utils.repack_vb(if_cuda, feature, label, action)
        x=autograd.Variable(torch.LongTensor(list()))
        loss, pre_action, right_num = ner_model.forward(fea_v, x)
        right_relation = compute_ents(len(fea_v), action_v.squeeze(1).data.tolist(), id2action)
        pred_relation = compute_ents(len(fea_v), pre_action, id2action)
        confusion=torch.LongTensor(len(label2id),len(label2id)).zero_()
        for i in range(len(fea_v)):
            ri = right_relation[i]
            pi = pred_relation[i]
            if not (ri == "0" or pi == "0"):
                rr = label2id[ri]
                pr = label2id[pi]
                confusion[pr][rr] +=1
    class_tp = torch.LongTensor(len(label2id)).zero_()
    class_fp = torch.LongTensor(len(label2id)).zero_()
    class_fn = torch.LongTensor(len(label2id)).zero_()
    #compute tp, fp and fn
    for i in range(len(label2id)):
        for j in range(len(label2id)):
            if i==j:
                class_tp[i] = confusion[i][j]
            else:
                class_fp[i] += confusion[i][j]
                class_fn[i] += confusion[j][i]
    #compute precision, recall, f1
    global_f1_score = 0
    global_pre_score = 0
    global_recall_score = 0
    pre =0
    recall =0
    f1 =0
    all_tp = 0
    all_fp = 0
    all_fn = 0
    for i in range(len(label2id)):
        all_tp+=class_tp[i]
        all_fn+=class_fn[i]
        all_fp+=class_fp[i]

        '''
        if not class_tp[i]+class_fp[i] ==0:
            pre = class_tp[i]/(class_tp[i]+class_fp[i])
        if not (class_tp[i]+class_fn[i]) ==0:
            recall = class_tp[i]/(class_tp[i]+class_fn[i])
        if not pre+recall==0:
            f1 = 2*pre*recall/(pre+recall)
        global_f1_score += f1 / len(label2id)
        global_pre_score += pre / len(label2id)
        global_recall_score += recall / len(label2id)
        '''
    pre = all_tp / (all_tp + all_fp)
    recall = all_tp / (all_tp + all_fn)
    f1 = 2 * pre * recall / (pre + recall)

    return pre, recall, f1

def calc_f1_score1(ner_model, dataset_loader, action2idx, if_cuda):

    idx2action = {v: k for k, v in action2idx.items()}
    ner_model.eval()
    correct = 0
    total_correct_entity = 0
    total_act = 0

    total_entity_in_gold = 0
    total_entity_in_pre = 0
    for feature, label, action in itertools.chain.from_iterable(dataset_loader):
        fea_v, tg_v, ac_v = utils.repack_vb(if_cuda, feature, label, action)

        x = autograd.Variable(torch.LongTensor(list()), requires_grad=False)
        loss, pre_action, right_num = ner_model.forward(fea_v, x, if_cuda)

        num_entity_in_real, num_entity_in_pre, correct_entity = to_entity(ac_v.squeeze(1).data.tolist(), pre_action, idx2action)

        total_correct_entity += correct_entity
        total_entity_in_gold += num_entity_in_real
        total_entity_in_pre += num_entity_in_pre


    if total_entity_in_pre > 0 :
        pre = total_correct_entity / float(total_entity_in_pre)
    else:
        pre = 0
    if total_entity_in_gold > 0 :
        rec = total_correct_entity / float(total_entity_in_gold)
    else:
        rec = 0
    if (pre + rec) > 0:
        f1 = 2 * pre * rec / float(pre + rec)
    else:
        f1 = 0
    return f1, pre, rec

def to_entity(real_action, predict_action, idx2action):
    flags = [False, False]
    entitys = [[],[]]
    actions = [real_action, predict_action]
    for idx in range(len(actions)):
        ner_start_pos = -1
        for ac_idx in range(len(actions[idx])):
            x=actions[idx]
            y=x[ac_idx]
            z=idx2action[y]
            if z.startswith('S') and ner_start_pos < 0:
                ner_start_pos = ac_idx
            elif z.startswith('O') and ner_start_pos >= 0:
                ner_start_pos = -1
            elif z.startswith('R') and ner_start_pos >= 0:
                entitys[idx].append(str(ner_start_pos)+'-'+str(ac_idx-1)+idx2action[actions[idx][ac_idx]])
                ner_start_pos = -1
    correct_entity = set(entitys[0]) & set(entitys[1])
    return len(entitys[0]), len(entitys[1]), len(correct_entity)



