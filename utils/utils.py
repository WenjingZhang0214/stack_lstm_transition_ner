import codecs
import csv
import itertools
from functools import reduce

import numpy as np
import shutil
import torch
import json
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.init
from utils.ner_dataset import *



def shrink_features(feature_map, features, thresholds):
    """
    filter un-common features by threshold
    """
    feature_count = {k: 0 for (k, v) in iter(feature_map.items())}
    for feature_list in features:
        for feature in feature_list:
            feature_count[feature] += 1
    shrinked_feature_count = [k for (k, v) in iter(feature_count.items()) if v >= thresholds]
    feature_map = {shrinked_feature_count[ind]: (ind + 1) for ind in range(0, len(shrinked_feature_count))}

    #inserting unk to be 0 encoded
    feature_map['<unk>'] = 0
    #inserting eof
    feature_map['<eof>'] = len(feature_map)
    return feature_map

def generate_corpus(lines, if_shrink_feature=False, thresholds=1):
    """
    generate label, feature, word dictionary and label dictionary

    args:
        lines : corpus
        if_shrink_feature: whether shrink word-dictionary
        threshold: threshold for shrinking word-dictionary

    """
    features = list()#word
    labels = list()#pos
    actions = list()

    tmp_fl = list()
    tmp_ll = list()
    tmp_al = list()

    feature_map = dict()
    label_map = dict()
    action_map = dict()

    wordcount_map = dict()

    count = -1
    initial = 0  # false
    first = 1  # true
    doc_start = 0


    for line in lines:
        line = line.rstrip('\n')
        if len(line) == 0:
            count = 0;
            if not first and not doc_start:
                features.append(tmp_fl)
                labels.append(tmp_ll)
                actions.append(tmp_al)
            initial = 1
            doc_start = 0
            tmp_fl = list()
            tmp_ll = list()
            tmp_al = list()
        elif count == 0 :
            first = 0
            count = 1
            if (line[4:14] == '-DOCSTART-'):
                doc_start = 1
                initial=0
                continue
            if initial:
                line = line[3:len(line) - 1]
                words = line.replace('\'','').split()
                for word in words:
                    # print(word)
                    if word[-1]==',':
                        word = word[0:len(word)-1]
                    posIndex=word.rfind("-")
                    feature = word[:posIndex].strip()  # German
                    label = word[posIndex+1:].strip()  # JJ
                    if label not in label_map:
                        label_map[label] = len(label_map)
                    tmp_ll.append(label)
                    # new word
                    if feature not in feature_map:
                        feature_map[feature] = len(feature_map)+1# 0 is for unk
                        wordcount_map[feature] = 1
                    else:
                        wordcount_map[feature] += 1
                    tmp_fl.append(feature)
            initial = 0
        elif count == 1:
            count = 0
            if doc_start:
                continue
            tmp_al.append(line)
            if line not in action_map:
                action_map[line] = len(action_map)


    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)
        actions.append(tmp_al)
    if if_shrink_feature:
        feature_map = shrink_features(feature_map, features, thresholds)
    else:
        # inserting unk to be 0 encoded
        feature_map['<unk>'] = 0
        # inserting eof
        feature_map['<eof>'] = len(feature_map)

    singletons = list()
    for k,v in wordcount_map.items():
        if v==1:
            singletons.append(k)

    return features, labels, actions, feature_map, label_map, action_map, singletons

def read_corpus(lines):
    """
    convert corpus into features and labels and actions
    """
    features = list()  # word
    labels = list()  # pos
    actions = list()

    tmp_fl = list()
    tmp_ll = list()
    tmp_al = list()

    count = -1
    initial = 0  # false
    first = 1  # true
    doc_start = 0

    for line in lines:
        line = line.rstrip('\n')
        if len(line) == 0:
            count = 0;
            if not first and not doc_start:
                features.append(tmp_fl)
                labels.append(tmp_ll)
                actions.append(tmp_al)
            initial = 1
            doc_start = 0
            tmp_fl = list()
            tmp_ll = list()
            tmp_al = list()
        elif count == 0:
            first = 0
            count = 1
            if (line[4:14] == '-DOCSTART-'):
                doc_start = 1
                initial = 0
                continue
            if initial:
                line = line[3:len(line) - 1]
                words = line.replace('\'', '').split()
                for word in words:
                    # print(word)
                    if word[-1] == ',':
                        word = word[0:len(word) - 1]
                    posIndex = word.rfind("-")
                    feature = word[:posIndex].strip()  # German
                    label = word[posIndex + 1:].strip()  # JJ
                    tmp_ll.append(label)
                    tmp_fl.append(feature)
            initial = 0
        elif count == 1:
            count = 0
            if doc_start:
                continue
            tmp_al.append(line)

    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)
        actions.append(tmp_al)

    return features, labels, actions

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)


def load_embedding_wlm(emb_file, delimiter, feature_map, full_feature_set, caseless, unk, emb_len,
                       shrink_to_train=False, shrink_to_corpus=False):
    """
    load embedding, indoc words would be listed before outdoc words

    args:
        emb_file: path to embedding file
        delimiter: delimiter of lines
        feature_map: word dictionary
        full_feature_set: all words in the corpus
        caseless: convert into casesless style
        unk: string for unknown token
        emb_len: dimension of embedding vectors
        shrink_to_train: whether to shrink out-of-training set or not
        shrink_to_corpus: whether to shrink out-of-corpus or not
    """
    if caseless:
        feature_set = set([key.lower() for key in feature_map])
        full_feature_set = set([key.lower() for key in full_feature_set])
    else:
        feature_set = set([key for key in feature_map])
        full_feature_set = set([key for key in full_feature_set])

    # ensure <unk> is 0
    word_dict = {v: (k + 1) for (k, v) in enumerate(feature_set - set(['<unk>']))}
    word_dict['<unk>'] = 0

    in_doc_freq_num = len(word_dict)
    rand_embedding_tensor = torch.FloatTensor(in_doc_freq_num, emb_len)
    init_embedding(rand_embedding_tensor)

    indoc_embedding_array = list()
    indoc_word_array = list()
    outdoc_embedding_array = list()
    outdoc_word_array = list()

    for line in open(emb_file, 'r'):
        line = line.split(delimiter)
        if len(line)==2:
            continue
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        if shrink_to_train and line[0] not in feature_set:
            continue

        if line[0] == unk:
            rand_embedding_tensor[0] = torch.FloatTensor(vector)  # unk is 0
        elif line[0] in word_dict:
            rand_embedding_tensor[word_dict[line[0]]] = torch.FloatTensor(vector)
        elif line[0] in full_feature_set:
            indoc_embedding_array.append(vector)
            indoc_word_array.append(line[0])
        elif not shrink_to_corpus:
            outdoc_word_array.append(line[0])
            outdoc_embedding_array.append(vector)

    embedding_tensor_0 = torch.FloatTensor(np.asarray(indoc_embedding_array))

    if not shrink_to_corpus:
        embedding_tensor_1 = torch.FloatTensor(np.asarray(outdoc_embedding_array))
        word_emb_len = embedding_tensor_0.size(1)
        assert (word_emb_len == emb_len)

    if shrink_to_corpus:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0], 0)
    else:
        embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0, embedding_tensor_1], 0)

    for word in indoc_word_array:
        word_dict[word] = len(word_dict)
    in_doc_num = len(word_dict)
    if not shrink_to_corpus:
        for word in outdoc_word_array:
            word_dict[word] = len(word_dict)

    return word_dict, embedding_tensor, in_doc_num

def encode_safe(input_lines, word_dict, unk, singletons, singleton_ratio):
    """
    encode list of strings with unk
    """
    lines = list()
    for line in input_lines:
        tmp_sen = list()
        for word in line:
            if word in singletons and torch.rand(1).numpy()[0] < singleton_ratio:
                tmp_sen.append(unk)
            elif word in word_dict:
                tmp_sen.append(word_dict[word])
            else:
                tmp_sen.append(unk)
        lines.append(tmp_sen)
    return lines

def encode(input_lines, word_dict):
    """
    encode list of strings into word-level representation
    """
    lines = list(map(lambda t: list(map(lambda m: word_dict[m], t)), input_lines))
    return lines

def construct_dataset(input_features, input_label, input_actions, word_dict, label_dict, action_dict, singletons, singleton_ratio, caseless):
    # encode and padding
    if caseless:
        input_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), input_features))

    features = encode_safe(input_features, word_dict, word_dict['<unk>'], singletons, singleton_ratio)
    labels = encode(input_label, label_dict)
    actions = encode(input_actions, action_dict)
    feature_tensor = []
    label_tensor = []
    action_tensor = []
    for feature, label, action in zip(features,labels,actions):
        feature_tensor.append(torch.LongTensor(feature))
        label_tensor.append(torch.LongTensor(label))
        action_tensor.append(torch.LongTensor(action))

    dataset = [TranDataset(feature_tensor, label_tensor, action_tensor)]

    return dataset


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform(input_embedding, -bias, bias)

def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()



def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))  # weight.size=[200,100],weight.size(0)=200
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))  # weight.size=[200,50]
        nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def repack_vb(if_cuda, feature, label, action):
    """packer for viterbi loss

    args:
        feature (Batch_size,Seq_len): input feature
        label (Batch_size,Seq_len): output target
        action (Batch_size,Seq_len): padding mask
    return:
        feature (Batch_size,Seq_len), label (Seq_len, Batch_size), action (Seq_len, Batch_size)
    """

    if if_cuda:
        fea_v = autograd.Variable(feature.transpose(0, 1)).cuda()
        tg_v = autograd.Variable(label.transpose(0, 1)).unsqueeze(2).cuda()
        mask_v = autograd.Variable(action.transpose(0, 1)).cuda()
    else:
        fea_v = autograd.Variable(feature.transpose(0, 1))
        tg_v = autograd.Variable(label.transpose(0, 1)).contiguous().unsqueeze(2)
        mask_v = autograd.Variable(action.transpose(0, 1)).contiguous()
    return fea_v, tg_v, mask_v

def IsActionForbidden(action, buffer_size, stack_size):
    is_shift = False
    is_output = False
    is_reduce = False
    if action.startswith("S"):
        is_shift = True
    elif action.startswith("O"):
        is_output = True
    elif action.startswith("R"):
        is_reduce = True
    if is_shift and buffer_size==1:#shift被禁止了
        return True
    if is_reduce and stack_size==1:
        return True
    if is_output and buffer_size==1:
        return True

    return False

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def to_scalar(var):
    """change the first element of a tensor to scalar
    """
    return var.view(-1).data.tolist()[0]


def save_checkpoint(state, track_list, filename):
    """
    save checkpoint
    """
    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')






