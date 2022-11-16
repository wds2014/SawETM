#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2021/9/3 下午8:59
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : mydataset.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import scipy.io as sio
import torch


class read_doc(Dataset):
    def __init__(self, path='dataset/20News/20ng.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.train_data = data['data_2000']
        self.voc = data['voc2000']
        self.N, self.voc_size = self.train_data.shape

    def __getitem__(self, index):
        try:
            return np.squeeze(self.train_data[index].toarray())
        except:
            return np.squeeze(self.train_data[index])

    def __len__(self):
        return self.N


def get_train_loader_tm(data_file='20ng.pkl', batch_size=200, shuffle=True, num_workers=2):
    dataset = read_doc(path=data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=False), dataset.voc_size, dataset.voc


def gen_ppl_doc(x, ratio=0.8):
    """
    inputs:
        x: N x V, np array,
        ratio: float or double,
    returns:
        x_1: N x V, np array, the first half docs whose length equals to ratio * doc length,
        x_2: N x V, np array, the second half docs whose length equals to (1 - ratio) * doc length,
    """
    import random
    x_1, x_2 = np.zeros_like(x), np.zeros_like(x)
    # indices_x, indices_y = np.nonzero(x)
    for doc_idx, doc in enumerate(x):
        indices_y = np.nonzero(doc)[0]
        l = []
        for i in range(len(indices_y)):
            value = doc[indices_y[i]]
            for _ in range(int(value)):
                l.append(indices_y[i])
        random.seed(2020)
        random.shuffle(l)
        l_1 = l[:int(len(l) * ratio)]
        l_2 = l[int(len(l) * ratio):]
        for l1_value in l_1:
            x_1[doc_idx][l1_value] += 1
        for l2_value in l_2:
            x_2[doc_idx][l2_value] += 1
    return x_1, x_2


class CustomDataset_txt_ppl(Dataset):
    def __init__(self, data_file, ratio=0.8):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        data_all = data['bow'].toarray()
        # data_all = data['data_2000'].toarray()
        self.train_data, self.test_data = gen_ppl_doc(data_all.astype("int32"), ratio=ratio)
        self.voc = data['voc']
        # self.voc = data['voc2000']
        self.N, self.vocab_size = self.train_data.shape
        for i in range(self.N):
            if np.sum(self.train_data[i]) >2 and np.sum(self.test_data[i]) > 2:
                self.idx = i
                break

    def __getitem__(self, index):
        train_data = self.train_data[index]
        test_data = self.test_data[index]
        # if np.sum(train_data) > 2 and np.sum(test_data) >2:
        return np.squeeze(train_data), np.squeeze(test_data)
        # else:
        #     return np.squeeze(self.train_data[self.idx]), np.squeeze(self.test_data[self.idx])

    def __len__(self):
        return self.N

def get_loader_txt_ppl(topic_data_file, batch_size=200, shuffle=True, num_workers=2, ratio=0.8):
    dataset = CustomDataset_txt_ppl(topic_data_file, ratio=ratio)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=False), dataset.vocab_size, dataset.voc


class CustomDataset_cluster(Dataset):
    def __init__(self, data_file, dataname='20ng', mode='train'):

        with open(data_file,'rb') as f:
            data = pickle.load(f)
        self.data = data['data_2000']
        self.voc = data['voc2000']
        self.label = np.squeeze(data['label'])

        if mode == 'train':
            self.data = self.data[data['train_id']]
            self.label = self.label[data['train_id']]
        else:
            self.data = self.data[data['test_id']]
            self.label = self.label[data['test_id']]

        self.N, self.vocab_size = self.data.shape

    def __getitem__(self, index):
        return np.squeeze(self.data[index].toarray()), np.squeeze(np.array(self.label[index]))

    def __len__(self):
        return self.N

def cluster_loader_cluster(topic_data_file, dataname='20ng', mode='train', batch_size=200, shuffle=True, num_workers=2):
    dataset = CustomDataset_cluster(topic_data_file, dataname=dataname, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=True), dataset.vocab_size, dataset.voc