#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2021/9/2 下午4:03
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : utils.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
from collections import Counter
from sklearn.cluster import k_means
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np
import os


def purity(labels, clustered):
    # find the set of cluster ids
    ### see http://www.cse.chalmers.se/~richajo/dit862/L13/Text%20clustering.html
    cluster_ids = set(clustered)
    N = len(clustered)
    majority_sum = 0
    for cl in cluster_ids:
        # for this cluster, we compute the frequencies of the different human labels we encounter
        # the result will be something like { 'camera':1, 'books':5, 'software':3 } etc.
        labels_cl = Counter(l for l, c in zip(labels, clustered) if c == cl)

        # we select the *highest* score and add it to the total sum
        majority_sum += max(labels_cl.values())

    # the purity score is the sum of majority counts divided by the total number of items
    return majority_sum / N


def normalization(data):
    _range = np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True)
    return (data - np.min(data, axis=1, keepdims=True)) / _range


def standardization(data):
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    return (data - mu) / sigma

def vision_phi(Phi, voc, train_num, outpath='phi_output', top_n=30):
    def get_top_n(phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += voc[index]
            top_n_words += ' '
        return top_n_words

    outpath = outpath + '/' + str(train_num)
    # Phi = Phi[::-1]
    if voc is not None:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        phi = 1
        for num, phi_layer in enumerate(Phi):
            phi = np.dot(phi, phi_layer)
            phi_k = phi.shape[1]
            path = os.path.join(outpath, 'phi' + str(num) + '.txt')
            f = open(path, 'w')
            for each in range(phi_k):
                top_n_words = get_top_n(phi[:, each], top_n)
                f.write(top_n_words)
                f.write('\n')
            f.close()
    else:
        print('voc need !!')