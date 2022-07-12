#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2021/9/3 下午9:50
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : cluster_main.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>


from model import Sawtooth
from mydataset import get_loader_txt_ppl, cluster_loader_cluster
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from cluster_clc import cluster_clc
from utils import *


if __name__ == "__main__":
    device = 'cuda:0'
    data_file = '20ng_2000.pkl'
    clc_num=20
    batch_size = 200
    k = [100, 64, 32]
    h = [100, 100, 100]
    layer_num = len(k)
    n_epoch = 1000
    test_fre = 20
    save_path = 'result_sawtooth_clc'
    train_loader, voc_size, voc = cluster_loader_cluster(data_file, '20ng', mode='train',batch_size=200)
    test_loader, voc_size, voc = cluster_loader_cluster(data_file, '20ng', mode='test', batch_size=200)


    model = Sawtooth(k=k, h=h, v=voc_size, emb_dim=100, add_embedding=True, device=device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    kl_weight = [1e-1 for _ in range(layer_num)]
    n_epoch = 1000
    train_num = 0
    test_fre = 50
    save_path = 'result_sawtooth_clc'

    best_cluster_purity = 0.
    best_cluster_nmi = 0.
    best_cluster_ami = 0.
    best_f1_score = 0.
    best_micro_prec = 0.
    best_micro_recall = 0.

    for epoch in range(n_epoch):
        model.train()
        loss_t = []
        kl_loss_t = [[] for _ in range(layer_num)]
        theta_entropy = [[] for _ in range(layer_num)]
        likelihood_t = []
        likelihood_test = []
        ppl_test = []
        num_data = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total=num_data)
        for i, (train_data, train_label) in pbar:
            train_data = train_data.to(device)

            rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = model(train_data.float())

            kl_part = [weight * kl for weight, kl in zip(kl_weight, kl_loss_list)]
            rec_loss = rec_list[0]
            loss = torch.stack(kl_part).sum() + rec_loss

            optimizer.zero_grad()
            loss.backward()

            for p in model.parameters():
                p.grad = p.grad.where(~torch.isnan(p.grad), torch.tensor(0., device=p.grad.device))
                p.grad = p.grad.where(~torch.isinf(p.grad), torch.tensor(0., device=p.grad.device))
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            loss_t.append(loss.item())
            likelihood_t.append(rec_loss.item())
            theta = theta_list[0]
            train_num += 1
            pbar.set_description(
                f'epoch:{epoch}|{n_epoch}, loss:{np.mean(loss_t):.4f}, likelihood:{np.mean(likelihood_t):.4f}')

        if epoch % test_fre == 0:
            model.eval()
            train_theta = None
            train_label = None
            test_theta = None
            test_label = None

            with torch.no_grad():
                for data, label in train_loader:
                    data = data.to(device)
                    rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = model(data.float())
                    theta = theta_list[0]
                    if train_theta is None:
                        train_theta = theta.cpu().detach().numpy()
                        train_label = label.numpy()
                    else:
                        train_theta = np.concatenate((train_theta, theta.cpu().detach().numpy()))
                        train_label = np.concatenate((train_label, label.numpy()))

                for data, label in test_loader:
                    data = data.to(device)
                    rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = model(data.float())
                    theta = theta_list[0]
                    if test_theta is None:
                        test_theta = theta.cpu().detach().numpy()
                        test_label = label.numpy()
                    else:
                        test_theta = np.concatenate((test_theta, theta.cpu().detach().numpy()))
                        test_label = np.concatenate((test_label, label.numpy()))

            purity_value, nmi_value, ami_value, micro_prec, micro_recall, micro_f1_score = cluster_clc(train_theta, train_label, test_theta, test_label, 20)
            vision_phi(phi_list, voc, epoch, outpath=save_path, top_n=20)
            if purity_value > best_cluster_purity:
                best_cluster_purity = purity_value
                best_cluster_nmi = nmi_value
                best_cluster_ami = ami_value
            if micro_f1_score > best_f1_score:
                best_f1_score = micro_f1_score
                best_micro_recall = micro_recall
                best_micro_prec = micro_prec

            print('*' * 88)
            print('     test   summary')
            print('*' * 88)
            print(f'epoch: {epoch}|{n_epoch}, purity: {purity_value:.4f} / {best_cluster_purity:.4f}, '
                  f'nmi: {nmi_value:.4f} / {best_cluster_nmi:.4f}, '
                  f'ami: {ami_value:.4f} / {best_cluster_ami:.4f}')
            print(f'F1: {micro_f1_score:.4f} / {best_f1_score:.4f}, prec: {micro_prec:.4f} / {best_micro_prec:.4f}, recall: {micro_recall:.4f} / {best_micro_recall:.4f}\n')
