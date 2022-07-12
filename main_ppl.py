#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/7/12 下午3:25
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : main_v1.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
from model import Sawtooth
from mydataset import get_loader_txt_ppl
import torch
from utils import vision_phi
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def _ppl(x, x_res):
    ## return ppl
    ## x: K1 * N
    X1 = x_res  ## V * N
    X1sum = torch.sum(X1, 0)
    X2 = torch.div(X1, X1sum)
    ppl = torch.sum(x * torch.log(X2)) / torch.sum(x)
    return torch.exp(-ppl)

def log_max(x):
    real_min = torch.tensor(1e-30).to(x.device)
    return torch.log(torch.max(x, real_min))

if __name__ == "__main__":
    device = 'cuda:0'
    dataset_dir = '20ng_2000.pkl'
    batch_size = 200
    k = [256,128,62]
    h = [100, 100, 100]
    layer_num = len(k)
    data_loader, vocab_size, voc = get_loader_txt_ppl(dataset_dir, batch_size=batch_size, ratio=0.8)


    model = Sawtooth(k=k, h=h, v=vocab_size, emb_dim=100, add_embedding=True, device=device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    kl_weight = [1e-0 for _ in range(layer_num)]
    n_epoch = 1000
    best_ppl = 99999
    train_num = 0
    test_fre = 20
    save_path = 'result_sawtooth_ppl'

    for epoch in range(n_epoch):
        model.train()
        loss_t = []
        kl_loss_t = [[] for _ in range(layer_num)]
        theta_entropy = [[] for _ in range(layer_num)]
        likelihood_t = []
        likelihood_test = []
        ppl_test = []
        num_data = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=num_data)
        for i, (train_data, test_data) in pbar:
            train_data = train_data.to(device)
            test_data = test_data.t().to(device)

            rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = model(train_data)

            kl_part = [weight * kl for weight, kl in zip(kl_weight, kl_loss_list)]
            rec_loss = rec_list[0]
            loss = 1.0 * torch.stack(kl_part).sum() + rec_loss
            optimizer.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.grad = p.grad.where(~torch.isnan(p.grad), torch.tensor(0., device=p.grad.device))
                p.grad = p.grad.where(~torch.isinf(p.grad), torch.tensor(0., device=p.grad.device))
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            loss_t.append(loss.item())
            likelihood_t.append(rec_loss.item())

            train_num += 1
            pbar.set_description(
                f'epoch: {epoch}|{n_epoch}, loss:{np.mean(loss_t):.4f}, likelihood:{np.mean(likelihood_t):.4f}')

        if epoch % test_fre == 0:

            model.eval()
            likelihood_test = []
            ppl_test = []
            pbar_test = tqdm(enumerate(data_loader), total=num_data)
            with torch.no_grad():
                # get_phi()
                for i, (train_data, test_data) in pbar_test:
                    train_data = train_data.to(device)
                    test_data = test_data.t().to(device)

                    rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = model(train_data)
                    likelihood_test.append(
                        (-torch.sum(test_data * log_max(phi_theta_list[0]) - phi_theta_list[0] - torch.lgamma(test_data + 1.0)) /
                         test_data.shape[1]).item())
                    ppl_test.append(_ppl(test_data, phi_theta_list[0]).item())
                    pbar_test.set_description(
                        f'test summary {epoch} epoch: likelihood:{np.mean(likelihood_test):.4f}, ppl:{np.mean(ppl_test):.4f}')
                vision_phi(phi_list, voc, epoch, outpath=save_path, top_n=20)
                # if best_ppl > np.mean(ppl_test):
                #     best_ppl = np.mean(ppl_test)
                # print(
                #     f'\ntest likelihood : \t {np.mean(likelihood_test):.4f}, test_ppl:{np.mean(ppl_test):.4f}, best_ppl:{best_ppl:.4f}\n')