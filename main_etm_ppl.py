#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2022/11/16 下午4:54
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : main_etm_ppl.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>

from model import Sawtooth, ETM
from mydataset import get_loader_txt_ppl
import torch
from utils import vision_phi
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np


if __name__ == "__main__":
    device = 'cuda:0'
    dataset_dir = '20ng_2000.pkl'
    # dataset_dir = '/home/wds/2021/DEEP_WeTe_ppl/dataset/20ng.pkl'
    batch_size = 200
    k = 100
    h = 100
    data_loader, vocab_size, voc = get_loader_txt_ppl(dataset_dir, batch_size=batch_size, ratio=0.8)

    model = ETM(num_topics=k, vocab_size=len(voc), t_hidden_size=h, rho_size=100, emsize=100, enc_drop=0.0, device=device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_epoch = 1000
    best_ppl = 99999
    train_num = 0
    test_fre = 20
    save_path = 'result_etm_ppl'

    for epoch in range(n_epoch):
        model.train()
        loss_t = []
        likelihood_t = []
        likelihood_test = []
        ppl_test = []
        num_data = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=num_data)
        for i, (train_data, test_data) in pbar:
            train_data = train_data.to(device)
            test_data = test_data.t().to(device)

            rec, kl_loss, phi, phi_theta, theta = model(train_data)

            loss = 1.0 * kl_loss + rec
            optimizer.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.grad = p.grad.where(~torch.isnan(p.grad), torch.tensor(0., device=p.grad.device))
                p.grad = p.grad.where(~torch.isinf(p.grad), torch.tensor(0., device=p.grad.device))
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            loss_t.append(loss.item())
            likelihood_t.append(rec.item())

            train_num += 1
            pbar.set_description(
                f'epoch: {epoch}|{n_epoch}, loss:{np.mean(loss_t):.4f}, likelihood:{np.mean(likelihood_t):.4f}')

        if epoch % test_fre == 0:
            ### calculate ppl as https://github.com/adjidieng/ETM
            model.eval()
            likelihood_test = []
            ppl_test = []
            pbar_test = tqdm(enumerate(data_loader), total=num_data)
            with torch.no_grad():
                # get_phi()
                for i, (train_data, test_data) in pbar_test:
                    train_data = train_data.to(device)
                    test_data = test_data.to(device)

                    rec, kl_loss, phi, phi_theta, theta = model(train_data)

                    sums_2 = test_data.sum(1)
                    likelihood_test.append(rec.item())
                    recon_ppl = -(torch.log(phi_theta + 1e-10) * test_data).sum(1)
                    loss_ppl = recon_ppl / sums_2
                    loss_ppl = np.nanmean(loss_ppl.cpu().detach().numpy())
                    ppl_test.append(loss_ppl)

                    pbar_test.set_description(
                        f'test summary {epoch} epoch: likelihood:{np.mean(likelihood_test):.4f}, ppl:{math.exp(np.mean(ppl_test)):.4f}')
                vision_phi([phi], voc, epoch, outpath=save_path, top_n=20)
