
from model import Sawtooth
from mydataset import get_train_loader_tm
import torch
from utils import vision_phi
import os
import numpy as np


if __name__ == "__main__":
    device = 'cuda:0'
    data_file = '20ng_2000.pkl'
    batch_size = 200
    k = [100, 64, 32]
    h = [100, 100, 100]
    n_epoch = 1000
    test_fre = 20
    save_path = 'result_sawtooth'

    dataloader, voc_size, voc = get_train_loader_tm(data_file=data_file, batch_size=batch_size)
    model = Sawtooth(k=k, h=h, v=voc_size, emb_dim=100, add_embedding=True, device=device)
    trainable_paras = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            print(k)
            trainable_paras.append(v)

    model = model.to(device)
    optimizer = torch.optim.AdamW(trainable_paras, lr=1e-2)
    kl_weight = [1.0 for _ in range(len(k))]

    for epoch in range(n_epoch):
        loss_t = []
        kl_loss_t = []
        rl_loss_t = []
        for i, data in enumerate(dataloader):
            data = data.to(device).float()
            rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list = model(data)
            rec_loss = rec_list[0]
            kl_part = torch.sum(torch.tensor([weight * kl_loss for weight, kl_loss in zip(kl_weight, kl_loss_list)]))
            loss = 1.0 * kl_part + rec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_t.append(loss.item())
            kl_loss_t.append(kl_part.item())
            rl_loss_t.append(rec_loss.item())
        print(
            f'epoch: {epoch}|{n_epoch} loss: {np.mean(loss_t)}, kl_loss: {np.mean(kl_loss_t)}, rl_loss: {np.mean(rl_loss_t)}')
        if epoch % test_fre == 0:
            vision_phi(phi_list, voc, epoch, outpath=save_path, top_n=20)
