#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
#                    _          _
#                .__(.)<  ??  >(.)__.
#                 \___)        (___/ 
# @Time    : 2021/6/24 上午10:57
# @Author  : wds -->> hellowds2014@gmail.com
# @File    : sawtooth.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~----->>>
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
#from utils import *
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os

class SAWtooth_layer(nn.Module):
    def __init__(self, alpha, rho, d_in=768, d_dim=256, pre_topic_k=100, is_top=False, is_bottom=False, learn_prior=True, device='cuda:0'):
        super(SAWtooth_layer, self).__init__()
        self.real_min = torch.tensor(1e-30)
        self.wei_shape_max = torch.tensor(100.0).float()
        self.wei_shape_min = torch.tensor(0.1).float()
        self.theta_max = torch.tensor(1000.0).float()
        self.is_top = is_top
        self.is_bottom = is_bottom
        self.device = device

        self.alpha = alpha   ### v,d
        self.rho = rho         #### k,d
        self.topic_k = self.rho.shape[0]
        self.h_encoder = nn.Linear(d_in, d_dim)    ### h_hiddent encoder
        self.bn_layer = nn.BatchNorm1d(d_dim)
        if is_top:
            self.shape_scale_encoder = nn.Linear(d_dim, 2*self.topic_k)
            self.k_prior = torch.ones((1, self.topic_k))
            self.l_prior = 0.1 * torch.ones((1, self.topic_k))
            if torch.cuda.is_available():
                self.k_prior = self.k_prior.to(self.device)
                self.l_prior = self.l_prior.to(self.device)
            if learn_prior:
                self.k_prior = nn.Parameter(self.k_prior)
                self.l_prior = nn.Parameter(self.l_prior)
        else:
            self.shape_scale_encoder = nn.Linear(d_dim+pre_topic_k, 2*self.topic_k)     ### for h + phi_{t+1}*theta_{t+1}

    def log_max(self, x):
        return torch.log(torch.max(x, self.real_min.to(self.device)))

    def reparameterize(self, Wei_shape, Wei_scale, sample_num=50):
        eps = torch.FloatTensor(sample_num, Wei_shape.shape[0], Wei_shape.shape[1]).uniform_(0.0, 1.).to(self.device)
        theta = torch.unsqueeze(Wei_scale, axis=0).repeat(sample_num, 1, 1) * torch.pow(-self.log_max(1 - eps), \
                            torch.unsqueeze(1 / Wei_shape, axis=0).repeat(sample_num, 1, 1))
        return torch.mean(torch.clamp(theta, self.real_min.to(self.device), self.theta_max), dim=0, keepdim=False)

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):
        eulergamma = torch.tensor(0.5772, dtype=torch.float32)
        part1 = Gam_shape * self.log_max(Wei_scale) - eulergamma.to(self.device) * Gam_shape * 1 / Wei_shape + self.log_max(
            1 / Wei_shape)
        part2 = - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))
        part3 = eulergamma.to(self.device) + 1 + Gam_shape * self.log_max(Gam_scale) - torch.lgamma(Gam_shape)
        KL = part1 + part2 + part3
        return - torch.sum(KL) / Wei_scale.shape[1]

    def compute_loss(self, x, re_x):
        likelihood = torch.sum(x * self.log_max(re_x) - re_x - torch.lgamma(x + 1.))
        return -likelihood / x.shape[1]

    def get_phi(self):
        w = torch.mm(self.alpha, self.rho.t())   ### v,k
        return torch.softmax(w+self.real_min.to(self.device), dim=0)
    def decoder(self, x):
        phi = self.get_phi()
        return torch.mm(phi, x), phi

    def forward(self, x, prior=None, bow=None):
        rec_loss = None
        kl_loss = None
        hidden = F.relu(self.bn_layer(self.h_encoder(x)))
        if not self.is_top:
            hidden = torch.cat((hidden, prior.t()), 1)     ### batch, (d_dim+pre_topic_k)

        k_temp, l_temp = torch.chunk(self.shape_scale_encoder(hidden), 2, dim=1)
        k = torch.clamp(F.softplus(k_temp), self.wei_shape_min, self.wei_shape_max)
        l_temp = F.softplus(l_temp) / torch.exp(torch.lgamma(1 + 1 / k))
        l = torch.clamp(l_temp, self.real_min, 9999.0)

        theta = self.reparameterize(k, l)       ### n,k
        phi_theta, phi = self.decoder(theta.t())   #### v,n
        if self.is_top:
            kl_loss = self.KL_GamWei(self.k_prior.t(),
                                     self.l_prior.t(),k.t(), l.t())
        else:
            kl_loss = self.KL_GamWei(prior, torch.tensor(1.0, dtype=torch.float32).to(self.device),
                           k.t(), l.t())
        if self.is_bottom:
            rec_loss = self.compute_loss(bow.t(), phi_theta)
        return rec_loss, kl_loss, phi.cpu().detach().numpy(), phi_theta, theta


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx, device='cuda:0'):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf).to(device)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf).to(device))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class Sawtooth(nn.Module):
    ## doneeeeeee
    def __init__(self, k=[256,128,64], h=[256,128,64], v=2000, emb_dim=100, add_embedding=True, device='cuda:0'):
        super(Sawtooth, self).__init__()
        self.voc_size = v
        self.hidden_size = h
        self.bn_layer = nn.ModuleList([nn.BatchNorm1d(hidden_size) for hidden_size in self.hidden_size])
        self.layer_num = len(k)
        self.dropout = nn.Dropout(p=0.1)
        self.add_embedding = add_embedding
        self.device = device
        if add_embedding:
            self.embed_layer = Conv1D(self.hidden_size[0], 1, self.voc_size, device=self.device)

        h_encoder = [Conv1D(self.hidden_size[0], 1, self.hidden_size[0], device=self.device)]
        for i in range(self.layer_num - 1):
            h_encoder.append(Conv1D(self.hidden_size[i + 1], 1, self.hidden_size[i], device=self.device))
        self.h_encoder = nn.ModuleList(h_encoder)
        self.alpha = self.init_alpha([v] + k, emb_dim)
        self.layer_num = len(k)
        sawtooth_layer = []
        for idx, each_k in enumerate(k):
            if idx == 0:
                if len(k) == 1:
                    sawtooth_layer.append(SAWtooth_layer(self.alpha[0], self.alpha[1], d_in=h[0],
                                                         d_dim=h[0], pre_topic_k=k[idx],
                                                         is_top=True, is_bottom=True, device=self.device))
                else:
                    sawtooth_layer.append(SAWtooth_layer(self.alpha[0], self.alpha[1], d_in=h[0],
                                                         d_dim=h[0], pre_topic_k=k[idx],
                                                         is_top=False, is_bottom=True, device=self.device))
            elif idx == len(k) - 1:
                sawtooth_layer.append(SAWtooth_layer(self.alpha[-2], self.alpha[-1], d_in=h[idx],
                                                     d_dim=h[idx], pre_topic_k=None,
                                                     is_top=True, is_bottom=False, device=self.device))
            else:
                sawtooth_layer.append(SAWtooth_layer(self.alpha[idx], self.alpha[idx + 1], d_in=h[idx],
                                                     d_dim=h[idx], pre_topic_k=k[idx],
                                                     is_top=False, is_bottom=False, device=self.device))
        self.sawtooth_layer = nn.ModuleList(sawtooth_layer)

    def init_alpha(self, k, emb_dim):
        w_para = []
        for idx, each_topic in enumerate(k):
            w = torch.ones(each_topic, emb_dim).to(self.device)
            nn.init.normal_(w, std=0.02)
            # w_para.append(torch.Tensor.requires_grad_(w))
            w_para.append(Parameter(w))
        return w_para

    def res_block(self, x, layer_num):
        ### res block for hidden path
        x1 = self.h_encoder[layer_num](x)
        try:
            out = x + x1
        except:
            out = x1
        return self.dropout(F.relu(self.bn_layer[layer_num](out)))

    def forward(self, x):
        hidden_list = [0] * self.layer_num
        theta_list = [0] * self.layer_num
        phi_list = [0] * self.layer_num
        phi_theta_list = [0] * (self.layer_num+1)
        kl_loss_list = [0] * self.layer_num

        rec_list = [0] * self.layer_num
        #### upward path
        if self.add_embedding:
            x_embed = self.embed_layer(1.0*x)
        else:
            x_embed = 1.0*x
        for t in range(self.layer_num):
            if t == 0:
                hidden_list[t] = self.res_block(x_embed, t)
            else:
                hidden_list[t] = self.res_block(hidden_list[t-1], t)
        #### downward path
        for t in range(self.layer_num-1, -1, -1):
            rec_list[t], kl_loss_list[t], phi_list[t], phi_theta_list[t], \
            theta_list[t] = self.sawtooth_layer[t](hidden_list[t], phi_theta_list[t+1], x)

        return rec_list, kl_loss_list, phi_list, phi_theta_list, theta_list

