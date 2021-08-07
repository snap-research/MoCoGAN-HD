"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


class RNNModule(nn.Module):
    def __init__(self,
                 pca_comp_path,
                 pca_stdev_path,
                 z_dim=512,
                 h_dim=384,
                 n_pca=384,
                 w_residual=0.2):
        super(RNNModule, self).__init__()
        pca_comp = np.load(pca_comp_path)
        pca_stdev = np.load(pca_stdev_path)
        self.pca_comp = torch.tensor(pca_comp[:n_pca], dtype=torch.float32)
        self.pca_stdev = torch.tensor(pca_stdev[:n_pca], dtype=torch.float32)

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_pca = n_pca
        self.w_residual = w_residual

        self.enc_cell = nn.LSTMCell(z_dim, h_dim)
        self.cell = nn.LSTMCell(z_dim, h_dim)
        self.w = nn.Parameter(torch.FloatTensor(h_dim, n_pca))
        self.b = nn.Parameter(torch.FloatTensor(n_pca))
        self.fc1 = nn.Linear(h_dim * 2, z_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(z_dim, z_dim)

        self.init_weights()

    def init_optim(self, lr, beta1, beta2):
        self.optim = optim.Adam(params=self.parameters(),
                                lr=lr,
                                betas=(beta1, beta2),
                                weight_decay=0,
                                eps=1e-8)

    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.LSTMCell)):
                for name, param in module.named_parameters():
                    if ('weight_ih' in name) or ('weight_hh' in name):
                        mul = param.shape[0] // 4
                        for idx in range(4):
                            init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                    elif 'bias' in name:
                        param.data.fill_(0)
            if (isinstance(module, nn.Linear)):
                init.orthogonal_(module.weight)

        nn.init.normal_(self.w, std=0.02)
        self.b.data.fill_(0.0)

    def forward(self, z, n_frame):
        pca_comp = self.pca_comp.cuda(z.get_device())
        pca_stdev = self.pca_stdev.cuda(z.get_device())
        pca_stdev = pca_stdev.view(-1, 1).repeat(1, pca_comp.shape[-1])
        pca_mul = pca_comp * pca_stdev

        out = [z]
        h_, c_ = self.enc_cell(z)
        h = [h_]
        c = [c_]
        e = []
        for i in range(n_frame - 1):
            e_ = self.get_initial_state_z(z.shape[0])
            h_, c_ = self.cell(e_, (h[-1], c[-1]))
            mul = torch.matmul(h_, self.w) + self.b
            mul = torch.tanh(mul)
            e.append(e_)
            h.append(h_)
            c.append(c_)
            out_ = out[-1] + self.w_residual * torch.matmul(mul, pca_mul)
            out.append(out_)

        out = [item.unsqueeze(1) for item in out]

        out = torch.cat(out, dim=1).view(-1, self.z_dim)

        e = [item.unsqueeze(1) for item in e]
        e = torch.cat(e, dim=1).view(-1, self.z_dim)

        hh = h[1:]
        hh = [item.unsqueeze(1) for item in hh]
        hh = torch.cat(hh, dim=1).view(-1, self.h_dim)

        cc = c[1:]
        cc = [item.unsqueeze(1) for item in cc]
        cc = torch.cat(cc, dim=1).view(-1, self.h_dim)

        hc = torch.cat((hh, cc), dim=1)
        e_rec = self.fc2(self.relu(self.fc1(hc)))

        return out, e, e_rec

    def get_initial_state_z(self, batchSize):
        return torch.cuda.FloatTensor(batchSize, self.z_dim).normal_()
