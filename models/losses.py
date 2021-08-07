"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn


def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def compute_gradient_penalty_T(real_B, fake_B, modelD, opt):
    alpha = torch.rand(list(real_B.size())[0], 1, 1, 1, 1)
    alpha = alpha.expand(real_B.size()).cuda(real_B.get_device())

    interpolates = alpha * real_B.data + (1 - alpha) * fake_B.data
    interpolates = torch.tensor(interpolates, requires_grad=True)

    pred_interpolates = modelD(interpolates)

    gradient_penalty = 0
    if isinstance(pred_interpolates, list):
        for cur_pred in pred_interpolates:
            gradients = torch.autograd.grad(outputs=cur_pred[-1],
                                            inputs=interpolates,
                                            grad_outputs=torch.ones(
                                                cur_pred[-1].size()).cuda(
                                                    real_B.get_device()),
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)[0]

            gradient_penalty += ((gradients.norm(2, dim=1) - 1)**2).mean()
    else:
        sys.exit('output is not list!')

    gradient_penalty = (gradient_penalty / opt.num_D) * 10
    return gradient_penalty


class GANLoss(nn.Module):
    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = torch.tensor(real_tensor,
                                                   requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = torch.tensor(fake_tensor,
                                                   requires_grad=False)
            target_tensor = self.fake_label_var

        if input.is_cuda:
            target_tensor = target_tensor.cuda()
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class Relativistic_Average_LSGAN(GANLoss):
    '''
        Relativistic average LSGAN
    '''
    def __call__(self, input_1, input_2, target_is_real):
        if isinstance(input_1[0], list):
            loss = 0
            for input_i, _input_i in zip(input_1, input_2):
                pred = input_i[-1]
                _pred = _input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred - torch.mean(_pred), target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input_1[-1], target_is_real)
            return self.loss(input_1[-1] - torch.mean(input_2[-1]),
                             target_tensor)
