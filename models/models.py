"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .rnn import RNNModule
from models.stylegan2 import model


def load_checkpoints(path, gpu):
    if gpu is None:
        ckpt = torch.load(path)
    else:
        loc = 'cuda:{}'.format(gpu)
        ckpt = torch.load(path, map_location=loc)
    return ckpt


def model_to_gpu(model, opt):
    if opt.isTrain:
        if opt.gpu is not None:
            model.cuda(opt.gpu)
            model = DDP(model,
                        device_ids=[opt.gpu],
                        find_unused_parameters=True)
        else:
            model.cuda()
            model = DDP(model, find_unused_parameters=True)
    else:
        model.cuda()
        model = nn.DataParallel(model)

    return model


def create_model(opt):
    ckpt = load_checkpoints(opt.img_g_weights, opt.gpu)

    modelG = model.Generator(size=opt.style_gan_size,
                             style_dim=opt.latent_dimension,
                             n_mlp=opt.n_mlp)
    modelG.load_state_dict(ckpt['g_ema'], strict=False)
    modelG.eval()

    for p in modelG.parameters():
        p.requires_grad = False

    if opt.isPCA:
        modelS = modelG.style
        modelS.eval()
        if opt.gpu is not None:
            modelS.cuda(opt.gpu)
        return modelS

    pca_com_path = os.path.join(opt.save_pca_path, 'pca_comp.npy')
    pca_stdev_path = os.path.join(opt.save_pca_path, 'pca_stdev.npy')
    modelR = RNNModule(pca_com_path,
                       pca_stdev_path,
                       z_dim=opt.latent_dimension,
                       h_dim=opt.h_dim,
                       n_pca=opt.n_pca,
                       w_residual=opt.w_residual)

    if opt.isTrain:
        from .D_3d import ModelD_3d

        modelR.init_optim(opt.lr, opt.beta1, opt.beta2)
        modelG.modelR = modelR

        modelD_3d = ModelD_3d(opt)
        if opt.cross_domain:
            from .D_img import ModelD_img
        else:
            from .D import ModelD_img
        modelD_img = ModelD_img(opt)

        modelG = model_to_gpu(modelG, opt)
        modelD_3d = model_to_gpu(modelD_3d, opt)
        modelD_img = model_to_gpu(modelD_img, opt)

        if opt.load_pretrain_path != 'None' and opt.load_pretrain_epoch > -1:
            opt.checkpoints_dir = opt.load_pretrain_path
            m_name = '/modelR_epoch_%d.pth' % (opt.load_pretrain_epoch)
            ckpt = load_checkpoints(opt.load_pretrain_path + m_name, opt.gpu)
            modelG.module.modelR.load_state_dict(ckpt)

            m_name = '/modelD_img_epoch_%d.pth' % (opt.load_pretrain_epoch)
            ckpt = load_checkpoints(opt.load_pretrain_path + m_name, opt.gpu)
            modelD_img.load_state_dict(ckpt)

            m_name = '/modelD_3d_epoch_%d.pth' % (opt.load_pretrain_epoch)
            ckpt = load_checkpoints(opt.load_pretrain_path + m_name, opt.gpu)
            modelD_3d.load_state_dict(ckpt)
        return [modelG, modelD_img, modelD_3d]
    else:
        modelR.eval()
        for p in modelR.parameters():
            p.requires_grad = False
        modelG.modelR = modelR
        modelG = model_to_gpu(modelG, opt)

        if opt.load_pretrain_path != 'None' and opt.load_pretrain_epoch > -1:
            m_name = '/modelR_epoch_%d.pth' % (opt.load_pretrain_epoch)
            ckpt = load_checkpoints(opt.load_pretrain_path + m_name, opt.gpu)
            modelG.module.modelR.load_state_dict(ckpt)
        return modelG
