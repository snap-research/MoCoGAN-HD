"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import random
import numpy as np

import torch
import torch.nn.functional as F

from models import losses


def warp_with_flip_batch(x):
    out = []
    for ii in range(x.shape[0]):
        out.append(warp_with_flip(x[ii]))
    return torch.cat(out, dim=0)


def warp_with_flip(x):
    num = random.randint(0, 1)
    if num == 1:
        return torch.flip(x, [-1]).unsqueeze(0)
    else:
        return x.unsqueeze(0)


def warp_with_color_batch(x):
    out = []
    for ii in range(x.shape[0]):
        out.append(warp_with_color(x[ii]))
    return torch.cat(out, dim=0)


def warp_with_color(x):
    c_shift = torch.rand(1) - 0.5
    c_shift = c_shift.cuda(x.get_device())
    m = torch.zeros_like(x)
    m = m.cuda(x.get_device())
    num = random.randint(0, 3)
    if num == 0:
        m.data += c_shift
    elif num == 1:
        m[0].data += c_shift
    elif num == 2:
        m[1].data += c_shift
    else:
        m[2].data += c_shift

    out = x + m
    return out.unsqueeze(0)


def warp_with_cutout_batch_real(x):
    out = []
    for ii in range(x.shape[0]):
        out.append(warp_with_cutout_real(x[ii]))
    return torch.cat(out, dim=0)


def warp_with_cutout_real(x, max_ratio=0.25):
    c, h, w = x.size()
    m = np.ones((c, h, w), np.float32)

    ratio = random.uniform(max_ratio / 2, max_ratio)
    num = random.randint(0, 3)
    if num == 0:
        h_start = random.uniform(0, max_ratio - ratio)
        w_start = random.uniform(0, 1 - max_ratio)
    elif num == 1:
        h_start = random.uniform(1 - max_ratio, 1 - ratio)
        w_start = random.uniform(0, 1 - max_ratio)
    elif num == 2:
        w_start = random.uniform(0, max_ratio - ratio)
        h_start = random.uniform(0, 1 - max_ratio)
    else:
        w_start = random.uniform(1 - max_ratio, 1 - ratio)
        h_start = random.uniform(0, 1 - max_ratio)

    h_s = round(h_start * (h - 1) - 0.5)
    w_s = round(w_start * (w - 1) - 0.5)
    length = round(h * ratio - 0.5)

    m[:, h_s:h_s + length, w_s:w_s + length] = 0.
    m = torch.from_numpy(m).cuda(x.get_device())
    out = x * m
    return out.unsqueeze(0)


def warp_with_affine(x, angle=180, trans=0.1, scale=0.05):
    angle = np.pi * angle / 180.

    pa = torch.FloatTensor(4)
    th = torch.FloatTensor(2, 3)

    pa[0].uniform_(-angle, angle)
    pa[1].uniform_(-trans, trans)
    pa[2].uniform_(-trans, trans)
    pa[3].uniform_(1. - scale, 1. + scale)

    th[0][0] = pa[3] * torch.cos(pa[0])
    th[0][1] = pa[3] * torch.sin(-pa[0])
    th[0][2] = pa[1]
    th[1][0] = pa[3] * torch.sin(pa[0])
    th[1][1] = pa[3] * torch.cos(pa[0])
    th[1][2] = pa[2]

    x = x.unsqueeze(0)
    th = th.unsqueeze(0)
    grid = F.affine_grid(th, x.size()).cuda(x.get_device())
    out = F.grid_sample(x, grid, padding_mode="reflection")
    return out


def warp(x):
    out = warp_with_cutout_batch_real(x)
    out_list = []
    for ii in range(out.shape[0]):
        num = random.randint(0, 2)
        if num == 0:
            out_list.append(warp_with_flip(out[ii]))
        elif num == 1:
            out_list.append(warp_with_color(out[ii]))
        else:
            out_list.append(warp_with_affine(out[ii]))
    return torch.cat(out_list, dim=0)


def flip_video(x):
    num = random.randint(0, 1)
    if num == 0:
        return torch.flip(x, [2])
    else:
        return x


def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.required_grad = on_or_off


def D_step(opt, modelG, modelD_img, modelD_3d, x, z):
    z.data.normal_()

    x_fake, _, _ = modelG([z], opt.n_frames_G, use_noise=True)
    x_fake = x_fake.view(opt.batchSize, opt.n_frames_G, opt.nc,
                         opt.style_gan_size, opt.style_gan_size)
    kernel_size = int(opt.style_gan_size / opt.video_frame_size)
    x_fake = F.avg_pool3d(x_fake, (1, kernel_size, kernel_size))

    x_in = x
    x_fake_in = x_fake

    D_fake_3d = modelD_3d(flip_video(
        torch.transpose(x_fake_in, 1, 2).detach()))
    D_real_3d = modelD_3d(flip_video(torch.transpose(x_in, 1, 2)))

    criterionGAN = losses.Relativistic_Average_LSGAN()
    D_loss_real_3d = criterionGAN(D_real_3d, D_fake_3d, True)
    D_loss_fake_3d = criterionGAN(D_fake_3d, D_real_3d, False)

    D_loss_3d = (D_loss_real_3d + D_loss_fake_3d) * 0.5

    loss_GP_3d = losses.compute_gradient_penalty_T(
        torch.transpose(x_in, 1, 2), torch.transpose(x_fake_in, 1, 2),
        modelD_3d, opt)
    D_loss_3d += loss_GP_3d

    modelD_3d.module.optim.zero_grad()
    D_loss_3d.backward(retain_graph=True)
    modelD_3d.module.optim.step()

    real_id = random.randint(1, opt.n_frames_G - 1)
    fake_id = random.randint(1, opt.n_frames_G - 1)

    aug_real2 = warp(torch.tensor(x_fake[:, 0]))
    aug_real = warp(x_fake[:, 0])
    aug_fake2 = warp(torch.tensor(x_fake[:, fake_id]))
    aug_fake = warp(x_fake[:, fake_id])

    D_real, logits_real = modelD_img(aug_real.detach())
    D_real2, logits_real2 = modelD_img(aug_real2.detach())
    D_fake, logits_fake = modelD_img(aug_fake.detach())
    D_fake2, logits_fake2 = modelD_img(aug_fake2.detach())

    # contrastive loss
    cntr_loss = modelD_img.module.get_cntr_loss_cross_domain(
        logits_real, logits_real2, logits_fake, logits_fake2)

    # l1 regularization  
    D_loss_real, D_loss_fake = losses.loss_hinge_dis(D_fake, D_real)
    D_loss = (D_loss_real + D_loss_fake) / 1. + 2. * cntr_loss

    modelD_img.module.optim.zero_grad()
    D_loss.backward()
    modelD_img.module.optim.step()

    with torch.no_grad():
        modelD_img.module._momentum_update_dis()

    return D_loss_real.item(), D_loss_fake.item(), D_loss_real_3d.item(
    ), D_loss_fake_3d.item(), cntr_loss.item()


def G_step(opt, modelG, modelD_img, modelD_3d, x, z):
    z.data.normal_()

    x_fake, rand_in, rand_rec = modelG([z], opt.n_frames_G, use_noise=True)

    x_fake = x_fake.view(opt.batchSize, opt.n_frames_G, 3, opt.style_gan_size,
                         opt.style_gan_size)
    kernel_size = int(opt.style_gan_size / opt.video_frame_size)
    x_fake = F.avg_pool3d(x_fake, (1, kernel_size, kernel_size))

    l_mutual = -torch.mean(F.cosine_similarity(rand_rec, rand_in.detach()))

    fake_id = random.randint(1, opt.n_frames_G - 1)
    warped_fake = warp(x_fake[:, fake_id])
    warped_real = warp(x_fake[:, 0])

    # momentum decoder (MoCo)
    with torch.no_grad():
        logits_real = modelD_img(warped_real, ema=True, proj_only=True)
        logits_fake = modelD_img(warped_fake, ema=True, proj_only=True)

        modelD_img.module.update_memory_bank(logits_real, logits_fake)

    D_fake, l_fake = modelD_img(warped_fake)
    D_real, l_real = modelD_img(warped_real)

    # Feature matching 
    cos_sim = F.cosine_similarity(l_fake, l_real)
    l_match = -cos_sim.mean()

    G_loss_2d = losses.loss_hinge_gen(D_fake)
    
    x_in = x
    x_fake_in = x_fake

    D_real_3d = modelD_3d(flip_video(torch.transpose(x_in, 1, 2)))
    D_fake_3d = modelD_3d(flip_video(torch.transpose(x_fake_in, 1, 2)))

    criterionGAN = losses.Relativistic_Average_LSGAN()
    G_loss_3d = (criterionGAN(D_fake_3d, D_real_3d, True) +
                 criterionGAN(D_real_3d, D_fake_3d, False)) * 0.5

    G_loss = G_loss_3d + G_loss_2d + opt.w_match * l_match + l_mutual

    modelG.module.modelR.optim.zero_grad()
    G_loss.backward()
    modelG.module.modelR.optim.step()

    return G_loss_2d.item(), G_loss_3d.item(), l_match.item(), l_mutual.item()


def GD_step(opt, modelG, modelD_img, modelD_3d, data, x, z):
    x.data.copy_(data['real_img'])

    for i in range(opt.G_step):
        G_loss, G_loss_3d, l_match, l_mutual = G_step(opt, modelG, modelD_img,
                                                      modelD_3d, x, z)

    D_loss_real, D_loss_fake, D_loss_real_3d, D_loss_fake_3d, cntr_loss_D = D_step(
        opt, modelG, modelD_img, modelD_3d, x, z)

    loss_names = [
        'D_real', 'D_fake', 'D_real_3d', 'D_fake_3d', 'cntr_D', 'G', 'G_3d',
        'l_match', 'l_mutual'
    ]

    loss_all = [
        D_loss_real, D_loss_fake, D_loss_real_3d, D_loss_fake_3d, cntr_loss_D,
        G_loss, G_loss_3d, l_match, l_mutual
    ]
    return loss_all, loss_names
