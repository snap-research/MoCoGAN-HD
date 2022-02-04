"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.BigGAN import BigGAN_D


def pair_cos_sim(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ModelD_img(nn.Module):
    def __init__(self, opt, D_B1=0.0, D_B2=0.999, adam_eps=1e-6):
        super(ModelD_img, self).__init__()

        self.moco_m = opt.moco_m

        self.modelD = BigGAN_D.Discriminator(resolution=opt.video_frame_size,
                                             proj_dim=opt.l_len)
        self.modelD_ema = BigGAN_D.Discriminator(
            resolution=opt.video_frame_size, proj_dim=opt.l_len)

        self.register_buffer("q_real", torch.randn(opt.q_len, opt.l_len))
        self.q_real = F.normalize(self.q_real, dim=0)
        self.register_buffer("q_fake", torch.randn(opt.q_len, opt.l_len))
        self.q_fake = F.normalize(self.q_fake, dim=0)

        self.q_len = opt.q_len // opt.world_batch_size * opt.world_batch_size

        self.register_buffer("q_ptr", torch.zeros(1, dtype=torch.long))

        self.T = opt.moco_t
        self.batchSize = opt.batchSize

        for param, param_ema in zip(self.modelD.parameters(),
                                    self.modelD_ema.parameters()):
            param_ema.data.copy_(param.data)
            param_ema.requires_grad = False

        self.optim = optim.Adam(params=self.modelD.parameters(),
                                lr=opt.lr,
                                betas=(D_B1, D_B2),
                                weight_decay=0,
                                eps=adam_eps)

    @torch.no_grad()
    def _momentum_update_dis(self):
        """
        Momentum update of the discriminator
        """
        for p, p_ema in zip(self.modelD.parameters(),
                            self.modelD_ema.parameters()):
            p_ema.data = p_ema.data * self.moco_m + p.data * (1. - self.moco_m)

    @torch.no_grad()
    def update_memory_bank(self, logits_real, logits_fake):
        logits_real = concat_all_gather(logits_real)
        logits_fake = concat_all_gather(logits_fake)

        batch_size_t = logits_real.shape[0]

        ptr = int(self.q_ptr)
        self.q_real[ptr:ptr + batch_size_t, :] = logits_real
        self.q_fake[ptr:ptr + batch_size_t, :] = logits_fake
        ptr = (ptr + batch_size_t) % self.q_len
        self.q_ptr[0] = ptr

    def get_cntr_loss_cross_domain(self, logits_real, logits_real2,
                                   logits_fake, logits_fake2):
        T = self.T
        cos_sim_real = pair_cos_sim(
            torch.cat((logits_real, logits_real2), dim=0),
            torch.cat(
                (logits_real, logits_real2, self.q_real[:self.q_len].detach()),
                dim=0))
        m = torch.ones_like(cos_sim_real) / T
        m.fill_diagonal_(0.)

        cos_sim_reg_real = F.softmax(cos_sim_real * m)

        cos_sim_fake = pair_cos_sim(
            torch.cat((logits_fake, logits_fake2), dim=0),
            torch.cat(
                (logits_fake, logits_fake2, self.q_fake[:self.q_len].detach()),
                dim=0))
        cos_sim_reg_fake = F.softmax(cos_sim_fake * m)

        cntr_loss_real = 0.

        for i in range(self.batchSize):
            cntr_loss_real += -torch.log(
                cos_sim_reg_real[i][i + self.batchSize])
            cntr_loss_real += -torch.log(
                cos_sim_reg_real[i + self.batchSize][i])

        cntr_loss_real = cntr_loss_real / (2. * self.batchSize)

        cntr_loss_fake = 0.

        for i in range(self.batchSize):
            cntr_loss_fake += -torch.log(
                cos_sim_reg_fake[i][i + self.batchSize])
            cntr_loss_fake += -torch.log(
                cos_sim_reg_fake[i + self.batchSize][i])

        cntr_loss_fake = cntr_loss_fake / (2. * self.batchSize)

        cntr_loss = cntr_loss_real + cntr_loss_fake
        return cntr_loss

    def forward(self, x, ema=False, proj_only=False):
        if ema:
            return self.modelD_ema(x, proj_only)
        return self.modelD(x, proj_only)
