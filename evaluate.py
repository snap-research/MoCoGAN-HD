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
from torchvision.io import write_video

from options.test_options import TestOptions
from models.models import create_model


def test():

    opt = TestOptions().parse(save=False)

    ### initialize models
    modelG = create_model(opt)

    z = torch.FloatTensor(1, opt.latent_dimension)
    z = z.cuda()

    def create_and_save(z, modelG, opt, use_noise, prefix):
        x_fake, _, _ = modelG(styles=[z],
                              n_frame=opt.n_frames_G,
                              use_noise=use_noise,
                              interpolation=opt.interpolation)
        x_fake = x_fake.view(1, -1, 3, opt.style_gan_size,
                             opt.style_gan_size).data
        x_fake = x_fake.clamp(-1, 1)

        video = x_fake[0].cpu()
        video = ((video + 1.) / 2. * 255).type(torch.uint8).permute(0, 2, 3, 1)
        write_video(os.path.join(opt.results_dir, prefix + '.mp4'),
                    video,
                    fps=opt.fps)

    os.makedirs(opt.results_dir, exist_ok=True)

    with torch.no_grad():
        for j in range(opt.num_test_videos):
            z.data.normal_()
            prefix = opt.name + '_' + str(
                opt.load_pretrain_epoch) + '_' + str(j) + '_noise'
            create_and_save(z, modelG, opt, True, prefix)

            prefix = opt.name + '_' + str(
                opt.load_pretrain_epoch) + '_' + str(j)
            create_and_save(z, modelG, opt, False, prefix)

        print(opt.name + ' Finished!')


if __name__ == "__main__":
    test()
