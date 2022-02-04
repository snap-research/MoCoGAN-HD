"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import argparse
import os
from time import gmtime, strftime


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name',
                                 type=str,
                                 default='video-gen',
                                 help='name of the experiment.')
        self.parser.add_argument('--gpu',
                                 type=int,
                                 default=None,
                                 help='gpu id to use')
        self.parser.add_argument('--batchSize',
                                 type=int,
                                 default=8,
                                 help='input batch size')
        self.parser.add_argument('--workers',
                                 default=16,
                                 type=int,
                                 help='# data loading workers')
        self.parser.add_argument('--save_pca_path',
                                 type=str,
                                 default='pca_stats/ffhq_256',
                                 help='folder to save pca statistics')

        # parameters for StyleGAN
        self.parser.add_argument('--latent_dimension',
                                 type=int,
                                 default=512,
                                 help='dimension of latent code')
        self.parser.add_argument(
            '--style_gan_size',
            type=int,
            default=256,
            help='spatial size for the output of generator')
        self.parser.add_argument('--n_mlp',
                                 type=int,
                                 default=8,
                                 help='number of mlp in stylegan')
        self.parser.add_argument('--img_g_weights',
                                 type=str,
                                 default='pretrained_models/ffhq_256.pt',
                                 help='weights for pretrained image generator')
        # parameters for RNN
        self.parser.add_argument('--load_pretrain_path',
                                 type=str,
                                 default='pretrained_models',
                                 help='path to the pretrained model')
        self.parser.add_argument('--load_pretrain_epoch',
                                 type=int,
                                 default=-1,
                                 help='epoch for the pretrained model')
        self.parser.add_argument(
            '--w_residual',
            type=float,
            default=0.2,
            help='the weight for calculating residual in RNN')
        self.parser.add_argument('--h_dim',
                                 type=int,
                                 default=384,
                                 help='hidden dimension for RNN')
        self.parser.add_argument('--n_pca',
                                 type=int,
                                 default=384,
                                 help='number of pca components')

        # added 
        self.parser.add_argument(
            '--style_gan3',
            default=True,
            help=
            'Use StyleGAN3 Generator instead of StyleGAN2 Generator'
        )
        
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        self.opt.isPCA = self.isPCA

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to disk
        if save:
            self.opt.checkpoints_dir = os.path.join(
                self.opt.checkpoints_dir,
                self.opt.name + strftime("_%y_%m_%d_%H_%M_%S", gmtime()))

            os.makedirs(self.opt.checkpoints_dir, exist_ok=True)

            file_name = os.path.join(self.opt.checkpoints_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt
