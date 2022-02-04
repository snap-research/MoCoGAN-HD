"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # displays and saving
        self.parser.add_argument(
            '--display_freq',
            type=int,
            default=100,
            help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq',
                                 type=int,
                                 default=5,
                                 help='frequency of priting training results')
        self.parser.add_argument('--save_latest_freq',
                                 type=int,
                                 default=1000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq',
                                 type=int,
                                 default=1,
                                 help='frequency of saving checkpoints')

        # optimizer
        self.parser.add_argument('--beta1',
                                 type=float,
                                 default=0.5,
                                 help='momentum term of adam')
        self.parser.add_argument('--beta2',
                                 type=float,
                                 default=0.999,
                                 help='momentum term of adam')
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=0.0001,
                                 help='initial learning rate for adam')

        # contrastive loss
        self.parser.add_argument(
            '--q_len',
            type=int,
            default=4096,
            help='size of queue to save logits used in constrastive loss')
        self.parser.add_argument('--l_len',
                                 type=int,
                                 default=256,
                                 help='size of logits in contrastive loss')
        self.parser.add_argument(
            '--moco_m',
            default=0.999,
            type=float,
            help='moco momentum of updating discriminator')
        self.parser.add_argument('--moco_t',
                                 default=0.07,
                                 type=float,
                                 help='softmax temperature')
        self.parser.add_argument('--w_match',
                                 default=1.0,
                                 type=float,
                                 help='the weight for feat match loss')

        # spatial size
        self.parser.add_argument(
            '--video_frame_size',
            type=int,
            default=128,
            help='spatial size of video frames for training')

        # training setting
        self.parser.add_argument('--cross_domain',
                                 action='store_true',
                                 help='in-domain or cross-domain training')
        self.parser.add_argument('--G_step',
                                 type=int,
                                 default=5,
                                 help='number of training iterations for G')
        self.parser.add_argument('--total_epoch',
                                 type=int,
                                 default=5,
                                 help='training epochs')

        self.parser.add_argument('--checkpoints_dir',
                                 type=str,
                                 default='./checkpoints',
                                 help='models are saved here')
        self.parser.add_argument(
            '--n_frames_G',
            type=int,
            default=16,
            help='number of input frames forwarded into generator')

        # 3D discriminators
        self.parser.add_argument('--num_D',
                                 type=int,
                                 default=2,
                                 help='number of discriminators to use')
        self.parser.add_argument('--norm_D_3d',
                                 type=str,
                                 default='instance',
                                 help='instance norm or batch nom for D_3d')
        self.parser.add_argument('--nc',
                                 type=int,
                                 default=3,
                                 help='# of input channels for D_3d')

        # dataloader setting
        self.parser.add_argument('--dataroot',
                                 type=str,
                                 default='/path/to/dataset/')
        self.parser.add_argument(
            '--time_step',
            type=int,
            default=2,
            help='the spacing between neighboring frames.')

        #  distributed training
        self.parser.add_argument(
            '--world_size',
            default=-1,
            type=int,
            help='number of nodes for distributed training')
        self.parser.add_argument('--rank',
                                 default=-1,
                                 type=int,
                                 help='node rank for distributed training')
        self.parser.add_argument(
            '--dist_url',
            default='tcp://localhost:10001',
            type=str,
            help='url used to set up distributed training')
        self.parser.add_argument('--dist_backend',
                                 default='nccl',
                                 type=str,
                                 help='distributed backend')
        self.parser.add_argument(
            '--multiprocessing_distributed',
            action='store_true',
            help=
            'Use multi-processing distributed training to launch N processes per node, which has N GPUs.'
        )

        
        self.isTrain = True
        self.isPCA = False
