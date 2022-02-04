"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import os

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir',
                                 type=str,
                                 default='./results/',
                                 help='saves results here.')
        self.parser.add_argument('--num_test_videos',
                                 type=int,
                                 default=1,
                                 help='num of videos to generate')
        self.parser.add_argument('--interpolation',
                                 action='store_true',
                                 help='generate test videos by interpolation')
        self.parser.add_argument(
            '--n_frames_G',
            type=int,
            default=16,
            help='number of input frames forwarded into generator')
        self.parser.add_argument('--fps', type=int, default=10, help='fps')

        self.isTrain = False
        self.isPCA = False
