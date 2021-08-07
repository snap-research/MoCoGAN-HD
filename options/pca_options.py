"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
from .base_options import BaseOptions


class PCAOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument(
            '--pca_iterations',
            type=int,
            default=250,
            help='number of iterations to get latent code')

        self.parser.add_argument(
            '--fake_img_size',
            type=int,
            default=512,
            help='spatial size for the output of generator')

        self.isTrain = False
        self.isPCA = True
