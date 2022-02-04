"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import torch.utils.data

from .video_dataset import VideoDataset


class VideoDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        dataset = VideoDataset(opt)
        self.dataset = dataset

        if opt.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.dataset)
        else:
            self.train_sampler = None

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=(self.train_sampler is None),
            num_workers=opt.workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)


def CreateDataLoader(opt):
    data_loader = VideoDatasetDataLoader(opt)
    return data_loader
