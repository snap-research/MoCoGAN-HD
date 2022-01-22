from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import sklearn.datasets
import time
from PIL import ImageFile
from PIL import Image
from video_folder import VideoFolder
from torch.utils.data import DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True
parser = argparse.ArgumentParser()
parser.add_argument('--datatrain', default = './sky_train', 
	help='path to trainset')
parser.add_argument('--datatest', default = './sky_test', 
	help='path to testset')
parser.add_argument('--outf', default='./model/', 
	help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, 
	help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=128, 
	help='the height and width of the input image to network')
parser.add_argument('--nframes', type=int, default=32, 
	help='number of frames in each video clip')
opt = parser.parse_args()
localtime = time.asctime( time.localtime(time.time()) )
print('\n start new program! ')
print(localtime)
print(opt) 
trainset = VideoFolder(root=opt.datatrain, 
					   nframes = opt.nframes,
                       transform=transforms.Compose([
                           transforms.Resize( (opt.imageSize, opt.imageSize) ),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       	])
                       )
testset = VideoFolder( root=opt.datatest, 
					   nframes = opt.nframes,
                       transform=transforms.Compose([
                           transforms.Resize((opt.imageSize, opt.imageSize)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])
                     )
print('trainset size ' + str(len(trainset)))
print('testset size ' + str(len(testset)))

train_loader = DataLoader(trainset,
                      batch_size=opt.batchSize,
                      num_workers=int(opt.workers), 
                      shuffle=True,
                      drop_last = True, 
                      pin_memory=True
                      )


valid_loader = DataLoader(testset,
                      batch_size=64,
                      num_workers=1,
                      shuffle=True,
                      drop_last = True, 
                      pin_memory=False
                      )

print('trainloader size: ' + str(len(train_loader)))
print('validloader size: ' + str(len(valid_loader)))







