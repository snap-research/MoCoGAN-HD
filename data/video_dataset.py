"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import os.path
import random

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def preprocess(image):
    # [0, 1] => [-1, 1]
    img = image * 2.0 - 1.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    return img


class VideoDataset(data.Dataset):
    def load_video_frames(self, dataroot):
        data_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            frames = sorted(meta[2], key=lambda item: item.split('.')[0])
            frames = [
                os.path.join(root, item) for item in frames
                if is_image_file(item)
            ]
            if len(frames) > self.opt.n_frames_G * self.opt.time_step:
                data_all.append(frames)
        self.video_num = len(data_all)
        return data_all

    def __init__(self, opt):
        self.opt = opt
        # dataroot에서 (n_frames_G*time_step 넘어가는) 모든 비디오 load 
        self.data_all = self.load_video_frames(opt.dataroot)

    def __getitem__(self, index):
        batch_data = self.getTensor(index)
        return_list = {'real_img': batch_data}

        return return_list

    def getTensor(self, index):
        n_frames = self.opt.n_frames_G

        video = self.data_all[index]
        video_len = len(video)

        # 필요한 frame 수
        n_frames_interval = n_frames * self.opt.time_step
        # 시작 인덱스 : 0~ (총 비디오 길이-1 - 필요한 프레임수) 중 랜덤하게 
        start_idx = random.randint(0, video_len - 1 - n_frames_interval)
        img = Image.open(video[0])
        h, w = img.height, img.width

        # 정사각형으로 crop
        if h > w:
            half = (h - w) // 2
            cropsize = (0, half, w, half + w)  # left, upper, right, lower
        elif w > h:
            half = (w - h) // 2
            cropsize = (half, 0, half + h, h)

        images = []
        for i in range(start_idx, start_idx + n_frames_interval,
                       self.opt.time_step):
            path = video[i]
            img = Image.open(path)

            if h != w:
                img = img.crop(cropsize)

            # 안티엘리어싱 : 높은 해상도 -> 낮은 해상도 깨지는거 해결
            img = img.resize(
                (self.opt.video_frame_size, self.opt.video_frame_size),
                Image.ANTIALIAS)
            img = np.asarray(img, dtype=np.float32)
            img /= 255.
            # (h, w, c) -> (c, h, w) -> (frame, c, h, w)
            img_tensor = preprocess(img).unsqueeze(0)
            images.append(img_tensor)

        video_clip = torch.cat(images)
        return video_clip

    def __len__(self):
        return self.video_num

    def name(self):
        return 'VideoDataset'
