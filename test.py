#!/usr/bin/python3
# coding=utf-8

import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import src.dataset as dataset
from src.net6_10_3 import F3Net


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg = Dataset.Config(datapath=path, snapshot='./out/try6_10_32_sifnet1-32', mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r, a = self.net(image)
                out = out2u

                plt.subplot(221)
                plt.imshow(np.uint8(image[0].permute(1, 2, 0).cpu().numpy() * self.cfg.std + self.cfg.mean))
                plt.subplot(222)
                plt.imshow(mask[0].cpu().numpy())
                plt.subplot(223)
                plt.imshow(out[0, 0].cpu().numpy())
                plt.subplot(224)
                plt.imshow(torch.sigmoid(out[0, 0]).cpu().numpy())
                plt.show()
                input()

    def save(self):
        with torch.no_grad():
            c = 0
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                import time
                a = time.time()
                out1u, out2u, out2r, out3= self.net(image, shape)
                b = time.time()
                c = b - a + c
                out = out1u
                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                head = '../eval/maps/try6_10_32_sifnet1-CAMO/' + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))
            print(c)
            print(2016 / c)


if __name__ == '__main__':
    torch.cuda.set_device(2)
    # HCOD
    for path in ['/media/zdc/dz/data/salience/CAMO/']:
        t = Test(dataset, F3Net, path)
        t.save()
        # t.show()
