import os
import time
import random
import argparse

import torch
import torch.nn as nn
from torch._C import device
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import torchvision
import numpy as np
import cv2

from model import selectNet
from dataset import custom_dataset


train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    Model = selectNet(opt.RGB).to(train_device)
    Model.train()

    train_data = custom_dataset(opt.trainPath)  # use RawDataset
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1,
        shuffle=True,
        num_workers=int(0),
        pin_memory=True)

    optimizer = optim.Adam(Model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(train_device)

    for epoch in range(opt.epoch):
        iter = 0
        avgLoss = 0.0
        for img, label in train_loader:
            pred = Model(img.to(train_device), 1)

            cost = criterion(pred, label.to(train_device))
            avgLoss += cost.item()

            Model.zero_grad()
            cost.backward()
            optimizer.step()

            iter += 1

            if iter % 100 == 0:
                print(epoch, '-', iter)
                print('avgLoss', avgLoss / 100)
                avgLoss = 0.0
        
        torch.save(Model.state_dict(), os.path.join('model/'+str(opt.savecode), 'Select_'+str(epoch)+'.pth'))

            # p / b / c / n / b c / b n / c n / b c n



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainPath', default='', help='train image path')
    parser.add_argument('--savecode', type=int, default=1000, help='for distint train result')
    parser.add_argument('--manualSeed', type=int, default=1122, help='for random seed setting')
    parser.add_argument('--RGB', type=int, default=1, help='RGB')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--epoch', type=int, help='number of epoch', default=200)

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=1.0 for Adam')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')

    opt = parser.parse_args()

    """ Seed and GPU setting """
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    train(opt)