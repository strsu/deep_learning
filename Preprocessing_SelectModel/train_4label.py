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

from model import selectNet_4label
from dataset import custom_dataset_one


train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    Model = selectNet_4label().to(train_device)
    Model.train()

    train_data = custom_dataset_one(opt.trainPath)  # use RawDataset
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1,
        shuffle=True,
        num_workers=int(0),
        pin_memory=True)

    optimizer = optim.Adam(Model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
    criterion = nn.BCELoss().to(train_device)

    for epoch in range(100):
        iter = 0
        avgLoss = 0.0
        print('Epoch :', epoch)
        for img, label in train_loader:
            pred = Model(img.to(train_device))

            ans_tensor = torch.zeros(3, dtype=torch.float, requires_grad = False).to(train_device)
            ans = label[0].split('_')
            if len(ans) == 1:
                if 'blur' in ans:
                    ans_tensor[0] = 1
                elif 'contrast' in ans:
                    ans_tensor[1] = 1
                elif 'noise' in ans:
                    ans_tensor[2] = 1
            elif len(ans) == 2:
                if 'blur' in ans and 'contrast' in ans:
                    ans_tensor[0] = 1
                    ans_tensor[1] = 1
                elif 'blur' in ans and 'noise' in ans:
                    ans_tensor[0] = 1
                    ans_tensor[2] = 1
                elif 'noise' in ans and 'contrast' in ans:
                    ans_tensor[1] = 1
                    ans_tensor[2] = 1 
            elif len(ans) == 3:
                ans_tensor[0] = 1
                ans_tensor[1] = 1
                ans_tensor[2] = 1
            ans_tensor = ans_tensor.unsqueeze(0)

            cost = criterion(pred, ans_tensor)
            avgLoss += cost.item()

            Model.zero_grad()
            cost.backward()
            optimizer.step()
            

            iter += 1

            if iter % 100 == 0:
                print(epoch, '-', iter)
                print('avgLoss', avgLoss / 100)
                avgLoss = 0.0
        
        torch.save(Model.state_dict(), os.path.join('model/select4_one/'+str(opt.savecode), 'Select4_one_'+str(epoch)+'.pth'))

            # p / b / c / n / b c / b n / c n / b c n



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainPath', default='', help='train image path')
    parser.add_argument('--savecode', type=int, default=1000, help='for distint train result')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
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
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    
    cudnn.benchmark = True
    cudnn.deterministic = True

    train(opt)