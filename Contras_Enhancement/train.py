import os
import time
import random
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import torchvision
import numpy as np

from model import CEModel
from dataset import custom_dataset, AlignCollate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):

    Model = CEModel(opt.RGB).to(device)
    Model.train()

    AlignCollate_demo = AlignCollate(opt)
    train_data = custom_dataset(opt)  # use RawDataset
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    optimizer = optim.Adam(Model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
    criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean').to(device)

    """ start training """    
    start_iter = 0
    if opt.save_path != '':
        try:
            path = os.path.join(opt.save_path, str(opt.savecode))
            for name in os.listdir(path):
                if '.pth' not in name:
                    continue
                elif start_iter < int(name.split('_')[-1].split('.')[0]):
                    start_iter = int(name.split('_')[-1].split('.')[0])
            print(os.path.join(path, 'contrastMap_'+str(start_iter)+'.pth'))
            Model.load_state_dict(torch.load(os.path.join(path, 'contrastMap_'+str(start_iter)+'.pth'), map_location=device))
            print(f'continue to train, start_iter: {start_iter}')
        except:
            print('Fail to load save model')
            start_iter = 0
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter
    losSum = 0.0
    echo = 0

    path = os.path.join(opt.save_path, str(opt.savecode))
    if os.path.isdir(path) == False:
        os.mkdir(os.path.join(opt.save_path, str(opt.savecode)))

    while(True):
        for ori_tensors, img_tensors, label_tensors in train_loader:
            map, out, res = Model(img_tensors.to(device))
            cost = criterion(res, ori_tensors.to(device))
            Model.zero_grad()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            cost.backward()

            optimizer.step()

            losSum += cost.item()
            iteration += 1

            if iteration % 500 == 0:
                print('cost:', losSum, iteration, echo)
                losSum = 0.0
                #torchvision.utils.save_image(ori_tensors[0], os.path.join(path, str(iteration)+'_original.jpg'))
                #torchvision.utils.save_image(img_tensors[0], os.path.join(path, str(iteration)+'_train.jpg'))
                #torchvision.utils.save_image(label_tensors[0], os.path.join(path, str(iteration)+'_mapGT.jpg'))
                #torchvision.utils.save_image(map[0], os.path.join(path, str(iteration)+'_map.jpg'))
                #torchvision.utils.save_image(out[0], os.path.join(path, str(iteration)+'_out.jpg'))
                #torchvision.utils.save_image(res[0], os.path.join(path, str(iteration)+'_restore.jpg'))
            if iteration % 5000 == 0:
                torch.save(Model.state_dict(), os.path.join(path, 'contrastMap_'+str(iteration)+'.pth'))
        echo += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True, help='path to training dataset')
    parser.add_argument('--save_path', required=True, help='for random seed setting')
    parser.add_argument('--savecode', type=int, default=1000, help='for random seed setting')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--imgH', type=int, default=800, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=640, help='the width of the input image')
    parser.add_argument('--RGB', type=int, default=0, help='RGB')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

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