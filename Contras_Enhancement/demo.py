import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
from model import CEModel
from dataset import custom_dataset_d, AlignCollate_d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def demo(opt):

    Model = CEModel(opt.RGB).to(device)
    
    path = os.path.join(opt.save_path, str(opt.savecode))
    print('load model from', os.path.join(path, opt.savename+'.pth'))
    
    Model.load_state_dict(torch.load(os.path.join(path, opt.savename+'.pth'), map_location=device))

    Model.eval()

    AlignCollate_demo = AlignCollate_d(opt)
    demo_data = custom_dataset_d(opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True) # default : pin_memory = True
    
    path_ = os.path.join(opt.demo_path, opt.savename)
    if os.path.isdir(path_) == False:
        os.mkdir(path_)

    start_time = time.time()
    with torch.no_grad():
        for img_tensors, labels in demo_loader:
            map, out, res = Model(img_tensors.to(device))
            for j in range(res.size(0)):
                #print('pred_'+labels[j], res.size())
                #torchvision.utils.save_image(img_tensors[j], os.path.join(path_, 'resize_'+labels[j]))
                #torchvision.utils.save_image(map[j], os.path.join(path_, 'pres_map_'+labels[j]))
                img_ = res[j].view(1, 3, opt.imgH, opt.imgW)
                torchvision.utils.save_image(img_, os.path.join(path_, labels[j][:-4]+'_pred.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', required=True, help='path to training dataset')
    #parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--save_path', required=True, help='for random seed setting')
    parser.add_argument('--savecode', type=int, default=1000, help='for random seed setting')
    parser.add_argument('--savename', required=True, help='for random seed setting')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--model', default=2, help="path to model to continue training")
    parser.add_argument('--imgH', type=int, default=800, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=640, help='the width of the input image')
    parser.add_argument('--RGB', type=int, default=0, help='RGB')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)

    opt = parser.parse_args()

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    
    opt.savename = 'bestAccuracy'
    demo(opt)