import numpy as np
import cv2
from PIL import Image
import os
from torch.utils import data
import torchvision.transforms as transforms
import torch

class Resize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        #data = np.asarray(img)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        #img.sub(0.5)
        return img

class Resize_d(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        #img.sub_(0.5).div_(0.5)
        #img.sub(0.5)
        return img

class AlignCollate(object):

    def __init__(self, opt):
        self.imgH = opt.imgH
        self.imgW = opt.imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        oris, images, labels = zip(*batch)

        transform = Resize((self.imgW, self.imgH))
        ori_tensors = [transform(ori) for ori in oris]
        ori_tensors = torch.cat([t.unsqueeze(0) for t in ori_tensors], 0) # 0차원을 추가 - 배치 사이즈
        image_tensors = [transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        label_tensors = [transform(label) for label in labels]
        label_tensors = torch.cat([t.unsqueeze(0) for t in label_tensors], 0)

        return ori_tensors, image_tensors, label_tensors

class AlignCollate_d(object):

    def __init__(self, opt):
        self.imgH = opt.imgH
        self.imgW = opt.imgW

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        transform = Resize_d((self.imgW, self.imgH))
        image_tensors = [transform(image) for image in images]
        # 0차원을 추가 - 배치 사이즈
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

class custom_dataset(data.Dataset):
    def __init__(self, opt):
        super(custom_dataset, self).__init__()
        self.origin_files = []
        self.img_files = []
        self.gt_files = []
        self.opt = opt
        self.img_path = opt.train_path
        for fileName in os.listdir(self.img_path):
            '''
            # 1500 이전
            if 'label' not in fileName and '.png' not in fileName:
                self.origin_files.append(fileName[:fileName[:-5].rfind('_')]+'.png')
                self.img_files.append(fileName)
                self.gt_files.append(fileName[:-4]+'_label'+fileName[-4:])
            '''
            # 1500 이후 - trainImge3
            if 'trans' not in fileName:
                self.origin_files.append(fileName)
                self.img_files.append(fileName[:-4]+'_trans.jpg')
                self.gt_files.append(fileName)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.opt.RGB == 0:
            ori = Image.open(os.path.join(self.img_path, self.origin_files[index])).convert('RGB')  # for color image
            img = Image.open(os.path.join(self.img_path, self.img_files[index])).convert('RGB')  # for color image
            gt = Image.open(os.path.join(self.img_path, self.gt_files[index])).convert('RGB')  # for color image
        else:
            ori = Image.open(os.path.join(self.img_path, self.origin_files[index])).convert('L')  # for color image
            img = Image.open(os.path.join(self.img_path, self.img_files[index])).convert('L')  # for color image
            gt = Image.open(os.path.join(self.img_path, self.gt_files[index])).convert('L')  # for color image
        return ori, img, gt

class custom_dataset_d(data.Dataset):
    def __init__(self, opt):
        super(custom_dataset_d, self).__init__()
        self.img_files = []
        self.opt = opt
        self.img_path = opt.demo_path
        for fileName in os.listdir(self.img_path):
            if os.path.isdir(os.path.join(self.img_path, fileName)): 
                continue
            if '.txt' not in fileName:
                self.img_files.append(fileName)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.opt.RGB == 0:
            img = Image.open(os.path.join(self.img_path, self.img_files[index])).convert('RGB')  # for color image
        else:
            img = Image.open(os.path.join(self.img_path, self.img_files[index])).convert('L')  # for color image
        return img, self.img_files[index]