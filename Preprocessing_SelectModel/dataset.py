import torch
from torch.utils import data
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import random
import cv2
import os

class custom_dataset(data.Dataset):
    def __init__(self, inputImgPath):
        super(custom_dataset, self).__init__()
        self.img_files = []
        self.img_paths = []
        self.labels = []
        self.to_tensor = transforms.ToTensor()

        funcList = ['blur', 'contrast', 'noise']

        tmpItem = os.listdir(os.path.join(inputImgPath, 'plain'))
        random.shuffle(tmpItem)
        random.shuffle(tmpItem)

        thr = 900

        for idx, f in enumerate(tmpItem):
            if idx >= thr:
                break
            if 'jpg' in f:
                self.img_files.append(f)
                self.img_paths.append(os.path.join(inputImgPath, 'plain'))
                self.labels.append('none')
        print("Load 'plain' img - ", len(self.img_files))


        # 1개 선택
        thr = 900
        
        for i in range(len(funcList)):
            tmpItem = os.listdir(os.path.join(inputImgPath, funcList[i]))
            random.shuffle(tmpItem)
            random.shuffle(tmpItem)

            for idx, f in enumerate(tmpItem):
                if idx >= thr:
                    break
                if 'jpg' in f:
                    self.img_files.append(f)
                    self.img_paths.append(os.path.join(inputImgPath, funcList[i]))
                    self.labels.append(funcList[i])
        print("Load 'one' img - ", len(self.img_files))
        
        thr = 450

        # 2개 선택
        for i in range(len(funcList)):
            for j in range(i+1, len(funcList)):
                tmpItem_1 = os.listdir(os.path.join(inputImgPath, funcList[i]+'_'+funcList[j]))
                tmpItem_2 = os.listdir(os.path.join(inputImgPath, funcList[j]+'_'+funcList[i]))
                random.shuffle(tmpItem_1)
                random.shuffle(tmpItem_2)

                for idx, f in enumerate(tmpItem_1):
                    if idx >= thr:
                        break
                    if 'jpg' in f:
                        self.img_files.append(f)
                        self.img_paths.append(os.path.join(inputImgPath, funcList[i]+'_'+funcList[j]))
                        self.labels.append(funcList[i]+'_'+funcList[j])

                for idx, f in enumerate(tmpItem_2):
                    if idx >= thr:
                        break
                    if 'jpg' in f:
                        self.img_files.append(f)
                        self.img_paths.append(os.path.join(inputImgPath, funcList[j]+'_'+funcList[i]))
                        self.labels.append(funcList[i]+'_'+funcList[j])

        print("Load 'two' img - ", len(self.img_files))

        thr = 150

        # 3개 선택
        for i in range(len(funcList)):
            for j in range(i+1, len(funcList)):
                for k in range(j+1, len(funcList)):

                    tmpItem_1 = os.listdir(os.path.join(inputImgPath, funcList[i]+'_'+funcList[j]+'_'+funcList[k]))
                    tmpItem_2 = os.listdir(os.path.join(inputImgPath, funcList[i]+'_'+funcList[k]+'_'+funcList[j]))
                    tmpItem_3 = os.listdir(os.path.join(inputImgPath, funcList[k]+'_'+funcList[j]+'_'+funcList[i]))
                    tmpItem_4 = os.listdir(os.path.join(inputImgPath, funcList[k]+'_'+funcList[i]+'_'+funcList[j]))
                    tmpItem_5 = os.listdir(os.path.join(inputImgPath, funcList[j]+'_'+funcList[i]+'_'+funcList[k]))
                    tmpItem_6 = os.listdir(os.path.join(inputImgPath, funcList[j]+'_'+funcList[k]+'_'+funcList[i]))

                    random.shuffle(tmpItem_1)
                    random.shuffle(tmpItem_2)
                    random.shuffle(tmpItem_3)
                    random.shuffle(tmpItem_4)
                    random.shuffle(tmpItem_5)
                    random.shuffle(tmpItem_6)

                    for idx, f in enumerate(tmpItem_1):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[i]+'_'+funcList[j]+'_'+funcList[k]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])
                    
                    for idx, f in enumerate(tmpItem_2):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[i]+'_'+funcList[k]+'_'+funcList[j]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])

                    
                    for idx, f in enumerate(tmpItem_3):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[k]+'_'+funcList[j]+'_'+funcList[i]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])

                    for idx, f in enumerate(tmpItem_4):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[k]+'_'+funcList[i]+'_'+funcList[j]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])

                    for idx, f in enumerate(tmpItem_5):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[j]+'_'+funcList[i]+'_'+funcList[k]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])
                    
                    for idx, f in enumerate(tmpItem_6):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[j]+'_'+funcList[k]+'_'+funcList[i]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_paths[index], self.img_files[index])).convert('RGB')  # for color image
        img = img.resize((int(img.width / 2), int(img.height / 2)))
        img = self.to_tensor(img)

        if '_' not in self.labels[index]:
            val = 0
        else:
            val = 1

        if 'blur' in self.labels[index]:
            val += 1
        
        if 'contrast' in self.labels[index]:
            val += 2

        if 'noise' in self.labels[index]:
            val += 3
        
        # p b c n bc bn cn bcn
        # 0 1 2 3 4  5  6  7

        return img, torch.ones([1], dtype=torch.long, requires_grad=False)*val


class custom_dataset_one(data.Dataset):
    def __init__(self, inputImgFolderPath):
        super(custom_dataset_one, self).__init__()
        self.img_files = []
        self.img_paths = []
        self.labels = []
        self.to_tensor = transforms.ToTensor()

        funcList = ['blur', 'contrast', 'noise']

        folderList = os.listdir(inputImgFolderPath)

        for f in folderList:
            # 어떤 폴더인지 판단
            fcnCnt = 0
            for fcn in funcList:
                if fcn in f:
                    fcnCnt += 1
            if 'self' in f:
                continue
            # 적용 함수가 1개인 폴더
            elif fcnCnt == 1:
                thr = 900
                tmpItem = [fn for fn in os.listdir(os.path.join(inputImgFolderPath, f)) if '.jpg' in fn]
                random.shuffle(tmpItem)
                random.shuffle(tmpItem)
                for idx, imgf in enumerate(tmpItem):
                    if idx >= thr:
                        break
                    if '.txt' in imgf:
                        continue
                    else:
                        self.img_files.append(imgf)
                        self.img_paths.append(os.path.join(inputImgFolderPath, f))
                        self.labels.append(f)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_paths[index], self.img_files[index])).convert('RGB')  # for color image
        img = img.resize((int(img.width / 2), int(img.height / 2)))
        img = self.to_tensor(img)
        #img = img.unsqueeze(0)
        return img, self.labels[index]


class custom_dataset_eval(data.Dataset):
    def __init__(self, inputImgPath):
        super(custom_dataset_eval, self).__init__()
        self.img_files = []
        self.img_paths = []
        self.labels = []
        self.to_tensor = transforms.ToTensor()

        funcList = ['blur', 'contrast', 'noise']

        tmpItem = os.listdir(os.path.join(inputImgPath, 'plain'))
        random.shuffle(tmpItem)
        random.shuffle(tmpItem)

        thr = 180

        for idx, f in enumerate(tmpItem):
            if idx >= thr:
                break
            if 'jpg' in f:
                self.img_files.append(f)
                self.img_paths.append(os.path.join(inputImgPath, 'plain'))
                self.labels.append('none')


        # 1개 선택
        thr = 180
        
        for i in range(len(funcList)):
            tmpItem = os.listdir(os.path.join(inputImgPath, funcList[i]))
            random.shuffle(tmpItem)
            random.shuffle(tmpItem)

            for idx, f in enumerate(tmpItem):
                if idx >= thr:
                    break
                if 'jpg' in f:
                    self.img_files.append(f)
                    self.img_paths.append(os.path.join(inputImgPath, funcList[i]))
                    self.labels.append(funcList[i])
        
        thr = 90

        # 2개 선택
        for i in range(len(funcList)):
            for j in range(i+1, len(funcList)):
                tmpItem_1 = os.listdir(os.path.join(inputImgPath, funcList[i]+'_'+funcList[j]))
                tmpItem_2 = os.listdir(os.path.join(inputImgPath, funcList[j]+'_'+funcList[i]))
                random.shuffle(tmpItem_1)
                random.shuffle(tmpItem_2)

                for idx, f in enumerate(tmpItem_1):
                    if idx >= thr:
                        break
                    if 'jpg' in f:
                        self.img_files.append(f)
                        self.img_paths.append(os.path.join(inputImgPath, funcList[i]+'_'+funcList[j]))
                        self.labels.append(funcList[i]+'_'+funcList[j])

                for idx, f in enumerate(tmpItem_2):
                    if idx >= thr:
                        break
                    if 'jpg' in f:
                        self.img_files.append(f)
                        self.img_paths.append(os.path.join(inputImgPath, funcList[j]+'_'+funcList[i]))
                        self.labels.append(funcList[i]+'_'+funcList[j])

        thr = 30

        # 3개 선택
        for i in range(len(funcList)):
            for j in range(i+1, len(funcList)):
                for k in range(j+1, len(funcList)):

                    tmpItem_1 = os.listdir(os.path.join(inputImgPath, funcList[i]+'_'+funcList[j]+'_'+funcList[k]))
                    tmpItem_2 = os.listdir(os.path.join(inputImgPath, funcList[i]+'_'+funcList[k]+'_'+funcList[j]))
                    tmpItem_3 = os.listdir(os.path.join(inputImgPath, funcList[k]+'_'+funcList[j]+'_'+funcList[i]))
                    tmpItem_4 = os.listdir(os.path.join(inputImgPath, funcList[k]+'_'+funcList[i]+'_'+funcList[j]))
                    tmpItem_5 = os.listdir(os.path.join(inputImgPath, funcList[j]+'_'+funcList[i]+'_'+funcList[k]))
                    tmpItem_6 = os.listdir(os.path.join(inputImgPath, funcList[j]+'_'+funcList[k]+'_'+funcList[i]))

                    random.shuffle(tmpItem_1)
                    random.shuffle(tmpItem_2)
                    random.shuffle(tmpItem_3)
                    random.shuffle(tmpItem_4)
                    random.shuffle(tmpItem_5)
                    random.shuffle(tmpItem_6)

                    for idx, f in enumerate(tmpItem_1):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[i]+'_'+funcList[j]+'_'+funcList[k]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])
                    
                    for idx, f in enumerate(tmpItem_2):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[i]+'_'+funcList[k]+'_'+funcList[j]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])

                    
                    for idx, f in enumerate(tmpItem_3):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[k]+'_'+funcList[j]+'_'+funcList[i]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])

                    for idx, f in enumerate(tmpItem_4):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[k]+'_'+funcList[i]+'_'+funcList[j]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])

                    for idx, f in enumerate(tmpItem_5):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[j]+'_'+funcList[i]+'_'+funcList[k]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])
                    
                    for idx, f in enumerate(tmpItem_6):
                        if idx >= thr:
                            break
                        if 'jpg' in f:
                            self.img_files.append(f)
                            self.img_paths.append(os.path.join(inputImgPath, funcList[j]+'_'+funcList[k]+'_'+funcList[i]))
                            self.labels.append(funcList[i]+'_'+funcList[j]+'_'+funcList[k])
        
        #print("Loaded tot img cnt -", len(self.labels))


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_paths[index], self.img_files[index])).convert('RGB')  # for color image
        img = img.resize((int(img.width / 2), int(img.height / 2)))
        img = self.to_tensor(img)
        return img, self.labels[index]