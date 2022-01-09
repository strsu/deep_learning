import torch
import torch.nn as nn
import torch.nn.functional as F
 
import numpy as np
import math

def sppnet(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
        w_pad = int((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp
        
 
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias = False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias = False)
 
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias = False)
            )
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out += self.shortcut(x)
        return out
 
class selectNet(nn.Module):
    def __init__(self, rgb):
        super(selectNet, self).__init__()
 
        self.output_num = [4,2,1]
        self.sm = nn.Softmax(dim=1)
        
        in_ch = 1
        if rgb:
            in_ch = 3
        
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size = 3)
        self.rb1_1 = BasicBlock(64, 64)
        self.rb1_2 = BasicBlock(64, 128)
        self.mp1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size = 3)
        self.rb2_1 = BasicBlock(256, 256)
        self.rb2_2 = BasicBlock(256, 256)
        self.mp2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(256, 512, kernel_size = 3)
        self.rb3_1 = BasicBlock(512, 512)
        self.rb3_2 = BasicBlock(512, 512)
      
        self.fc1 = nn.Linear(10752, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 8)
        
 
    def forward(self, x, batch_size):
        out = F.relu(self.conv1(x))
        out = self.rb1_1(out)
        out = self.rb1_2(out)
        out = self.mp1(out)
       
        out = F.relu(self.conv2(out))
        out = self.rb2_1(out)
        out = self.rb2_2(out)
        out = self.mp2(out)

        out = F.relu(self.conv3(out))
        out = self.rb3_1(out)
        out = self.rb3_2(out)
        out = sppnet(out, batch_size, [out.size(2), out.size(3)], self.output_num)
        
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        
        #return self.sm(out)
        return out

class selectNet_4label(nn.Module):
    def __init__(self):
        super(selectNet_4label, self).__init__()
 
        self.output_num = [4,2,1]
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3)
        self.rb1_1 = BasicBlock(64, 64)
        self.rb1_2 = BasicBlock(64, 128)
        self.mp1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size = 3)
        self.rb2_1 = BasicBlock(256, 256)
        self.rb2_2 = BasicBlock(256, 256)
        self.mp2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(256, 512, kernel_size = 3)
        self.rb3_1 = BasicBlock(512, 512)
        self.rb3_2 = BasicBlock(512, 512)
      
        self.fc1 = nn.Linear(10752, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 3)
        
 
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.rb1_1(out)
        out = self.rb1_2(out)
        out = self.mp1(out)
       
        out = F.relu(self.conv2(out))
        out = self.rb2_1(out)
        out = self.rb2_2(out)
        out = self.mp2(out)

        out = F.relu(self.conv3(out))
        out = self.rb3_1(out)
        out = self.rb3_2(out)
        out = sppnet(out, 1, [out.size(2), out.size(3)], self.output_num)
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        #return self.sm(out)
        return torch.sigmoid(out)