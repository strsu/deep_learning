import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, skip_c):
        super().__init__()

        self.drop_prob = 0.3
        self.dropout = nn.Dropout(p=self.drop_prob)

        self.up = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(True))
        self.conv = torch.nn.Conv2d(skip_c+out_c, out_c, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class autoEncoder(nn.Module):
    def __init__(self, RGB=0):
        super(autoEncoder, self).__init__()
        
        self.drop_prob = 0.3
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.maxPool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        if RGB == 1:
            self.c = 1
        else:
            self.c = 3

        # Encoder
        self.E_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(self.c,  64, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(64), nn.LeakyReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(True))
        
        self.E_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(True))

        self.E_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(True),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(True))

        self.E_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(True))
        
        self.E_layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(True))
        # Decoder
        self.D_layer1 = decoder_block(256, 128, 256)
        self.D_layer2 = decoder_block(128, 64, 128)
        self.D_layer3 = decoder_block(64, 32, 64)
        self.D_layer4 = decoder_block(32, self.c, 64)
 
    def forward(self, input, isTanh=False, skip_1=False):
        if skip_1 == False:
            unpool_1 = self.E_layer1(input)
            Eout_1 = self.maxPool2d(unpool_1)
        else:
            unpool_1 = input
            Eout_1 = self.maxPool2d(unpool_1)
        
        unpool_2 = self.E_layer2(Eout_1)
        Eout_2 = self.maxPool2d(unpool_2)

        unpool_3 = self.E_layer3(Eout_2)
        Eout_3 = self.maxPool2d(unpool_3)

        unpool_4 = self.E_layer4(Eout_3)
        Eout_4 = self.maxPool2d(unpool_4)

        Eout_5 = self.E_layer5(Eout_4)
        
        Dout_1 = F.leaky_relu(self.D_layer1(Eout_5, unpool_4))
        Dout_2 = F.leaky_relu(self.D_layer2(Dout_1, unpool_3))
        Dout_3 = F.leaky_relu(self.D_layer3(Dout_2, unpool_2))

        if isTanh:
            Dout_4 = torch.tanh(self.D_layer4(Dout_3, unpool_1))
        else:
            Dout_4 = torch.sigmoid(self.D_layer4(Dout_3, unpool_1))
        return Dout_4


class Restore(nn.Module):
    def __init__(self):
        super(Restore, self).__init__()
        self.contrastEq = autoEncoder()
    
    def forward(self, input, map):
        out = torch.add(input, map)
        res = self.contrastEq(out)
        return res



class contrastMap(nn.Module):
    def __init__(self):
        super(contrastMap, self).__init__()      
        self.contrast_map = autoEncoder()
 
    def forward(self, input):
        return self.contrast_map(input, isTanh=True)


class CEModel(nn.Module):
    def __init__(self, RGB=0):
        super(CEModel, self).__init__()

        self.contrast_map = autoEncoder(RGB)
        self.contrast_restore = autoEncoder(RGB)
 
    def forward(self, input):
        map = self.contrast_map(input, isTanh=False)
        factor = torch.div(torch.mul((259/255), (torch.add(map, 1.0))), torch.mul(1.0, torch.sub((259/255), map)))
        out = torch.add(torch.mul(factor, torch.sub(input, (128/255))), (128/255))
        res = self.contrast_restore(out, isTanh=False)
        
        return map, out, res
  