# ResNet Wide Version as in Qiao's Paper
import torch.nn as nn
import math
from utils import Attention

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

class SimpleBlock(nn.Module):
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace = True)
        self.relu2 = nn.ReLU(inplace = True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, list_of_num_layers, list_of_out_dims):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride = 2, padding = 3, bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)

        return out

class ResNet_CLS(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ResNet_CLS, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(512 * 7 * 7, num_classes)
        if backbone == "ResNet10":
            self.resnet = ResNet10()
        elif backbone == "ResNet18":
            self.resnet = ResNet18()
        elif backbone == "ResNet34":
            self.resnet = ResNet34()
        else:
            raise "backbone error"
        self.attention = Attention()
    
    def forward(self, x):
        out_attention = self.attention.forward_attention(self.resnet, x, output_size=(x.shape[-2], x.shape[-1]), use_softmax = False)
        out = self.resnet(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out, out_attention

def ResNet10():
    return ResNet(SimpleBlock, [1,1,1,1], [64, 128, 256, 512])

def ResNet18():
    return ResNet(SimpleBlock, [2,2,2,2], [64,128,256,512])

def ResNet34():
    return ResNet(SimpleBlock, [3,4,6,3], [64,128,256,512])

def ResNet10_CLS(num_classes = 64):
    return ResNet_CLS(backbone = "ResNet10", num_classes = num_classes)

def ResNet18_CLS(num_classes = 64):
    return ResNet_CLS(backbone = "ResNet18", num_classes = num_classes)

def ResNet34_CLS(num_classes = 64):
    return ResNet_CLS(backbone = "ResNet34", num_classes = num_classes)
