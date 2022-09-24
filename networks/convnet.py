import torch.nn as nn
import torch
from utils import Attention

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2, stride = 2)
        )

def conv_block_no_pooling(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)

class ConvNet4(nn.Module):

    def __init__(self, x_dim = 3, hid_dim = 64, z_dim = 128, pooling = False):
        super(ConvNet4, self).__init__()
        self.layer1 = conv_block(x_dim, hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        if pooling:
            self.layer3 = conv_block(hid_dim, z_dim)
            self.layer4 = conv_block(z_dim, z_dim)
        else:
            self.layer3 = conv_block_no_pooling(hid_dim, z_dim)
            self.layer4 = conv_block_no_pooling(z_dim, z_dim)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

class AConvNet4(nn.Module):

    def __init__(self, x_dim = 3, hid_dim = 64, z_dim = 64, pooling = False):
        super(AConvNet4, self).__init__()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()

        self.layer1 = conv_block(x_dim, hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        if pooling:
            self.layer3 = conv_block(hid_dim, hid_dim)
            self.layer4 = conv_block(hid_dim, z_dim)
        else:
            self.layer3 = conv_block_no_pooling(hid_dim, hid_dim)
            self.layer4 = conv_block_no_pooling(hid_dim, z_dim)

    def forward(self, x):
        x1 = self.layer1(x)
        at1 = self.sa1(x1) 
        x1 = at1 * x1

        x2 = self.layer2(x1)
        at2 = self.sa2(x2) 
        x2 = at2 * x2

        x3 = self.layer3(x2)
        at3 = self.sa3(x3) 
        x3 = at3 * x3

        x4 = self.layer4(x3)
        at4 = self.sa4(x4)
        x4 = at4 * x4

        return x4, at1

class AttentionConvNet4(nn.Module):

    def __init__(self, x_dim = 3, hid_dim = 64, z_dim = 64, pooling = False):
        super(AttentionConvNet4, self).__init__()
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.ca1 = ChannelAttention(64)
        self.ca2 = ChannelAttention(64)
        self.ca3 = ChannelAttention(64)
        self.ca4 = ChannelAttention(64)

        self.layer1 = conv_block(x_dim, hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        if pooling:
            self.layer3 = conv_block(hid_dim, hid_dim)
            self.layer4 = conv_block(hid_dim, z_dim)
        else:
            self.layer3 = conv_block_no_pooling(hid_dim, hid_dim)
            self.layer4 = conv_block_no_pooling(hid_dim, z_dim)

    def forward(self, x):
        x1 = self.layer1(x)
        at1 = self.ca1(x1)
        x1 = at1 * x1
        at1 = self.sa1(at1) 
        x1 = at1 * x1

        x2 = self.layer2(x1)
        at2 = self.ca2(x2)
        x2 = at2 * x2
        at2 = self.sa2(at2) 
        x2 = at2 * x2

        x3 = self.layer3(x2)
        at3 = self.ca3(x3)
        x3 = at3 * x3
        at3 = self.sa3(at3) 
        x3 = at3 * x3

        x4 = self.layer4(x3)
        at4 = self.ca4(x4)
        x4 = at4 * x4
        at4 = self.sa4(x4)
        x4 = at4 * x4

        #return x4, x3, x2, x1
        return x4

class ConvNet6(nn.Module):
    
    def __init__(self, x_dim = 3, hid_dim = 64, z_dim = 64):
        super(ConvNet6, self).__init__()
        self.layer1 = conv_block(x_dim, hid_dim)
        self.layer2 = conv_block(hid_dim, hid_dim)
        self.layer3 = conv_block_no_pooling(hid_dim, hid_dim)
        self.layer4 = conv_block_no_pooling(hid_dim, hid_dim)
        self.layer5 = conv_block_no_pooling(hid_dim, hid_dim)
        self.layer6 = conv_block_no_pooling(hid_dim, z_dim)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x4)

        return x6

class ConvNet_CLS(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ConvNet_CLS, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(64 * 21 * 21, num_classes)
        if backbone == "ConvNet4":
            self.convnet = ConvNet4()
        elif backbone == "ConvNet6":
            self.convnet = ConvNet6()
        else:
            raise "backbone error"
        self.attention = Attention()

    def forward(self, x):
        out_attention = self.attention.forward_attention(self.convnet, x, output_size=(x.shape[-2], x.shape[-1]), use_softmax = False)
        out = self.convnet(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out, out_attention
