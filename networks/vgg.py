from torchvision import models
import torch.nn as nn
import torch

class VGG16(nn.Module):
    def __init__(self,pretrained = True):
        super(VGG16, self).__init__()
        model = models.vgg16(pretrained=pretrained)
        self.backbone = torch.nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, x):
        x = self.backbone(x)

        return x
