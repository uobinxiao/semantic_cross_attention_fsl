import numpy
from torchvision import transforms

def aug_transform(image_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((image_size, image_size)),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(numpy.array([0.485, 0.456, 0.406]), numpy.array([0.229, 0.224, 0.225])),])

    return transform

def noaug_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(numpy.array([0.485, 0.456, 0.406]), numpy.array([0.229, 0.224, 0.225])),])

    return transform
