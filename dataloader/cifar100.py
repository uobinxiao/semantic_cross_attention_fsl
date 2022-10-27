import os.path as osp
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/cub/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/cub/split')

DATA_PATH = "data/cifar100"

class CIFAR100(Dataset):

    def __init__(self, setname, image_size = 84, if_augmentation = False):
        if "train" in setname and if_augmentation:
            self.transform = transforms.Compose([
                #transforms.RandomResizedCrop((image_size, image_size)),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            self.transform = transforms.Compose([
                #transforms.Resize(int(image_size * 1.1)),
                #transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

        data_dict = {}

        data_dict = np.load(osp.join(DATA_PATH, "few-shot-"+setname+".npz"))
        self.data = data_dict["features"]
        self.targets = data_dict["targets"]
        #key is the global label of 100 classes, value is the reordered label
        label_dict = {}
        reversed_label_dict = {}

        start_index = 0
        for tmp_label in self.targets:
            if tmp_label not in label_dict.keys():
                label_dict[tmp_label] = start_index
                reversed_label_dict[start_index] = tmp_label
                start_index += 1

        label = []
        for tmp_label in self.targets:
            label.append(label_dict[tmp_label])

        self.reversed_label_dict = reversed_label_dict
        self.label = label

        text_dict = {}
        with open(osp.join(DATA_PATH, "classes.txt")) as f:
            lines = f.readlines()
            for line in lines:
                index, text = line.strip().split(" ")
                text_dict[int(index)] = text
        self.text_dict = text_dict

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        image, label = self.data[i], self.label[i]
        #image = np.transpose(image, ())
        tmp_image = Image.fromarray(image)
        image = self.transform(tmp_image)
        #image1 = self.transform(tmp_image)
        #image2 = self.transform(tmp_image)
        #print(image1.shape)
        #cv2.imwrite("image1.jpg", image1.cpu().numpy().transpose(1, 2, 0) * 255)
        #cv2.imwrite("image2.jpg", image2.cpu().numpy().transpose(1, 2, 0) * 255)
        #exit()
        tmp_image.close()
        text_label = self.text_dict[self.reversed_label_dict[label]]

        return image, label, text_label
