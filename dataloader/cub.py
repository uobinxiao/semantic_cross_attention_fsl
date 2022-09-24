import os.path as osp
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/cub/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/cub/split')

class CUB(Dataset):

    def __init__(self, setname, image_size = 84, if_augmentation = False):
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
        lines = sorted(lines)

        data = []
        label = []
        lb = -1
        self.wnids = []

        if "train" in setname and if_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((image_size, image_size)),
                #transforms.RandAugment(num_ops = 2),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        data_dict = {}
        text_label = []

        for l in tqdm(lines):
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            text_name = wnid.split(".")[-1]
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                
            data.append(path)
            image = Image.open(path).convert("RGB")
            data_dict[path] = np.array(image)
            image.close()
            label.append(lb)
            text_label.append(text_name)

        self.data_dict = data_dict
        self.data = data
        self.label = label
        self.text_label = text_label
        self.num_class = np.unique(np.array(label)).shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        tmp_image = Image.fromarray(self.data_dict[path])
        image = self.transform(tmp_image)
        tmp_image.close()
        text_label = self.text_label[i]

        return image, label, text_label

