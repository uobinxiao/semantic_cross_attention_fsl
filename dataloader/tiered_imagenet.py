import os.path as osp
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from timm.data.auto_augment import rand_augment_transform
from tqdm import tqdm

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/tieredimagenet/images')

class TieredImageNet(Dataset):
    """ Usage: 
    """
    def __init__(self, setname, image_size, if_augmentation = False):

        # Transformation
        if "train" in setname and if_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((image_size, image_size)),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
                ])

        data = []
        label = []
        text_label_dict = {}
        lb = -1
        data_dict = {}

        image_dir = os.path.join(IMAGE_PATH, setname)
        dir_list = glob.glob(os.path.join(image_dir, "*"))
        for dir_path in tqdm(dir_list):
            dir_name = dir_path.split("/")[-1]
            image_list = glob.glob(os.path.join(dir_path, "*"))
            for image_path in image_list:
                data.append(image_path)
                if dir_name not in text_label_dict.values():
                    lb = lb + 1
                    text_label_dict[lb] = dir_name
                label.append(lb)

        self.data_dict = data_dict
        self.data = data
        self.label = label
        self.text_label_dict = text_label_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        tmp_image = Image.open(path).convert("RGB")
        image = self.transform(tmp_image)
        tmp_image.close()
        text_label = self.text_label_dict[label]

        return image, label, text_label

