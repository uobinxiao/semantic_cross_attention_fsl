import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')

class MiniImageNet(Dataset):
    """ Usage: 
    """
    def __init__(self, setname, image_size, if_augmentation = False):

        if "train" in setname and if_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((image_size, image_size)),
                #transforms.RandAugment(num_ops = 5),
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

        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        lines = sorted(lines)

        data = []
        label = []
        text_label = []
        data_dict = {}
        lb = -1

        self.wnids = []

        for l in tqdm(lines):
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            image = Image.open(path).convert("RGB")
            data_dict[path] = np.array(image)
            image.close()
            label.append(lb)
            text_label.append(wnid)

        self.data_dict = data_dict
        self.data = data
        self.label = label
        self.text_label = text_label
        self.num_class = len(set(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label, text_label = self.data[i], self.label[i], self.text_label[i]
        tmp_image = Image.fromarray(self.data_dict[path])
        image = self.transform(tmp_image)
        tmp_image.close()

        return image, label, text_label
