import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from PIL import Image
from misc import config as cfg
from misc import utils
from pdb import set_trace as st

def default_loader(path):
    return Image.open(path).convert('RGB')


class TASKCV_T(Dataset):
    def __init__(self, root, train, transform=None, loader=default_loader):

        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, 'taskcv', 'validation')
        self.train = train

        if self.train:
            fh = open(os.path.join(self.root, 'image_list.txt'), 'r')
        else:
            fh = open(os.path.join(self.root, 'image_list.txt'), 'r')

        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        fn = os.path.join(self.root, fn)
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


def get_taskcv_t(train, batch_size=cfg.batch_size):
    """Get USPS dataset loader."""
   
    pre_process = transforms.Compose([transforms.Resize([cfg.image_size, cfg.image_size]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=cfg.dataset_mean,
                                          std=cfg.dataset_std)])    
    # dataset and data loader
    taskcvt_dataset = TASKCV_T(root=cfg.data_root,
                        train=train,
                        transform=pre_process
                        )


    taskcvt_data_loader = torch.utils.data.DataLoader(
        dataset=taskcvt_dataset,
        batch_size=batch_size,
        shuffle=True)
    return taskcvt_data_loader
