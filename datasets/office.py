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


class OFFICE(Dataset):
    def __init__(self, root, name, transform=None, loader=default_loader):

        self.name = name
        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, self.name)


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


def get_office(name, batch_size=cfg.batch_size):
    """Get USPS dataset loader."""
   
    pre_process = transforms.Compose([transforms.Resize([cfg.image_size, cfg.image_size]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=cfg.dataset_mean,
                                          std=cfg.dataset_std)])    
    # dataset and data loader
    office_dataset = OFFICE(root=cfg.data_root,
                        name=name,
                        transform=pre_process
                        )


    office_data_loader = torch.utils.data.DataLoader(
        dataset=office_dataset,
        batch_size=batch_size,
        shuffle=True)
    return office_data_loader
