import os
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
from PIL import Image
from misc import utils
from pdb import set_trace as st
from misc import config as cfg

def default_loader(path):
    return Image.open(path).convert('RGB')


class USPS(Dataset):
    def __init__(self, root, train, transform=None, loader=default_loader):

        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, 'usps')
        self.train = train

        if self.train:
            fh = open(os.path.join(self.root,'usps_train_list.txt'), 'r')
        else:
            fh = open(os.path.join(self.root,'usps_test_list.txt'), 'r')

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


def get_usps(train, sample, batch_size=cfg.batch_size):
    """Get USPS dataset loader."""
    # image pre-processing
    # convert_to_gray = transforms.Lambda(
    # lambda x: (x[0, ...] * 0.299 + x[1, ...] * 0.587 + x[2, ...] * 0.114).unsqueeze(0))

    pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
                                      transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor()])     

    # pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
    #                                   transforms.Grayscale(num_output_channels=1),
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                       mean=cfg.dataset_mean,
    #                                       std=cfg.dataset_std)])    
    # dataset and data loader
    usps_dataset = USPS(root=cfg.data_root,
                        train=train,
                        transform=pre_process
                        )

    if sample:
       usps_data_loader = utils.get_sampled_data_loader(usps_dataset,
                                          1800, shuffle=True)
       return usps_data_loader

    else:
        usps_data_loader = torch.utils.data.DataLoader(
            dataset=usps_dataset,
            batch_size=batch_size,
            shuffle=True)
        return usps_data_loader
