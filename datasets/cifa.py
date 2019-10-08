"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

from misc import config as cfg


def get_cifa(train, get_dataset=False, batch_size=cfg.batch_size):
    """Get SVHN dataset loader."""
    #image pre-processing

    pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
                                      # transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=cfg.dataset_mean,
                                          std=cfg.dataset_std)])

    # dataset and data loader
    cifa_dataset = datasets.CIFAR10(root=cfg.data_root,
                                 train=train,
                                 transform=pre_process,
                                 download=False)

    if get_dataset:
        return cifa_dataset
    else:
        cifa_data_loader = torch.utils.data.DataLoader(
            dataset=cifa_dataset,
            batch_size=batch_size,
            shuffle=True)
        return cifa_data_loader



