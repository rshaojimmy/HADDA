"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms

from misc import config as cfg


def get_stl(train, get_dataset=False, batch_size=cfg.batch_size):
    """Get SVHN dataset loader."""
    #image pre-processing

    pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
                                      # transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=cfg.dataset_mean,
                                          std=cfg.dataset_std)])

    # dataset and data loader
    stl_dataset = datasets.STL10(root=cfg.data_root,
                                 split='train' if train else 'test',
                                 transform=pre_process,
                                 download=False)

    if get_dataset:
        return stl_dataset
    else:
        stl_data_loader = torch.utils.data.DataLoader(
            dataset=stl_dataset,
            batch_size=batch_size,
            shuffle=True)
        return stl_data_loader



