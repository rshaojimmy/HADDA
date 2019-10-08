"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
from misc import config as cfg

from pdb import set_trace as st
from misc import utils



# def get_mnist(train, get_dataset=False, batch_size=cfg.batch_size):

#     convert_to_3_channels = transforms.Lambda(
#         lambda x: torch.cat([x, x, x], 0))
#     pre_process = transforms.Compose([transforms.Scale(cfg.image_size),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize(
#                                           mean=cfg.dataset_mean,
#                                           std=cfg.dataset_std),
#                                       convert_to_3_channels])

#     # dataset and data loader
#     mnist_dataset = datasets.MNIST(root=cfg.data_root,
#                                    train=train,
#                                    transform=pre_process,
#                                    download=False)

#     if get_dataset:
#         return mnist_dataset
#     else:
#         mnist_data_loader = torch.utils.data.DataLoader(
#             dataset=mnist_dataset,
#             batch_size=batch_size,
#             shuffle=True)
#         return mnist_data_loader



def get_mnist(train, sample, batch_size=cfg.batch_size):
    """Get MNIST dataset loader."""
    # image pre-processing

    pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
                                      transforms.ToTensor()
                                        ])       
    # pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                       mean=(0.1307,),
    #                                       std=(0.3081,))
    #                                     ])    
    # pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                       mean=cfg.dataset_mean,
    #                                       std=cfg.dataset_std)
    #                                     ])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=cfg.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=False)

    if sample:
       mnist_data_loader = utils.get_sampled_data_loader(mnist_dataset,
                                          2000, shuffle=True)
       return mnist_data_loader
    else:
        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=batch_size,
            shuffle=True)
        return mnist_data_loader


