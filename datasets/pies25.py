from __future__ import print_function
import torch.utils.data as data
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import os.path
import numpy as np
# from .utils import download_url, check_integrity
from misc import utils
from pdb import set_trace as st
from misc import config as cfg

class PIE25(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`
    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        # 'train': ["GTSRB_train_32x32_old.mat"],
        # 'test': ["GTSRB_test_32x32_old.mat"]}        
        'train': ["PIE25_re.mat"],
        'test': ["PIE25_re.mat"]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        # self.url = self.split_list[split][0]
        self.filename = self.split_list[split][0]
        # self.file_md5 = self.split_list[split][2]


        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        self.data = (self.data*255).astype(np.uint8)
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()
        self.labels = self.labels - 1 

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        self.data = np.transpose(self.data, (3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    # def download(self):
    #     md5 = self.split_list[self.split][2]
    #     download_url(self.url, self.root, self.filename, md5)


def get_pie25(train, get_dataset=False, batch_size=cfg.batch_size):
    """Get SVHN dataset loader."""
    #image pre-processing

    # convert_to_gray = transforms.Lambda(
    # lambda x: (x[0, ...] * 0.299 + x[1, ...] * 0.587 + x[2, ...] * 0.114).unsqueeze(0))    
    # pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize(
    #                                       mean=(0.5, 0.5, 0.5),
    #                                       std=(0.5, 0.5, 0.5)),
    #                                   convert_to_gray]) 

    pre_process = transforms.Compose([transforms.Resize(cfg.image_size),
                                      transforms.Grayscale(num_output_channels=1),
                                      transforms.ToTensor()])

    # dataset and data loader
    pie25_dataset = PIE25(root=cfg.data_root,
                                 split='train' if train else 'test',
                                 transform=pre_process,
                                 download=False)

    if get_dataset:
        return pie25_dataset
    else:
        pie25_data_loader = torch.utils.data.DataLoader(
            dataset=pie25_dataset,
            batch_size=batch_size,
            shuffle=True)
        return pie25_data_loader