"""Utilities for ADDA."""

import os
import random
from PIL import Image
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn import init
from misc import config as cfg
from datasets import (get_mnist, get_mnist_m, get_svhn, get_usps, get_synth, get_synthsign, get_gtsrb, get_stl, get_cifa, get_pie05, get_pie27, get_pie09,
                     get_pie37, get_pie25, get_pie02, get_taskcv_s, get_taskcv_t, get_office)
from pdb import set_trace as st


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    # return Variable(tensor, volatile=volatile)
    return tensor


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        # init.normal(m.weight.data, std=1e-3)
        # init.uniform(m.bias.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)                


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loader(name, train=True, dataset = None, sample=False):
    """Get data loader by name."""
    if name == "MNIST":
        return get_mnist(train, sample)
    elif name == "MNIST-M":
        return get_mnist_m(train)    
    elif name == "SVHN":
        return get_svhn(train)
    elif name == "USPS":
        return get_usps(train, sample)    
    elif name == "SYNTH":
        return get_synth(train)    
    elif name == "SYNTHSIGN":
        return get_synthsign(train)    
    elif name == "GTSRB":
        return get_gtsrb(train)
    elif name == "STL":
        return get_stl(train)    
    elif name == "CIFA":
        return get_cifa(train)
    elif name == "PIE27":
        return get_pie27(train)
    elif name == "PIE05":
        return get_pie05(train)    
    elif name == "PIE09":
        return get_pie09(train)
    elif name == "PIE37":
        return get_pie37(train)
    elif name == "PIE25":
        return get_pie25(train)
    elif name == "PIE02":
        return get_pie02(train)    
    elif name == "taskcv_S":
        return get_taskcv_s(train)    
    elif name == "taskcv_T":
        return get_taskcv_t(train)    
    elif name == "OFFICE":
        return get_office(dataset)

def init_model(net, restore, init= True):
    """Init models with cuda and weights."""
    # init weights of model
    if init:
        init_weights(net, init_type=cfg.init_type)
    # net.apply(init_weights)
    
    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(cfg.model_root):
        os.makedirs(cfg.model_root)
    torch.save(net.state_dict(),
               os.path.join(cfg.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(cfg.model_root,
                                                             filename)))

def save_trainedmodel(net, filename):
    """Save trained model."""
    if not os.path.exists(os.path.join(cfg.model_root, cfg.namesave)):
        os.makedirs(os.path.join(cfg.model_root, cfg.namesave))
    torch.save(net.state_dict(),
               os.path.join(cfg.model_root, cfg.namesave, filename))
    print("save pretrained model to: {}".format(os.path.join(cfg.model_root, cfg.namesave,
                                                             filename)))


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def get_sampled_data_loader(dataset, candidates_num, shuffle=True):
    """Get data loader for sampled dataset."""
    # get indices
    indices = torch.arange(0, len(dataset))
    if shuffle:
        indices = torch.randperm(len(dataset))
    # slice indices
    candidates_num = min(len(dataset), candidates_num)
    excerpt = indices.narrow(0, 0, candidates_num).long()
    sampler = torch.utils.data.sampler.SubsetRandomSampler(excerpt)
    return make_data_loader(dataset, sampler=sampler, shuffle=False)


def make_data_loader(dataset, batch_size=cfg.batch_size,
                     shuffle=True, sampler=None):
    """Make dataloader from dataset."""
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler)
    return data_loader


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)    
    

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)        