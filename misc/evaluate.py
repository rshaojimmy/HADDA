"""Test script for ATDA."""

import torch.nn as nn
import torch

from misc.utils import make_variable
from misc import config as cfg
from pdb import set_trace as st

def evaluate(encoder, classifier, data_loader):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        # labels = make_variable(labels.squeeze_())
        labels = label_convert(labels)
        labels = make_variable(labels).long().squeeze()

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {:.5f}, Avg Accuracy = {:2.5%}".format(loss, acc))



def evaluate_target(encoder, classifier, images, labels):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    images = make_variable(images, volatile=True)
    # labels = make_variable(labels.squeeze_())
    labels = make_variable(labels).long().squeeze()

    preds = classifier(encoder(images))

    _, target_predicted = torch.max(preds.data, 1)
    target_correct = (target_predicted == labels.data).sum()
    acc = target_correct/labels.size()[0]

    return acc



def evaluate_target_fianl(encoder, classifier, data_loader):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        # labels = make_variable(labels.squeeze_())
        labels = make_variable(labels).long().squeeze()

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    
    print("Avg Loss = {:.5f}, Avg Accuracy = {:2.5%}".format(loss, acc))


def evaluate_step(encoder, classifier, datatst, label):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    pre = classifier(encoder(datatst))

    _, target_predicted = torch.max(pre.data, 1)
    target_correct = (target_predicted == label.data).sum()
    acc = 100*target_correct/label.size()[0]

    # print("Test Accuracy = {:2.5%}".format(acc))
    return acc


def eval_func(encoder, classifier, data_loader, sample = False):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        # labels[labels == 10] = 0
        labels = make_variable(labels).long().squeeze()
        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item() 
        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)



    # acc /= 1800
    # acc /= 2000
    # if sample:
    #     # acc /= (len(data_loader)*cfg.batch_size)
    #     acc /= 1800
    #     # acc /= 2000
    # else:
    #     # acc /= 1800
    #     # acc /= 2000
        # acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))




def eval_func_datacollect(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    labelsall = []
    predsall = []
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        # labels[labels == 10] = 0
        labels = make_variable(labels).long().squeeze()
        preds = classifier(encoder(images))
        pred_cls = preds.data.max(1)[1]

        labelsall.extend(labels.data.cpu().numpy())
        predsall.extend(pred_cls.cpu().numpy())

    return labelsall, predsall

