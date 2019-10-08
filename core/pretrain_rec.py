"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
import torch

from misc import config as cfg
from misc.utils import make_variable, save_model
from misc import evaluate
import itertools

from pdb import set_trace as st


def train_src_rec(encoder, classifier, generator, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    generator.train()

    # setup criterion and optimizer
    
    optimizer = optim.Adam(
        generator.parameters(),
        lr=cfg.learning_rate_apt,
        betas=(cfg.beta1, cfg.beta2))

    criterionRec = torch.nn.MSELoss()

    ####################
    # 2. train network #
    ####################
    for epoch in range(cfg.num_epochs_pre_rec):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            # labels[labels == 10] = 0
            labels = make_variable(labels).long().squeeze()

            # zero gradients for optimizer
            optimizer.zero_grad()
            # compute loss for critic
            feat= encoder(images)
            feat_reshape = (feat.unsqueeze(2)).unsqueeze(2)
            reconst = generator(feat_reshape)
            loss_rec = criterionRec(reconst, images)

            loss_rec.backward()
            optimizer.step()

            # print step info
            if (step+1) % cfg.log_step_pre == 0:
                print('Epoch [%d] ' 
                      'loss[%.2f] '
                    %(epoch,
                      loss_rec.data[0],
                      )
                    )


    # # save final model
    save_model(generator, "ADDA-source-generator-final.pt")

    return generator



