"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim

from misc import config as cfg
from misc.utils import make_variable, save_model
from misc import evaluate
import itertools

from pdb import set_trace as st


def train_src(encoder, classifier, data_loader, tgt_data_loader_eval):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    
    optimizer = optim.Adam(
        list(encoder.parameters())+ list(classifier.parameters()),
        lr=cfg.learning_rate_pre,
        betas=(cfg.beta1, cfg.beta2))
    criterion = nn.CrossEntropyLoss()    

    # optimizer = optim.Adam(
    #     itertools(encoder.parameters(), classifier.parameters()),
    #     lr=cfg.learning_rate_pre,
    #     betas=(cfg.beta1, cfg.beta2))
    # criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    for epoch in range(cfg.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            # labels[labels == 10] = 0
            labels = make_variable(labels).long().squeeze()

            # zero gradients for optimizer
            optimizer.zero_grad()
            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)
            # optimize source classifier
            loss.backward()
            optimizer.step()

            # st()

            acc = evaluate.evaluate_step(encoder, classifier, images, labels)

            # print step info
            if (step+1) % cfg.log_step_pre == 0:
                print('Epoch [%d] ' 
                      'loss[%.2f] '
                      'Source_Accuracy[%.2f] '                                           
                    %(epoch,
                      loss.data[0],
                      acc                                          
                      )
                    )

        # eval model on test set
        if ((epoch + 1) % cfg.eval_step_pre == 0):
            # evaluate.eval_func(encoder, classifier, tgt_data_loader_eval, sample=True)
            # evaluate.eval_func(encoder, classifier, data_loader)
            print(">>> source only <<<")
            evaluate.eval_func(encoder, classifier, tgt_data_loader_eval)

        # save model parameters
        if ((epoch + 1) % cfg.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier



