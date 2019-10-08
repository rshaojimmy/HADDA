"""Adversarial adaptation to train target encoder."""

import os

import torch
import torch.optim as optim
from torch import nn
from misc import config as cfg
from misc.utils import make_variable
from misc import evaluate
from collections import OrderedDict
import itertools
from models import loss
from torch.nn import DataParallel

from pdb import set_trace as st


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, src_classifier, tgt_classifier, tgt_data_loader_eval, 
              generator, discriminator, Saver, logger):
    """Train encoder for target domain."""
    torch.cuda.set_device(0)
    ####################
    # 1. setup network #
    ####################
    # set train state for Dropout and BN layers
    src_classifier.eval()
    src_encoder.eval()

    tgt_encoder.train()
    tgt_classifier.eval()
    
    critic.train()
    generator.train()
    discriminator.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    criterionRec = torch.nn.MSELoss()
    criterionGAN = loss.GANLoss()
  
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                                   lr=cfg.learning_rate_apt,
                                   betas=(cfg.beta1, cfg.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=cfg.learning_rate_apt_D,
                                  betas=(cfg.beta1, cfg.beta2))

    optimizer_autoencoder_ge = optim.Adam(list(tgt_encoder.parameters()) + list(generator.parameters()),
                                lr=cfg.learning_rate_apt,
                                betas=(cfg.beta1, cfg.beta2))
    optimizer_autoencoder_ad = optim.Adam(discriminator.parameters(),
                                lr=cfg.learning_rate_apt_D,
                                betas=(cfg.beta1, cfg.beta2))


    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################
    stepall = 0
    for epoch in range(cfg.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, labels_src), (images_tgt, labels_tgt)) in data_zip:

            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            labels_src = make_variable(labels_src).long().squeeze()
            labels_tgt = make_variable(labels_tgt).long().squeeze()

            ###########################
            # train discriminator #
            ###########################
            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            ############################
            # train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_tgt.zero_grad()

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()


            ###########################
            # train autoencoder #
            ###########################

            # generator

            optimizer_autoencoder_ge.zero_grad()
            
            feat_src = src_encoder(images_src)
            feat_src_reshape = (feat_src.unsqueeze(2)).unsqueeze(2)
            reconst_src = generator(feat_src_reshape)
            loss_ge_src = criterionRec(reconst_src, images_src)

            feat_tgt = tgt_encoder(images_tgt)
            feat_tgt_reshape = (feat_tgt.unsqueeze(2)).unsqueeze(2)
            reconst_tgt = generator(feat_tgt_reshape)
            # loss_ge_tgt = criterionRec(reconst_tgt, images_tgt)

            reconst_src_G = discriminator(reconst_src)
            reconst_tgt_G = discriminator(reconst_tgt)

            loss_reconst_src_G = criterionGAN(reconst_src_G, True)
            loss_reconst_tgt_G = criterionGAN(reconst_tgt_G, True)

            loss_autoencoder_ge = cfg.para_const*loss_ge_src + loss_reconst_src_G + loss_reconst_tgt_G

            loss_autoencoder_ge.backward()
            optimizer_autoencoder_ge.step()

            # discriminator
            optimizer_autoencoder_ad.zero_grad()

            reconst_src_D = discriminator(reconst_src.detach())
            reconst_tgt_D = discriminator(reconst_tgt.detach())

            loss_reconst_src_D = criterionGAN(reconst_src_D, True)
            loss_reconst_tgt_D = criterionGAN(reconst_tgt_D, False)            

            loss_autoencoder_ad = (loss_reconst_src_D + loss_reconst_tgt_D) * cfg.para_autoD

            loss_autoencoder_ad.backward()
            optimizer_autoencoder_ad.step()

            #######################
            # print step info #
            #######################

            acc_src = evaluate.evaluate_step(src_encoder, src_classifier, images_src, labels_src)
            acc_tgt = evaluate.evaluate_step(tgt_encoder, tgt_classifier, images_tgt, labels_tgt)

            infodic = OrderedDict([('loss_reconst', loss_ge_src.data[0]), ('feat_d_loss', loss_critic.data[0]), ('feat_g_loss', loss_tgt.data[0]), 
                ('auto_g_loss', loss_autoencoder_ge.data[0]), ('auto_d_loss', loss_autoencoder_ad.data[0]),
                ('acc_src', acc_src), ('acc_tgt', acc_tgt)])

            for tag, value in infodic.items():
                logger.scalar_summary(tag, value, stepall)  

            if ((step + 1) % cfg.log_step == 0):
                Saver.print_current_errors(epoch, (step + 1), infodic)


            stepall += 1      

            # eval model on test set
        evaluate.eval_func(tgt_encoder, tgt_classifier, tgt_data_loader_eval)
        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % cfg.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                cfg.model_root, cfg.name,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                cfg.model_root, cfg.name,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))
            torch.save(tgt_classifier.state_dict(), os.path.join(
                cfg.model_root, cfg.name,
                "ADDA-target-tgt-classifier-{}.pt".format(epoch + 1)))
            torch.save(generator.state_dict(), os.path.join(
                cfg.model_root, cfg.name,
                "ADDA-generator-{}.pt".format(epoch + 1)))           

    torch.save(critic.state_dict(), os.path.join(
        cfg.model_root, cfg.name,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        cfg.model_root, cfg.name,
        "ADDA-target-encoder-final.pt"))
    torch.save(tgt_classifier.state_dict(), os.path.join(
        cfg.model_root, cfg.name,
        "ADDA-target-tgt-classifier-final.pt"))
    torch.save(generator.state_dict(), os.path.join(
        cfg.model_root, cfg.name,
        "ADDA-generator-final.pt".format(epoch + 1)))


    return tgt_encoder, tgt_classifier
