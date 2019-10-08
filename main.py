"""Main script for ADDA."""
import os
from misc import config as cfg
from core import train_src, train_src_rec, train_tgt
from misc.evaluate import eval_func
from models import Discriminator_feat, Discriminator_img, LeNetClassifier, LeNetEncoder, LeNetGenerator
from misc.utils import get_data_loader, init_model, init_random_seed, mkdirs
from misc.saver import Saver
from logger import Logger

from pdb import set_trace as st

if __name__ == '__main__':
    # init random seed
    init_random_seed(cfg.manual_seed)
    Saver = Saver()
    Saver.print_config()

    logs_path = os.path.join(cfg.model_root, cfg.name, 'logs')
    mkdirs(logs_path)
    logger = Logger(logs_path)

    # load dataset SM
    # src_data_loader = get_data_loader(cfg.src_dataset)
    # src_data_loader_eval = get_data_loader(cfg.src_dataset, train=False)
    # tgt_data_loader = get_data_loader(cfg.tgt_dataset)
    # tgt_data_loader_eval = get_data_loader(cfg.tgt_dataset, train=False)
    
    # load dataset UM MU
    src_data_loader = get_data_loader(cfg.src_dataset, sample = True)
    src_data_loader_eval = get_data_loader(cfg.src_dataset, train=False)
    tgt_data_loader = get_data_loader(cfg.tgt_dataset, sample = True)
    tgt_data_loader_eval = tgt_data_loader

    # load models
    src_encoder = init_model(net=LeNetEncoder(cfg.inputc),
                             restore=cfg.src_encoder_restore)
    src_classifier = init_model(net=LeNetClassifier(ncls = cfg.ncls),
                                restore=cfg.src_classifier_restore)    
    tgt_classifier = init_model(net=LeNetClassifier(ncls = cfg.ncls),
                                restore=cfg.src_classifier_restore)
    tgt_encoder = init_model(net=LeNetEncoder(cfg.inputc),
                             restore=cfg.tgt_encoder_restore)
    critic = init_model(Discriminator_feat(input_dims=cfg.d_input_dims,
                                      hidden_dims=cfg.d_hidden_dims,
                                      output_dims=cfg.d_output_dims),
                        restore=cfg.d_model_restore)
    generator = init_model(net=LeNetGenerator(input_dims=cfg.g_input_dims, outputc = cfg.inputc),
                         restore=cfg.src_generator_restore)      
    # generator = init_model(net=LeNetGenerator(input_dims=cfg.g_input_dims, outputc = cfg.inputc),
    #                      restore=None)    
    discriminator = init_model(net=Discriminator_img(nc = cfg.inputc),
                         restore=None)
 
    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    # pre-train source encoder classifier
    if not (src_encoder.restored and src_classifier.restored and
            cfg.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader, tgt_data_loader_eval)
        tgt_classifier = init_model(net=LeNetClassifier(ncls = cfg.ncls),
                                restore=cfg.src_classifier_restore)

    # pre-train source generator   
    if not (generator.restored and cfg.src_model_trained):
        generator = train_src_rec(
            src_encoder, src_classifier, generator, src_data_loader)
        generator = init_model(net=LeNetGenerator(input_dims=cfg.g_input_dims, outputc = cfg.inputc),
                     restore=cfg.src_generator_restore)    

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_func(src_encoder, src_classifier, src_data_loader_eval)

    print(">>> source only <<<")
    eval_func(src_encoder, src_classifier, tgt_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)
    print(">>> Generator <<<")
    print(generator)
    print(">>> Discriminator <<<")
    print(discriminator)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())


    if not (tgt_encoder.restored and critic.restored and
            cfg.tgt_model_trained):
        tgt_encoder, tgt_classifier = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader, src_classifier, tgt_classifier,
                                tgt_data_loader_eval, generator, discriminator, Saver, logger)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_func(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    tgt_classifier = init_model(net=LeNetClassifier(ncls = cfg.ncls),
                            restore=cfg.tgt_classifier_restore) 
    eval_func(tgt_encoder, tgt_classifier, tgt_data_loader_eval)
