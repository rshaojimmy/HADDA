import numpy as np
import os
import ntpath
import time
from . import utils
from misc import config as cfg

from pdb import set_trace as st

class Saver():
    def __init__(self):
        # self.cfg = cfg
        self.name = cfg.name
        # self.img_dir = os.path.join(cfg.model_root, cfg.name, 'images')
        # print('create image directory %s...' % self.img_dir)
        # utils.mkdirs(self.img_dir)
        self.save_file = os.path.join(cfg.model_root, cfg.name)
        utils.mkdirs(self.save_file)
        self.log_name = os.path.join(self.save_file, 'loss_log.txt')
        # utils.mkdirs(self.log_name)

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def print_current_errors(self, epoch, i, errors):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    #save image to the disk
    # def save_images(self, visuals, image_path, epoch):
    #     image_dir = self.img_dir
    #     short_path = ntpath.basename(image_path[0])
    #     name = os.path.splitext(short_path)[0]

    #     for label, image_numpy in visuals.items():
    #         image_name = '%d_%s_%s.png' % (epoch, label, name)
    #         save_path = os.path.join(image_dir, image_name)
    #         utils.save_image(image_numpy, save_path)

        # save to the disk
    def print_config(self):
        args = vars(cfg)
        file_name = os.path.join(self.save_file, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

       

