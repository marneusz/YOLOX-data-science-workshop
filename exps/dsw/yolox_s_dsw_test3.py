#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/dsw"
        self.train_ann = "train_annotations.coco.json"
        self.val_ann = "val_annotations.coco.json"

        self.num_classes = 4

        self.max_epoch = 25
        self.no_aug_epochs = 15
        self.data_num_workers = 8
        self.eval_interval = 1
        self.print_interval = 25

        self.save_history_ckpt = False

        # ---------------
        self.basic_lr_per_img = 0.01 / 16.0

        # --------------- transform config ----------------- #
        self.scale = (0.1, 2)

        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.7, 1.3)

        self.enable_mixup = True
        self.mixup_prob = 0.5

        self.hsv_prob = 0.3

        self.flip_prob = 0.3

        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 5.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mixup_scale = (0.7, 1.3)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

