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
        self.data_num_workers = 8
        self.eval_interval = 1
        self.print_interval = 25

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # ---------------
        self.basic_lr_per_img = 0.01 / 32.0
