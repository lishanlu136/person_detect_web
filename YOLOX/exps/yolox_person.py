#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/14 15:09
# @Author  : lishanlu
# @File    : yolox_person.py
# @Software: PyCharm
# @Discription:

from __future__ import absolute_import, print_function, division
import os
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.num_classes = 1
        self.max_epoch = 300
        self.data_num_workers = 1
        self.eval_interval = 1
