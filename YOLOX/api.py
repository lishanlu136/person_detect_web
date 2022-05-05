#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/09/08 16:21
# @Author  : lishanlu
# @File    : api.py
# @Software: PyCharm
# @Discription:

from __future__ import absolute_import, print_function, division
import os
import time
#from loguru import logger
import numpy as np
import cv2
import shutil

import torch

from yolox.data.data_augment import preproc, preproc_legacy
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


def read_conf(conf_path):
    dic = {}
    with open(conf_path,'r', encoding='utf-8') as fs:
        lines = fs.readlines()
        for line in lines:
            key,value = line.strip().replace(' ','').replace('，',',').split('=')
            if key in ['classNames']:
                value = value.split(',')
            dic[key] = value
    return dic


def visual(output, img,  ratio, classes, cls_conf=0.35):
    if output is None:
        return img
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(img, bboxes, scores, cls, cls_conf, classes)
    return vis_res


def visual_res(img, result):
    for res in result:
        name = res['name']
        prop = res['prop']
        left = res['left']
        top = res['top']
        right = res['right']
        bottom = res['bottom']
        text = '%s: %.02f'%(name, prop)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 1.1, 1)[0]
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(img, (left, top + 1),(left + txt_size[0] + 1, top + int(1.5 * txt_size[1])), (0, 255, 0),-1)
        cv2.putText(img, text, (left, top + txt_size[1]), font, 1.1, (0, 0, 255), thickness=1)
    return img


class DETECTOR(object):
    def __init__(self, conf_path, exp, device=torch.device('cpu')):
        conf = read_conf(conf_path)
        weights = conf_path.replace('.conf', '.pth')
        self.exp = exp
        self.model = self.exp.get_model()
        self.test_size = (int(conf['height']), int(conf['width']))
        print("Model Summary: {}".format(get_model_info(self.model, self.test_size)))
        #print("loading checkpoint")
        ckpt = torch.load(weights, map_location=device)
        # load the model state dict
        self.model.load_state_dict(ckpt["model"])
        #print("loaded checkpoint done.")
        self.fp16 = device.type != 'cpu'  # half precision only supported on CUDA
        if self.fp16:
            self.model.cuda()
            self.model.half()  # to FP16
        self.device = device
        self.model.eval()
        self.default_nms_thres = float(conf['nms'])
        self.default_conf_thres = float(conf['conf'])
        self.class_names = conf['classNames']     # list
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = len(self.class_names)

    def detect(self, img, conf=None, nms=None, legacy=False):
        if not conf:
            conf = self.default_conf_thres
        if not nms:
            nms = self.default_nms_thres
        height, width = img.shape[:2]
        if legacy:
            img, ratio = preproc_legacy(img, self.test_size, self.rgb_means, self.std)
        else:
            img, ratio = preproc(img, self.test_size)
        #print("ratio:", ratio)

        img = torch.from_numpy(img).unsqueeze(0)
        if self.device != "cpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, conf, nms)[0]
            import pdb
            # pdb.set_trace()
            res = []
            if outputs is not None:
                outputs = outputs.cpu().detach().numpy()
                if len(outputs) > 0:
                    for obj in outputs:
                        bboxes = obj[0:4]
                        #print("bboxes:", bboxes)
                        # preprocessing: resize
                        bboxes /= ratio
                        cls = obj[6]
                        scores = obj[4] * obj[5]
                        dic = {
                            'name': self.class_names[int(cls)],
                            'prop': float(scores),
                            'left':max(0, int(bboxes[0])),
                            'top': max(0, int(bboxes[1])),
                            'right':min(int(bboxes[2]), width-1),
                            'bottom':min(int(bboxes[3]), height-1)
                        }
                        res.append(dic)

        return res


class PERSON(DETECTOR):
    def __init__(self, conf_path, device):
        from exps.yolox_person import Exp
        super().__init__(conf_path, exp=Exp(), device=device)


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    person_det = PERSON('./models/detect_person_yolox_s.conf', device)
    img_path = 'images/948abfcd-279a-42db-8222-4b202abc1426.jpg'
    img = cv2.imread(img_path)
    outputs = person_det.detect(img)
    print(outputs)
    img_res = visual_res(img, outputs)
    cv2.namedWindow('res', 0)
    cv2.resizeWindow('res', 1280, 960)
    cv2.imshow("res", img_res)
    cv2.waitKey()






