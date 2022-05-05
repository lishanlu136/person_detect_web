#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/17 11:22
# @Author  : lishanlu
# @File    : web_service.py
# @Software: PyCharm
# @Discription:

from __future__ import absolute_import, print_function, division
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('YOLOX')

import torch
from flask import Flask,request,flash,render_template,redirect,url_for
import json
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import math
import time
from functools import wraps
import logging
from shutil import move
from shutil import Error as SError
from PIL import Image
print("start import algorithm ...")

import YOLOX.api as yolox_api

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
person_detector = yolox_api.PERSON('YOLOX/models/person_best.conf', device)


app = Flask(__name__)

#允许上传type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', "PNG", "JPG", 'JPEG']) #大写的.JPG是不允许的

#check type
def allowed_file(filename):
    return '.' in filename and filename.split('.', 1)[1] in ALLOWED_EXTENSIONS

#upload path
UPLOAD_FOLDER = './uploads'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """"
    目前只支持上传英文名
    """
    if request.method == 'POST':
        #获取上传文件
        file = request.files['file']
        #print(dir(file))
        #检查文件对象是否存在且合法
        if file and allowed_file(file.filename):  #哪里规定file都有什么属性
            filename = secure_filename(file.filename)  #把汉字文件名抹掉了，所以下面多一道检查
            if filename != file.filename:
               flash("only support ASCII name")
               return render_template('upload.html')
            #save
            try:
                file.save(os.path.join(UPLOAD_FOLDER, filename)) #现在似乎不会出现重复上传同名文件的问题
            except FileNotFoundError:
                os.mkdir(UPLOAD_FOLDER)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
            return redirect(url_for('update', fileName=filename))
        else:
            return 'Upload Failed'
    else: #GET方法
        return render_template('upload.html')


def render_photo_as_page(filename):
    """每次调用都将上传的图片复制到static中"""
    img = Image.open(os.path.join(UPLOAD_FOLDER, filename))  #上传文件夹和static分离
    img.save(os.path.join('./static/images', filename)) #这里要求jpg还是png都必须保存成png，因为html里是写死的
    #predict
    cv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    res = person_detector.detect(cv_img)
    img_res = yolox_api.visual_res(cv_img, res)
    filename_ext = os.path.splitext(filename)[-1]
    filename_res = filename.replace(filename_ext, '_out'+filename_ext)
    cv2.imwrite(os.path.join('./static/images', filename_res), img_res)
    result = {}
    result["num_person"] = len(res)
    return result, filename_res


@app.route('/upload/<path:fileName>', methods=['POST', 'GET'])
def update(fileName):
    """输入url加载图片，并返回预测值；上传图片，也会重定向到这里"""
    result, fileName_res = render_photo_as_page(fileName)
    return render_template('show.html', fname='images/'+fileName_res, result=result, ori_fname='images/'+fileName)


@app.route('/thanks', methods=['POST', 'GET'])
def thanks():
    """
    根据用户反馈的结果，把检测错误的图片保存下来
    """
    category = request.form["Correctness"]    #True or False
    fileName = request.form['ori_filename']
    # 把检测不正确的图片另外保存起来
    if category == 'Incorrect':
        src = './static/{}'.format(fileName)
        dst = './static/error'
        try:
            os.mkdir(dst)
        except FileExistsError:
            pass
        try:
            move(src, dst)  #上传同名照片会报错   #这里能在移动的同时重命名吗？
        except SError:
            flash("File arealdy exists or has the same name")
    return render_template('thanks.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
